# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU investigation: B < fsdp micro-batch sharding for the Qwen3-VL trainer.

Root-cause probe for the residual GSPO importance-ratio explosion that survived
the packed-vision-slice fix (commit 1932d40). On the real TPU run the mesh is
(fsdp=8, tp=4), the rollout batch is B=8, and MICRO_BSZ=2. The model shards every
activation on the batch axis: ``ShardingConfig.act_btd = ('fsdp', None, 'tp')``
(model.py). A B=8 batch divides evenly across fsdp=8, but a B=2 micro-batch does
NOT — so the micro forward and the full forward use different batch layouts.

These tests demonstrate, on an 8-device CPU mesh (created via
``--xla_force_host_platform_device_count=8`` BEFORE importing jax):

  1. ``with_sharding_constraint(x, P('fsdp', None, 'tp'))`` with batch dim B
     divisible by fsdp matches the unsharded result (only bf16/fp32 rounding).
  2. The SAME constraint with B=2 < fsdp=8 raises ``IndivisibleError`` — the
     micro batch cannot be laid out on the fsdp axis at all. (Whatever a given
     XLA/JAX version does to "satisfy" an indivisible constraint — error, pad,
     or replicate — it necessarily differs from the divisible B=8 layout, which
     is precisely how full-vs-micro ``old != cur`` and the ratio blew up.)
  3. An exclusive-prefix ``cumsum`` over a *divisible* batch-sharded ``[B]``
     vector (the ``vis_offsets`` computation in model.py ``_inject`` /
     ``_apply_deepstack``) is itself correct — XLA inserts the cross-shard
     collective. cumsum is NOT the bug; indivisibility is.

Run::

    JAX_PLATFORMS=cpu python -m tunix.models.qwen3vl.micro_batch_fsdp_sharding_test
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
# 8 CPU "devices" so we can build an fsdp=8 mesh that mirrors the TPU slice.
# MUST be set before jax is imported.
_FLAGS = os.environ.get("XLA_FLAGS", "")
if "xla_force_host_platform_device_count" not in _FLAGS:
  os.environ["XLA_FLAGS"] = (
      _FLAGS + " --xla_force_host_platform_device_count=8"
  ).strip()

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


def _mesh(fsdp: int, tp: int) -> Mesh:
  assert fsdp * tp == len(jax.devices()), (fsdp, tp, len(jax.devices()))
  devs = np.array(jax.devices()).reshape(fsdp, tp)
  return Mesh(devs, axis_names=("fsdp", "tp"))


def _act_btd_forward(x, mesh):
  """Mirror model.shard(x, act_btd=('fsdp', None, 'tp')) inside a jit."""

  def f(v):
    v = jax.lax.with_sharding_constraint(
        v, NamedSharding(mesh, P("fsdp", None, "tp"))
    )
    return jnp.sum(v * v, axis=-1)

  with mesh:
    return jax.jit(f)(x)


def test_divisible_batch_sharding_matches_unsharded():
  """B divisible by fsdp: batch-sharded act forward == unsharded (rounding)."""
  D = 16
  x = jnp.asarray(
      np.random.default_rng(0).standard_normal((8, 4, D)), dtype=jnp.float32
  )
  ref = jnp.sum(x * x, axis=-1)
  for fsdp, tp in [(8, 1), (2, 4)]:
    out = _act_btd_forward(x, _mesh(fsdp, tp))
    worst = float(jnp.max(jnp.abs(out - ref)))
    print(f"[divisible] fsdp={fsdp} tp={tp} B=8 worst|diff|={worst:.3e}")
    assert worst < 1e-4, (fsdp, tp, worst)


def test_b2_over_fsdp8_is_indivisible():
  """B=2 micro-batch over fsdp=8 cannot be batch-sharded — the core defect.

  The model shards activations on ('fsdp', None, 'tp'); a B=2 micro slice over
  fsdp=8 is indivisible, so its forward layout is forced to differ from the
  divisible B=8 full forward. That layout difference is the residual source of
  iter-0 cur != old (now neutralized by computing `old` through the identical
  micro path; see micro_batch_vision_slice_test.test_old_via_micro_equals_cur).
  """
  D = 16
  x = jnp.asarray(
      np.random.default_rng(0).standard_normal((2, 4, D)), dtype=jnp.float32
  )
  raised = False
  try:
    _act_btd_forward(x, _mesh(8, 1))
  except Exception as ex:  # noqa: BLE001 - we assert on the type/text below
    raised = True
    print(f"[indivisible] B=2/fsdp=8 raised {type(ex).__name__}: {str(ex)[:120]}")
    assert "divis" in str(ex).lower() or "Indivisible" in type(ex).__name__, ex
  assert raised, (
      "expected B=2 over fsdp=8 to be rejected as indivisible; if a future JAX"
      " silently pads/replicates instead, the micro layout STILL differs from"
      " the B=8 layout — the old-via-micro fix is what makes this safe."
  )


def test_cumsum_over_divisible_batch_sharded_vector_is_correct():
  """vis_offsets = cumsum(vis_counts) - vis_counts over a sharded [B] vector.

  With B divisible by fsdp, the exclusive prefix sum is correct (XLA inserts the
  cross-shard collective) — so the offset arithmetic in _inject/_apply_deepstack
  is NOT the bug. Only the indivisible B<fsdp case is.
  """
  mesh = _mesh(8, 1)
  counts = jnp.array([3, 2, 4, 1, 5, 2, 3, 1], dtype=jnp.int32)  # B=8
  ref = jnp.cumsum(counts) - counts

  def f(c):
    c = jax.lax.with_sharding_constraint(c, NamedSharding(mesh, P("fsdp")))
    return jnp.cumsum(c) - c

  with mesh:
    out = jax.jit(f)(counts)
  print(f"[cumsum] exclusive-prefix sharded={np.asarray(out).tolist()}")
  assert bool(jnp.all(out == ref)), (np.asarray(out), np.asarray(ref))


if __name__ == "__main__":
  print(f"jax devices: {len(jax.devices())}")
  test_divisible_batch_sharding_matches_unsharded()
  test_b2_over_fsdp8_is_indivisible()
  test_cumsum_over_divisible_batch_sharded_vector_is_correct()
  print("all sharding-investigation checks passed")
