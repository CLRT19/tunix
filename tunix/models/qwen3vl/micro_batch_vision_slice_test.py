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

"""CPU regression test for micro-batch slicing of packed vision metadata.

Reproduces (and guards against) a micro-batch slicing inconsistency in the
Qwen3-VL GRPO/GSPO trainer: gradient accumulation slices a packed
``VisionGridData`` per micro-batch and the per-token log-probs of the sliced
micro forward must match the full-batch forward (up to fp tolerance).

Run on CPU with a tiny random-weight model::

    JAX_PLATFORMS=cpu python -m tunix.models.qwen3vl.micro_batch_vision_slice_test

The bug is weight-INDEPENDENT (pure index arithmetic in how the packed vision
batch is sliced), so random weights are sufficient and a tiny config keeps it
fast. Non-uniform image sizes are essential: uniform images hide the offset
errors.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np
from flax import nnx

from tunix.models.qwen3vl import grpo_example
from tunix.models.qwen3vl import model as model_lib
from tunix.models.qwen3vl import vision as vision_lib


# ---------------------------------------------------------------------------
# Tiny model config. mrope_section sums to head_dim/2 (= 16). merge=2.
# ---------------------------------------------------------------------------
def _tiny_config() -> model_lib.ModelConfig:
  return model_lib.ModelConfig(
      num_layers=2,
      vocab_size=512,
      embed_dim=64,
      hidden_dim=128,
      num_heads=2,
      head_dim=32,
      num_kv_heads=1,
      norm_eps=1e-6,
      rope_theta=5_000_000,
      use_tied_embedding=False,
      param_dtype=jnp.float32,
      vision_config=vision_lib.VisionModelConfig(
          hidden_size=32,
          out_hidden_size=64,
          depth=4,
          num_heads=2,
          intermediate_size=64,
          patch_size=16,
          temporal_patch_size=2,
          spatial_merge_size=2,
          window_size=32,
          in_channels=3,
          num_position_embeddings=64,  # 8x8 learned grid
          deepstack_visual_indexes=(1, 2),
          mrope_section=(6, 5, 5),  # sums to 16 = head_dim/2
          image_pad_id=151655,
      ),
  )


IMAGE_PAD = 151655
VISION_START = 151654
IMAGE_TOKEN = IMAGE_PAD
VIDEO_TOKEN = 151656


def _build_batch(cfg: model_lib.ModelConfig, rng: np.random.Generator):
  """B=4 sequences, each with ONE image of a DIFFERENT (h,w) grid."""
  vc = cfg.vision_config
  merge = vc.spatial_merge_size
  patch_vol = vc.in_channels * vc.temporal_patch_size * vc.patch_size**2

  # Non-uniform grids (t=1, h, w multiples of merge), all different.
  grids = np.array(
      [[1, 4, 4], [1, 6, 4], [1, 4, 8], [1, 8, 6]], dtype=np.int32
  )
  B = grids.shape[0]

  vis_tokens = [int(t * h * w // (merge * merge)) for t, h, w in grids]
  text_pre = 3
  text_post = 5

  seq_lens = [text_pre + 1 + n + text_post for n in vis_tokens]
  L = max(seq_lens)

  input_tokens = np.zeros((B, L), dtype=np.int32)
  padding_mask = np.zeros((B, L), dtype=np.int32)
  completion_mask = np.zeros((B, L), dtype=np.int32)

  for i in range(B):
    n = vis_tokens[i]
    seq = []
    seq += list(rng.integers(5, 400, size=text_pre))
    seq.append(VISION_START)
    seq += [IMAGE_PAD] * n
    seq += list(rng.integers(5, 400, size=text_post - 1))
    li = len(seq)
    input_tokens[i, :li] = seq
    padding_mask[i, :li] = 1
    completion_mask[i, text_pre + 1 + n : li] = 1

  total_patches = int(np.sum(grids[:, 0] * grids[:, 1] * grids[:, 2]))
  pixel_values = jnp.asarray(
      rng.standard_normal((total_patches, patch_vol)), dtype=jnp.float32
  )

  vision_grid = vision_lib.compute_grid_data(grids, vc)

  positions, _ = model_lib.get_rope_index(
      jnp.asarray(input_tokens),
      image_grid_thw=jnp.asarray(grids),
      video_grid_thw=None,
      attention_mask=jnp.asarray(padding_mask),
      spatial_merge_size=merge,
      image_token_id=IMAGE_TOKEN,
      video_token_id=VIDEO_TOKEN,
      vision_start_token_id=VISION_START,
  )

  advantages = jnp.asarray(rng.standard_normal(B), dtype=jnp.float32)

  return dict(
      input_tokens=jnp.asarray(input_tokens),
      positions=jnp.asarray(positions),
      pixel_values=pixel_values,
      vision_grid=vision_grid,
      padding_mask=jnp.asarray(padding_mask),
      completion_mask=jnp.asarray(completion_mask),
      advantages=advantages,
      grids=grids,
      total_patches=total_patches,
  )


def _per_seq_completion_mismatch(full_logp, micro_logp, comp_mask_slice):
  # NB: image-pad token ids exceed the tiny vocab, so log-prob lookups at those
  # (non-completion) positions are NaN. Use jnp.where (NOT * mask) so a NaN at a
  # masked-out position cannot poison the sum (nan * 0 == nan).
  m = comp_mask_slice[:, 1:].astype(jnp.bool_)
  diff = jnp.where(m, jnp.abs(micro_logp - full_logp), 0.0)
  denom = jnp.clip(m.sum(-1).astype(jnp.float32), min=1.0)
  return np.asarray(diff.sum(-1) / denom)


def _run(replace_field=None):
  """Run full vs micro forward.

  replace_field: optional name of a VisionGridData field to overwrite in the
  micro slice with the *exact* full-grid rows (the bisection probe).
  """
  cfg = _tiny_config()
  rng = np.random.default_rng(0)
  model = model_lib.Qwen3VL(cfg, rngs=nnx.Rngs(params=0))
  batch = _build_batch(cfg, rng)

  B = batch["input_tokens"].shape[0]
  MICRO = 2
  n_accum = B // MICRO

  full_logp = grpo_example._compute_per_token_logps(
      model,
      input_tokens=batch["input_tokens"],
      positions=batch["positions"],
      pixel_values=batch["pixel_values"],
      vision_grid=batch["vision_grid"],
      padding_mask=batch["padding_mask"],
  )

  patch_offsets, pos_embed_offsets = (
      grpo_example._vision_patch_offsets_per_sequence(
          batch["vision_grid"],
          batch_size=B,
          total_patches=batch["total_patches"],
      )
  )

  results = []
  for mb in range(n_accum):
    s, e = mb * MICRO, mb * MICRO + MICRO
    patch_start = int(patch_offsets[s])
    patch_end = int(patch_offsets[e])
    pe_start = int(pos_embed_offsets[s])
    pe_end = int(pos_embed_offsets[e])

    micro_grid = grpo_example._slice_vision_grid(
        batch["vision_grid"],
        seq_start=s,
        seq_end=e,
        patch_start=patch_start,
        patch_end=patch_end,
        pos_embed_start=pe_start,
        pos_embed_end=pe_end,
    )

    if replace_field is not None:
      fg = batch["vision_grid"]
      exact = {
          "cos": fg.cos[patch_start:patch_end],
          "sin": fg.sin[patch_start:patch_end],
          "cu_seqlens": fg.cu_seqlens[s : e + 1] - fg.cu_seqlens[s],
          "pos_embed_idx": fg.pos_embed_idx[:, pe_start:pe_end],
          "pos_embed_weights": fg.pos_embed_weights[:, pe_start:pe_end],
          "pos_embed_gather": (
              fg.pos_embed_gather[patch_start:patch_end] - pe_start
          ),
      }
      micro_grid = micro_grid.replace(**{replace_field: exact[replace_field]})

    micro_logp = grpo_example._compute_per_token_logps(
        model,
        input_tokens=batch["input_tokens"][s:e],
        positions=batch["positions"][:, s:e, :],
        pixel_values=batch["pixel_values"][patch_start:patch_end],
        vision_grid=micro_grid,
        padding_mask=batch["padding_mask"][s:e],
    )

    mm = _per_seq_completion_mismatch(
        full_logp[s:e], micro_logp, batch["completion_mask"][s:e]
    )
    results.append(mm)
  return results


def test_micro_slice_matches_full():
  results = _run()
  worst = max(float(np.max(r)) for r in results)
  for mb, r in enumerate(results):
    print(f"mb={mb} per-seq mismatch={r.tolist()}")
  assert worst < 1e-3, f"micro-sliced logps diverge from full batch: {worst}"


if __name__ == "__main__":
  print("=== baseline (current slicing) ===")
  base = _run()
  for mb, r in enumerate(base):
    print(f"  mb={mb} per-seq mismatch={[round(float(x),4) for x in r]}")

  print("\n=== bisection: replace one micro-sliced field with exact rows ===")
  for field in [
      "cos",
      "sin",
      "cu_seqlens",
      "pos_embed_idx",
      "pos_embed_weights",
      "pos_embed_gather",
  ]:
    res = _run(replace_field=field)
    worst = max(float(np.max(r)) for r in res)
    flat = [round(float(np.max(r)), 4) for r in res]
    print(f"  replace {field:18s} worst={worst:.4f}  per-mb-max={flat}")
