# Copyright 2025 Google LLC
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

"""vLLM rollout worker with Tunix sampler."""

import os
import pickle
import time
from typing import Any, Callable, Dict, Optional, Tuple

from absl import logging
from flax import nnx
import jax
from jax._src import distributed as jax_distributed
from jax.experimental import multihost_utils
import jaxtyping
import numpy as np
from tunix.generate import mappings
from tunix.generate import vllm_sampler
from tunix.rl.rollout import base_rollout


def _uses_global_vllm_mesh() -> bool:
  return os.environ.get("QWEN3VL_VLLM_GLOBAL_MESH", "").lower() in (
      "1",
      "true",
      "yes",
      "on",
  )


def _gather_global_rollout_inputs(
    prompts: list[str], images: list[Any] | None
) -> tuple[list[str], list[Any] | None, slice]:
  """Give every controller the same host-ordered global request batch.

  In-process vLLM keeps a Python scheduler on every JAX controller. A global
  TPU mesh therefore requires identical request order and multimodal values on
  every controller, even though GRPO's dataset is normally process-sharded.
  Serialize each local chunk, all-gather the variable-size payloads, and let
  the caller slice the global sampler output back to its original local rows.
  """
  process_count = jax.process_count()
  process_index = jax.process_index()
  local_count = len(prompts)
  counts = np.asarray(
      multihost_utils.process_allgather(
          np.asarray([local_count], dtype=np.int32), tiled=True
      )
  ).reshape(-1)
  if counts.shape != (process_count,) or not np.all(counts == local_count):
    raise ValueError(
        "Global vLLM requires the same local rollout batch size on every "
        f"process; process {process_index} saw counts={counts.tolist()}."
    )

  payload = pickle.dumps(
      (list(prompts), None if images is None else list(images)),
      protocol=pickle.HIGHEST_PROTOCOL,
  )
  lengths = np.asarray(
      multihost_utils.process_allgather(
          np.asarray([len(payload)], dtype=np.int64), tiled=True
      )
  ).reshape(-1)
  if lengths.shape != (process_count,) or np.any(lengths <= 0):
    raise ValueError(
        "Invalid global vLLM serialized payload lengths: "
        f"{lengths.tolist()}."
    )
  padded_length = int(lengths.max())
  local_bytes = np.zeros((padded_length,), dtype=np.uint8)
  local_bytes[: len(payload)] = np.frombuffer(payload, dtype=np.uint8)
  gathered = np.asarray(
      multihost_utils.process_allgather(local_bytes, tiled=True),
      dtype=np.uint8,
  ).reshape(process_count, padded_length)

  global_prompts: list[str] = []
  global_images: list[Any] | None = [] if images is not None else None
  for owner, length in enumerate(lengths.tolist()):
    owner_prompts, owner_images = pickle.loads(
        gathered[owner, :length].tobytes()
    )
    if len(owner_prompts) != local_count:
      raise ValueError(
          f"Global vLLM payload {owner} contained {len(owner_prompts)} "
          f"prompts; expected {local_count}."
      )
    if (owner_images is None) != (global_images is None):
      raise ValueError(
          "Global vLLM requires every process to agree on whether a rollout "
          "contains multimodal inputs."
      )
    if owner_images is not None and len(owner_images) != local_count:
      raise ValueError(
          f"Global vLLM payload {owner} contained {len(owner_images)} images; "
          f"expected {local_count}."
      )
    global_prompts.extend(owner_prompts)
    if global_images is not None:
      global_images.extend(owner_images)

  local_start = process_index * local_count
  return (
      global_prompts,
      global_images,
      slice(local_start, local_start + local_count),
  )


def _gather_non_addressable_leaf(value: Any) -> Any:
  """Fully replicate a global jax.Array before the mapped vLLM resharding.

  The trainer actor params are FSDP-sharded across hosts (not fully
  addressable). ``transfer_state_with_mappings`` reshards to the vLLM target
  sharding but never gathers, so non-addressable leaves must be all-gathered
  first. No-op for already-addressable arrays (single-host / replicated).
  """
  if isinstance(value, jax.Array) and not value.is_fully_addressable:
    return multihost_utils.process_allgather(value, tiled=True)
  return value


def _gather_non_addressable_params(params: jaxtyping.PyTree) -> jaxtyping.PyTree:
  return jax.tree.map(_gather_non_addressable_leaf, params)


def _wait_at_vllm_barrier(suffix: str, label: str) -> None:
  """Synchronize vLLM phases through the host coordination service."""
  if jax.process_count() == 1:
    return
  client = jax_distributed.global_state.client
  if client is None:
    raise RuntimeError(
        "Multi-host vLLM rollout has no JAX coordination client."
    )
  run_id = os.environ.get("QWEN3VL_RUN_ID") or os.environ.get("RUN_NAME")
  if not run_id:
    raise RuntimeError(
        "QWEN3VL_RUN_ID or RUN_NAME is required for the vLLM init barrier."
    )
  barrier_name = f"{run_id}_{suffix}"
  timeout_ms = int(
      os.environ.get("QWEN3VL_MODEL_LOAD_BARRIER_TIMEOUT_MS", "1800000")
  )
  started = time.perf_counter()
  logging.info(
      "[%s] entering %s proc=%d/%d",
      label,
      barrier_name,
      jax.process_index(),
      jax.process_count(),
  )
  client.wait_at_barrier(barrier_name, timeout_ms)
  logging.info(
      "[%s] exited %s proc=%d elapsed_sec=%.2f",
      label,
      barrier_name,
      jax.process_index(),
      time.perf_counter() - started,
  )


def _wait_for_all_actor_weights() -> None:
  """Finish the global actor gather before any host-local TPU execution."""
  _wait_at_vllm_barrier(
      "vllm_actor_weights_ready_before_engine_init",
      "vllm-weight-gather barrier",
  )


def _wait_for_all_vllm_engines() -> None:
  """Finish every host's independent local engine initialization."""
  _wait_at_vllm_barrier(
      "vllm_engine_ready_before_weight_load",
      "vllm-init barrier",
  )


def _wait_for_all_vllm_weight_loads() -> None:
  """Finish every local weight load before returning to global training."""
  _wait_at_vllm_barrier(
      "vllm_local_weights_ready_before_training",
      "vllm-weight-load barrier",
  )


def _initialize_vllm_sampler_in_waves(
    factory: Callable[[], vllm_sampler.VllmSampler],
) -> vllm_sampler.VllmSampler:
  """Initialize host-local TPU engines in coordination-service waves.

  This keeps multiple independent local PJRT executions from starting at the
  same instant across a large pod.  Inactive hosts wait only in the host-side
  coordination service, so they dispatch no TPU work while a wave initializes.
  """
  process_count = jax.process_count()
  if process_count == 1:
    return factory()
  if _uses_global_vllm_mesh():
    # A global TPU program requires every JAX controller to dispatch the same
    # operation. Initializing it in per-host waves would deadlock immediately.
    return factory()

  client = jax_distributed.global_state.client
  if client is None:
    raise RuntimeError(
        "Multi-host vLLM rollout has no JAX coordination client."
    )
  run_id = os.environ.get("QWEN3VL_RUN_ID") or os.environ.get("RUN_NAME")
  if not run_id:
    raise RuntimeError(
        "QWEN3VL_RUN_ID or RUN_NAME is required for vLLM init waves."
    )
  wave_size = int(os.environ.get("QWEN3VL_VLLM_INIT_WAVE_SIZE", "0"))
  if wave_size <= 0:
    wave_size = process_count
  wave_size = min(wave_size, process_count)
  timeout_ms = int(
      os.environ.get("QWEN3VL_MODEL_LOAD_BARRIER_TIMEOUT_MS", "1800000")
  )
  process_index = jax.process_index()
  sampler = None
  for wave_start in range(0, process_count, wave_size):
    wave_end = min(wave_start + wave_size, process_count)
    active = wave_start <= process_index < wave_end
    if active:
      logging.info(
          "[vllm-init wave] starting proc=%d wave=[%d,%d)",
          process_index,
          wave_start,
          wave_end,
      )
      sampler = factory()
      logging.info(
          "[vllm-init wave] initialized proc=%d wave=[%d,%d)",
          process_index,
          wave_start,
          wave_end,
      )
    barrier_name = f"{run_id}_vllm_init_wave_{wave_start}_{wave_end}"
    client.wait_at_barrier(barrier_name, timeout_ms)
    logging.info(
        "[vllm-init wave] barrier exited proc=%d wave=[%d,%d)",
        process_index,
        wave_start,
        wave_end,
    )
  if sampler is None:
    raise RuntimeError(
        f"Process {process_index} was not assigned a vLLM initialization wave."
    )
  return sampler


class VllmRollout(base_rollout.BaseRollout):
  """vLLM rollout worker."""

  def __init__(
      self,
      model: Any,
      tokenizer: Any,
      cache_config_or_size: base_rollout.CacheConfig | int,
      mesh: jax.sharding.Mesh,
      rollout_config: base_rollout.RolloutConfig,
  ):
    mapping_config = mappings.MappingConfig.build(
        mapping_obj=rollout_config.rollout_mapping_config,
        model=model,
        backend="vllm_jax",
    )
    sampler_config = vllm_sampler.VllmConfig(
            server_mode=rollout_config.rollout_vllm_server_mode,
            mapping_config=mapping_config,
            return_logprobs=rollout_config.return_logprobs,
            init_with_random_weights=rollout_config.rollout_vllm_init_with_random_weights,
            tpu_backend_type=rollout_config.rollout_vllm_tpu_backend_type,
            additional_config=rollout_config.rollout_vllm_additional_config,
            enable_dp_attention=rollout_config.rollout_vllm_enable_dp_attention,
            hbm_utilization=rollout_config.rollout_vllm_hbm_utilization,
            lora_config=rollout_config.rollout_vllm_lora_config,
            mesh=mesh,
            tensor_parallel_size=rollout_config.tensor_parallel_size,
            data_parallel_size=rollout_config.data_parallel_size,
            expert_parallel_size=rollout_config.expert_parallel_size,
            engine_kwargs={
                "model": rollout_config.rollout_vllm_model_version,
                "max_model_len": cache_config_or_size,
                "swap_space": rollout_config.rollout_vllm_swap_space_size_gb,
                "async_scheduling": (
                    rollout_config.rollout_vllm_async_scheduling
                ),
                "max_num_batched_tokens": (
                    rollout_config.rollout_vllm_max_num_batched_tokens
                ),
                "max_num_seqs": rollout_config.rollout_vllm_max_num_seqs,
                "hf_config_path": rollout_config.rollout_vllm_hf_config_path,
                "max_logprobs": (
                    1
                ),  # We only need the logprobs of the sampled tokens
                **rollout_config.rollout_vllm_kwargs,
            },
            sampling_kwargs=rollout_config.rollout_vllm_sampling_kwargs,
    )
    skip_initial_load = (
        os.environ.get("TUNIX_VLLM_SKIP_INITIAL_LOAD_CHECKPOINT") == "1"
    )
    gathered_actor_state = None
    if skip_initial_load:
      logging.warning(
          "Skipping initial actor->vLLM checkpoint load because "
          "TUNIX_VLLM_SKIP_INITIAL_LOAD_CHECKPOINT=1. This is intended for "
          "diagnostics only; rollout weights will remain at vLLM init state."
      )
    else:
      if _uses_global_vllm_mesh():
        # Both actor and rollout arrays span the same global device set, so
        # Tunix can reshard directly without first replicating 8B parameters
        # onto every host.
        logging.info(
            "[vllm-weight-gather] retaining global actor state for global "
            "vLLM reshard"
        )
        gathered_actor_state = nnx.state(model)
        jax.block_until_ready(gathered_actor_state)
      else:
        # This is the last global TPU collective before host-local vLLM work.
        # Local engines execute a host-dependent number of programs and
        # advance PJRT launch IDs independently. Gather to host memory first,
        # then all remaining initialization operations are strictly local.
        logging.info(
            "[vllm-weight-gather] gathering actor state before engine init"
        )
        gathered_actor_state = _gather_non_addressable_params(nnx.state(model))
      _wait_for_all_actor_weights()

    self._sampler = _initialize_vllm_sampler_in_waves(
        lambda: vllm_sampler.VllmSampler(
            tokenizer=tokenizer,
            config=sampler_config,
        )
    )
    # Initial actor->vLLM weight push. Gather FSDP-sharded (non-addressable)
    # params first — load_checkpoint goes straight to the sampler's
    # update_params (no allgather there), and reshard can't handle a
    # non-fully-addressable input.
    _wait_for_all_vllm_engines()
    if gathered_actor_state is not None:
      self._sampler.load_checkpoint(gathered_actor_state)
    _wait_for_all_vllm_weight_loads()
    self._pending_images = None

  @property
  def mesh(self) -> jax.sharding.Mesh:
    return self._sampler.mesh

  def set_pending_images(self, images) -> None:
    """Stage per-prompt multimodal inputs for the next generate() call.

    Mirrors ``Qwen3VLVanillaRollout``: ``rl_cluster.generate`` forwards only
    ``prompts``, so multimodal inputs (e.g. PIL images, one per prompt) are
    staged here by the launcher/dataset pipe before ``generate`` runs. Pass
    ``None`` for a text-only batch.
    """
    self._pending_images = images

  def consume_pending_images(self):
    # Some callers construct a rollout with ``__new__`` for lightweight
    # contract tests, and older checkpoints may restore an object created
    # before pending multimodal inputs were introduced. Treat the missing
    # attribute the same as an empty pending-image slot.
    images = getattr(self, "_pending_images", None)
    self._pending_images = None
    return images

  def generate(
      self,
      prompts: list[str],
      rollout_config: base_rollout.RolloutConfig,
      **kwargs,
  ) -> base_rollout.RolloutOutput:
    """Generates samples from the model."""
    sampling_kwargs = dict(kwargs)
    images = sampling_kwargs.pop("images", None)
    if images is None:
      images = self.consume_pending_images()
    if images is not None and len(images) != len(prompts):
      raise ValueError(
          f"Number of images ({len(images)}) must match number of prompts "
          f"({len(prompts)})."
      )
    local_slice = slice(None)
    gathered_global_batch = False
    if _uses_global_vllm_mesh() and jax.process_count() > 1:
      prompts, images, local_slice = _gather_global_rollout_inputs(
          prompts, None if images is None else list(images)
      )
      gathered_global_batch = True
    if rollout_config.eos_tokens is not None:
      sampling_kwargs["stop_token_ids"] = rollout_config.eos_tokens

    global_output = self._sampler(
        input_strings=prompts,
        max_generation_steps=rollout_config.max_tokens_to_generate,
        max_prompt_length=rollout_config.max_prompt_length,
        temperature=rollout_config.temperature,
        top_p=rollout_config.top_p,
        top_k=rollout_config.top_k,
        # The vLLM JAX/TPU backend rejects a per-request seed
        # (tpu_platform.validate_request). Stochastic sampling at temperature>0
        # still varies completions across steps via vLLM's own RNG, so GRPO
        # exploration is preserved.
        seed=None,
        echo=False,
        pad_output=True,
        images=list(images) if images is not None else None,
        **sampling_kwargs,
    )
    if gathered_global_batch:
      self.output = type(global_output)(
          text=global_output.text[local_slice],
          logits=(
              None
              if global_output.logits is None
              else global_output.logits[local_slice]
          ),
          tokens=global_output.tokens[local_slice],
          padded_prompt_tokens=global_output.padded_prompt_tokens[local_slice],
          logprobs=(
              None
              if global_output.logprobs is None
              else global_output.logprobs[local_slice]
          ),
      )
    else:
      self.output = global_output

    output = base_rollout.RolloutOutput(
        text=self.output.text,
        logits=None,
        tokens=self.output.tokens,
        left_padded_prompt_tokens=self.output.padded_prompt_tokens,
        logprobs=self.output.logprobs,
        # Forced/padded prompt width — the trainer asserts this is uniform
        # across hosts (SPMD). Matches vanilla rollout's contract.
        prompt_seq_len=int(self.output.padded_prompt_tokens.shape[1]),
    )
    # RolloutOutput predates termination metadata, but standalone evaluation
    # needs to distinguish EOS from max-token truncation. Preserve it as a
    # backward-compatible dynamic attribute, matching Qwen3VLRolloutOutput.
    terminated = getattr(self._sampler, "last_terminated", None)
    if terminated:
      output.terminated = terminated[0][local_slice]
    return output

  def get_per_token_logps(
      self,
      prompt_tokens: jax.Array,
      completion_tokens: jax.Array,
      completion_mask: jax.Array | None = None,
  ) -> jax.Array:
    """Returns per-token log probabilities from the rollout policy."""
    # b/428730696, we cannot return self.output.logprobs yet
    # May need to validate if there will be any difference from recalculation
    return self.output.logprobs

  def update_params(
      self,
      params: jaxtyping.PyTree,
      filter_types: Optional[Tuple[Any, ...]] = None,
  ) -> None:
    # Gather FSDP-sharded actor params to fully-replicated before the mapped
    # sync (transfer_state_with_mappings reshards but never gathers). No-op
    # when params are already addressable.
    if not _uses_global_vllm_mesh():
      params = _gather_non_addressable_params(params)
    self._sampler.update_params(params, filter_types)

  def sync_from_actor(self, actor_params: jaxtyping.PyTree) -> None:
    """Push fresh actor weights into the vLLM engine (called each train step).

    Mirrors Qwen3VLVanillaRollout.sync_from_actor so the standalone GRPO loop
    can drive either engine identically.
    """
    self.update_params(actor_params, filter_types=nnx.Param)

  def pad_id(self) -> int:
    return self._sampler.tokenizer.pad_id()

  def eos_id(self) -> int:
    return self._sampler.tokenizer.eos_id()

  def model(self) -> nnx.Module:
    return self._sampler.transformer
