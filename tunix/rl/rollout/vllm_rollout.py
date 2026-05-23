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

from typing import Any, Dict, Optional, Tuple

from flax import nnx
import jax
from jax.experimental import multihost_utils
import jaxtyping
from tunix.generate import mappings
from tunix.generate import vllm_sampler
from tunix.rl.rollout import base_rollout


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
    self._sampler = vllm_sampler.VllmSampler(
        tokenizer=tokenizer,
        config=vllm_sampler.VllmConfig(
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
        ),
    )
    # Initial actor->vLLM weight push. Gather FSDP-sharded (non-addressable)
    # params first — load_checkpoint goes straight to the sampler's
    # update_params (no allgather there), and reshard can't handle a
    # non-fully-addressable input.
    state = _gather_non_addressable_params(nnx.state(model))
    self._sampler.load_checkpoint(state)
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
    images = self._pending_images
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
    if rollout_config.eos_tokens is not None:
      sampling_kwargs["stop_token_ids"] = rollout_config.eos_tokens

    self.output = self._sampler(
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

    return base_rollout.RolloutOutput(
        text=self.output.text,
        logits=None,
        tokens=self.output.tokens,
        left_padded_prompt_tokens=self.output.padded_prompt_tokens,
        logprobs=self.output.logprobs,
        # Forced/padded prompt width — the trainer asserts this is uniform
        # across hosts (SPMD). Matches vanilla rollout's contract.
        prompt_seq_len=int(self.output.padded_prompt_tokens.shape[1]),
    )

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
