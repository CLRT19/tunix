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

"""Vanilla rollout for Qwen3-VL using ``Qwen3VLSampler`` from PR #1177.

This wraps the existing single-host Qwen3-VL sampler in the
``BaseRollout`` interface so the GRPO learner can drive multimodal
rollouts without reaching into the sampler internals.

The wrapper accepts a per-batch image list via two channels:

  * The ``images`` kwarg on ``generate(...)``.
  * A ``set_pending_images(images)`` setter that the launcher / dataset
    pipe calls before invoking ``rl_cluster.generate()`` — needed because
    ``rl_cluster.generate`` itself only forwards ``prompts`` and not
    arbitrary kwargs through to the rollout.

For the smoke we run GRPO with ``num_iterations=1`` and ``beta=0`` so
``get_per_token_logps`` (which would need a multimodal forward pass) is
never called on this rollout; it raises ``NotImplementedError`` for now.
"""

from __future__ import annotations

import functools
import operator
from typing import Any, Optional, Sequence, Tuple

from flax import nnx
import jax
import jaxtyping
import numpy as np
from tunix.models.qwen3vl import sampler as qwen3vl_sampler_lib
from tunix.rl import reshard
from tunix.rl import utils
from tunix.rl.rollout import base_rollout


class Qwen3VLVanillaRollout(base_rollout.BaseRollout):
  """Vanilla rollout worker for Qwen3-VL.

  Wraps ``Qwen3VLSampler`` (PR #1177) and exposes the ``BaseRollout`` API.
  """

  def __init__(
      self,
      model: nnx.Module,
      processor: Any,
      cache_config_or_size: base_rollout.CacheConfig | int,
  ):
    cache_size = (
        cache_config_or_size.cache_size
        if isinstance(cache_config_or_size, base_rollout.CacheConfig)
        else int(cache_config_or_size)
    )
    self._sampler = qwen3vl_sampler_lib.Qwen3VLSampler(
        model, processor, cache_size=cache_size
    )
    self._processor = processor
    self._pending_images: Sequence[Any] | None = None

  # ------------------------------------------------------------------
  # Image staging (called by the launcher before rl_cluster.generate)
  # ------------------------------------------------------------------

  def set_pending_images(self, images: Sequence[Any] | None) -> None:
    """Stage the per-prompt image list for the next generate() call.

    Args:
      images: A list of PIL images, one per prompt in the upcoming batch,
        or ``None`` for a text-only batch.
    """
    self._pending_images = images

  def consume_pending_images(self) -> Sequence[Any] | None:
    images = self._pending_images
    self._pending_images = None
    return images

  # ------------------------------------------------------------------
  # BaseRollout interface
  # ------------------------------------------------------------------

  def generate(
      self,
      prompts: list[str],
      rollout_config: base_rollout.RolloutConfig,
      **kwargs,
  ) -> base_rollout.RolloutOutput:
    """Generate completions for ``prompts``, optionally with images."""
    images = kwargs.get('images')
    if images is None:
      images = self.consume_pending_images()
    if images is not None and len(images) != len(prompts):
      raise ValueError(
          f'Number of images ({len(images)}) must match number of prompts'
          f' ({len(prompts)}).'
      )

    top_p = rollout_config.top_p
    top_k = rollout_config.top_k
    temperature = rollout_config.temperature
    if temperature == 0.0:
      top_p = None
      top_k = None

    seed = rollout_config.seed
    if seed is None:
      seed_int = 0
    elif isinstance(seed, int):
      seed_int = seed
    else:  # jax.Array — fold into a Python int for the sampler's PRNGKey path.
      seed_int = int(np.array(jax.random.bits(seed, shape=(), dtype=jax.numpy.uint32)))

    out = self._sampler.generate_with_tokens(
        prompts=prompts,
        max_new_tokens=rollout_config.max_tokens_to_generate,
        images=list(images) if images is not None else None,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        eos_tokens=rollout_config.eos_tokens,
        seed=seed_int,
    )

    target_prompt_len = rollout_config.max_prompt_length
    prompt_tokens = out['prompt_tokens']
    if target_prompt_len > prompt_tokens.shape[1]:
      pad_value = self.pad_id()
      pad_width = target_prompt_len - prompt_tokens.shape[1]
      prompt_tokens = np.pad(
          prompt_tokens,
          ((0, 0), (pad_width, 0)),
          constant_values=pad_value,
      )
    elif target_prompt_len < prompt_tokens.shape[1]:
      prompt_tokens = prompt_tokens[:, -target_prompt_len:]

    return base_rollout.RolloutOutput(
        text=out['texts'],
        logits=None,
        tokens=out['completion_tokens'],
        left_padded_prompt_tokens=prompt_tokens,
        logprobs=None,
    )

  def get_per_token_logps(
      self,
      prompt_tokens: jax.Array,
      completion_tokens: jax.Array,
      completion_mask: jax.Array | None = None,
  ) -> jax.Array:
    raise NotImplementedError(
        'Qwen3VLVanillaRollout.get_per_token_logps is not implemented for'
        ' the smoke configuration. Run GRPO with num_iterations=1 and'
        ' beta=0 so this path is never exercised, or implement a'
        ' multimodal forward here.'
    )

  def update_params(
      self,
      params: jaxtyping.PyTree,
      filter_types: Optional[Tuple[Any, ...]] = None,
  ) -> None:
    if filter_types is not None:
      dst_params = nnx.state(self.model(), filter_types)
      resharded_params = reshard.reshard_pytree(params, dst_params)
    else:
      resharded_params = params
    flat_new_params, _ = utils.to_flat_dict(resharded_params)
    new_params_precision = jax.tree.leaves(flat_new_params)[0].dtype
    rollout_precision = jax.tree.leaves(
        self._sampler._flattened_model_state  # pylint: disable=protected-access
    )[0].dtype
    if new_params_precision != rollout_precision:
      flat_new_params = jax.tree.map(
          lambda x: x.astype(rollout_precision), flat_new_params
      )
    flat_old_params, tree_def = utils.to_flat_dict(
        self._sampler._flattened_model_state  # pylint: disable=protected-access
    )
    merged_params = functools.reduce(
        operator.ior, [flat_old_params, flat_new_params], {}
    )
    merged_params = jax.tree.unflatten(tree_def, merged_params.values())
    new_model = nnx.merge(
        self._sampler._model_graphdef,  # pylint: disable=protected-access
        merged_params,
    )
    self._sampler._flattened_model_state = jax.tree.leaves(  # pylint: disable=protected-access
        nnx.variables(new_model, nnx.Param),
        is_leaf=lambda x: isinstance(x, nnx.Variable),
    )

  def pad_id(self) -> int:
    tok = self._processor.tokenizer
    return tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

  def eos_id(self) -> int:
    return self._processor.tokenizer.eos_token_id

  def model(self) -> nnx.Module:
    return self._sampler._model  # pylint: disable=protected-access
