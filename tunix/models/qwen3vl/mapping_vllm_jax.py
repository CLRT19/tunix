# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""vLLM JAX backend weight mappings for the dense Qwen3-VL model.

This maps the tunix vanilla Qwen3-VL nnx parameter tree (the GRPO actor) onto the
flat parameter paths of the tpu-inference `Qwen3VLForConditionalGeneration` JAX
model that vLLM serves, so the trainer can push fresh weights into the vLLM
rollout engine each step (`VllmSampler.update_params` ->
`utils.transfer_state_with_mappings`).

Source keys are tunix nnx param paths (see `qwen3vl/params.py`, which loads HF
weights *into* these names). Target keys are the tpu-inference model's flat state
paths (see `tpu-inference/tpu_inference/models/jax/qwen3_vl.py:load_weights`,
the HF->jax `mappings` dict). The two trees were cross-checked leaf-by-leaf:

  * Text attention projections share an identical head-split layout on both
    sides -- q `[D, N, H]`, k/v `[D, K, H]`, o `[N, H, D]` -- so NO transpose is
    needed. (Any TP head-dim padding on the vLLM side is repaired by
    `_align_shape`, which `update_params` feeds `num_kv_heads`/`head_dim`.)
  * All vision linears are stored `[in, out]` on both sides -> no transpose.
  * `embedder.input_embedding` and `visual.pos_embed.embedding` are `[rows, hidden]`
    on both sides -> no transpose (so `to_hf_transpose_keys` stays None; a blanket
    {'embedding': (1, 0)} like the text-only Qwen3 map would wrongly hit BOTH).
  * Several vision submodules are named differently between the trees; the map
    bridges them: out_proj->proj, linear1/2->fc1/2, deepstack_mergers->
    deepstack_merger_list.

The single genuine layout transform is the vision patch embedding: tunix keeps it
as a flattened linear kernel `[C*T*P*P, hidden]`, while tpu-inference wants a 3D
`nnx.Conv` kernel `[T, P, P, C, hidden]`. `_patch_embed_to_conv5d` performs that
reshape+transpose; it is registered as a `to_hf_hook_fn` keyed on the tunix source
path and runs before `_align_shape`.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import jax.numpy as jnp

Sharding = Tuple[str | None, ...]
MappingEntry = Tuple[str, Sharding]

# Qwen3-VL fixed patchify geometry (RGB, temporal patch 2, spatial patch 16).
# These are architecture constants shared by the 4B and 8B variants.
_PATCH_IN_CHANNELS = 3
_PATCH_TEMPORAL = 2
_PATCH_SIZE = 16


def _patch_embed_to_conv5d(v: jnp.ndarray) -> jnp.ndarray:
  """Reshape tunix's flat patch-embed kernel into a tpu-inference Conv3D kernel.

  tunix stores `visual.patch_embed.proj.kernel` as a flattened linear weight of
  shape `[C*T*P*P, hidden]`, row-major over (in_channels, temporal_patch,
  patch_h, patch_w) -- the inverse of the HF Conv3D weight `[out, in, T, P, P]`
  permuted `(1, 2, 3, 4, 0)` then flattened (see `qwen3vl/params.py`).

  tpu-inference's `nnx.Conv(in=C, out=hidden, kernel=(T, P, P))` expects a
  channels-last kernel `[T, P, P, C, hidden]` (its `load_weights` applies
  transpose `(2, 3, 4, 1, 0)` to the HF Conv3D weight to reach the same layout).

  So: un-flatten to `[C, T, P, P, hidden]`, then move the leading in-channels
  axis to position 3 -> `[T, P, P, C, hidden]`.
  """
  c, t, p = _PATCH_IN_CHANNELS, _PATCH_TEMPORAL, _PATCH_SIZE
  flat, hidden = v.shape
  assert flat == c * t * p * p, (
      f'patch_embed kernel first dim {flat} != C*T*P*P={c * t * p * p}; '
      'Qwen3-VL patchify geometry assumption violated -- check vision_config.'
  )
  v = v.reshape(c, t, p, p, hidden)  # [in, T, P, P, out]
  return jnp.transpose(v, (1, 2, 3, 0, 4))  # -> [T, P, P, in, out]


def _to_hf_mappings() -> Dict[str, MappingEntry]:
  """tunix nnx param path -> (tpu-inference flat state path, sharding).

  The sharding tuple is informational here: `transfer_state_with_mappings`
  reshards to the *live* target sharding, and `build_flat_dict` only inspects
  this tuple for the literal 'layer' axis (scanned stacking) -- which the
  tpu-inference model never uses (its layers/blocks are unrolled nnx.List).
  Values below name the TP-sharded ('model') axis for readability only.
  """
  m: Dict[str, MappingEntry] = {
      # ----- text decoder -----
      'embedder.input_embedding': (
          'language_model.embed_tokens.weight',
          ('model', None),
      ),
      'layers.*.input_layernorm.w': (
          'language_model.layers.*.input_layernorm.weight',
          (None,),
      ),
      'layers.*.post_attention_layernorm.w': (
          'language_model.layers.*.post_attention_layernorm.weight',
          (None,),
      ),
      'layers.*.attn.q_proj.w': (
          'language_model.layers.*.self_attn.q_proj.weight',
          (None, 'model', None),
      ),
      'layers.*.attn.k_proj.w': (
          'language_model.layers.*.self_attn.k_proj.weight',
          (None, 'model', None),
      ),
      'layers.*.attn.v_proj.w': (
          'language_model.layers.*.self_attn.v_proj.weight',
          (None, 'model', None),
      ),
      'layers.*.attn.o_proj.w': (
          'language_model.layers.*.self_attn.o_proj.weight',
          ('model', None, None),
      ),
      'layers.*.attn.q_norm.w': (
          'language_model.layers.*.self_attn.q_norm.weight',
          (None,),
      ),
      'layers.*.attn.k_norm.w': (
          'language_model.layers.*.self_attn.k_norm.weight',
          (None,),
      ),
      'layers.*.mlp.gate_proj.kernel': (
          'language_model.layers.*.mlp.gate_proj.weight',
          (None, 'model'),
      ),
      'layers.*.mlp.up_proj.kernel': (
          'language_model.layers.*.mlp.up_proj.weight',
          (None, 'model'),
      ),
      'layers.*.mlp.down_proj.kernel': (
          'language_model.layers.*.mlp.down_proj.weight',
          ('model', None),
      ),
      'final_norm.w': ('language_model.norm.weight', (None,)),
      'lm_head.w': ('lm_head.weight', (None, 'model')),
      # ----- vision: patch + position embeddings -----
      'visual.patch_embed.proj.kernel': (
          'visual.patch_embed.proj.kernel',
          (None, None, None, None, 'model'),
      ),
      'visual.patch_embed.proj.bias': (
          'visual.patch_embed.proj.bias',
          ('model',),
      ),
      'visual.pos_embed.embedding': (
          'visual.pos_embed.embedding',
          (None, 'model'),
      ),
      # ----- vision: transformer blocks -----
      'visual.blocks.*.attn.qkv_proj.kernel': (
          'visual.blocks.*.attn.qkv_proj.kernel',
          (None, 'model'),
      ),
      'visual.blocks.*.attn.qkv_proj.bias': (
          'visual.blocks.*.attn.qkv_proj.bias',
          ('model',),
      ),
      'visual.blocks.*.attn.out_proj.kernel': (
          'visual.blocks.*.attn.proj.kernel',
          ('model', None),
      ),
      'visual.blocks.*.attn.out_proj.bias': (
          'visual.blocks.*.attn.proj.bias',
          (None,),
      ),
      'visual.blocks.*.mlp.linear1.kernel': (
          'visual.blocks.*.mlp.fc1.kernel',
          (None, 'model'),
      ),
      'visual.blocks.*.mlp.linear1.bias': (
          'visual.blocks.*.mlp.fc1.bias',
          ('model',),
      ),
      'visual.blocks.*.mlp.linear2.kernel': (
          'visual.blocks.*.mlp.fc2.kernel',
          ('model', None),
      ),
      'visual.blocks.*.mlp.linear2.bias': (
          'visual.blocks.*.mlp.fc2.bias',
          (None,),
      ),
      'visual.blocks.*.norm1.scale': (
          'visual.blocks.*.norm1.scale',
          (None,),
      ),
      'visual.blocks.*.norm1.bias': (
          'visual.blocks.*.norm1.bias',
          (None,),
      ),
      'visual.blocks.*.norm2.scale': (
          'visual.blocks.*.norm2.scale',
          (None,),
      ),
      'visual.blocks.*.norm2.bias': (
          'visual.blocks.*.norm2.bias',
          (None,),
      ),
      # ----- vision: deepstack mergers (tunix deepstack_mergers -> vLLM deepstack_merger_list) -----
      'visual.deepstack_mergers.*.norm.scale': (
          'visual.deepstack_merger_list.*.norm.scale',
          (None,),
      ),
      'visual.deepstack_mergers.*.norm.bias': (
          'visual.deepstack_merger_list.*.norm.bias',
          (None,),
      ),
      'visual.deepstack_mergers.*.linear_fc1.kernel': (
          'visual.deepstack_merger_list.*.linear_fc1.kernel',
          (None, 'model'),
      ),
      'visual.deepstack_mergers.*.linear_fc1.bias': (
          'visual.deepstack_merger_list.*.linear_fc1.bias',
          ('model',),
      ),
      'visual.deepstack_mergers.*.linear_fc2.kernel': (
          'visual.deepstack_merger_list.*.linear_fc2.kernel',
          ('model', None),
      ),
      'visual.deepstack_mergers.*.linear_fc2.bias': (
          'visual.deepstack_merger_list.*.linear_fc2.bias',
          (None,),
      ),
      # ----- vision: final merger -----
      'visual.merger.norm.scale': (
          'visual.merger.norm.scale',
          (None,),
      ),
      'visual.merger.norm.bias': (
          'visual.merger.norm.bias',
          (None,),
      ),
      'visual.merger.linear_fc1.kernel': (
          'visual.merger.linear_fc1.kernel',
          (None, 'model'),
      ),
      'visual.merger.linear_fc1.bias': (
          'visual.merger.linear_fc1.bias',
          ('model',),
      ),
      'visual.merger.linear_fc2.kernel': (
          'visual.merger.linear_fc2.kernel',
          ('model', None),
      ),
      'visual.merger.linear_fc2.bias': (
          'visual.merger.linear_fc2.bias',
          (None,),
      ),
  }
  return m


def _to_hf_hook_fns() -> Dict[str, Any] | None:
  """Per-source-key value transforms applied during weight sync."""
  return {
      'visual.patch_embed.proj.kernel': _patch_embed_to_conv5d,
  }


VLLM_JAX_MAPPING: Dict[str, Any] = {
    'to_hf_mappings': _to_hf_mappings(),
    # LoRA sync not supported for the Qwen3-VL GRPO actor (full fine-tune).
    'lora_to_hf_mappings': None,
    # No blanket transposes: every non-patch-embed leaf shares layout across the
    # two trees (see module docstring).
    'to_hf_transpose_keys': None,
    'to_hf_hook_fns': _to_hf_hook_fns(),
}

__all__ = [
    'VLLM_JAX_MAPPING',
]
