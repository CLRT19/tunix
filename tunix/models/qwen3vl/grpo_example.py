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

"""GRPO smoke for Qwen3-VL on ChartQA.

Standalone training loop modeled on ``train_example.py`` (SFT). Bypasses
``rl_cluster`` / ``PeftTrainer`` because their model-call signatures
assume the text-only Qwen3 interface; here we call Qwen3VL with its
native ``(input_tokens, positions_3d, pixel_values, vision_grid, cache,
padding_mask)`` signature.

Flow per step:
  1. Sample ``NUM_PROMPTS`` prompts (image + question + label).
  2. Tile to G generations per prompt.
  3. Roll out completions via Qwen3VLVanillaRollout (wraps Qwen3VLSampler).
  4. Score completions with chartqa.check_answer (string match + ±1%).
  5. Compute group-relative advantages.
  6. Re-tokenize prompt+completion as a multi-turn conversation via
     encode_messages (gives 3D M-RoPE positions and the completion mask).
  7. JIT grad step on ``-advantages * per_token_logps``, masked to
     completion tokens.

Designed for the smoke configuration: ``num_iterations=1``, ``beta=0`` —
no reference model, no clipping, no KL. We can layer those in once the
end-to-end loop is green.

Usage::

    python -m tunix.models.qwen3vl.grpo_example
"""

from __future__ import annotations

import logging
import os
from typing import Any

import datasets
from flax import nnx
from grain import python as grain
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from transformers import AutoProcessor
from tunix.cli.reward_fn import chartqa as chartqa_reward
from tunix.models.qwen3vl import model as model_lib
from tunix.models.qwen3vl import params as params_lib
from tunix.models.qwen3vl.train_example import _maybe_initialize_jax_distributed
from tunix.models.qwen3vl.train_example import resolve_model_dir
from tunix.models.qwen3vl.utils import encode_messages
from tunix.models.qwen3vl.utils import load_processor
from tunix.models.qwen3vl.vision import VisionGridData
from tunix.rl.rollout import base_rollout
from tunix.rl.rollout import qwen3vl_vanilla_rollout
from tunix.sft.utils import show_hbm_usage

logging.basicConfig(level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hyperparameters (env-overridable)
# ---------------------------------------------------------------------------

MODEL_ID = os.environ.get('QWEN3VL_MODEL_DIR', 'Qwen/Qwen3-VL-4B-Instruct')
DATASET_ID = os.environ.get('QWEN3VL_DATASET', 'HuggingFaceM4/ChartQA')
DATASET_SPLIT = os.environ.get('QWEN3VL_DATASET_SPLIT', 'train')

NUM_PROMPTS = int(os.environ.get('QWEN3VL_NUM_PROMPTS', '2'))
NUM_GENERATIONS = int(os.environ.get('QWEN3VL_NUM_GENERATIONS', '4'))
MAX_NEW_TOKENS = int(os.environ.get('QWEN3VL_MAX_NEW_TOKENS', '128'))
MAX_SEQ_LEN = int(os.environ.get('QWEN3VL_MAX_SEQ_LEN', '1536'))
ROLLOUT_CACHE_SIZE = int(os.environ.get('QWEN3VL_ROLLOUT_CACHE_SIZE', '1536'))
MAX_IMAGE_SIZE = int(os.environ.get('QWEN3VL_MAX_IMAGE_SIZE', '512'))

LEARNING_RATE = float(os.environ.get('QWEN3VL_LR', '1e-6'))
MAX_STEPS = int(os.environ.get('QWEN3VL_MAX_STEPS', '4'))
CKPT_EVERY_N_STEPS = int(os.environ.get('QWEN3VL_CKPT_EVERY_N_STEPS', '2'))

CKPT_DIR = os.environ.get(
    'QWEN3VL_CKPT_DIR', '/tmp/qwen3vl_grpo_chartqa_ckpts'
)
MESH_SHAPE = tuple(
    int(x) for x in os.environ.get('QWEN3VL_MESH_SHAPE', '1,1').split(',')
)

SYSTEM_PROMPT = os.environ.get(
    'QWEN3VL_SYSTEM_PROMPT',
    'You are given a chart image and a question about it. Think step by '
    'step inside <think>...</think> tags, then give only the final answer '
    'inside <answer>...</answer> tags.',
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class _PrepareChartQA(grain.MapTransform):
  """Convert one ChartQA row to a dict of (prompt_text, image, label)."""

  def map(self, element: dict[str, Any]) -> dict[str, Any]:
    image = element['image'].convert('RGB')
    # Fixed square resize: required under multi-host so every JAX process
    # produces the same vision-patch count and the SPMD JIT sees one graph.
    image = image.resize((MAX_IMAGE_SIZE, MAX_IMAGE_SIZE))
    label = element['label']
    if isinstance(label, str):
      label = [label]
    return {
        'image': image,
        'question': str(element['query']),
        'label': list(label),
    }


def create_dataset() -> grain.DataLoader:
  hf_ds = datasets.load_dataset(DATASET_ID, split=DATASET_SPLIT)
  shard_options = (
      grain.ShardByJaxProcess(drop_remainder=True)
      if jax.process_count() > 1
      else grain.NoSharding()
  )
  return grain.DataLoader(
      data_source=hf_ds,
      sampler=grain.IndexSampler(
          num_records=len(hf_ds),
          num_epochs=1000,  # effectively infinite for a smoke
          shard_options=shard_options,
      ),
      operations=[_PrepareChartQA()],
      worker_count=0,  # AutoProcessor is not fork-safe.
  )


# ---------------------------------------------------------------------------
# Prompt + rollout helpers
# ---------------------------------------------------------------------------


def _build_prompt_conversation(question: str, image) -> list[dict[str, Any]]:
  """Build the conversation up to (but not including) the assistant turn.

  ``apply_chat_template(..., add_generation_prompt=True)`` adds the
  trailing ``<|im_start|>assistant\n`` so the sampler decodes the
  assistant response.
  """
  return [
      {'role': 'system', 'content': SYSTEM_PROMPT},
      {
          'role': 'user',
          'content': [
              {'type': 'image', 'image': image},
              {'type': 'text', 'text': question},
          ],
      },
  ]


def _apply_chat_template(processor: AutoProcessor, question: str) -> str:
  """Render the prompt as a single string with <|image_pad|> placeholder.

  The image is wired in later via the sampler's image processor.
  """
  return processor.apply_chat_template(
      [
          {'role': 'system', 'content': SYSTEM_PROMPT},
          {
              'role': 'user',
              'content': [
                  {'type': 'image'},
                  {'type': 'text', 'text': question},
              ],
          },
      ],
      tokenize=False,
      add_generation_prompt=True,
  )


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def grpo_loss_fn(
    model: model_lib.Qwen3VL,
    input_tokens: jax.Array,  # [B, L]
    positions: jax.Array,  # [3, B, L]
    pixel_values: jax.Array,  # [P, C]
    vision_grid: VisionGridData,
    padding_mask: jax.Array,  # [B, L]
    completion_mask: jax.Array,  # [B, L] — 1 where loss applies
    advantages: jax.Array,  # [B]
) -> jax.Array:
  """Advantage-weighted per-token NLL on completion tokens.

  Single-iteration GRPO with no KL: loss = -E[A * log pi(a|s)], masked to
  completion tokens. Equivalent to REINFORCE with group-relative baseline.
  """
  logits, _ = model(
      input_tokens,
      positions,
      pixel_values,
      vision_grid,
      cache=None,
      padding_mask=padding_mask,
  )
  logits = logits[:, :-1, :].astype(jnp.float32)  # [B, L-1, V]
  targets = input_tokens[:, 1:]  # [B, L-1]
  log_probs = jax.nn.log_softmax(logits, axis=-1)
  per_token_logps = jnp.take_along_axis(
      log_probs, targets[..., None], axis=-1
  )[..., 0]  # [B, L-1]

  mask = completion_mask[:, 1:].astype(jnp.float32)  # [B, L-1]
  per_token_loss = -jnp.expand_dims(advantages, 1) * per_token_logps
  denom = jnp.clip(jnp.sum(mask), min=1.0)
  return jnp.sum(per_token_loss * mask) / denom


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------


def _train_step(
    model: model_lib.Qwen3VL,
    optimizer: nnx.Optimizer,
    batch_kwargs: dict,
    advantages: jax.Array,
):
  """Single grad step. Mutates `model` and `optimizer` in place."""

  def loss_only(m):
    return grpo_loss_fn(m, **batch_kwargs, advantages=advantages)

  loss, grads = nnx.value_and_grad(loss_only)(model)
  optimizer.update(model, grads)
  return loss


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
  _maybe_initialize_jax_distributed()
  jax.config.update('jax_compilation_cache_dir', '/tmp/jax_cache')
  jax.config.update('jax_explain_cache_misses', True)
  is_primary = jax.process_index() == 0
  if is_primary:
    os.makedirs(CKPT_DIR, exist_ok=True)

  # --- Mesh + model ---
  config = model_lib.ModelConfig.qwen3vl_4b()
  config.remat_config = model_lib.RematConfig.BLOCK
  model_dir = resolve_model_dir(MODEL_ID)
  logger.info(
      'Loading Qwen3-VL from %s on mesh %s (proc %d/%d)',
      model_dir,
      MESH_SHAPE,
      jax.process_index(),
      jax.process_count(),
  )
  if jax.process_count() > 1 or any(s > 1 for s in MESH_SHAPE):
    devices = np.asarray(jax.devices(), dtype=object).reshape(tuple(MESH_SHAPE))
    mesh = jax.sharding.Mesh(devices, ('fsdp', 'tp'))
  else:
    mesh = jax.make_mesh(MESH_SHAPE, ('fsdp', 'tp'))
  model = params_lib.create_model_from_safe_tensors(
      model_dir, config, mesh=mesh, dtype=jnp.bfloat16
  )
  show_hbm_usage()

  processor = load_processor(model_dir)

  # --- Rollout adapter ---
  rollout = qwen3vl_vanilla_rollout.Qwen3VLVanillaRollout(
      model=model,
      processor=processor,
      cache_config_or_size=ROLLOUT_CACHE_SIZE,
  )
  rollout_config = base_rollout.RolloutConfig(
      max_tokens_to_generate=MAX_NEW_TOKENS,
      max_prompt_length=MAX_SEQ_LEN - MAX_NEW_TOKENS,
      temperature=1.0,
      top_p=0.95,
  )

  # --- Optimizer ---
  optimizer = nnx.Optimizer(
      model, optax.adamw(LEARNING_RATE), wrt=nnx.Param
  )

  # --- Dataset ---
  data_iter = iter(create_dataset())

  # --- Checkpoint manager (process-0 writes; others participate in collective save) ---
  ckpt_mgr = ocp.CheckpointManager(
      directory=CKPT_DIR,
      options=ocp.CheckpointManagerOptions(
          save_interval_steps=CKPT_EVERY_N_STEPS,
          max_to_keep=3,
      ),
  )

  for step in range(1, MAX_STEPS + 1):
    # 1. Sample NUM_PROMPTS prompts on each process.
    batch = [next(data_iter) for _ in range(NUM_PROMPTS)]

    # 2. Tile to NUM_GENERATIONS per prompt.
    questions = [b['question'] for b in batch for _ in range(NUM_GENERATIONS)]
    images = [b['image'] for b in batch for _ in range(NUM_GENERATIONS)]
    labels = [b['label'] for b in batch for _ in range(NUM_GENERATIONS)]
    prompt_strs = [_apply_chat_template(processor, q) for q in questions]

    # 3. Rollout.
    logger.info('[step %d] rolling out %d completions', step, len(prompt_strs))
    rollout_out = rollout.generate(
        prompts=prompt_strs,
        rollout_config=rollout_config,
        images=images,
    )

    # 4. Score.
    rewards = np.array(
        chartqa_reward.check_answer(
            prompts=questions,
            completions=rollout_out.text,
            label=labels,
        ),
        dtype=np.float32,
    )

    # 5. Group-relative advantages: (r - mean) / (std + eps).
    grouped = rewards.reshape(NUM_PROMPTS, NUM_GENERATIONS)
    mean = grouped.mean(axis=-1, keepdims=True)
    std = grouped.std(axis=-1, ddof=1, keepdims=True)
    advantages_np = ((grouped - mean) / (std + 1e-4)).reshape(-1)

    logger.info(
        '[step %d] rewards mean=%.3f min=%.3f max=%.3f advantages |range|=[%.3f, %.3f]',
        step,
        float(rewards.mean()),
        float(rewards.min()),
        float(rewards.max()),
        float(advantages_np.min()),
        float(advantages_np.max()),
    )

    # 6. Build training inputs: prompt + completion conversations.
    conversations = []
    for q, img, comp in zip(questions, images, rollout_out.text):
      conversations.append([
          {'role': 'system', 'content': SYSTEM_PROMPT},
          {
              'role': 'user',
              'content': [
                  {'type': 'image', 'image': img},
                  {'type': 'text', 'text': q},
              ],
          },
          {'role': 'assistant', 'content': comp},
      ])
    encoded = encode_messages(
        processor,
        conversations,
        loss_roles={'assistant'},
        vcfg=config.vision_config,
        max_seq_len=MAX_SEQ_LEN,
        padding='max_length',
        truncation=True,
    )

    batch_kwargs = dict(
        input_tokens=jnp.array(encoded.input_tokens),
        positions=jnp.array(encoded.positions),
        pixel_values=jnp.array(encoded.pixel_values, dtype=jnp.bfloat16),
        vision_grid=encoded.vision_grid,
        padding_mask=jnp.array(encoded.input_mask).astype(jnp.bool_),
        completion_mask=jnp.array(encoded.completion_mask),
    )
    advantages = jnp.array(advantages_np, dtype=jnp.float32)

    # 7. Grad step.
    with mesh:
      loss = _train_step(model, optimizer, batch_kwargs, advantages)
    logger.info('[step %d] loss=%.4f', step, float(loss))

    # Save checkpoint at the configured cadence.
    if step % CKPT_EVERY_N_STEPS == 0:
      logger.info('[step %d] saving checkpoint', step)
      ckpt_mgr.save(
          step,
          args=ocp.args.PyTreeSave(item=nnx.state(model)),
      )
      ckpt_mgr.wait_until_finished()

  ckpt_mgr.close()
  logger.info('GRPO smoke complete. Checkpoints in %s', CKPT_DIR)


if __name__ == '__main__' and '__file__' in globals():
  main()
