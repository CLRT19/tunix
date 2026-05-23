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
import time
from typing import Any

import datasets
from flax import nnx
from grain import python as grain
import jax
import jax.numpy as jnp
from jax.experimental import multihost_utils
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
# Model size selects the tunix ModelConfig preset (qwen3vl_4b / qwen3vl_8b); it
# MUST match the weights at QWEN3VL_MODEL_DIR. 8B is untied (has lm_head); 4B is tied.
MODEL_SIZE = os.environ.get('QWEN3VL_MODEL_SIZE', '4b').lower()
# Rollout engine: 'vanilla' (proven pure-JAX sampler, default) or 'vllm'
# (in-process vLLM, jax 0.9.2 env). Vanilla stays the fallback.
ROLLOUT_ENGINE = os.environ.get('QWEN3VL_ROLLOUT_ENGINE', 'vanilla').lower()
DATASET_ID = os.environ.get('QWEN3VL_DATASET', 'HuggingFaceM4/ChartQA')
DATASET_SPLIT = os.environ.get('QWEN3VL_DATASET_SPLIT', 'train')

NUM_PROMPTS = int(os.environ.get('QWEN3VL_NUM_PROMPTS', '2'))
NUM_GENERATIONS = int(os.environ.get('QWEN3VL_NUM_GENERATIONS', '4'))
MAX_NEW_TOKENS = int(os.environ.get('QWEN3VL_MAX_NEW_TOKENS', '128'))
MAX_SEQ_LEN = int(os.environ.get('QWEN3VL_MAX_SEQ_LEN', '1536'))
ROLLOUT_CACHE_SIZE = int(os.environ.get('QWEN3VL_ROLLOUT_CACHE_SIZE', '1536'))
# Forced prompt length for the sampler. Required for multi-host SPMD:
# each process pads to the same boundary, so the decode JIT compiles a
# single graph cluster-wide. Should cover system+chat-template tokens +
# vision image-pad tokens (256 for 512x512 image) + question text +
# generation-prompt suffix. Default 768 leaves ~256 tokens for the
# question and template overhead.
ROLLOUT_PROMPT_LEN = int(os.environ.get('QWEN3VL_ROLLOUT_PROMPT_LEN', '768'))
MAX_IMAGE_SIZE = int(os.environ.get('QWEN3VL_MAX_IMAGE_SIZE', '512'))

LEARNING_RATE = float(os.environ.get('QWEN3VL_LR', '1e-6'))
# Global-norm gradient clip. 0 disables. A small clip (~1.0) keeps the
# overfit curve from overshooting into divergence once advantages spike.
GRAD_CLIP = float(os.environ.get('QWEN3VL_GRAD_CLIP', '0'))
MAX_STEPS = int(os.environ.get('QWEN3VL_MAX_STEPS', '4'))
CKPT_EVERY_N_STEPS = int(os.environ.get('QWEN3VL_CKPT_EVERY_N_STEPS', '2'))

CKPT_DIR = os.environ.get(
    'QWEN3VL_CKPT_DIR', '/tmp/qwen3vl_grpo_chartqa_ckpts'
)
# Optional resume-from-checkpoint: point at a manager directory (the
# parent of step subdirs) and pass the step number separately. Only
# model params are restored — optimizer state is NOT saved by this
# trainer, so the resumed run starts with a fresh AdamW. Model-correct
# but not trajectory-identical (acceptable for smoke verification).
RESUME_FROM = os.environ.get('QWEN3VL_RESUME_FROM') or None
RESUME_FROM_STEP_STR = os.environ.get('QWEN3VL_RESUME_FROM_STEP')
RESUME_FROM_STEP = int(RESUME_FROM_STEP_STR) if RESUME_FROM_STEP_STR else None
# Overfit mode: when set, capture NUM_PROMPTS prompts once at start-up
# and reuse the same batch for every training step. Used to demonstrate
# the GRPO loop actually optimises (reward should climb toward 1.0 over
# a few dozen steps) — mirrors gs1693's qwen3 simplereward_overfit_128
# experiment in spirit, scaled down for this trainer.
OVERFIT = os.environ.get('QWEN3VL_OVERFIT', '0') in ('1', 'true', 'True')
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
  # In overfit mode every host must train on the *same* fixed prompts so
  # the per-host gradients reinforce instead of fighting (the multi-host
  # different-prompt-per-host interference that caused the step-5 reward
  # collapse). NoSharding makes all processes iterate the identical
  # sequence, so each captures the same fixed_batch. The forced prompt
  # length + fixed image resize already keep shapes SPMD-uniform.
  if OVERFIT or jax.process_count() == 1:
    shard_options = grain.NoSharding()
  else:
    shard_options = grain.ShardByJaxProcess(drop_remainder=True)
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


def _make_local_rollout_mesh() -> jax.sharding.Mesh:
  """Single local-device mesh for per-host rollout generation."""
  device = jax.local_devices()[0]
  return jax.sharding.Mesh(
      np.asarray([device], dtype=object).reshape((1, 1)), ('fsdp', 'tp')
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


def _train_step_impl(
    model: model_lib.Qwen3VL,
    optimizer: nnx.Optimizer,
    input_tokens: jax.Array,
    positions: jax.Array,
    pixel_values: jax.Array,
    vision_grid,
    padding_mask: jax.Array,
    completion_mask: jax.Array,
    advantages: jax.Array,
) -> jax.Array:
  """Single grad step. Mutates `model` and `optimizer` in place."""

  def loss_only(m):
    return grpo_loss_fn(
        m,
        input_tokens=input_tokens,
        positions=positions,
        pixel_values=pixel_values,
        vision_grid=vision_grid,
        padding_mask=padding_mask,
        completion_mask=completion_mask,
        advantages=advantages,
    )

  loss, grads = nnx.value_and_grad(loss_only)(model)
  optimizer.update(model, grads)
  return loss


# Jitted version, donating the optimizer state in place. Mirrors
# tunix.sft.peft_trainer's jit_train_and_eval_step pattern.
_train_step = nnx.jit(_train_step_impl, donate_argnames=('optimizer',))


def _shard_optimizer_state(optimizer: nnx.Optimizer, mesh: jax.sharding.Mesh):
  """Apply sharding constraints to the optimizer state. Mirrors
  ``tunix.sft.peft_trainer.PeftTrainer._shard_optimizer``. Without this,
  the first jit call compiles twice and may put state on a single device.
  """
  if mesh.empty:
    return
  optimizer_state = nnx.state(optimizer, nnx.optimizer.OptState)
  optimizer_pspecs = nnx.get_partition_spec(optimizer_state)
  optimizer_sharded_state = jax.lax.with_sharding_constraint(
      optimizer_state, optimizer_pspecs
  )
  nnx.update(optimizer, optimizer_sharded_state)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
  _maybe_initialize_jax_distributed()
  # Persistent compile cache. Prefer a native gs:// path so the cache is
  # shared across all hosts and survives reboots; JAX uses etils.epath
  # for paths containing `://`. A gcsfuse mount is treated as local (no
  # `://`) and would skip locking — unsafe under multi-host writes.
  # See jax/_src/lru_cache.py:39 + jax/_src/compiler.py:796 (writes only
  # from process_id == 0, so cross-host write contention is mitigated).
  jax.config.update(
      'jax_compilation_cache_dir',
      os.environ.get('JAX_COMPILATION_CACHE_DIR', '/tmp/jax_cache'),
  )
  jax.config.update('jax_explain_cache_misses', True)
  is_primary = jax.process_index() == 0
  if is_primary:
    os.makedirs(CKPT_DIR, exist_ok=True)

  # --- Mesh + model ---
  config = getattr(model_lib.ModelConfig, f'qwen3vl_{MODEL_SIZE}')()
  logger.info('[model] size=%s (from QWEN3VL_MODEL_SIZE)', MODEL_SIZE)
  # nnx.remat conflicts with the sampler's jax.lax.while_loop (decode loop):
  # the inner forward pass mutates Param state at a different trace level
  # inside the while_loop body and raises TraceContextError. The SFT path
  # disables remat only for sample generation and restores it; GRPO does
  # rollouts every step, so we disable it for the whole run. 4B params at
  # bf16 on 32 v5p chips fits comfortably without remat for our smoke.
  config.remat_config = model_lib.RematConfig.NONE
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

  # Optional warm-start from a previous checkpoint. Restores ONLY model
  # params (the saver writes plain PyTreeSave(nnx.state(model))), so the
  # AdamW state begins fresh. Sharding for the restored arrays is derived
  # from the live nnx.state(model) target via construct_restore_args —
  # this mirrors tunix/sft/checkpoint_manager.py:178.
  if RESUME_FROM:
    if RESUME_FROM_STEP is None:
      raise RuntimeError(
          'QWEN3VL_RESUME_FROM is set but QWEN3VL_RESUME_FROM_STEP is not.'
      )
    logger.info(
        'Resuming model params from %s step %d',
        RESUME_FROM,
        RESUME_FROM_STEP,
    )
    resume_mgr = ocp.CheckpointManager(directory=RESUME_FROM)
    abstract_params = nnx.state(model)
    restore_args = ocp.checkpoint_utils.construct_restore_args(
        target=abstract_params
    )
    restored = resume_mgr.restore(
        RESUME_FROM_STEP,
        args=ocp.args.PyTreeRestore(
            item=abstract_params, restore_args=restore_args
        ),
    )
    nnx.update(model, restored)
    resume_mgr.close()
    show_hbm_usage()

  processor = load_processor(model_dir)

  # --- Rollout adapter ---
  _temp = float(os.environ.get('QWEN3VL_TEMPERATURE', '1.0'))
  _top_p_env = os.environ.get('QWEN3VL_TOP_P', '0.95')
  _top_p = None if _top_p_env in ('', 'none', 'None') else float(_top_p_env)
  rollout_config = base_rollout.RolloutConfig(
      max_tokens_to_generate=MAX_NEW_TOKENS,
      max_prompt_length=ROLLOUT_PROMPT_LEN,
      temperature=_temp,
      top_p=_top_p,
  )

  if ROLLOUT_ENGINE == 'vllm':
    # In-process vLLM rollout (jax 0.9.2 env). vLLM loads its own copy of the
    # model from model_dir and colocates on the trainer mesh. A v5p-64 (32
    # chips) maps naturally to DP=hosts x TP=chips-per-host, which also yields
    # the per-host *replicated* generation the multimodal prefill path needs.
    # Set DP/mesh explicitly (tunix would otherwise infer dp from tp). The
    # actor->vLLM weight sync uses the qwen3vl vllm_jax mapping (auto-resolved
    # from the model's BackendMappingMixin) + caller-side allgather in
    # VllmRollout.update_params. NOTE: needs on-hardware validation.
    from tunix.rl.rollout import vllm_rollout  # local import: jax-0.9.2 env only
    # In-process vLLM: each host runs an INDEPENDENT TP=tp engine on its LOCAL
    # chips; data-parallelism across hosts comes from the 8 separate JAX
    # processes, NOT vLLM DP. So dp=1 + local device ids — otherwise the TPU
    # worker is handed a global device index it doesn't own (KeyError in
    # tpu_worker.init_device). Mirrors the vanilla per-host replicated rollout.
    os.environ.setdefault('TUNIX_VLLM_USE_LOCAL_TPU_DEVICE_IDS', '1')
    tp = MESH_SHAPE[1] if len(MESH_SHAPE) > 1 else 1
    dp = int(os.environ.get('QWEN3VL_VLLM_DP', '1'))
    vllm_rollout_config = base_rollout.RolloutConfig(
        max_tokens_to_generate=MAX_NEW_TOKENS,
        max_prompt_length=ROLLOUT_PROMPT_LEN,
        temperature=_temp,
        top_p=_top_p,
        tensor_parallel_size=tp,
        data_parallel_size=dp,
        rollout_vllm_model_version=model_dir,
        rollout_vllm_hf_config_path=model_dir,
        rollout_vllm_tpu_backend_type='jax',
        rollout_vllm_init_with_random_weights=True,
        rollout_vllm_hbm_utilization=float(
            os.environ.get('QWEN3VL_VLLM_HBM', '0.3')
        ),
        rollout_vllm_max_num_seqs=NUM_PROMPTS * NUM_GENERATIONS,
        # Must be >= the per-image multimodal token budget (vLLM rejects engine
        # init otherwise: "max_tokens_per_mm_item > max_num_batched_tokens").
        # Qwen3-VL images expand to up to ~16k tokens.
        rollout_vllm_max_num_batched_tokens=int(
            os.environ.get('QWEN3VL_VLLM_MAX_BATCHED_TOKENS', '16384')
        ),
    )
    logger.info(
        '[rollout] engine=vllm tp=%d dp=%d hbm_util=%s model_dir=%s',
        tp, dp, os.environ.get('QWEN3VL_VLLM_HBM', '0.3'), model_dir,
    )
    rollout = vllm_rollout.VllmRollout(
        model,
        processor.tokenizer,
        cache_config_or_size=ROLLOUT_CACHE_SIZE,
        mesh=mesh,
        rollout_config=vllm_rollout_config,
    )
  else:
    # Vanilla rollout (default). Intentionally local/replicated: the Qwen3-VL
    # multimodal prefill path scatters vision embeddings into hidden states;
    # doing that with the actor's hidden dim sharded over tp corrupts image
    # tokens on TPU. Keep the train actor sharded, generate per-host.
    rollout_mesh = _make_local_rollout_mesh()
    logger.info(
        'Loading local replicated rollout copy on %s (proc %d)',
        jax.local_devices()[0],
        jax.process_index(),
    )
    rollout_model = params_lib.create_model_from_safe_tensors(
        model_dir, config, mesh=rollout_mesh, dtype=jnp.bfloat16
    )
    rollout = qwen3vl_vanilla_rollout.Qwen3VLVanillaRollout(
        model=rollout_model,
        processor=processor,
        cache_config_or_size=ROLLOUT_CACHE_SIZE,
    )
    if RESUME_FROM:
      rollout.sync_from_actor(nnx.state(model, nnx.Param))

  # --- Optimizer (constructed and sharded inside the mesh) ---
  tx = optax.adamw(LEARNING_RATE)
  if GRAD_CLIP > 0:
    # Clip first, then adam — bounds the per-step update so a reward spike
    # can't kick the policy into divergence (the step-5 overfit collapse).
    tx = optax.chain(optax.clip_by_global_norm(GRAD_CLIP), tx)
    logger.info('[optim] grad clip_by_global_norm=%.3g, lr=%.3g',
                GRAD_CLIP, LEARNING_RATE)
  with mesh:
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
    _shard_optimizer_state(optimizer, mesh)

  # --- Dataset ---
  data_iter = iter(create_dataset())
  fixed_batch: list[dict[str, Any]] | None = None
  if OVERFIT:
    fixed_batch = [next(data_iter) for _ in range(NUM_PROMPTS)]
    logger.info(
        '[overfit] captured %d fixed prompts; will reuse for every step',
        len(fixed_batch),
    )

  # --- Checkpoint manager (process-0 writes; others participate in collective save) ---
  ckpt_mgr = ocp.CheckpointManager(
      directory=CKPT_DIR,
      options=ocp.CheckpointManagerOptions(
          save_interval_steps=CKPT_EVERY_N_STEPS,
          max_to_keep=3,
      ),
  )

  for step in range(1, MAX_STEPS + 1):
    # 1. Sample NUM_PROMPTS prompts on each process (or reuse the fixed
    # batch when running in overfit mode).
    if fixed_batch is not None:
      batch = fixed_batch
    else:
      batch = [next(data_iter) for _ in range(NUM_PROMPTS)]

    # 2. Tile to NUM_GENERATIONS per prompt.
    questions = [b['question'] for b in batch for _ in range(NUM_GENERATIONS)]
    images = [b['image'] for b in batch for _ in range(NUM_GENERATIONS)]
    labels = [b['label'] for b in batch for _ in range(NUM_GENERATIONS)]
    prompt_strs = [_apply_chat_template(processor, q) for q in questions]

    # 3. Rollout. Vary the seed per step so we actually explore — without
    # this the sampler is deterministic and overfit mode (fixed prompts)
    # would loop at the same completions forever.
    rollout_config.seed = step
    logger.info('[step %d] rolling out %d completions', step, len(prompt_strs))
    _t_roll = time.perf_counter()
    rollout_out = rollout.generate(
        prompts=prompt_strs,
        rollout_config=rollout_config,
        images=images,
    )
    logger.info(
        '[step %d] rollout_sec=%.2f (engine=%s, %d completions)',
        step, time.perf_counter() - _t_roll, ROLLOUT_ENGINE, len(prompt_strs),
    )

    # 3b. One-time cluster-wide assertion that every host padded the
    # prompt to the same forced length. The Phase 4 silent-death failure
    # mode was per-process prompt-length divergence → different compiled
    # decode graphs → the run hangs ~7 min in with no traceback. This
    # check makes the regression loud at step 1 instead of silent later.
    if step == 1:
      local_len = jnp.asarray(rollout_out.prompt_seq_len, dtype=jnp.int32)
      all_lens = multihost_utils.process_allgather(local_len)
      all_lens_np = np.asarray(all_lens)
      if not np.all(all_lens_np == ROLLOUT_PROMPT_LEN):
        raise RuntimeError(
            f'prompt_seq_len mismatch across hosts: got {all_lens_np.tolist()},'
            f' expected {ROLLOUT_PROMPT_LEN}. Sampler is no longer respecting'
            ' forced_prompt_length — multi-host SPMD will silent-die.'
        )
      logger.info(
          '[step 1] prompt_seq_len OK across %d hosts: %d',
          int(all_lens_np.size),
          ROLLOUT_PROMPT_LEN,
      )

    # 4. Score. Combine answer-match (1.0) and format-shape (0.1) so the
    # reward is dense early — getting just <think>...<answer> right gets
    # 0.1 even if the answer is wrong, which produces nonzero advantages
    # before the model stumbles onto correct answers.
    answer_rewards = np.array(
        chartqa_reward.check_answer(
            prompts=questions,
            completions=rollout_out.text,
            label=labels,
        ),
        dtype=np.float32,
    )
    format_rewards = np.array(
        chartqa_reward.check_format(
            prompts=questions, completions=rollout_out.text
        ),
        dtype=np.float32,
    )
    rewards = answer_rewards + format_rewards

    # Diagnostic: show what the model actually produced for the first
    # completion so we can tell empty/truncated/no-format apart.
    if is_primary:
      _first = rollout_out.text[0] if rollout_out.text else ''
      logger.info(
          '[step %d] completion[0] len=%d label=%r ans_r=%.2f fmt_r=%.2f'
          ' text=%r',
          step,
          len(_first),
          labels[0],
          float(answer_rewards[0]),
          float(format_rewards[0]),
          _first[:300],
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

    advantages = jnp.array(advantages_np, dtype=jnp.float32)

    # 7. Grad step (jitted, optimizer donated in place).
    with mesh:
      loss = _train_step(
          model,
          optimizer,
          input_tokens=jnp.array(encoded.input_tokens),
          positions=jnp.array(encoded.positions),
          pixel_values=jnp.array(encoded.pixel_values, dtype=jnp.bfloat16),
          vision_grid=encoded.vision_grid,
          padding_mask=jnp.array(encoded.input_mask).astype(jnp.bool_),
          completion_mask=jnp.array(encoded.completion_mask),
          advantages=advantages,
      )
    rollout.sync_from_actor(nnx.state(model, nnx.Param))
    logger.info('[step %d] loss=%.4f', step, float(loss))

    # 7b. Characterize peak HBM after step 1 (rollout + train step both
    # done) so we know our margin to OOM. Cheap; one-shot.
    if step == 1:
      show_hbm_usage()

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
