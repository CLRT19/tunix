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
  7. ``NUM_ITERATIONS`` JIT grad step(s) on the configured policy loss
     (``QWEN3VL_LOSS_ALGO``), masked to completion tokens.

Two policy losses (``QWEN3VL_LOSS_ALGO``):
  * ``grpo`` (default): single-iteration ``-advantages * per_token_logps``,
    ``beta=0`` — REINFORCE with group baseline, no reference model / KL / clip.
  * ``gspo``: GSPO-token (sequence-level, length-normalized importance ratio
    with PPO clip), matching zlab-princeton/vero. Needs ``NUM_ITERATIONS>=2``
    to exercise the off-policy clip; the first inner step is on-policy.

Usage::

    python -m tunix.models.qwen3vl.grpo_example
"""

from __future__ import annotations

import json
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

# Metrics are written as a JSONL stream by the primary process and pushed to
# Weights & Biases by a SEPARATE sidecar (scripts/multihost/wandb_sync_jsonl.py).
# wandb is deliberately NOT imported/run in the training process: wandb 0.27
# forks a service process (start_method is ignored) that corrupts worker 0's
# libtpu state and hangs the multi-host model-load. File handle set in main().
_METRICS_FH = None

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
# Gradient checkpointing: 'block' (per-layer remat, needed to fit long seqs on
# the vLLM path) or 'none'. Only safe with the vLLM rollout (see main()).
QWEN3VL_REMAT = os.environ.get('QWEN3VL_REMAT', 'none').lower()
DATASET_ID = os.environ.get('QWEN3VL_DATASET', 'HuggingFaceM4/ChartQA')
DATASET_SPLIT = os.environ.get('QWEN3VL_DATASET_SPLIT', 'train')

NUM_PROMPTS = int(os.environ.get('QWEN3VL_NUM_PROMPTS', '2'))
NUM_GENERATIONS = int(os.environ.get('QWEN3VL_NUM_GENERATIONS', '4'))
TRAIN_BATCH_SIZE = NUM_PROMPTS * NUM_GENERATIONS
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

# Policy-loss algorithm. "grpo" = the original single-iteration
# REINFORCE-with-group-baseline loss. "gspo" = GSPO-token (Zheng et al.
# 2507.18071): a *sequence-level*, length-normalized importance ratio with
# token-level gradient and PPO-style clipping. GSPO only differs from GRPO
# once the policy drifts from the rollout (behavior) policy, so it needs
# NUM_ITERATIONS > 1 — the first inner step is on-policy (ratio == 1) and
# subsequent steps reuse the same rollout off-policy, where the clip bites.
# Matches zlab-princeton/vero gspo_llmjudge_shared.yaml:
#   loss_mode=gspo, clip_ratio_low=3e-4, clip_ratio_high=4e-4,
#   clip_ratio_c=10.0, use_kl_loss=false, loss_agg_mode=seq-mean-token-mean.
_loss_algo_raw = os.environ.get('QWEN3VL_LOSS_ALGO', 'grpo').lower()
if _loss_algo_raw in ('gspo', 'gspo-token'):
  LOSS_ALGO = 'gspo'  # tunix calls it "gspo-token"; alias to our internal flag.
elif _loss_algo_raw == 'grpo':
  LOSS_ALGO = 'grpo'
else:
  raise ValueError(
      "QWEN3VL_LOSS_ALGO must be 'grpo', 'gspo', or 'gspo-token'; got"
      f' {_loss_algo_raw!r}'
  )
# Optimizer steps per rollout (verl ppo_epochs * num_minibatches analogue).
# 1 keeps the original on-policy behavior; >=2 activates GSPO's off-policy clip.
NUM_ITERATIONS = int(os.environ.get('QWEN3VL_NUM_ITERATIONS', '1'))
if NUM_ITERATIONS < 1:
  raise ValueError(f'QWEN3VL_NUM_ITERATIONS must be >= 1; got {NUM_ITERATIONS}')
MICRO_BSZ = int(os.environ.get('QWEN3VL_MICRO_BSZ', '0'))
if MICRO_BSZ < 0:
  raise ValueError(f'QWEN3VL_MICRO_BSZ must be >= 0; got {MICRO_BSZ}')
if 0 < MICRO_BSZ < TRAIN_BATCH_SIZE and TRAIN_BATCH_SIZE % MICRO_BSZ != 0:
  raise ValueError(
      'QWEN3VL_MICRO_BSZ must divide B=NUM_PROMPTS*NUM_GENERATIONS when'
      f' accumulation is active; got MICRO_BSZ={MICRO_BSZ},'
      f' B={TRAIN_BATCH_SIZE}. Set QWEN3VL_MICRO_BSZ=0 to disable.'
  )
# GSPO clip range. Tiny by design — the sequence-level ratio is a
# length-normalized geometric mean, so its variance is far smaller than a
# token-level ratio and the clip must be correspondingly tight.
CLIP_LOW = float(os.environ.get('QWEN3VL_CLIP_LOW', '3e-4'))
CLIP_HIGH = float(os.environ.get('QWEN3VL_CLIP_HIGH', '4e-4'))
# Safety bound on the log importance ratio before exp (verl clip_ratio_c).
CLIP_C = float(os.environ.get('QWEN3VL_CLIP_C', '10.0'))

# Metrics JSONL path (primary process writes one JSON object per step). The
# sidecar pushes it to wandb. Empty / WANDB_MODE=disabled turns metrics off.
# Default sits next to the checkpoints (on the bucket).
METRICS_JSONL = os.environ.get('QWEN3VL_METRICS_JSONL', '')
METRICS_ENABLED = os.environ.get('WANDB_MODE', '').lower() != 'disabled'

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

# Verbatim from zlab-princeton/vero examples/prompts/system_prompt_chatting.txt.
# The vero reward router needs <answer>...\boxed{result}...</answer>; this is the
# prompt that elicits it (rule-based rewards are 0 without the boxed format).
_VERO_SYSTEM_PROMPT = r"""You are a helpful, conversational assistant tasked with answering a question about an image.

Your response must include two parts:

1. **Reasoning**: A detailed, free-flowing chain of thought enclosed in `<think>` and `</think>` tags.
2. **Final Answer**: A clear, conversational response enclosed in `<answer>` and `</answer>` tags, using \boxed{} notation when the question has a definitive answer.

---

### Reasoning Instructions

* The reasoning section must be inside `<think>` … `</think>` tags.
* The reasoning should resemble a stream of consciousness: explore, test hypotheses, backtrack if necessary, reflect, and refine.
* Let the reasoning flow naturally while progressing toward a conclusion.
* Use reasoning strategies such as:
  * **Planning** – outline possible approaches before committing.
  * **Exploration** – consider multiple image regions or interpretations, even unlikely ones.
  * **Evaluation** – compare alternatives and verify against visual evidence.
  * **Reflection** – revisit earlier ideas if they may still be viable.
* Thoroughly examine and cross-check relevant image regions before narrowing down.
* If the image is ambiguous, make a reasonable inference based on visual and contextual cues.
* End the reasoning once you are confident in the conclusion.

---

### Final Answer Instructions

* The answer section must be enclosed in `<answer>` … `</answer>` tags.
* The `<answer>` section should stand on its own as a response to the user: it must provide necessary context and justification so that a reader can understand and verify the conclusion without reading `<think>`.
  - Do NOT refer to the `<think>` section (avoid phrases like "as explained above" or "from the reasoning").
* Boxed result:
    * If the question has a definitive, concise answer (a number, word, phrase, or label), include a conversational, natural response followed by exactly one boxed result using LaTeX: \boxed{final_result}.
    * If the question is open-ended, subjective, or does not yield a concise final result, omit the boxed notation."""
SYSTEM_PROMPT = os.environ.get('QWEN3VL_SYSTEM_PROMPT', _VERO_SYSTEM_PROMPT)

# ---------------------------------------------------------------------------
# Vero multi-domain (optional). When QWEN3VL_VERO_JSONL is set we switch
# from the single-domain ChartQA HF dataset path to a JSONL-backed dataset
# that carries `domain`, `reward_type`, `ground_truth`, and `extra_info`
# per row, routed through the vero_router reward fn. Empty defaults keep
# the ChartQA fallback bit-identical for Phase 4+5 runs.
# ---------------------------------------------------------------------------
QWEN3VL_VERO_JSONL = os.environ.get('QWEN3VL_VERO_JSONL', '')
QWEN3VL_VERO_IMAGE_ROOT = os.environ.get('QWEN3VL_VERO_IMAGE_ROOT', '')
QWEN3VL_VERO_LOCAL_DIR = os.environ.get(
    'QWEN3VL_VERO_LOCAL_DIR', '/tmp/qwen3vl_vero_smoke'
)
# Parquet streaming over the Vero-600k HF snapshot. When PARQUET_ROOT is
# set it takes precedence over the JSONL path; the chartqa fallback
# remains the default when neither is configured.
QWEN3VL_VERO_PARQUET_ROOT = os.environ.get('QWEN3VL_VERO_PARQUET_ROOT', '')
QWEN3VL_VERO_PARQUET_LOCAL_DIR = os.environ.get(
    'QWEN3VL_VERO_PARQUET_LOCAL_DIR', '/tmp/qwen3vl_vero_parquet'
)
QWEN3VL_VERO_PARQUET_SHARDS_PER_DOMAIN = int(
    os.environ.get('QWEN3VL_VERO_PARQUET_SHARDS_PER_DOMAIN', '2')
)
# Cap rows decoded into RAM per domain (0 = unlimited). shards_per_domain is a
# PER-SUBSET cap; a domain has many subset dirs, so this bounds the one-time
# startup decode + RAM. ~2000 covers a 1000+ step run; smokes use less.
QWEN3VL_VERO_MAX_ROWS_PER_DOMAIN = int(
    os.environ.get('QWEN3VL_VERO_MAX_ROWS_PER_DOMAIN', '0')
)
# In-scope 5 domains for the Vero-600k parquet snapshot (captioning_IF
# is intentionally excluded — out of scope per the investigation report).
DEFAULT_VERO_DOMAINS = (
    'chart_ocr',
    'counting_grounding_search',
    'knowledge_recognition',
    'spatial_action',
    'stem',
)
QWEN3VL_DOMAINS = [
    s.strip()
    for s in os.environ.get('QWEN3VL_DOMAINS', '').split(',')
    if s.strip()
] or None
_mw_raw = os.environ.get('QWEN3VL_MIX_WEIGHTS', '').strip()
QWEN3VL_MIX_WEIGHTS = (
    [float(s) for s in _mw_raw.split(',') if s.strip()] if _mw_raw else None
)
QWEN3VL_FORMAT_SCORE = float(os.environ.get('QWEN3VL_FORMAT_SCORE', '0.2'))


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


def _gs_to_gcsfuse(uri: str) -> str:
  """Map a ``gs://<bucket>/<path>`` URI to its gcsfuse mountpoint so we can
  read it as a local path WITHOUT copying to the boot disk. The bucket is
  mounted at ``QWEN3VL_GCSFUSE_MOUNT`` (default ``/home/linrong/bucket``)."""
  mount = os.environ.get('QWEN3VL_GCSFUSE_MOUNT', '/home/linrong/bucket')
  bucket = os.environ.get('QWEN3VL_GCS_BUCKET', 'linrong-vlm-tpu-us-central1-a')
  prefix = f'gs://{bucket}/'
  if uri.startswith(prefix):
    return os.path.join(mount, uri[len(prefix):])
  return uri  # already a local / gcsfuse path


def create_dataset():
  # Dataset selection precedence (highest first):
  #   1. Vero-600k parquet snapshot (PARQUET_ROOT set) — streams shards.
  #   2. Vero multi-domain JSONL (VERO_JSONL set) — preexisting path.
  #   3. ChartQA HF dataset fallback (bit-identical to pre-vero behavior).
  if QWEN3VL_VERO_PARQUET_ROOT:
    from tunix.models.qwen3vl import vero_dataset
    # Read parquet DIRECTLY from the gcsfuse-mounted bucket — do NOT copy
    # shards to the local boot disk (small + shared). shards_per_domain caps
    # how many shards per subset are read into RAM at startup; per-step reads
    # then come from RAM, not the bucket.
    parquet_root = _gs_to_gcsfuse(QWEN3VL_VERO_PARQUET_ROOT)
    logger.info('[vero-parquet] reading from bucket (no local copy): %s',
                parquet_root)
    return vero_dataset.build_vero_parquet_dataset(
        parquet_root,
        MAX_IMAGE_SIZE,
        QWEN3VL_DOMAINS,
        QWEN3VL_MIX_WEIGHTS,
        shuffle_seed=0,
        shards_per_domain=QWEN3VL_VERO_PARQUET_SHARDS_PER_DOMAIN,
        max_rows_per_domain=QWEN3VL_VERO_MAX_ROWS_PER_DOMAIN or None,
    )
  if QWEN3VL_VERO_JSONL:
    from tunix.models.qwen3vl import vero_dataset
    local_jsonl, local_img_root = vero_dataset.download_smoke_assets(
        QWEN3VL_VERO_JSONL,
        QWEN3VL_VERO_IMAGE_ROOT,
        QWEN3VL_VERO_LOCAL_DIR,
    )
    return vero_dataset.build_vero_jsonl_dataset(
        local_jsonl,
        local_img_root,
        MAX_IMAGE_SIZE,
        QWEN3VL_DOMAINS,
        QWEN3VL_MIX_WEIGHTS,
        shuffle_seed=0,
    )
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


def _compute_per_token_logps(
    model: model_lib.Qwen3VL,
    input_tokens: jax.Array,  # [B, L]
    positions: jax.Array,  # [3, B, L]
    pixel_values: jax.Array,  # [P, C]
    vision_grid: VisionGridData,
    padding_mask: jax.Array,  # [B, L]
) -> jax.Array:
  """Forward pass → per-token log-prob of the realized next token. [B, L-1]."""
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
  return jnp.take_along_axis(log_probs, targets[..., None], axis=-1)[..., 0]


# Jitted, grad-free forward for capturing the behavior-policy log-probs that
# GSPO's importance ratio is measured against. Only used on the gspo path.
_compute_old_logps = nnx.jit(_compute_per_token_logps)


def policy_loss_fn(
    model: model_lib.Qwen3VL,
    input_tokens: jax.Array,  # [B, L]
    positions: jax.Array,  # [3, B, L]
    pixel_values: jax.Array,  # [P, C]
    vision_grid: VisionGridData,
    padding_mask: jax.Array,  # [B, L]
    completion_mask: jax.Array,  # [B, L] — 1 where loss applies
    advantages: jax.Array,  # [B]
    old_per_token_logps: jax.Array,  # [B, L-1] — behavior policy (gspo only)
) -> jax.Array:
  """GRPO or GSPO-token policy-gradient loss on completion tokens.

  GRPO (``LOSS_ALGO != 'gspo'``): ``loss = -E[A * log pi(a|s)]`` with a global
  token-mean — the original Phase-4/6 REINFORCE-with-group-baseline path,
  numerically unchanged.

  GSPO-token (``LOSS_ALGO == 'gspo'``, Zheng et al. 2507.18071): a
  sequence-level, length-normalized importance ratio with token-level gradient
  and PPO clipping. On the first inner iteration ``old == current`` so the
  ratio is 1 and it coincides with GRPO; off-policy inner iterations are where
  the clip applies. Mirrors tunix ``grpo_learner`` gspo-token + verl
  ``loss_mode=gspo`` / ``loss_agg_mode=seq-mean-token-mean``.
  """
  per_token_logps = _compute_per_token_logps(
      model,
      input_tokens=input_tokens,
      positions=positions,
      pixel_values=pixel_values,
      vision_grid=vision_grid,
      padding_mask=padding_mask,
  )  # [B, L-1]
  mask = completion_mask[:, 1:].astype(jnp.float32)  # [B, L-1]
  adv = jnp.expand_dims(advantages, 1)  # [B, 1]

  if LOSS_ALGO != 'gspo':
    per_token_loss = -adv * per_token_logps
    denom = jnp.clip(jnp.sum(mask), min=1.0)
    return jnp.sum(per_token_loss * mask) / denom

  # Mask with jnp.where (NOT * mask): per_token_logps is computed over ALL
  # tokens incl prompt/padding, and any inf/nan there would poison the masked
  # sums via `nan * 0 = nan`. Zeroing the log-probs on non-completion positions
  # cuts those tokens out of BOTH value and gradient, so padding can never leak.
  mask_b = mask > 0
  cur = jnp.where(mask_b, per_token_logps, 0.0)  # [B, L-1], grad only on compl.
  old = jnp.where(mask_b, old_per_token_logps, 0.0)
  denom_tok = jnp.clip(mask.sum(-1), min=1.0)  # [B]
  # Sequence-level, length-normalized log importance ratio (completion only).
  seq_log_ratio = jnp.where(mask_b, cur - old, 0.0).sum(-1) / denom_tok  # [B]
  # GSPO-token: sequence-level weight carried with a token-level gradient.
  si = cur - jax.lax.stop_gradient(cur) + jnp.expand_dims(
      jax.lax.stop_gradient(seq_log_ratio), 1
  )  # [B, L-1]
  si = jnp.clip(si, max=CLIP_C)
  coef_1 = jnp.exp(si)
  coef_2 = jnp.clip(coef_1, 1.0 - CLIP_LOW, 1.0 + CLIP_HIGH)
  per_token_loss = -jnp.minimum(coef_1 * adv, coef_2 * adv)  # [B, L-1]
  per_token_loss = jnp.where(mask_b, per_token_loss, 0.0)  # drop non-completion
  # seq-mean-token-mean: per-sequence token-mean, then batch-mean. With adv==0
  # every per_token_loss is exactly 0 -> loss 0 (no nan).
  seq_loss = per_token_loss.sum(-1) / denom_tok
  return seq_loss.mean()


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
    old_per_token_logps: jax.Array,
) -> jax.Array:
  """Single grad step. Mutates `model` and `optimizer` in place."""

  def loss_only(m):
    return policy_loss_fn(
        m,
        input_tokens=input_tokens,
        positions=positions,
        pixel_values=pixel_values,
        vision_grid=vision_grid,
        padding_mask=padding_mask,
        completion_mask=completion_mask,
        advantages=advantages,
        old_per_token_logps=old_per_token_logps,
    )

  loss, grads = nnx.value_and_grad(loss_only)(model)
  optimizer.update(model, grads)
  return loss


# Jitted version, donating the optimizer state in place. Mirrors
# tunix.sft.peft_trainer's jit_train_and_eval_step pattern.
_train_step = nnx.jit(_train_step_impl, donate_argnames=('optimizer',))


def _micro_grad_impl(
    model: model_lib.Qwen3VL,
    micro_input_tokens: jax.Array,
    micro_positions: jax.Array,
    micro_pixel_values: jax.Array,
    micro_vision_grid: VisionGridData,
    micro_padding_mask: jax.Array,
    micro_completion_mask: jax.Array,
    micro_advantages: jax.Array,
    micro_old_per_token_logps: jax.Array,
) -> tuple[jax.Array, Any]:
  """Compute one micro-batch loss and grads without updating optimizer state."""

  def loss_only(m):
    return policy_loss_fn(
        m,
        input_tokens=micro_input_tokens,
        positions=micro_positions,
        pixel_values=micro_pixel_values,
        vision_grid=micro_vision_grid,
        padding_mask=micro_padding_mask,
        completion_mask=micro_completion_mask,
        advantages=micro_advantages,
        old_per_token_logps=micro_old_per_token_logps,
    )

  return nnx.value_and_grad(loss_only)(model)


_micro_grad_fn = nnx.jit(_micro_grad_impl)


def _apply_accumulated_grads_impl(
    model: model_lib.Qwen3VL,
    optimizer: nnx.Optimizer,
    grads: Any,
) -> jax.Array:
  """Apply already-accumulated gradients once, donating optimizer state."""
  optimizer.update(model, grads)
  return jnp.asarray(0.0, dtype=jnp.float32)


_apply_accumulated_grads = nnx.jit(
    _apply_accumulated_grads_impl, donate_argnames=('optimizer',)
)


def _micro_accumulation_active(batch_size: int) -> bool:
  return MICRO_BSZ > 0 and MICRO_BSZ < batch_size


def _vision_patch_offsets_per_sequence(
    vision_grid: VisionGridData,
    *,
    batch_size: int,
    total_patches: int,
) -> np.ndarray:
  """Return [B+1] patch offsets for this trainer's one-image-per-sequence pack."""
  offsets = np.asarray(vision_grid.cu_seqlens, dtype=np.int64)
  if offsets.ndim != 1:
    raise ValueError(
        'QWEN3VL_MICRO_BSZ requires 1-D vision_grid.cu_seqlens; got'
        f' shape={offsets.shape}.'
    )
  # encode_messages flattens images in sequence order, and grpo_example builds
  # exactly one image per sequence. Under that constraint each cu_seqlens segment
  # is the patch span for the matching sequence, so patch counts may vary without
  # assuming uniform images. Multi-image or video rows need an explicit
  # image-to-sequence map before they can be micro-batched here.
  if offsets.shape[0] != batch_size + 1:
    raise ValueError(
        'QWEN3VL_MICRO_BSZ currently requires exactly one image frame per'
        ' sequence so vision_grid.cu_seqlens has B+1 entries; got'
        f' cu_seqlens={offsets.shape[0]} entries for B={batch_size}.'
    )
  if int(offsets[0]) != 0 or int(offsets[-1]) != total_patches:
    raise ValueError(
        'vision_grid.cu_seqlens does not match packed pixel_values:'
        f' first={int(offsets[0])}, last={int(offsets[-1])},'
        f' total_patches={total_patches}.'
    )
  return offsets


def _slice_vision_grid(
    vision_grid: VisionGridData,
    *,
    seq_start: int,
    seq_end: int,
    patch_start: int,
    patch_end: int,
) -> VisionGridData:
  """Slice precomputed vision metadata to match a contiguous pixel patch span."""
  cu_seqlens = vision_grid.cu_seqlens[seq_start : seq_end + 1]
  cu_seqlens = cu_seqlens - cu_seqlens[0]
  pos_embed_idx = vision_grid.pos_embed_idx[:, patch_start:patch_end]
  pos_embed_weights = vision_grid.pos_embed_weights[:, patch_start:patch_end]
  pos_embed_gather = vision_grid.pos_embed_gather[patch_start:patch_end]
  if pos_embed_idx.shape[1] > 0:
    pos_embed_gather = pos_embed_gather - patch_start
  return VisionGridData(
      cos=vision_grid.cos[patch_start:patch_end],
      sin=vision_grid.sin[patch_start:patch_end],
      cu_seqlens=cu_seqlens,
      pos_embed_idx=pos_embed_idx,
      pos_embed_weights=pos_embed_weights,
      pos_embed_gather=pos_embed_gather,
  )


def _train_step_accum(
    model: model_lib.Qwen3VL,
    optimizer: nnx.Optimizer,
    input_tokens: jax.Array,
    positions: jax.Array,
    pixel_values: jax.Array,
    vision_grid: VisionGridData,
    padding_mask: jax.Array,
    completion_mask: jax.Array,
    advantages: jax.Array,
    old_per_token_logps: jax.Array,
    patch_offsets: np.ndarray,
) -> jax.Array:
  """Micro-batch one logical train step and apply the averaged gradients once."""
  batch_size = input_tokens.shape[0]
  if not _micro_accumulation_active(batch_size):
    return _train_step(
        model,
        optimizer,
        input_tokens=input_tokens,
        positions=positions,
        pixel_values=pixel_values,
        vision_grid=vision_grid,
        padding_mask=padding_mask,
        completion_mask=completion_mask,
        advantages=advantages,
        old_per_token_logps=old_per_token_logps,
    )
  if batch_size % MICRO_BSZ != 0:
    raise ValueError(
        'QWEN3VL_MICRO_BSZ must divide the encoded batch size when'
        f' accumulation is active; got MICRO_BSZ={MICRO_BSZ}, B={batch_size}.'
    )

  n_accum = batch_size // MICRO_BSZ
  total_loss = jnp.asarray(0.0, dtype=jnp.float32)
  accumulated_grads = None
  if LOSS_ALGO != 'gspo':
    total_token_count = jnp.clip(
        jnp.sum(completion_mask[:, 1:].astype(jnp.float32)), min=1.0
    )

  for mb_idx in range(n_accum):
    mb_start = mb_idx * MICRO_BSZ
    mb_end = mb_start + MICRO_BSZ
    patch_start = int(patch_offsets[mb_start])
    patch_end = int(patch_offsets[mb_end])
    micro_completion_mask = completion_mask[mb_start:mb_end]
    micro_loss, micro_grads = _micro_grad_fn(
        model,
        input_tokens[mb_start:mb_end],
        positions[:, mb_start:mb_end, :],
        pixel_values[patch_start:patch_end],
        _slice_vision_grid(
            vision_grid,
            seq_start=mb_start,
            seq_end=mb_end,
            patch_start=patch_start,
            patch_end=patch_end,
        ),
        padding_mask[mb_start:mb_end],
        micro_completion_mask,
        advantages[mb_start:mb_end],
        old_per_token_logps[mb_start:mb_end],
    )

    if LOSS_ALGO == 'gspo':
      # GSPO is seq-mean-token-mean. Because startup validation requires equal
      # sequence-count micro-batches, averaging micro losses/grads is exactly the
      # gradient of the full batch mean over B sequences.
      loss_scale = jnp.asarray(1.0 / n_accum, dtype=jnp.float32)
    else:
      # GRPO is a global token mean: full loss is sum(all token losses) divided
      # by total completion tokens. A micro loss has already divided by its own
      # token count, so weight its loss/grads by micro_tokens / total_tokens.
      micro_token_count = jnp.sum(
          micro_completion_mask[:, 1:].astype(jnp.float32)
      )
      loss_scale = micro_token_count / total_token_count

    micro_grads = jax.tree.map(lambda grad: grad * loss_scale, micro_grads)
    accumulated_grads = (
        micro_grads
        if accumulated_grads is None
        else jax.tree.map(
            lambda lhs, rhs: lhs + rhs, accumulated_grads, micro_grads
        )
    )
    total_loss = total_loss + micro_loss * loss_scale

  _apply_accumulated_grads(model, optimizer, accumulated_grads)
  return total_loss


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

  # Metrics — primary process only, written as a JSONL stream (NO wandb in the
  # training process; see _METRICS_FH note above). A sidecar pushes it to wandb.
  global _METRICS_FH
  metrics_path = METRICS_JSONL or os.path.join(
      os.path.dirname(CKPT_DIR.rstrip('/')), 'metrics.jsonl'
  )
  if is_primary and METRICS_ENABLED and metrics_path:
    try:
      os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
      _METRICS_FH = open(metrics_path, 'a', buffering=1)  # line-buffered
      _cfg = {
          'loss_algo': LOSS_ALGO, 'num_iterations': NUM_ITERATIONS,
          'clip_low': CLIP_LOW, 'clip_high': CLIP_HIGH, 'clip_c': CLIP_C,
          'lr': LEARNING_RATE, 'grad_clip': GRAD_CLIP,
          'micro_bsz': MICRO_BSZ,
          'num_prompts': NUM_PROMPTS, 'num_generations': NUM_GENERATIONS,
          'max_new_tokens': MAX_NEW_TOKENS, 'max_seq_len': MAX_SEQ_LEN,
          'max_steps': MAX_STEPS, 'mesh_shape': os.environ.get('QWEN3VL_MESH_SHAPE', ''),
          'model_size': MODEL_SIZE, 'rollout_engine': ROLLOUT_ENGINE,
          'format_score': QWEN3VL_FORMAT_SCORE,
          'dataset': 'vero-parquet' if QWEN3VL_VERO_PARQUET_ROOT else (
              'vero-jsonl' if QWEN3VL_VERO_JSONL else DATASET_ID),
      }
      _METRICS_FH.write(json.dumps({'_config': _cfg}) + '\n')
      logger.info('[metrics] JSONL stream -> %s', metrics_path)
    except Exception as e:  # never let logging kill training
      logger.warning('[metrics] could not open %s: %s', metrics_path, e)
      _METRICS_FH = None

  # Active dataset banner. Make the vero vs chartqa choice loud at start
  # so it's obvious in the launcher log which path the run is on.
  if QWEN3VL_VERO_PARQUET_ROOT:
    logger.info(
        '[dataset] mode=vero-parquet root=%s local_dir=%s'
        ' shards_per_domain=%d domains_filter=%s mix_weights=%s'
        ' format_score=%.3f',
        QWEN3VL_VERO_PARQUET_ROOT,
        QWEN3VL_VERO_PARQUET_LOCAL_DIR,
        QWEN3VL_VERO_PARQUET_SHARDS_PER_DOMAIN,
        QWEN3VL_DOMAINS or list(DEFAULT_VERO_DOMAINS),
        QWEN3VL_MIX_WEIGHTS,
        QWEN3VL_FORMAT_SCORE,
    )
  elif QWEN3VL_VERO_JSONL:
    logger.info(
        '[dataset] mode=vero jsonl=%s image_root=%s local_dir=%s'
        ' domains_filter=%s mix_weights=%s format_score=%.3f',
        QWEN3VL_VERO_JSONL,
        QWEN3VL_VERO_IMAGE_ROOT,
        QWEN3VL_VERO_LOCAL_DIR,
        QWEN3VL_DOMAINS,
        QWEN3VL_MIX_WEIGHTS,
        QWEN3VL_FORMAT_SCORE,
    )
  else:
    logger.info(
        '[dataset] mode=chartqa dataset_id=%s split=%s',
        DATASET_ID,
        DATASET_SPLIT,
    )

  # --- Mesh + model ---
  config = getattr(model_lib.ModelConfig, f'qwen3vl_{MODEL_SIZE}')()
  logger.info('[model] size=%s (from QWEN3VL_MODEL_SIZE)', MODEL_SIZE)
  # Gradient checkpointing (per-block remat). nnx.remat conflicts with the
  # VANILLA sampler's jax.lax.while_loop (TraceContextError), so it must stay
  # NONE on the vanilla path. But with the vLLM rollout the training model is
  # ONLY used for the train step (forward+backward) and the old-logps forward —
  # never for sampling — so remat is safe AND necessary: without it the train
  # step needs ~80 GiB/chip at seq 1024 and OOMs (RuntimeProgramAllocationFailure)
  # at the longer seq lengths needed to let the model emit a boxed answer.
  if QWEN3VL_REMAT == 'block':
    if ROLLOUT_ENGINE != 'vllm':
      logger.warning('[remat] BLOCK requested but rollout=%s — remat breaks the'
                     ' vanilla sampler while_loop; forcing NONE.', ROLLOUT_ENGINE)
      config.remat_config = model_lib.RematConfig.NONE
    else:
      config.remat_config = model_lib.RematConfig.BLOCK
      logger.info('[remat] gradient checkpointing ON (RematConfig.BLOCK)')
  else:
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
  # Barrier so all hosts enter the global-sharding load together. Without it,
  # fast (local-disk) loads let hosts reach the sharded device_put at very
  # different times, exposing load-time skew on the multi-host mesh.
  if jax.process_count() > 1:
    multihost_utils.sync_global_devices('before_qwen3vl_model_load')
  model = params_lib.create_model_from_safe_tensors(
      model_dir, config, mesh=mesh, dtype=jnp.bfloat16
  )
  if jax.process_count() > 1:
    multihost_utils.sync_global_devices('after_qwen3vl_model_load')
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
  # Force vero's sampling and OVERRIDE the model's generation_config.json
  # (which otherwise clamps to temp=0.7/top_k=20/top_p=0.8 -> low diversity, so
  # the model rarely samples the <think>/<answer> tag path). vero uses
  # temp=1.0, top_p=1.0, top_k=-1 (all tokens). -1 disables top-k in vLLM.
  _top_k = int(os.environ.get('QWEN3VL_TOP_K', '-1'))
  rollout_config = base_rollout.RolloutConfig(
      max_tokens_to_generate=MAX_NEW_TOKENS,
      max_prompt_length=ROLLOUT_PROMPT_LEN,
      temperature=_temp,
      top_p=_top_p,
      top_k=_top_k,
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
        top_k=_top_k,
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

  if LOSS_ALGO == 'gspo':
    logger.info(
        '[policy-loss] algo=gspo (seq-mean-token-mean) num_iterations=%d'
        ' clip_low=%.2g clip_high=%.2g clip_c=%.2g lr=%.2g',
        NUM_ITERATIONS, CLIP_LOW, CLIP_HIGH, CLIP_C, LEARNING_RATE,
    )
    if NUM_ITERATIONS < 2:
      logger.warning(
          '[policy-loss] GSPO with num_iterations=%d is on-policy and'
          ' numerically identical to GRPO — set QWEN3VL_NUM_ITERATIONS>=2'
          ' for the importance-ratio clip to take effect.',
          NUM_ITERATIONS,
      )
  else:
    logger.info(
        '[policy-loss] algo=grpo (REINFORCE + group baseline)'
        ' num_iterations=%d lr=%.2g', NUM_ITERATIONS, LEARNING_RATE,
    )

  if _micro_accumulation_active(TRAIN_BATCH_SIZE):
    logger.info(
        '[micro-batch] gradient accumulation enabled: micro_bsz=%d B=%d'
        ' n_accum=%d',
        MICRO_BSZ, TRAIN_BATCH_SIZE, TRAIN_BATCH_SIZE // MICRO_BSZ,
    )

  # Multi-host SPMD requires identical policy-loss config on every process:
  # LOSS_ALGO drives a jit-time Python branch, while NUM_ITERATIONS and MICRO_BSZ
  # drive Python loop counts, so a per-host divergence compiles different
  # programs and silent-dies (same failure class as the step-1 prompt_seq_len
  # drift guard). Assert uniformity loudly before the first jit.
  if jax.process_count() > 1:
    local_cfg = jnp.asarray(
        [1.0 if LOSS_ALGO == 'gspo' else 0.0, float(NUM_ITERATIONS),
         CLIP_LOW, CLIP_HIGH, CLIP_C, float(MICRO_BSZ)],
        dtype=jnp.float32,
    )
    all_cfg = np.asarray(multihost_utils.process_allgather(local_cfg))
    if not np.all(all_cfg == all_cfg[0]):
      raise RuntimeError(
          'policy-loss config differs across hosts (rows ='
          ' [algo, iters, clip_low, clip_high, clip_c, micro_bsz]):'
          f' {all_cfg.tolist()}.'
          ' All workers must export identical QWEN3VL_LOSS_ALGO /'
          ' NUM_ITERATIONS / CLIP_* / QWEN3VL_MICRO_BSZ — divergence'
          ' silent-dies under SPMD.'
      )
    logger.info('[policy-loss] config uniform across %d hosts', all_cfg.shape[0])

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
    if QWEN3VL_VERO_JSONL:
      # Vero JSONL rows carry `ground_truth` + domain/reward_type/extra_info.
      # Tile each by NUM_GENERATIONS, same fan-out pattern as questions/images,
      # so per-completion indices stay aligned.
      ground_truths = [
          b['ground_truth'] for b in batch for _ in range(NUM_GENERATIONS)
      ]
      domains = [b['domain'] for b in batch for _ in range(NUM_GENERATIONS)]
      reward_types = [
          b['reward_type'] for b in batch for _ in range(NUM_GENERATIONS)
      ]
      extra_infos = [
          b['extra_info'] for b in batch for _ in range(NUM_GENERATIONS)
      ]
      labels = None  # ChartQA-only field; not used on the vero path.
    else:
      labels = [b['label'] for b in batch for _ in range(NUM_GENERATIONS)]
      ground_truths = None
      domains = None
      reward_types = None
      extra_infos = None
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

    # 4. Score. Vero-mode dispatches to the multi-domain reward router and
    # combines via verl-style `(1 - fs) * acc + fs * fmt`. ChartQA-mode keeps
    # the additive `answer + format` shape that drove the Phase 4 overfit
    # curve — touching that convention would change its dynamics.
    if QWEN3VL_VERO_JSONL:
      from tunix.cli.reward_fn import vero_format
      from tunix.cli.reward_fn import vero_router
      answer_rewards = np.array(
          vero_router.score_batch(
              domains,
              reward_types,
              ground_truths,
              rollout_out.text,
              extra_infos,
          ),
          dtype=np.float32,
      )
      format_rewards = np.array(
          vero_format.check_format_batch(rollout_out.text),
          dtype=np.float32,
      )
      rewards = (
          (1.0 - QWEN3VL_FORMAT_SCORE) * answer_rewards
          + QWEN3VL_FORMAT_SCORE * format_rewards
      )

      # Per-domain reward stats BEFORE group-relative tile-reduction. Useful
      # for spotting one domain dominating or collapsing the policy.
      if is_primary:
        unique_domains = sorted(set(domains))
        for d in unique_domains:
          idx = np.array(
              [i for i, dd in enumerate(domains) if dd == d], dtype=np.int64
          )
          if idx.size == 0:
            continue
          dr = rewards[idx]
          logger.info(
              '[step %d][domain %s] n=%d reward mean=%.3f std=%.3f'
              ' min=%.3f max=%.3f',
              step,
              d,
              int(idx.size),
              float(dr.mean()),
              float(dr.std()),
              float(dr.min()),
              float(dr.max()),
          )

      # Diagnostic: surface the first completion + its domain/gt.
      if is_primary:
        _first = rollout_out.text[0] if rollout_out.text else ''
        logger.info(
            '[step %d] completion[0] len=%d domain=%s rt=%s gt=%r'
            ' acc_r=%.2f fmt_r=%.2f text=%r',
            step,
            len(_first),
            domains[0],
            reward_types[0],
            ground_truths[0],
            float(answer_rewards[0]),
            float(format_rewards[0]),
            _first[:300],
        )
        # Per-completion format diagnostics: how often does the model emit the
        # vero <think>/<answer>/\boxed structure? (drives the format reward.)
        for _ci, _t in enumerate(rollout_out.text):
          logger.info(
              '[step %d] compl[%d] start_think=%s has_thinkclose=%s'
              ' has_answer=%s has_boxed=%s acc=%.2f fmt=%.2f head=%r',
              step, _ci,
              _t.lstrip()[:7] == '<think>',
              '</think>' in _t, '<answer>' in _t, '\\boxed' in _t,
              float(answer_rewards[_ci]), float(format_rewards[_ci]),
              _t.lstrip()[:60],
          )
    else:
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
    enc_input_tokens = jnp.array(encoded.input_tokens)
    enc_positions = jnp.array(encoded.positions)
    enc_pixel_values = jnp.array(encoded.pixel_values, dtype=jnp.bfloat16)
    enc_padding_mask = jnp.array(encoded.input_mask).astype(jnp.bool_)
    enc_completion_mask = jnp.array(encoded.completion_mask)
    batch_size = enc_input_tokens.shape[0]
    micro_accum_active = _micro_accumulation_active(batch_size)
    if micro_accum_active:
      if batch_size != TRAIN_BATCH_SIZE:
        raise ValueError(
            'encoded train batch size differs from NUM_PROMPTS*NUM_GENERATIONS;'
            f' encoded B={batch_size}, configured B={TRAIN_BATCH_SIZE}.'
        )
      micro_patch_offsets = _vision_patch_offsets_per_sequence(
          encoded.vision_grid,
          batch_size=batch_size,
          total_patches=enc_pixel_values.shape[0],
      )
    else:
      micro_patch_offsets = None

    # 7. Grad step(s) (jitted, optimizer donated in place).
    with mesh:
      # GSPO measures its importance ratio against the behavior policy, so
      # capture the pre-update log-probs ONCE before the inner loop. The
      # GRPO path never reads this; pass a correctly-shaped zero so the
      # jitted train step keeps a single, stable signature.
      if LOSS_ALGO == 'gspo':
        old_per_token_logps = _compute_old_logps(
            model,
            input_tokens=enc_input_tokens,
            positions=enc_positions,
            pixel_values=enc_pixel_values,
            vision_grid=encoded.vision_grid,
            padding_mask=enc_padding_mask,
        )
      else:
        old_per_token_logps = jnp.zeros(
            (enc_input_tokens.shape[0], enc_input_tokens.shape[1] - 1),
            dtype=jnp.float32,
        )
      # NUM_ITERATIONS optimizer steps reusing this rollout. Iteration 0 is
      # on-policy (ratio == 1); iterations >=1 are off-policy, where GSPO's
      # sequence-level clip takes effect. GRPO uses NUM_ITERATIONS=1.
      for _inner_it in range(NUM_ITERATIONS):
        if micro_accum_active:
          loss = _train_step_accum(
              model,
              optimizer,
              input_tokens=enc_input_tokens,
              positions=enc_positions,
              pixel_values=enc_pixel_values,
              vision_grid=encoded.vision_grid,
              padding_mask=enc_padding_mask,
              completion_mask=enc_completion_mask,
              advantages=advantages,
              old_per_token_logps=old_per_token_logps,
              patch_offsets=micro_patch_offsets,
          )
        else:
          loss = _train_step(
              model,
              optimizer,
              input_tokens=enc_input_tokens,
              positions=enc_positions,
              pixel_values=enc_pixel_values,
              vision_grid=encoded.vision_grid,
              padding_mask=enc_padding_mask,
              completion_mask=enc_completion_mask,
              advantages=advantages,
              old_per_token_logps=old_per_token_logps,
          )
    rollout.sync_from_actor(nnx.state(model, nnx.Param))
    logger.info(
        '[step %d] loss=%.4f (algo=%s iters=%d)',
        step, float(loss), LOSS_ALGO, NUM_ITERATIONS,
    )

    if _METRICS_FH is not None:
      metrics = {
          'step': step,
          'train/loss': float(loss),
          'reward/mean': float(rewards.mean()),
          'reward/min': float(rewards.min()),
          'reward/max': float(rewards.max()),
          'reward/std': float(rewards.std()),
          'reward/answer_mean': float(answer_rewards.mean()),
          'reward/format_mean': float(format_rewards.mean()),
          'advantage/min': float(advantages_np.min()),
          'advantage/max': float(advantages_np.max()),
          'advantage/abs_mean': float(np.abs(advantages_np).mean()),
      }
      if domains is not None:
        for d in sorted(set(domains)):
          idx = np.array(
              [i for i, dd in enumerate(domains) if dd == d], dtype=np.int64
          )
          if idx.size:
            metrics[f'reward_by_domain/{d}'] = float(rewards[idx].mean())
      try:
        _METRICS_FH.write(json.dumps(metrics) + '\n')
      except Exception as e:  # logging must never crash training
        logger.warning('[metrics] write failed at step %d: %s', step, e)

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
  if _METRICS_FH is not None:
    try:
      _METRICS_FH.close()
    except Exception:  # pylint: disable=broad-except
      pass
  logger.info('GRPO smoke complete. Checkpoints in %s', CKPT_DIR)


if __name__ == '__main__' and '__file__' in globals():
  main()
