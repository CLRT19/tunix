#!/usr/bin/env bash
# Smoke-test variant of run_qwen3_simplereward.sh.
#
# Differences from the production script:
#   * rollout_engine="vanilla" (no vLLM dependency).
#   * Mesh shapes default to (8,4) for v5p-64 (32 chips, 8 hosts), and can
#     be overridden via SMOKE_MESH_SHAPE for single-host debugging.
#   * Tiny training budget: batch_size=8, num_batches=1, train_fraction=0.25
#     -> max_steps=2 so the run finishes in minutes.
#   * Reduced generation length, eval batches, and LoRA rank to keep the
#     smoke run fast.
#
# Invoke via launch_qwen3_simplereward_smoke_tpu.sh, which handles file
# sync + token plumbing across all TPU workers. For single-host debugging use
# launch_qwen3_simplereward_smoke_single_host_tpu.sh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
HF_TOKEN_ENV_FILE="${HF_TOKEN_ENV_FILE:-$HOME/.cache/tunix/hf_token_env}"
DATE_TIME_STR=$(date -u +%Y%m%dT%H%M%SZ)
SMOKE_LOG_DIR="${SMOKE_LOG_DIR:-/home/gs1693/repo/tunix/examples/rl/grpo/logs/${DATE_TIME_STR}_qwen3_simplereward_smoke}"
mkdir -p "$SMOKE_LOG_DIR"
SMOKE_LOG_STEM="${SMOKE_LOG_STEM:-qwen3_simplereward_smoke_$(date -u +%Y%m%dT%H%M%SZ)}"
if [[ -n "${TPU_WORKER_MODE:-}" ]]; then
    SMOKE_LOG_ROLE="worker"
else
    SMOKE_LOG_ROLE="launcher"
fi
SMOKE_LOG_HOST="$(hostname -s 2>/dev/null || hostname)"
SMOKE_LOG_FILE="${SMOKE_LOG_FILE:-$SMOKE_LOG_DIR/${SMOKE_LOG_STEM}_${SMOKE_LOG_ROLE}_${SMOKE_LOG_HOST}.log}"

mkdir -p "$SMOKE_LOG_DIR"
exec > >(tee -a "$SMOKE_LOG_FILE") 2>&1
echo "[smoke] Logging shell output to $SMOKE_LOG_FILE"

if [[ -f "$HF_TOKEN_ENV_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$HF_TOKEN_ENV_FILE"
fi

if [[ -z "${TPU_WORKER_MODE:-}" ]]; then
    TPU_ENV=$(curl -sf "http://metadata.google.internal/computeMetadata/v1/instance/attributes/tpu-env" \
        -H "Metadata-Flavor: Google")
    NODE_ID=$(echo "$TPU_ENV" | grep "^NODE_ID:" | awk '{print $2}' | tr -d "'")
    ZONE=$(echo "$TPU_ENV" | grep "^ZONE:" | awk '{print $2}' | tr -d "'")

    PASSTHROUGH="TPU_WORKER_MODE=1"
    for var in HF_TOKEN_ENV_FILE model_name SMOKE_LOG_DIR SMOKE_LOG_STEM SMOKE_MESH_SHAPE SMOKE_MESH_AXIS_NAMES SMOKE_BATCH_SIZE SMOKE_NUM_BATCHES SMOKE_NUM_TRAIN_EPOCHS SMOKE_TRAIN_FRACTION SMOKE_MAX_STEPS SMOKE_NUM_GENERATIONS SMOKE_TOTAL_GENERATION_STEPS SMOKE_MAX_PROMPT_LENGTH SMOKE_NUM_TEST_BATCHES; do
        if [[ -n "${!var:-}" ]]; then
            printf -v quoted_value "%q" "${!var}"
            PASSTHROUGH="$PASSTHROUGH $var=$quoted_value"
        fi
    done

    SCRIPT=$(realpath "$0")
    printf -v quoted_script "%q" "$SCRIPT"
    REMOTE_CMD="$PASSTHROUGH bash $quoted_script"
    printf -v quoted_remote_cmd "%q" "$REMOTE_CMD"
    echo "Launching smoke run on all workers of $NODE_ID ($ZONE)..."
    gcloud compute tpus tpu-vm ssh "$NODE_ID" \
        --zone="$ZONE" \
        --worker=all \
        --command="bash -lc $quoted_remote_cmd"
    exit $?
fi

set -x

MINICONDA_DIR="$HOME/miniconda3"
CONDA_ENV_NAME="tunix"
PYTHON_BIN="$MINICONDA_DIR/envs/$CONDA_ENV_NAME/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python env not found: $PYTHON_BIN" >&2
  exit 1
fi

if [[ -f "$MINICONDA_DIR/etc/profile.d/conda.sh" ]]; then
  source "$MINICONDA_DIR/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV_NAME"
else
  export CONDA_PREFIX="$MINICONDA_DIR/envs/$CONDA_ENV_NAME"
  export CONDA_DEFAULT_ENV="$CONDA_ENV_NAME"
  export PATH="$CONDA_PREFIX/bin:${PATH:-}"
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is required. Run launch_qwen3_simplereward_smoke_tpu.sh with HF_TOKEN set." >&2
  exit 1
fi

export TPU_LOG_DIR="$SMOKE_LOG_DIR"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
cd "$REPO_ROOT"

model_name=${model_name:-"Qwen3-1.7B-base"}
mesh_shape=${SMOKE_MESH_SHAPE:-"(8,4)"}
mesh_axis_names=${SMOKE_MESH_AXIS_NAMES:-"('fsdp','tp')"}
batch_size=${SMOKE_BATCH_SIZE:-8}
num_batches=${SMOKE_NUM_BATCHES:-1}
num_train_epochs=${SMOKE_NUM_TRAIN_EPOCHS:-1}
warmup_ratio=0.0
train_fraction=${SMOKE_TRAIN_FRACTION:-0.25}  # default max_steps = 8 * 1 * 1 * 0.25 = 2
num_test_batches=${SMOKE_NUM_TEST_BATCHES:-2}
num_generations=${SMOKE_NUM_GENERATIONS:-4}
total_generation_steps=${SMOKE_TOTAL_GENERATION_STEPS:-128}
max_prompt_length=${SMOKE_MAX_PROMPT_LENGTH:-256}

echo "[smoke] Using parameters:"
echo "  Model: $model_name"
echo "  Batch Size: $batch_size"
echo "  Num Batches: $num_batches"
echo "  Train Fraction: $train_fraction"
echo "  Num Generations: $num_generations"
echo "  Generation Steps: $total_generation_steps"
echo "  Max Prompt Length: $max_prompt_length"
echo "  Mesh: $mesh_shape $mesh_axis_names"

if [[ -n "${SMOKE_MAX_STEPS:-}" ]]; then
  max_steps="$SMOKE_MAX_STEPS"
else
  max_steps_float=$(awk "BEGIN {print $batch_size * $num_batches * $num_train_epochs * $train_fraction}")
  max_steps=$(printf "%.0f" "$max_steps_float")
fi
warmup_steps=0

echo "  Max steps: $max_steps"

if (( max_steps < 1 )); then
  echo "Computed max_steps=$max_steps; smoke test needs at least one step." >&2
  exit 1
fi

"$PYTHON_BIN" -m tunix.cli.grpo_main \
  base_config.yaml \
  model_config.model_name=${model_name} \
  model_config.model_id=Qwen/${model_name} \
  model_config.model_source=huggingface \
  model_config.intermediate_ckpt_dir="/tmp/intermediate_ckpt/${model_name}" \
  model_config.mesh.shape="$mesh_shape" \
  model_config.mesh.axis_names="$mesh_axis_names" \
  model_config.rng_seed=42 \
  actor_model_config.lora_config.rank=8 \
  actor_model_config.lora_config.alpha=16.0 \
  actor_model_config.lora_config.module_path=".*q_einsum|.*kv_einsum|.*gate_proj|.*down_proj|.*up_proj|.*attn_vec_einsum" \
  actor_model_config.mesh.shape="$mesh_shape" \
  actor_model_config.mesh.axis_names="$mesh_axis_names" \
  rollout_model_config.mesh.shape="$mesh_shape" \
  rollout_model_config.mesh.axis_names="$mesh_axis_names" \
  tokenizer_config.tokenizer_path=Qwen/${model_name} \
  tokenizer_config.tokenizer_type=huggingface \
  tokenizer_config.add_bos=false \
  dataset_name="gsm8k" \
  batch_size=$batch_size \
  num_batches=$num_batches \
  num_test_batches=$num_test_batches \
  num_train_epochs=$num_train_epochs \
  rl_training_config.mini_batch_size=$batch_size \
  rl_training_config.train_micro_batch_size=$batch_size \
  rl_training_config.rollout_micro_batch_size=$batch_size \
  rl_training_config.compute_logps_micro_batch_size=$batch_size \
  rl_training_config.actor_optimizer_config.opt_type="adamw" \
  rl_training_config.actor_optimizer_config.peak_value=3e-6 \
  rl_training_config.actor_optimizer_config.schedule_type="warmup_cosine_decay_schedule" \
  rl_training_config.actor_optimizer_config.init_value=0.0 \
  rl_training_config.actor_optimizer_config.end_value=0.0 \
  rl_training_config.actor_optimizer_config.warmup_ratio=$warmup_ratio \
  rl_training_config.actor_optimizer_config.warmup_steps=$warmup_steps \
  rl_training_config.actor_optimizer_config.decay_steps=$max_steps \
  rl_training_config.actor_optimizer_config.b1=0.9 \
  rl_training_config.actor_optimizer_config.b2=0.99 \
  rl_training_config.actor_optimizer_config.weight_decay=0.1 \
  rl_training_config.actor_optimizer_config.max_grad_norm=0.1 \
  rl_training_config.eval_every_n_steps=100 \
  rl_training_config.max_steps=$max_steps \
  rl_training_config.metrics_logging_options.log_dir="/tmp/tensorboard/${model_name}_smoke" \
  rl_training_config.metrics_logging_options.flush_every_n_steps=1 \
  rl_training_config.checkpointing_options.save_interval_steps=1000 \
  rl_training_config.checkpointing_options.max_to_keep=1 \
  rl_training_config.profiler_options={} \
  rollout_config.total_generation_steps=$total_generation_steps \
  rollout_config.max_prompt_length=$max_prompt_length \
  rollout_config.temperature=0.9 \
  rollout_config.top_p=1.0 \
  rollout_config.top_k=50 \
  rollout_engine="vanilla" \
  offload_to_cpu=false \
  grpo_config.num_generations=$num_generations \
  grpo_config.num_iterations=1 \
  grpo_config.beta=0.08 \
  grpo_config.epsilon=0.2 \
  reward_functions="['tunix/cli/reward_fn/simple_math.py']"
