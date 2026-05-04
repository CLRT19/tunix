#!/usr/bin/env bash
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


set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
HF_TOKEN_ENV_FILE="${HF_TOKEN_ENV_FILE:-$HOME/.cache/tunix/hf_token_env}"
DATE_TIME_STR="${DATE_TIME_STR:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_LOG_DIR="${RUN_LOG_DIR:-/home/gs1693/repo/tunix/examples/rl/grpo/logs/${DATE_TIME_STR}_qwen3_simplereward}"
mkdir -p "$RUN_LOG_DIR"

if [[ -n "${TPU_WORKER_MODE:-}" ]]; then
    RUN_LOG_ROLE="worker"
else
    RUN_LOG_ROLE="launcher"
fi
RUN_LOG_HOST="$(hostname)"
RUN_LOG_STEM="${RUN_LOG_STEM:-qwen3_simplereward_${DATE_TIME_STR}}"
RUN_LOG_FILE="${RUN_LOG_FILE:-$RUN_LOG_DIR/${RUN_LOG_STEM}_${RUN_LOG_ROLE}_${RUN_LOG_HOST}.log}"

mkdir -p "$RUN_LOG_DIR"
exec > >(tee -a "$RUN_LOG_FILE") 2>&1
echo "Logging to $RUN_LOG_FILE"

# Prefer the wrapper-created env file so HF_TOKEN does not need to be embedded
# in the gcloud SSH command line.
if [[ -f "$HF_TOKEN_ENV_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$HF_TOKEN_ENV_FILE"
fi

# If not already running as a TPU worker, re-launch on all hosts via gcloud.
# Workers re-enter with TPU_WORKER_MODE=1 and skip this block.
if [[ -z "${TPU_WORKER_MODE:-}" ]]; then
    TPU_ENV=$(curl -sf "http://metadata.google.internal/computeMetadata/v1/instance/attributes/tpu-env" \
        -H "Metadata-Flavor: Google")
    NODE_ID=$(echo "$TPU_ENV" | grep "^NODE_ID:" | awk '{print $2}' | tr -d "'")
    ZONE=$(echo "$TPU_ENV" | grep "^ZONE:" | awk '{print $2}' | tr -d "'")

    # Pass through any user-overridden params
    PASSTHROUGH="TPU_WORKER_MODE=1"
    for var in HF_TOKEN_ENV_FILE model_name batch_size num_batches num_train_epochs warmup_ratio train_fraction rollout_mesh_shape rollout_tensor_parallel_size rollout_data_parallel_size TUNIX_VLLM_USE_LOCAL_TPU_DEVICE_IDS RUN_LOG_DIR RUN_LOG_STEM; do
        if [[ -n "${!var:-}" ]]; then
            printf -v quoted_value "%q" "${!var}"
            PASSTHROUGH="$PASSTHROUGH $var=$quoted_value"
        fi
    done

    SCRIPT=$(realpath "$0")
    printf -v quoted_script "%q" "$SCRIPT"
    REMOTE_CMD="$PASSTHROUGH bash $quoted_script"
    printf -v quoted_remote_cmd "%q" "$REMOTE_CMD"
    echo "Launching on all workers of $NODE_ID ($ZONE)..."
    gcloud compute tpus tpu-vm ssh "$NODE_ID" \
        --zone="$ZONE" \
        --worker=all \
        --command="bash -lc $quoted_remote_cmd"
    exit $?
fi

set -x # Enable xtrace

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
  echo "HF_TOKEN is required. Run launch_qwen3_simplereward_tpu.sh with HF_TOKEN set." >&2
  exit 1
fi

mkdir -p "$RUN_LOG_DIR"
export TPU_LOG_DIR="$RUN_LOG_DIR"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
cd "$REPO_ROOT"

# specify at cmd line to override defaults, e.g.
model_name=${model_name:-"Qwen3-1.7B-base"}
batch_size=${batch_size:-1}
num_batches=${num_batches:-3738}
num_train_epochs=${num_train_epochs:-1}
warmup_ratio=${warmup_ratio:-0.1}
train_fraction=${train_fraction:-1.0}
rollout_mesh_shape=${rollout_mesh_shape:-"(1,4)"}
rollout_tensor_parallel_size=${rollout_tensor_parallel_size:-4}
rollout_data_parallel_size=${rollout_data_parallel_size:-1}
export TUNIX_VLLM_USE_LOCAL_TPU_DEVICE_IDS="${TUNIX_VLLM_USE_LOCAL_TPU_DEVICE_IDS:-1}"

echo "Using parameters:"
echo "  Batch Size: $batch_size"
echo "  Num Batches: $num_batches"
echo "  Num Epochs: $num_train_epochs"
echo "  Warmup Ratio: $warmup_ratio"
echo "  Train Fraction: $train_fraction"
echo "  Rollout Mesh Shape: $rollout_mesh_shape"
echo "  Rollout TP/DP: $rollout_tensor_parallel_size/$rollout_data_parallel_size"

max_steps_float=$(awk "BEGIN {print $batch_size * $num_batches * $num_train_epochs * $train_fraction}")
max_steps=$(printf "%.0f" "$max_steps_float")
warmup_steps=$(awk "BEGIN {printf \"%.0f\", $warmup_ratio * $max_steps}")

echo "Max steps: $max_steps"
echo "Rounded warmup steps: $warmup_steps"

if (( max_steps < 1 )); then
  echo "Computed max_steps=$max_steps; increase num_batches or train_fraction so at least one training step runs." >&2
  exit 1
fi

"$PYTHON_BIN" -m tunix.cli.grpo_main \
  base_config.yaml \
  model_config.model_name=${model_name} \
  model_config.model_id=Qwen/${model_name} \
  model_config.model_source=huggingface \
  model_config.intermediate_ckpt_dir="/tmp/intermediate_ckpt/${model_name}" \
  model_config.mesh.shape="(2,4)" \
  model_config.mesh.axis_names="('fsdp','tp')" \
  model_config.rng_seed=42 \
  actor_model_config.lora_config.rank=64 \
  actor_model_config.lora_config.alpha=64.0 \
  actor_model_config.lora_config.module_path=".*q_einsum|.*kv_einsum|.*gate_proj|.*down_proj|.*up_proj|.*attn_vec_einsum" \
  actor_model_config.mesh.shape="(2,4)" \
  actor_model_config.mesh.axis_names="('fsdp','tp')" \
  rollout_model_config.mesh.shape="$rollout_mesh_shape" \
  rollout_model_config.mesh.axis_names="('fsdp','tp')" \
  tokenizer_config.tokenizer_path=Qwen/${model_name} \
  tokenizer_config.tokenizer_type=huggingface \
  tokenizer_config.add_bos=false \
  dataset_name="gsm8k" \
  batch_size=$batch_size \
  num_batches=$num_batches \
  num_test_batches=100 \
  num_train_epochs=$num_train_epochs \
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
  rl_training_config.eval_every_n_steps=10 \
  rl_training_config.max_steps=$max_steps \
  rl_training_config.metrics_logging_options.log_dir="/tmp/tensorboard/${model_name}" \
  rl_training_config.metrics_logging_options.flush_every_n_steps=20 \
  rl_training_config.checkpointing_options.save_interval_steps=500 \
  rl_training_config.checkpointing_options.max_to_keep=4 \
  rl_training_config.profiler_options={} \
  rollout_config.total_generation_steps=768 \
  rollout_config.max_prompt_length=256 \
  rollout_config.temperature=0.9 \
  rollout_config.top_p=1.0 \
  rollout_config.top_k=50 \
  rollout_config.rollout_vllm_model_version=Qwen/${model_name} \
  rollout_config.tensor_parallel_size=$rollout_tensor_parallel_size \
  rollout_config.data_parallel_size=$rollout_data_parallel_size \
  rollout_engine="vllm" \
  offload_to_cpu=false \
  grpo_config.num_generations=4 \
  grpo_config.num_iterations=1 \
  grpo_config.beta=0.08 \
  grpo_config.epsilon=0.2 \
  reward_functions="['tunix/cli/reward_fn/simple_math.py']"
