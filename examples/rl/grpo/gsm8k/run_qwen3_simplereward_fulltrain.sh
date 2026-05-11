#!/usr/bin/env bash
# Multi-host GRPO full fine-tuning run for Qwen3 GSM8K simple reward.
#
# Invoke through launch_qwen3_simplereward_fulltrain_tpu.sh from worker 0. This
# script re-enters on all TPU workers, mounts the configured GCS bucket, writes
# checkpoints/TensorBoard/trajectory CSVs to the bucket, and writes shell logs
# locally under examples/rl/grpo/logs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
MOUNT_SCRIPT="$REPO_ROOT/mount_bucket.sh"
HF_TOKEN_ENV_FILE="${HF_TOKEN_ENV_FILE:-$HOME/.cache/tunix/hf_token_env}"
DATE_TIME_STR="${DATE_TIME_STR:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_LOG_DIR="${RUN_LOG_DIR:-/home/gs1693/repo/tunix/examples/rl/grpo/logs/${DATE_TIME_STR}_qwen3_simplereward_fulltrain_multihost}"
RUN_LOG_STEM="${RUN_LOG_STEM:-qwen3_simplereward_fulltrain_multihost_${DATE_TIME_STR}}"

mkdir -p "$RUN_LOG_DIR"
if [[ -n "${TPU_WORKER_MODE:-}" ]]; then
  RUN_LOG_ROLE="worker"
else
  RUN_LOG_ROLE="launcher"
fi
RUN_LOG_HOST="$(hostname -s 2>/dev/null || hostname)"
RUN_LOG_FILE="${RUN_LOG_FILE:-$RUN_LOG_DIR/${RUN_LOG_STEM}_${RUN_LOG_ROLE}_${RUN_LOG_HOST}.log}"

exec > >(tee -a "$RUN_LOG_FILE") 2>&1
echo "[fulltrain] Logging shell output to $RUN_LOG_FILE"

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
  for var in HF_TOKEN_ENV_FILE DATE_TIME_STR RUN_LOG_DIR RUN_LOG_STEM BUCKET_NAME BUCKET_MOUNT_POINT BUCKET_RUN_ROOT CHECKPOINT_ROOT TENSORBOARD_LOG_DIR TRAJECTORY_LOG_DIR TUNIX_JAX_DISTRIBUTED_AUTO_INIT TUNIX_JAX_DISTRIBUTED_INIT_TIMEOUT_SECONDS TUNIX_SKIP_FINAL_CHECKPOINT TUNIX_DISABLE_SAMPLER_STATE_SHARDING TUNIX_SAMPLER_DEBUG_SHARDS model_name batch_size mini_batch_size train_micro_batch_size actor_train_completion_micro_batch_size rollout_micro_batch_size compute_logps_micro_batch_size num_batches num_train_epochs warmup_ratio max_steps model_mesh_shape model_mesh_axis_names rollout_engine rollout_mesh_shape rollout_tensor_parallel_size rollout_data_parallel_size TUNIX_VLLM_USE_LOCAL_TPU_DEVICE_IDS total_generation_steps max_prompt_length num_generations rollout_vllm_max_num_seqs rollout_vllm_max_num_batched_tokens rollout_temperature rollout_top_p rollout_top_k num_test_batches eval_every_n_steps metrics_flush_every_n_steps save_interval_steps max_to_keep trajectory_max_rows_per_step; do
    if [[ -v "$var" ]]; then
      printf -v quoted_value "%q" "${!var}"
      PASSTHROUGH="$PASSTHROUGH $var=$quoted_value"
    fi
  done

  SCRIPT=$(realpath "$0")
  printf -v quoted_script "%q" "$SCRIPT"
  REMOTE_CMD="$PASSTHROUGH bash $quoted_script"
  printf -v quoted_remote_cmd "%q" "$REMOTE_CMD"
  echo "[fulltrain] Launching on all workers of $NODE_ID ($ZONE)..."
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
  echo "HF_TOKEN is required. Run launch_qwen3_simplereward_fulltrain_tpu.sh with HF_TOKEN set." >&2
  exit 1
fi

if [[ ! -f "$MOUNT_SCRIPT" ]]; then
  echo "Bucket mount script not found: $MOUNT_SCRIPT" >&2
  exit 1
fi

BUCKET_NAME="${BUCKET_NAME:-linrong-vlm-tpu-us-central1-a}"
BUCKET_MOUNT_POINT="${BUCKET_MOUNT_POINT:-$HOME/bucket}"
bash "$MOUNT_SCRIPT" "$BUCKET_NAME" "$BUCKET_MOUNT_POINT"

BUCKET_RUN_ROOT="${BUCKET_RUN_ROOT:-$BUCKET_MOUNT_POINT/tunix_runs/grpo_gsm8k_qwen3_simplereward_fulltrain/${DATE_TIME_STR}}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$BUCKET_RUN_ROOT/checkpoints}"
TENSORBOARD_LOG_DIR="${TENSORBOARD_LOG_DIR:-$BUCKET_RUN_ROOT/tensorboard}"
TRAJECTORY_LOG_DIR="${TRAJECTORY_LOG_DIR:-$BUCKET_RUN_ROOT/trajectories}"

mkdir -p "$RUN_LOG_DIR" "$CHECKPOINT_ROOT" "$TENSORBOARD_LOG_DIR" "$TRAJECTORY_LOG_DIR"
export TPU_LOG_DIR="$RUN_LOG_DIR"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export TUNIX_JAX_DISTRIBUTED_AUTO_INIT="${TUNIX_JAX_DISTRIBUTED_AUTO_INIT:-1}"
export TUNIX_JAX_DISTRIBUTED_INIT_TIMEOUT_SECONDS="${TUNIX_JAX_DISTRIBUTED_INIT_TIMEOUT_SECONDS:-300}"
export TUNIX_SKIP_FINAL_CHECKPOINT="${TUNIX_SKIP_FINAL_CHECKPOINT:-0}"
export TUNIX_DISABLE_SAMPLER_STATE_SHARDING="${TUNIX_DISABLE_SAMPLER_STATE_SHARDING:-0}"
export TUNIX_GRPO_TRAJECTORY_LOG_DIR="$TRAJECTORY_LOG_DIR"
export TUNIX_GRPO_TRAJECTORY_MAX_ROWS_PER_STEP="${trajectory_max_rows_per_step:-64}"
cd "$REPO_ROOT"

model_name=${model_name:-"Qwen3-1.7B-base"}
batch_size=${batch_size:-32}
mini_batch_size=${mini_batch_size:-$batch_size}
train_micro_batch_size=${train_micro_batch_size:-8}
actor_train_completion_micro_batch_size=${actor_train_completion_micro_batch_size:-1}
rollout_micro_batch_size=${rollout_micro_batch_size:-$train_micro_batch_size}
compute_logps_micro_batch_size=${compute_logps_micro_batch_size:-$train_micro_batch_size}
num_batches=${num_batches:-4000}
num_train_epochs=${num_train_epochs:-4}
warmup_ratio=${warmup_ratio:-0.1}
max_steps=${max_steps:-4000}
model_mesh_shape=${model_mesh_shape:-"(8,4)"}
model_mesh_axis_names=${model_mesh_axis_names:-"('fsdp','tp')"}
rollout_engine=${rollout_engine:-"vanilla"}
rollout_mesh_shape_was_set="${rollout_mesh_shape+x}"
rollout_mesh_shape=${rollout_mesh_shape:-"(1,4)"}
rollout_tensor_parallel_size=${rollout_tensor_parallel_size:-4}
rollout_data_parallel_size=${rollout_data_parallel_size:-1}
total_generation_steps=${total_generation_steps:-8096}
max_prompt_length=${max_prompt_length:-256}
num_generations=${num_generations:-4}
rollout_vllm_max_num_seqs=${rollout_vllm_max_num_seqs:-4}
rollout_vllm_max_num_batched_tokens=${rollout_vllm_max_num_batched_tokens:-32768}
rollout_temperature=${rollout_temperature:-0.9}
rollout_top_p=${rollout_top_p-1.0}
rollout_top_k=${rollout_top_k-50}
num_test_batches=${num_test_batches:-100}
eval_every_n_steps=${eval_every_n_steps:-100}
metrics_flush_every_n_steps=${metrics_flush_every_n_steps:-5}
save_interval_steps=${save_interval_steps:-250}
max_to_keep=${max_to_keep:-4}
export TUNIX_VLLM_USE_LOCAL_TPU_DEVICE_IDS="${TUNIX_VLLM_USE_LOCAL_TPU_DEVICE_IDS:-1}"
disable_sampler_state_sharding_normalized="$(printf '%s' "$TUNIX_DISABLE_SAMPLER_STATE_SHARDING" | tr '[:upper:]' '[:lower:]')"
if [[ "$rollout_engine" == "vanilla" ]] && [[ "$disable_sampler_state_sharding_normalized" =~ ^(1|true|yes|on)$ ]]; then
  if [[ -z "$rollout_mesh_shape_was_set" ]]; then
    rollout_mesh_shape="(4,8)"
  fi
  effective_rollout_mesh_shape="$rollout_mesh_shape"
elif [[ "$rollout_engine" == "vanilla" ]]; then
  effective_rollout_mesh_shape="$model_mesh_shape"
else
  effective_rollout_mesh_shape="$rollout_mesh_shape"
fi

warmup_steps=$(awk "BEGIN {printf \"%.0f\", $warmup_ratio * $max_steps}")

actor_train_args=()
if [[ -n "$actor_train_completion_micro_batch_size" ]]; then
  actor_train_args+=(
    rl_training_config.actor_train_completion_micro_batch_size="$actor_train_completion_micro_batch_size"
  )
fi
rollout_sampling_args=(rollout_config.temperature="$rollout_temperature")
if [[ -n "$rollout_top_p" ]]; then
  rollout_sampling_args+=(rollout_config.top_p="$rollout_top_p")
fi
if [[ -n "$rollout_top_k" ]]; then
  rollout_sampling_args+=(rollout_config.top_k="$rollout_top_k")
fi

echo "[fulltrain] Using parameters:"
echo "  Model: $model_name"
echo "  Global Prompt Batch Size/Data Step: $batch_size"
echo "  Global PPO Mini Batch Size: $mini_batch_size"
echo "  Global Train/Rollout/Logps Micro Batch Sizes: $train_micro_batch_size/$rollout_micro_batch_size/$compute_logps_micro_batch_size"
echo "  Local Actor Completion Micro Batch Size: ${actor_train_completion_micro_batch_size:-disabled}"
echo "  Num Generations: $num_generations"
echo "  Effective PPO Mini Batch Completions: $((mini_batch_size * num_generations))"
echo "  Global Rollout Batch Completions/Data Step: $((batch_size * num_generations))"
echo "  Num Batches/Data Steps: $num_batches"
echo "  Max Steps: $max_steps"
echo "  Num Epochs: $num_train_epochs"
echo "  Warmup Ratio/Steps: $warmup_ratio/$warmup_steps"
echo "  Generation Steps: $total_generation_steps"
echo "  Max Prompt Length: $max_prompt_length"
echo "  vLLM Max Seqs/Batched Tokens: $rollout_vllm_max_num_seqs/$rollout_vllm_max_num_batched_tokens"
echo "  Actor/Reference Mesh: $model_mesh_shape $model_mesh_axis_names"
echo "  Rollout Engine: $rollout_engine"
echo "  Rollout Mesh: $effective_rollout_mesh_shape"
echo "  Rollout TP/DP: $rollout_tensor_parallel_size/$rollout_data_parallel_size"
echo "  Rollout Sampling: temperature=$rollout_temperature top_p=${rollout_top_p:-<omitted>} top_k=${rollout_top_k:-<omitted>}"
echo "  Disable Sampler State Sharding: $TUNIX_DISABLE_SAMPLER_STATE_SHARDING"
echo "  Local shell logs: $RUN_LOG_DIR"
echo "  Bucket run root: $BUCKET_RUN_ROOT"
echo "  Checkpoints: $CHECKPOINT_ROOT"
echo "  TensorBoard: $TENSORBOARD_LOG_DIR"
echo "  Trajectory CSVs: $TRAJECTORY_LOG_DIR"

if (( max_steps < 1 )); then
  echo "Computed max_steps=$max_steps; increase max_steps or num_batches." >&2
  exit 1
fi
if (( batch_size % mini_batch_size != 0 )); then
  echo "batch_size=$batch_size must be divisible by mini_batch_size=$mini_batch_size." >&2
  exit 1
fi
if (( mini_batch_size % train_micro_batch_size != 0 )); then
  echo "mini_batch_size=$mini_batch_size must be divisible by train_micro_batch_size=$train_micro_batch_size." >&2
  exit 1
fi

"$PYTHON_BIN" -m tunix.cli.grpo_main \
  base_config.yaml \
  model_config.model_name=${model_name} \
  model_config.model_id=Qwen/${model_name} \
  model_config.model_source=huggingface \
  model_config.intermediate_ckpt_dir="/tmp/intermediate_ckpt/${model_name}" \
  model_config.mesh.shape="$model_mesh_shape" \
  model_config.mesh.axis_names="$model_mesh_axis_names" \
  model_config.rng_seed=42 \
  actor_model_config.lora_config={} \
  actor_model_config.mesh.shape="$model_mesh_shape" \
  actor_model_config.mesh.axis_names="$model_mesh_axis_names" \
  rollout_model_config.mesh.shape="$effective_rollout_mesh_shape" \
  rollout_model_config.mesh.axis_names="$model_mesh_axis_names" \
  tokenizer_config.tokenizer_path=Qwen/${model_name} \
  tokenizer_config.tokenizer_type=huggingface \
  tokenizer_config.add_bos=false \
  dataset_name="gsm8k" \
  batch_size=$batch_size \
  num_batches=$num_batches \
  num_test_batches=$num_test_batches \
  num_train_epochs=$num_train_epochs \
  rl_training_config.mini_batch_size=$mini_batch_size \
  rl_training_config.train_micro_batch_size=$train_micro_batch_size \
  "${actor_train_args[@]}" \
  rl_training_config.rollout_micro_batch_size=$rollout_micro_batch_size \
  rl_training_config.compute_logps_micro_batch_size=$compute_logps_micro_batch_size \
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
  rl_training_config.eval_every_n_steps=$eval_every_n_steps \
  rl_training_config.max_steps=$max_steps \
  rl_training_config.checkpoint_root_directory="$CHECKPOINT_ROOT" \
  rl_training_config.metrics_logging_options.log_dir="$TENSORBOARD_LOG_DIR" \
  rl_training_config.metrics_logging_options.flush_every_n_steps=$metrics_flush_every_n_steps \
  rl_training_config.checkpointing_options.save_interval_steps=$save_interval_steps \
  rl_training_config.checkpointing_options.max_to_keep=$max_to_keep \
  rl_training_config.profiler_options={} \
  rollout_config.total_generation_steps=$total_generation_steps \
  rollout_config.max_prompt_length=$max_prompt_length \
  rollout_config.rollout_vllm_max_num_seqs=$rollout_vllm_max_num_seqs \
  rollout_config.rollout_vllm_max_num_batched_tokens=$rollout_vllm_max_num_batched_tokens \
  "${rollout_sampling_args[@]}" \
  rollout_config.rollout_vllm_model_version=Qwen/${model_name} \
  rollout_config.tensor_parallel_size=$rollout_tensor_parallel_size \
  rollout_config.data_parallel_size=$rollout_data_parallel_size \
  rollout_engine="$rollout_engine" \
  offload_to_cpu=false \
  grpo_config.num_generations=$num_generations \
  grpo_config.num_iterations=1 \
  grpo_config.beta=0.08 \
  grpo_config.epsilon=0.2 \
  reward_functions="['tunix/cli/reward_fn/simple_math.py']"
