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
#
# Smoke variant of run_qwen3_simplereward.sh: same code path, tiny inputs.
# Use to debug a Pathways multi-host setup end-to-end in minutes:
#   1) preflight prints jax.devices() so missing/extra hosts surface fast
#   2) 2 training steps, short prompts, no eval/checkpoint overhead

set -x

# All knobs overridable, but defaults are tuned for "fail fast".
model_name=${model_name:-"Qwen3-1.7B-base"}
batch_size=${batch_size:-1}
num_batches=${num_batches:-2}
num_train_epochs=${num_train_epochs:-1}
warmup_ratio=${warmup_ratio:-0.0}
train_fraction=${train_fraction:-1.0}
mesh_shape=${mesh_shape:-"(2,4)"}
total_generation_steps=${total_generation_steps:-64}
max_prompt_length=${max_prompt_length:-128}
num_generations=${num_generations:-2}

max_steps_float=$(awk "BEGIN {print $batch_size * $num_batches * $num_train_epochs * $train_fraction}")
max_steps=$(printf "%.0f" "$max_steps_float")
warmup_steps=0

echo "[smoke] max_steps=${max_steps} mesh=${mesh_shape} model=${model_name}"

# --- Preflight: confirm Pathways is wired up before pulling weights -----------
python3 -c "
import jax
devs = jax.devices()
print('[smoke] jax.device_count =', jax.device_count())
print('[smoke] jax.process_count =', jax.process_count())
print('[smoke] jax.process_index =', jax.process_index())
print('[smoke] devices =', devs)
assert jax.device_count() > 0, 'no JAX devices visible to Pathways'
" || { echo "[smoke] preflight FAILED"; exit 1; }

# --- Tiny training run --------------------------------------------------------
python3 -m tunix.cli.grpo_main \
  base_config.yaml \
  model_config.model_name=${model_name} \
  model_config.model_id=Qwen/${model_name} \
  model_config.model_source=huggingface \
  model_config.use_flash_attention=true \
  model_config.flash_attention_block_size=256 \
  model_config.intermediate_ckpt_dir="/tmp/intermediate_ckpt/${model_name}-smoke" \
  model_config.mesh.shape="${mesh_shape}" \
  model_config.mesh.axis_names="('fsdp','tp')" \
  model_config.rng_seed=42 \
  actor_model_config.lora_config.rank=64 \
  actor_model_config.lora_config.alpha=64.0 \
  actor_model_config.lora_config.module_path=".*q_einsum|.*kv_einsum|.*gate_proj|.*down_proj|.*up_proj|.*attn_vec_einsum" \
  actor_model_config.mesh.shape="${mesh_shape}" \
  actor_model_config.mesh.axis_names="('fsdp','tp')" \
  rollout_model_config.mesh.shape="${mesh_shape}" \
  rollout_model_config.mesh.axis_names="('fsdp','tp')" \
  tokenizer_config.tokenizer_path=Qwen/${model_name} \
  tokenizer_config.tokenizer_type=huggingface \
  tokenizer_config.add_bos=false \
  dataset_name="gsm8k" \
  batch_size=$batch_size \
  num_batches=$num_batches \
  num_test_batches=1 \
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
  rl_training_config.eval_every_n_steps=999999 \
  rl_training_config.max_steps=$max_steps \
  rl_training_config.metrics_logging_options.log_dir="/tmp/tensorboard/${model_name}-smoke" \
  rl_training_config.metrics_logging_options.flush_every_n_steps=1 \
  rl_training_config.checkpointing_options.save_interval_steps=999999 \
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

echo "[smoke] OK"
