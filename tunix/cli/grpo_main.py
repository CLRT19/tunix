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

"""Main entry point for GRPO training."""

import csv
import dataclasses
import os
import pathlib

from absl import app
from absl import flags
from absl import logging
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from tunix.cli import config
from tunix.cli.utils import data as data_lib
from tunix.cli.utils import model as model_lib
from tunix.examples.data import math_dataset as example_data
from tunix.perf import export as perf_export
from tunix.perf import metrics as perf_metrics
from tunix.perf.experimental import export as perf_export_v2
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.grpo import grpo_learner
from tunix.rl.rollout import base_rollout

GrpoConfig = grpo_learner.GrpoConfig

_PATHWAYS_BNS = flags.DEFINE_string(
    "pathways_bns", None, "BNS address of the Pathways server."
)


def _maybe_initialize_jax_distributed():
  """Initializes JAX distributed runtime when direct TPU-VM launch requests it."""
  if os.environ.get("TUNIX_JAX_DISTRIBUTED_AUTO_INIT", "").lower() not in (
      "1",
      "true",
      "yes",
  ):
    return
  if jax.distributed.is_initialized():
    return
  timeout_seconds = int(
      os.environ.get("TUNIX_JAX_DISTRIBUTED_INIT_TIMEOUT_SECONDS", "300")
  )
  logging.info(
      "Initializing JAX distributed runtime with automatic cluster detection."
  )
  jax.distributed.initialize(initialization_timeout=timeout_seconds)


def _flatten_metric_values(value):
  """Flattens buffered metric values without treating strings as iterables."""
  if isinstance(value, (str, bytes, np.str_)):
    return [value]
  if isinstance(value, (list, tuple)):
    flattened = []
    for item in value:
      flattened.extend(_flatten_metric_values(item))
    return flattened
  try:
    array_value = np.asarray(value)
  except Exception:  # pylint: disable=broad-exception-caught
    return [value]
  if array_value.ndim == 0:
    return [array_value.item()]
  return array_value.reshape(-1).tolist()


def _metric_values(metrics, metric_name):
  if metric_name not in metrics:
    return []
  values, _ = metrics[metric_name]
  return _flatten_metric_values(values)


def _value_at(values, index):
  return values[index] if index < len(values) else ""


def _create_trajectory_csv_logger(log_dir, max_rows_per_step):
  """Creates a per-process CSV logger for GRPO prompt/completion samples."""
  pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
  process_index = jax.process_index()
  csv_path = pathlib.Path(log_dir) / (
      f"grpo_trajectories_process_{process_index}.csv"
  )
  fieldnames = [
      "global_step",
      "mode",
      "process_index",
      "row",
      "reward_sum",
      "reward_mean",
      "reward_min",
      "reward_max",
      "completion_token_ids",
      "completion_token_count",
      "completion_nonpad_count",
      "completion_first_eos_index",
      "pad_id",
      "eos_id",
      "decoded_from_logged_token_ids",
      "prompt",
      "completion",
  ]

  def logger(metrics_buffer):
    metrics = metrics_buffer.metrics
    prompts = _metric_values(metrics, "prompts")
    completions = _metric_values(metrics, "completions")
    reward_sum = _metric_values(metrics, "rewards/sum")
    reward_mean = _metric_values(metrics, "rewards/mean")
    reward_min = _metric_values(metrics, "rewards/min")
    reward_max = _metric_values(metrics, "rewards/max")
    completion_token_ids = _metric_values(metrics, "completions/token_ids")
    completion_token_count = _metric_values(
        metrics, "completions/token_count"
    )
    completion_nonpad_count = _metric_values(
        metrics, "completions/nonpad_count"
    )
    completion_first_eos_index = _metric_values(
        metrics, "completions/first_eos_index"
    )
    pad_id = _metric_values(metrics, "completions/pad_id")
    eos_id = _metric_values(metrics, "completions/eos_id")
    decoded_from_logged_token_ids = _metric_values(
        metrics, "completions/decoded_from_token_ids"
    )

    row_count = max(
        len(prompts),
        len(completions),
        len(reward_sum),
        len(reward_mean),
        len(reward_min),
        len(reward_max),
        len(completion_token_ids),
        len(completion_token_count),
        len(completion_nonpad_count),
        len(completion_first_eos_index),
        len(pad_id),
        len(eos_id),
        len(decoded_from_logged_token_ids),
    )
    if row_count == 0:
      return
    if max_rows_per_step > 0:
      row_count = min(row_count, max_rows_per_step)

    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="", encoding="utf-8") as csv_file:
      writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
      if write_header:
        writer.writeheader()
      for row_index in range(row_count):
        writer.writerow({
            "global_step": metrics_buffer.global_steps,
            "mode": metrics_buffer.mode,
            "process_index": process_index,
            "row": row_index,
            "reward_sum": _value_at(reward_sum, row_index),
            "reward_mean": _value_at(reward_mean, row_index),
            "reward_min": _value_at(reward_min, row_index),
            "reward_max": _value_at(reward_max, row_index),
            "completion_token_ids": _value_at(
                completion_token_ids, row_index
            ),
            "completion_token_count": _value_at(
                completion_token_count, row_index
            ),
            "completion_nonpad_count": _value_at(
                completion_nonpad_count, row_index
            ),
            "completion_first_eos_index": _value_at(
                completion_first_eos_index, row_index
            ),
            "pad_id": _value_at(pad_id, row_index),
            "eos_id": _value_at(eos_id, row_index),
            "decoded_from_logged_token_ids": _value_at(
                decoded_from_logged_token_ids, row_index
            ),
            "prompt": _value_at(prompts, row_index),
            "completion": _value_at(completions, row_index),
        })

  return logger


class GrpoPipeline(config.HyperParameters):
  """Class for running the GRPO trainer."""

  def create_rollout_config(self):
    rollout_config = self.config["rollout_config"]

    # Get all valid field names from RolloutConfig
    valid_fields = {
        f.name for f in dataclasses.fields(base_rollout.RolloutConfig)
    }

    # Filter rollout_config to only include valid keys
    filtered_config = {
        k: v for k, v in rollout_config.items() if k in valid_fields
    }

    # Apply explicit recomputed/renamed values
    if "total_generation_steps" in rollout_config:
      filtered_config["max_tokens_to_generate"] = rollout_config[
          "total_generation_steps"
      ]
    if (
        "max_prompt_length" in rollout_config
        and "total_generation_steps" in rollout_config
    ):
      filtered_config["kv_cache_size"] = (
          rollout_config["max_prompt_length"]
          + rollout_config["total_generation_steps"]
          + 256
      )

    return base_rollout.RolloutConfig(**filtered_config)

  def create_role_to_mesh(self):
    default_mesh = self.create_mesh("actor_model_config")
    actor_mesh = reference_mesh = rollout_mesh = default_mesh
    if "reference_model_config" in self.config:
      reference_mesh = self.create_mesh("reference_model_config")
    if "rollout_model_config" in self.config:
      rollout_mesh = self.create_mesh("rollout_model_config")
    return {
        rl_cluster_lib.Role.ACTOR: actor_mesh,
        rl_cluster_lib.Role.REFERENCE: reference_mesh,
        rl_cluster_lib.Role.ROLLOUT: rollout_mesh,
    }

  def create_cluster_config(self):
    return rl_cluster_lib.ClusterConfig(
        role_to_mesh=self.create_role_to_mesh(),
        rollout_engine=self.config["rollout_engine"],
        offload_to_cpu=self.config["offload_to_cpu"],
        training_config=self.create_rl_training_config(),
        rollout_config=self.create_rollout_config(),
    )

  def create_rl_training_config(self):
    base_key = "rl_training_config"
    constructed_rl_training_config = self.obtain_training_config_dict(base_key)

    base_config = self.config[base_key]
    if base_config.get("actor_optimizer_config"):
      constructed_rl_training_config["actor_optimizer"] = self.create_optimizer(
          base_key, "actor_optimizer_config"
      )
    if base_config.get("critic_optimizer_config"):
      constructed_rl_training_config["critic_optimizer"] = (
          self.create_optimizer(base_key, "critic_optimizer_config")
      )

    return rl_cluster_lib.RLTrainingConfig(**constructed_rl_training_config)

  def create_perf_config(self, cluster_config: rl_cluster_lib.ClusterConfig):
    perf_metrics_options = cluster_config.training_config.perf_metrics_options
    if not perf_metrics_options:
      return None

    perf_config = perf_metrics.PerfMetricsConfig()

    if perf_metrics_options.enable_perf_v1:
      custom_export_fn_path = perf_metrics_options.custom_export_fn_path
      if custom_export_fn_path:
        perf_config.custom_export_fn = self._get_function_from_path(
            custom_export_fn_path
        )
        if perf_config.custom_export_fn is None:
          raise ValueError(
              "Could not load custom export function from"
              f" {custom_export_fn_path}"
          )
      else:
        perf_config.custom_export_fn = (
            perf_export.PerfMetricsExport.from_cluster_config(cluster_config)
        )

    if perf_metrics_options.enable_perf_v2:
      custom_export_fn_path_v2 = perf_metrics_options.custom_export_fn_path_v2
      if custom_export_fn_path_v2:
        perf_config.custom_export_fn_v2 = self._get_function_from_path(
            custom_export_fn_path_v2
        )
        if perf_config.custom_export_fn_v2 is None:
          raise ValueError(
              "Could not load custom export function v2 from"
              f" {custom_export_fn_path_v2}"
          )
      else:
        perf_config.custom_export_fn_v2 = (
            perf_export_v2.PerfMetricsExport.from_cluster_config(
                cluster_config=cluster_config,
                enable_trace_writer=perf_metrics_options.enable_trace_writer,
                trace_dir=perf_metrics_options.trace_dir,
            ).export_metrics
        )
    return perf_config

  def create_rl_cluster(self):
    # Should not use LoRA for reference model.
    if self.config["reference_model_config"].get("lora_config"):
      logging.warning(
          "LoRA config is not supported for the reference model. Disabling"
          " LoRA."
      )
      del self.config["reference_model_config"]["lora_config"]
    reference_model, tokenizer_path = model_lib.create_model(
        self.config["reference_model_config"],
        self.config["tokenizer_config"],
        self.create_mesh("reference_model_config"),
    )
    if self.config["actor_model_config"].get("lora_config", None):
      actor_model = model_lib.apply_lora_to_model(
          reference_model,
          self.create_mesh("actor_model_config"),
          self.config["actor_model_config"]["lora_config"],
      )
    else:
      graph_def, params = nnx.split(reference_model)
      actor_model = nnx.merge(
          graph_def,
          jax.tree.map(jnp.copy, params),
      )

    tokenizer = model_lib.create_tokenizer(
        self.config["tokenizer_config"], tokenizer_path
    )

    cluster_config = self.create_cluster_config()
    perf_config = self.create_perf_config(cluster_config)
    return rl_cluster_lib.RLCluster(
        actor=actor_model,
        reference=reference_model,
        tokenizer=tokenizer,
        cluster_config=cluster_config,
        perf_config=perf_config,
    )

  def run_grpo_trainer(self):
    grpo_trainer = grpo_learner.GrpoLearner(
        rl_cluster=self.create_rl_cluster(),
        reward_fns=self.obtain_reward_fn(),
        algo_config=GrpoConfig(**self.config["grpo_config"]),
    )
    trajectory_log_dir = os.environ.get("TUNIX_GRPO_TRAJECTORY_LOG_DIR")
    if trajectory_log_dir:
      max_rows_per_step = int(
          os.environ.get("TUNIX_GRPO_TRAJECTORY_MAX_ROWS_PER_STEP", "64")
      )
      grpo_trainer.rl_cluster.with_external_metrics_logger(
          _create_trajectory_csv_logger(
              trajectory_log_dir, max_rows_per_step
          )
      )
      logging.info(
          "Writing GRPO trajectory CSV samples to %s, max_rows_per_step=%d",
          trajectory_log_dir,
          max_rows_per_step,
      )

    tokenizer = grpo_trainer.rl_cluster.tokenizer
    if self.config.get("data_module", None):
      dataset = data_lib.get_dataset_from_module(
          self.config["data_module"],
          tokenizer,
      )
    elif self.config["data_source"] == "local":
      dataset = example_data.create_dataset(
          data_source=self.config["data_source"],
          dataset=self.config["data_directory"],
          tokenizer=tokenizer,
      )
    else:
      dataset = example_data.create_dataset(
          data_source="tfds",
          dataset=self.config["dataset_name"],
          tokenizer=tokenizer,
          tfds_download=self.config["tfds_download"],
      )
    dataset, _ = data_lib.post_init_dataset(
        dataset,
        tokenizer,
        batch_size=self.config["batch_size"],
        num_batches=self.config.get("num_batches", None),
        max_prompt_length=self.config["rollout_config"].get(
            "max_prompt_length", None
        ),
        shard_by_process=jax.process_count() > 1,
        batch_size_is_global=True,
    )
    grpo_trainer.train(dataset)


def _setup_jax_pathways(pathways_bns: str):
  """Sets up Jax with Pathways."""
  flags.FLAGS.pathways_ifrt = True
  jax.config.update("jax_xla_backend", "pathways")
  jax.config.update("jax_backend_target", pathways_bns)


def main(argv, **kwargs):
  if _PATHWAYS_BNS.value:
    _setup_jax_pathways(_PATHWAYS_BNS.value)
  else:
    _maybe_initialize_jax_distributed()
  pipeline = GrpoPipeline(argv, **kwargs)
  logging.info(
      "--- Launching GRPO pipeline with following config ---\n"
      "%r\n--------------------------",
      pipeline.config,
  )
  pipeline.run_grpo_trainer()


if __name__ == "__main__":
  app.run(main)
