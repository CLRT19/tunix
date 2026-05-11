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

"""RL Trainer."""

import functools
from typing import Any, Callable, Optional

from flax import nnx
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike  # pylint: disable=g-importing-member
import optax
from tunix.rl import common
from tunix.sft import peft_trainer
from typing_extensions import override
from tunix.perf import trace as perf_trace
from tunix.perf.experimental import tracer as perf_tracer_lib
from tunix.sft.metrics_logger import MetricsLogger  # pylint: disable=unused-import


class Trainer(peft_trainer.PeftTrainer):
  """Handles additional RL metrics logging and display."""

  def __init__(
      self,
      model: nnx.Module,
      optimizer: optax.GradientTransformation,
      training_config: peft_trainer.TrainingConfig,
      custom_checkpoint_metadata_fn: Callable[[], dict[str, Any]],
      metrics_logger: Optional[MetricsLogger] = None,
      perf_tracer: Optional[perf_trace.Tracer] = None,
      perf_tracer_v2: Optional[perf_tracer_lib.Tracer] = None,
  ):
    super().__init__(
        model,
        optimizer,
        training_config,
        metrics_logger,
        perf_tracer,
        perf_tracer_v2,
    )
    self.rl_metrics_to_log = {}  # Metric name -> key in aux.
    self.tqdm_metrics_to_display = []
    self.custom_checkpoint_metadata_fn = custom_checkpoint_metadata_fn
    self._jitted_completion_grad_step_fn = None
    self._jitted_apply_grads_step_fn = None
    self._completion_microbatch_skip_jit = False
    self._completion_microbatch_cache_nnx_graph = True

  def with_rl_metrics_to_log(
      self,
      rl_metrics_to_log: dict[str, Callable[[ArrayLike], ArrayLike]],
  ) -> None:
    self.rl_metrics_to_log = rl_metrics_to_log

  def with_tqdm_metrics_to_display(
      self, tqdm_metrics_to_display: list[str | Callable[[], str]]
  ) -> None:
    self.tqdm_metrics_to_display = tqdm_metrics_to_display

  def _weighted_aux(self, aux: Any, loss_scale: ArrayLike) -> Any:
    if aux is None:
      return None
    return jax.tree.map(lambda x: x * loss_scale, aux)

  def _add_trees(self, lhs: Any | None, rhs: Any) -> Any:
    if lhs is None:
      return rhs
    return jax.tree.map(lambda x, y: x + y, lhs, rhs)

  @override
  def clear_jit_cache(self):
    super().clear_jit_cache()
    if hasattr(self, "_jitted_completion_grad_step_fn"):
      self._jitted_completion_grad_step_fn = None
      self._jitted_apply_grads_step_fn = None

  @override
  def jit_train_and_eval_step(
      self, skip_jit: bool = False, cache_nnx_graph: bool = True
  ):
    self._completion_microbatch_skip_jit = skip_jit
    self._completion_microbatch_cache_nnx_graph = cache_nnx_graph
    return super().jit_train_and_eval_step(skip_jit, cache_nnx_graph)

  @override
  def _train_step(
      self, model: nnx.Module, optimizer: nnx.Optimizer, inputs: Any
  ) -> tuple[ArrayLike, Any | None, ArrayLike]:
    """Runs a regular train step."""
    if not isinstance(inputs, common.TrainExampleMicroBatch):
      return super()._train_step(model, optimizer, inputs)
    raise ValueError(
        "TrainExampleMicroBatch must be handled by _call_train_step so each"
        " completion chunk is compiled separately."
    )

  def _completion_microbatch_grad_step(
      self, model: nnx.Module, inputs: Any
  ) -> tuple[ArrayLike, Any | None, Any]:
    """Computes one chunk's loss and gradients without updating the optimizer."""
    inputs = self.gen_model_input_fn(inputs)

    grad_fn = nnx.value_and_grad(
        self.loss_fn,
        argnums=nnx.DiffState(0, nnx.LoRAParam) if self._lora_enabled else 0,
        has_aux=self._has_aux,
    )
    out, grads = grad_fn(model, **inputs)
    if self._has_aux:
      loss, aux = out
      return loss, aux, grads
    return out, None, grads

  def _apply_accumulated_grads_step(
      self, model: nnx.Module, optimizer: nnx.Optimizer, grads: Any
  ) -> ArrayLike:
    grad_norm = optax.global_norm(grads)
    optimizer.update(model, grads)
    return grad_norm

  def _completion_microbatch_step_fns(
      self,
  ) -> tuple[
      Callable[[Any], tuple[ArrayLike, Any | None, Any]],
      Callable[[Any], ArrayLike],
  ]:
    if self._completion_microbatch_skip_jit:
      return (
          functools.partial(self._completion_microbatch_grad_step, self.model),
          functools.partial(
              self._apply_accumulated_grads_step, self.model, self.optimizer
          ),
      )

    if self._jitted_completion_grad_step_fn is None:
      grad_step = nnx.jit(self._completion_microbatch_grad_step)
      apply_grads_step = nnx.jit(
          self._apply_accumulated_grads_step, donate_argnames=("optimizer",)
      )

      def maybe_cache_and_partial(f, *args):
        if self._completion_microbatch_cache_nnx_graph:
          return functools.partial(nnx.cached_partial(f, *args))
        return functools.partial(f, *args)

      self._jitted_completion_grad_step_fn = maybe_cache_and_partial(
          grad_step, self.model
      )
      self._jitted_apply_grads_step_fn = maybe_cache_and_partial(
          apply_grads_step, self.model, self.optimizer
      )

    return (
        self._jitted_completion_grad_step_fn,
        self._jitted_apply_grads_step_fn,
    )

  @override
  def _call_train_step(
      self,
      train_step_fn: Callable[[Any], tuple[ArrayLike, Any | None, ArrayLike]],
      train_example: Any,
  ) -> tuple[ArrayLike, Any | None, ArrayLike]:
    if not isinstance(train_example, common.TrainExampleMicroBatch):
      return super()._call_train_step(train_step_fn, train_example)
    if not train_example.chunks:
      raise ValueError("TrainExampleMicroBatch must contain at least one chunk.")

    total_loss = jnp.asarray(0.0, dtype=jnp.float32)
    total_aux = None
    total_grads = None
    grad_step, apply_grads_step = self._completion_microbatch_step_fns()

    for chunk in train_example.chunks:
      loss_scale = getattr(chunk, "loss_scale", 1.0)
      loss, aux, grads = grad_step(chunk)
      grads = jax.tree.map(lambda grad: grad * loss_scale, grads)
      total_grads = self._add_trees(total_grads, grads)
      total_loss = total_loss + loss * loss_scale
      total_aux = self._add_trees(
          total_aux, self._weighted_aux(aux, loss_scale)
      )

    grad_norm = apply_grads_step(total_grads)
    return total_loss, total_aux, grad_norm

  @override
  def _eval_step(
      self, model: nnx.Module, inputs: Any
  ) -> ArrayLike | tuple[ArrayLike, Any]:
    if not isinstance(inputs, common.TrainExampleMicroBatch):
      return super()._eval_step(model, inputs)
    raise ValueError(
        "TrainExampleMicroBatch must be handled by _call_eval_step so each"
        " completion chunk is compiled separately."
    )

  @override
  def _call_eval_step(
      self,
      eval_step_fn: Callable[[Any], tuple[ArrayLike, Any | None]],
      eval_example: Any,
  ) -> tuple[ArrayLike, Any | None]:
    if not isinstance(eval_example, common.TrainExampleMicroBatch):
      return super()._call_eval_step(eval_step_fn, eval_example)
    if not eval_example.chunks:
      raise ValueError("TrainExampleMicroBatch must contain at least one chunk.")

    total_loss = jnp.asarray(0.0, dtype=jnp.float32)
    total_aux = None
    for chunk in eval_example.chunks:
      loss_scale = getattr(chunk, "loss_scale", 1.0)
      loss, aux = eval_step_fn(chunk)
      total_loss = total_loss + loss * loss_scale
      total_aux = self._add_trees(
          total_aux, self._weighted_aux(aux, loss_scale)
      )

    return total_loss, total_aux

  @override
  def custom_checkpoint_metadata(self) -> dict[str, Any]:
    return self.custom_checkpoint_metadata_fn()

  def restored_global_step(self) -> int:
    return self._restored_custom_metadata.get("global_step", 0)

  @override
  def _post_process_train_step(self, aux: Any) -> None:
    assert self._buffered_train_metrics is not None
    for metric_name, op in self.rl_metrics_to_log.items():
      if metric_name not in self._buffered_train_metrics.additional_metrics:
        self._buffered_train_metrics.additional_metrics[metric_name] = (
            [aux[metric_name]],
            op,
        )
      else:
        self._buffered_train_metrics.additional_metrics[metric_name][0].append(
            aux[metric_name]
        )

  @override
  def _post_process_eval_step(self, aux: Any) -> None:
    assert self._buffered_eval_metrics is not None
    for metric_name, op in self.rl_metrics_to_log.items():
      if metric_name not in self._buffered_eval_metrics.additional_metrics:
        self._buffered_eval_metrics.additional_metrics[metric_name] = (
            [aux[metric_name]],
            op,
        )
      else:
        self._buffered_eval_metrics.additional_metrics[metric_name][0].append(
            aux[metric_name]
        )

  def _get_additional_tqdm_metrics(self) -> list[str]:
    metrics = set()
    for key_or_fn in self.tqdm_metrics_to_display:
      if isinstance(key_or_fn, str):
        metrics.add(key_or_fn)
      elif val := key_or_fn():
        metrics.add(val)
    return list(metrics)

  @property
  def _tqdm_train_metrics(self) -> list[str]:
    metrics = super()._tqdm_train_metrics
    metrics.extend(self._get_additional_tqdm_metrics())
    return metrics
