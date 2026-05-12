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

"""Tests for RL trainer behavior."""

from absl.testing import absltest
from flax import nnx
import jax.numpy as jnp
import numpy as np
import optax
from tunix.rl import common
from tunix.rl import trainer as rl_trainer
from tunix.sft import peft_trainer


class ScalarModel(nnx.Module):

  def __init__(self):
    self.w = nnx.Param(jnp.asarray(1.0, dtype=jnp.float32))


def _train_example() -> common.TrainExample:
  return common.TrainExample(
      prompt_ids=jnp.ones((4, 2), dtype=jnp.int32),
      prompt_mask=jnp.ones((4, 2), dtype=jnp.int32),
      completion_ids=jnp.array([
          [1.0, 0.0, 0.0],
          [2.0, 4.0, 0.0],
          [1.0, 3.0, 5.0],
          [7.0, 0.0, 0.0],
      ]),
      completion_mask=jnp.array([
          [1, 0, 0],
          [1, 1, 0],
          [1, 1, 1],
          [1, 0, 0],
      ]),
      advantages=jnp.arange(4, dtype=jnp.float32),
      ref_per_token_logps=None,
      old_per_token_logps=None,
  )


def _loss_fn(model, train_example, loss_agg_mode):
  per_token_loss = model.w[...] * train_example.completion_ids
  loss = common.aggregate_loss(
      per_token_loss, train_example.completion_mask, loss_agg_mode
  )
  return loss, {"weighted_completion_id": loss}


class TrainerTest(absltest.TestCase):

  def _make_trainer(self, loss_agg_mode: str) -> rl_trainer.Trainer:
    model = ScalarModel()
    trainer = rl_trainer.Trainer(
        model=model,
        optimizer=optax.sgd(1.0),
        training_config=peft_trainer.TrainingConfig(
            eval_every_n_steps=1,
            max_steps=1,
        ),
        custom_checkpoint_metadata_fn=lambda: {},
    )
    trainer.with_loss_fn(
        lambda model, train_example: _loss_fn(
            model, train_example, loss_agg_mode
        ),
        has_aux=True,
    )
    trainer.with_gen_model_input_fn(lambda x: {"train_example": x})
    return trainer

  def test_completion_microbatch_update_matches_full_batch_token_mean(self):
    example = _train_example()
    full_trainer = self._make_trainer("token-mean")
    chunked_trainer = self._make_trainer("token-mean")
    chunks = common.split_train_example(
        example, chunk_size=1, loss_agg_mode="token-mean"
    )
    full_train_step, full_eval_step = full_trainer.jit_train_and_eval_step(
        skip_jit=True
    )
    chunked_train_step, chunked_eval_step = (
        chunked_trainer.jit_train_and_eval_step(skip_jit=True)
    )

    full_eval_loss, full_eval_aux = full_trainer._call_eval_step(
        full_eval_step, example
    )
    chunked_eval_loss, chunked_eval_aux = chunked_trainer._call_eval_step(
        chunked_eval_step,
        common.TrainExampleMicroBatch(chunks=tuple(chunks)),
    )
    np.testing.assert_allclose(
        chunked_eval_loss, full_eval_loss, rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        chunked_eval_aux["weighted_completion_id"],
        full_eval_aux["weighted_completion_id"],
        rtol=1e-6,
        atol=1e-6,
    )

    full_loss, full_aux, full_grad_norm = full_trainer._call_train_step(
        full_train_step, example
    )
    chunked_loss, chunked_aux, chunked_grad_norm = (
        chunked_trainer._call_train_step(
            chunked_train_step,
            common.TrainExampleMicroBatch(chunks=tuple(chunks)),
        )
    )

    with self.assertRaisesRegex(ValueError, "handled by _call_train_step"):
      chunked_trainer._train_step(
          chunked_trainer.model,
          chunked_trainer.optimizer,
          common.TrainExampleMicroBatch(chunks=tuple(chunks)),
      )
    with self.assertRaisesRegex(ValueError, "handled by _call_eval_step"):
      chunked_trainer._eval_step(
          chunked_trainer.model,
          common.TrainExampleMicroBatch(chunks=tuple(chunks)),
      )

    np.testing.assert_allclose(chunked_loss, full_loss, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        chunked_aux["weighted_completion_id"],
        full_aux["weighted_completion_id"],
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        chunked_grad_norm, full_grad_norm, rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        chunked_trainer.model.w[...],
        full_trainer.model.w[...],
        rtol=1e-6,
        atol=1e-6,
    )

  def test_completion_microbatch_jitted_hook_matches_full_batch_token_mean(self):
    example = _train_example()
    full_trainer = self._make_trainer("token-mean")
    chunked_trainer = self._make_trainer("token-mean")
    chunks = common.split_train_example(
        example, chunk_size=1, loss_agg_mode="token-mean"
    )
    full_train_step, _ = full_trainer.jit_train_and_eval_step(skip_jit=False)
    chunked_train_step, _ = chunked_trainer.jit_train_and_eval_step(
        skip_jit=False
    )

    full_loss, _, _ = full_trainer._call_train_step(full_train_step, example)
    chunked_loss, _, _ = chunked_trainer._call_train_step(
        chunked_train_step,
        common.TrainExampleMicroBatch(chunks=tuple(chunks)),
    )

    np.testing.assert_allclose(chunked_loss, full_loss, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        chunked_trainer.model.w[...],
        full_trainer.model.w[...],
        rtol=1e-6,
        atol=1e-6,
    )

  def test_completion_microbatch_update_matches_full_batch_sequence_mean(self):
    example = _train_example()
    full_trainer = self._make_trainer("sequence-mean-token-mean")
    chunked_trainer = self._make_trainer("sequence-mean-token-mean")
    full_train_step, _ = full_trainer.jit_train_and_eval_step(skip_jit=True)
    chunked_train_step, _ = chunked_trainer.jit_train_and_eval_step(
        skip_jit=True
    )

    _, _, _ = full_trainer._call_train_step(full_train_step, example)
    chunks = common.split_train_example(
        example, chunk_size=1, loss_agg_mode="sequence-mean-token-mean"
    )
    _, _, _ = chunked_trainer._call_train_step(
        chunked_train_step,
        common.TrainExampleMicroBatch(chunks=tuple(chunks)),
    )

    np.testing.assert_allclose(
        chunked_trainer.model.w[...],
        full_trainer.model.w[...],
        rtol=1e-6,
        atol=1e-6,
    )


if __name__ == "__main__":
  absltest.main()
