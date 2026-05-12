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

"""Tests for rollout stop token pass-through."""

from absl.testing import absltest
import numpy as np
from tunix.rl.rollout import base_rollout


class _FakeSamplerOutput:

  text = ["ok"]
  tokens = [np.array([1])]
  padded_prompt_tokens = np.array([[0, 1]])
  logprobs = None


class _FakeSampler:

  def __init__(self):
    self.call_kwargs = None

  def __call__(self, **kwargs):
    self.call_kwargs = kwargs
    return _FakeSamplerOutput()


class RolloutStopTokensTest(absltest.TestCase):

  def test_vllm_rollout_passes_eos_tokens_as_stop_token_ids(self):
    # pylint: disable=g-import-not-at-top
    from tunix.rl.rollout import vllm_rollout
    # pylint: enable=g-import-not-at-top

    rollout = vllm_rollout.VllmRollout.__new__(vllm_rollout.VllmRollout)
    fake_sampler = _FakeSampler()
    rollout._sampler = fake_sampler

    rollout.generate(
        ["prompt"],
        base_rollout.RolloutConfig(eos_tokens=[151643, 151645]),
    )

    self.assertEqual(
        fake_sampler.call_kwargs["stop_token_ids"], [151643, 151645]
    )

  def test_sglang_jax_rollout_passes_eos_tokens_as_stop_token_ids(self):
    try:
      # pylint: disable=g-import-not-at-top
      from tunix.rl.rollout import sglang_jax_rollout
      # pylint: enable=g-import-not-at-top
    except ModuleNotFoundError as exc:
      if exc.name == "sgl_jax":
        self.skipTest("sgl_jax is not installed")
      raise

    rollout = sglang_jax_rollout.SglangJaxRollout.__new__(
        sglang_jax_rollout.SglangJaxRollout
    )
    fake_sampler = _FakeSampler()
    rollout._sampler = fake_sampler

    rollout.generate(
        ["prompt"],
        base_rollout.RolloutConfig(eos_tokens=[151643, 151645]),
    )

    self.assertEqual(
        fake_sampler.call_kwargs["stop_token_ids"], [151643, 151645]
    )


if __name__ == "__main__":
  absltest.main()
