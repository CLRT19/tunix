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
"""Unit tests for Qwen3VLVanillaRollout.

These tests stub out the underlying Qwen3VLSampler so they run on CPU in
under a second — they verify the adapter contract (RolloutOutput shape,
prompt padding, image staging), not the model logic.
"""

import types
import unittest
from unittest import mock

import numpy as np

# We import the module under test after defining the stubs so that
# ``Qwen3VLSampler`` can be patched.
from tunix.rl.rollout import base_rollout
from tunix.rl.rollout import qwen3vl_vanilla_rollout as rollout_lib


class _FakeTokenizer:

  def __init__(self):
    self.pad_token_id = 0
    self.eos_token_id = 999

  def decode(self, ids, skip_special_tokens=True):
    return ' '.join(str(int(i)) for i in ids)


class _FakeProcessor:

  def __init__(self):
    self.tokenizer = _FakeTokenizer()


class _FakeSampler:
  """Minimal stand-in for Qwen3VLSampler: returns canned token arrays."""

  def __init__(self, model, processor, cache_size):
    del model, cache_size
    self._processor = processor
    self.last_kwargs = None
    self._flattened_model_state = ()
    self._model_graphdef = None
    self._model = object()  # opaque; only identity checked

  def generate_with_tokens(self, **kwargs):
    self.last_kwargs = kwargs
    prompts = kwargs['prompts']
    bsz = len(prompts)
    prompt_tokens = np.array(
        [[0, 0, 1, 2, 3]] * bsz, dtype=np.int32  # 2 left-pad zeros
    )
    completion_tokens = [np.array([10, 11, 999, 12], dtype=np.int32)] * bsz
    # Pretend the sampler trimmed at EOS (999): completion is [10, 11].
    trimmed = [c[:2] for c in completion_tokens]
    texts = ['10 11'] * bsz
    return {
        'texts': texts,
        'prompt_tokens': prompt_tokens,
        'completion_tokens': trimmed,
        'token_buffer': None,
        'prompt_seq_len': prompt_tokens.shape[1],
    }


class RolloutTest(unittest.TestCase):

  def _make(self):
    with mock.patch.object(
        rollout_lib.qwen3vl_sampler_lib, 'Qwen3VLSampler', _FakeSampler
    ):
      proc = _FakeProcessor()
      cache_cfg = base_rollout.CacheConfig(
          cache_size=128, num_layers=1, num_kv_heads=1, head_dim=1
      )
      return rollout_lib.Qwen3VLVanillaRollout(
          model=types.SimpleNamespace(), processor=proc, cache_config_or_size=cache_cfg
      )

  def test_generate_returns_rollout_output(self):
    r = self._make()
    cfg = base_rollout.RolloutConfig(
        max_tokens_to_generate=4,
        max_prompt_length=8,
        temperature=0.0,
    )
    out = r.generate(prompts=['p1', 'p2'], rollout_config=cfg)
    self.assertEqual(len(out.text), 2)
    self.assertEqual(out.text[0], '10 11')
    self.assertEqual(len(out.tokens), 2)
    np.testing.assert_array_equal(out.tokens[0], [10, 11])
    # Prompt should be left-padded to 8 (pad_value=0 from fake tokenizer).
    self.assertEqual(out.left_padded_prompt_tokens.shape, (2, 8))
    np.testing.assert_array_equal(
        out.left_padded_prompt_tokens[0], [0, 0, 0, 0, 0, 1, 2, 3]
    )

  def test_generate_truncates_prompts_longer_than_max(self):
    r = self._make()
    cfg = base_rollout.RolloutConfig(
        max_tokens_to_generate=4,
        max_prompt_length=3,
        temperature=0.0,
    )
    out = r.generate(prompts=['p1'], rollout_config=cfg)
    self.assertEqual(out.left_padded_prompt_tokens.shape, (1, 3))
    np.testing.assert_array_equal(
        out.left_padded_prompt_tokens[0], [1, 2, 3]
    )

  def test_pending_images_forwarded_to_sampler(self):
    r = self._make()
    cfg = base_rollout.RolloutConfig(
        max_tokens_to_generate=4, max_prompt_length=8, temperature=0.0,
    )
    fake_imgs = [object(), object()]
    r.set_pending_images(fake_imgs)
    r.generate(prompts=['a', 'b'], rollout_config=cfg)
    self.assertEqual(r._sampler.last_kwargs['images'], fake_imgs)
    # After one generate(), pending_images should be consumed (=None).
    self.assertIsNone(r._pending_images)

  def test_explicit_images_kwarg_overrides_pending(self):
    r = self._make()
    cfg = base_rollout.RolloutConfig(
        max_tokens_to_generate=4, max_prompt_length=8, temperature=0.0,
    )
    fake_pending = [object(), object()]
    fake_explicit = [object(), object()]
    r.set_pending_images(fake_pending)
    r.generate(prompts=['a', 'b'], rollout_config=cfg, images=fake_explicit)
    self.assertEqual(r._sampler.last_kwargs['images'], fake_explicit)

  def test_image_count_mismatch_raises(self):
    r = self._make()
    cfg = base_rollout.RolloutConfig(
        max_tokens_to_generate=4, max_prompt_length=8, temperature=0.0,
    )
    with self.assertRaisesRegex(ValueError, 'must match'):
      r.generate(
          prompts=['a', 'b'], rollout_config=cfg, images=[object()]
      )

  def test_pad_and_eos_ids(self):
    r = self._make()
    self.assertEqual(r.pad_id(), 0)
    self.assertEqual(r.eos_id(), 999)

  def test_get_per_token_logps_raises_until_implemented(self):
    r = self._make()
    with self.assertRaises(NotImplementedError):
      r.get_per_token_logps(
          prompt_tokens=np.zeros((1, 1)),
          completion_tokens=np.zeros((1, 1)),
      )


if __name__ == '__main__':
  unittest.main()
