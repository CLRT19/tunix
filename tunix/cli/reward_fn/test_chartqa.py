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
"""Unit tests for tunix.cli.reward_fn.chartqa."""

import unittest

from tunix.cli.reward_fn import chartqa


def _wrap(answer_text: str) -> str:
  return f"<think>reasoning</think><answer>{answer_text}</answer>"


class CheckAnswerTest(unittest.TestCase):

  def test_exact_match_case_insensitive(self):
    rewards = chartqa.check_answer(
        prompts=[None, None],
        completions=[_wrap("December"), _wrap("december")],
        label=[["December"], ["December"]],
    )
    self.assertEqual(rewards, [1.0, 1.0])

  def test_numeric_tolerance_within_1pct(self):
    # 42.5 vs 42.8 -> diff 0.3, rel 0.71% -> match.
    rewards = chartqa.check_answer(
        prompts=[None],
        completions=[_wrap("42.8")],
        label=[["42.5"]],
    )
    self.assertEqual(rewards, [1.0])

  def test_numeric_tolerance_beyond_1pct(self):
    # 42.5 vs 43.5 -> diff 1.0, rel 2.3% -> no match.
    rewards = chartqa.check_answer(
        prompts=[None],
        completions=[_wrap("43.5")],
        label=[["42.5"]],
    )
    self.assertEqual(rewards, [0.0])

  def test_percent_and_dollar_normalization(self):
    rewards = chartqa.check_answer(
        prompts=[None, None],
        completions=[_wrap("42.5%"), _wrap("$1,234.50")],
        label=[["42.5"], ["1234.5"]],
    )
    self.assertEqual(rewards, [1.0, 1.0])

  def test_multiple_acceptable_labels_one_matches(self):
    rewards = chartqa.check_answer(
        prompts=[None],
        completions=[_wrap("yes")],
        label=[["Yes", "true", "1"]],
    )
    self.assertEqual(rewards, [1.0])

  def test_label_is_string_not_list(self):
    # The HF schema is list[str] but some callers may pass a single str.
    rewards = chartqa.check_answer(
        prompts=[None],
        completions=[_wrap("December")],
        label=["December"],
    )
    self.assertEqual(rewards, [1.0])

  def test_no_answer_tags_falls_back_to_raw_completion(self):
    rewards = chartqa.check_answer(
        prompts=[None],
        completions=["The answer is December."],
        label=[["December"]],
    )
    # Raw completion stripped/lowered doesn't equal "december" (has prefix),
    # but the leading numeric extraction returns None, so no match.
    self.assertEqual(rewards, [0.0])

  def test_empty_label_yields_zero(self):
    rewards = chartqa.check_answer(
        prompts=[None],
        completions=[_wrap("December")],
        label=[[]],
    )
    self.assertEqual(rewards, [0.0])

  def test_garbage_completion_yields_zero(self):
    rewards = chartqa.check_answer(
        prompts=[None],
        completions=[_wrap("banana")],
        label=[["42"]],
    )
    self.assertEqual(rewards, [0.0])

  def test_trailing_punctuation_normalized(self):
    rewards = chartqa.check_answer(
        prompts=[None],
        completions=[_wrap("December.")],
        label=[["December"]],
    )
    self.assertEqual(rewards, [1.0])

  def test_custom_reward_magnitude(self):
    rewards = chartqa.check_answer(
        prompts=[None],
        completions=[_wrap("December")],
        label=[["December"]],
        r=0.5,
    )
    self.assertEqual(rewards, [0.5])

  def test_custom_rel_tol(self):
    # 100 vs 105 -> 5% diff. With rel_tol=0.1 should match; default no.
    completions = [_wrap("105")]
    label = [["100"]]
    self.assertEqual(
        chartqa.check_answer(prompts=[None], completions=completions, label=label),
        [0.0],
    )
    self.assertEqual(
        chartqa.check_answer(
            prompts=[None], completions=completions, label=label, rel_tol=0.1,
        ),
        [1.0],
    )


class CheckFormatTest(unittest.TestCase):

  def test_well_formed_completion(self):
    rewards = chartqa.check_format(
        prompts=[None],
        completions=["<think>r</think><answer>42</answer>"],
    )
    self.assertEqual(rewards, [0.1])

  def test_missing_answer_tags(self):
    rewards = chartqa.check_format(
        prompts=[None],
        completions=["<think>r</think>42"],
    )
    self.assertEqual(rewards, [0.0])


if __name__ == "__main__":
  unittest.main()
