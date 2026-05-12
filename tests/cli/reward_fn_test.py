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

"""Tests for configurable GRPO reward delimiters."""

from absl.testing import absltest
from tunix.cli.reward_fn import gsm8k
from tunix.cli.reward_fn import simple_math


class RewardDelimiterTest(absltest.TestCase):

  def test_simple_math_format_uses_think_by_default(self):
    completions = [
        "<think>work</think><answer>7</answer>",
        "<reasoning>work</reasoning><answer>7</answer>",
    ]

    self.assertEqual(
        simple_math.check_format(["p", "p"], completions),
        [0.1, 0],
    )

  def test_simple_math_format_can_use_reasoning_override(self):
    completions = ["<reasoning>work</reasoning><answer>7</answer>"]

    self.assertEqual(
        simple_math.check_format(
            ["p"],
            completions,
            reasoning_start="<reasoning>",
            reasoning_end="</reasoning>",
        ),
        [0.1],
    )

  def test_simple_math_answer_uses_custom_solution_tags(self):
    completions = ["<think>work</think><solution>42</solution>"]

    self.assertEqual(
        simple_math.check_answer(
            ["p"],
            completions,
            ["42"],
            solution_start="<solution>",
            solution_end="</solution>",
        ),
        [1],
    )

  def test_gsm8k_format_and_answer_use_think_by_default(self):
    completions = [
        "<think>work</think><answer>7</answer>",
        "<reasoning>work</reasoning><answer>7</answer>",
    ]

    self.assertEqual(
        gsm8k.match_format_exactly(["p", "p"], completions),
        [3.0, 0],
    )
    self.assertEqual(
        gsm8k.check_answer(["p", "p"], completions, ["7", "7"]),
        [3.0, 0],
    )

  def test_gsm8k_numbers_use_custom_solution_tags(self):
    completions = ["<think>work</think><solution>3.5</solution>"]

    self.assertEqual(
        gsm8k.check_numbers(
            ["p"],
            completions,
            ["3.5"],
            solution_start="<solution>",
            solution_end="</solution>",
        ),
        [1.5],
    )


if __name__ == "__main__":
  absltest.main()
