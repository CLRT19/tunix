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
"""DAPO overlong reward shaping wrapper for GRPO runs."""

import os
from typing import List

from tunix.rl.grpo.dapo_learner import reward_shaping


def overlong_reward_shaping(
    prompts: List[str],
    completions: List[str],
    mode,
    completion_token_counts: List[int] | None = None,
    **kwargs,
) -> List[float]:
  del kwargs
  overlong_buffer = {
      "enable": True,
      "overlong_buffer_length": int(
          os.environ.get("TUNIX_DAPO_OVERLONG_BUFFER_LENGTH", "2048")
      ),
      "overlong_buffer_penalty": float(
          os.environ.get("TUNIX_DAPO_OVERLONG_BUFFER_PENALTY", "1.0")
      ),
      "max_response_length": int(
          os.environ.get("TUNIX_DAPO_OVERLONG_MAX_RESPONSE_LENGTH", "10144")
      ),
  }
  if completion_token_counts is not None:
    overlong_buffer_length = overlong_buffer["overlong_buffer_length"]
    overlong_buffer_penalty = overlong_buffer["overlong_buffer_penalty"]
    max_response_length = overlong_buffer["max_response_length"]
    expected_response_length = max_response_length - overlong_buffer_length
    return [
        min(
            -(int(output_length) - expected_response_length)
            / overlong_buffer_length
            * overlong_buffer_penalty,
            0,
        )
        for output_length in completion_token_counts
    ]
  return reward_shaping(
      prompts=prompts,
      completions=completions,
      mode=mode,
      overlong_buffer=overlong_buffer,
  )
