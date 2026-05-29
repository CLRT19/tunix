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
"""Vero strict format check.

Standalone, domain-agnostic format reward used by the 5-domain Vero
router. Mirrors ``chartqa.check_format``'s ``<think>...</think>
<answer>...</answer>`` shape but is stricter: the completion must
*fullmatch* the pattern (no trailing/leading garbage beyond whitespace)
and both blocks must have non-empty contents.

Module-level constants are public so reward routers can re-export the
same tag literals (and avoid duplicating the regex).
"""

import re
from typing import List

REASONING_START = "<think>"
REASONING_END = "</think>"
ANSWER_START = "<answer>"
ANSWER_END = "</answer>"

_FORMAT_RE = re.compile(
    rf"^\s*"
    rf"{re.escape(REASONING_START)}(.+?){re.escape(REASONING_END)}\s*"
    rf"{re.escape(ANSWER_START)}(.+?){re.escape(ANSWER_END)}"
    rf"\s*$",
    flags=re.DOTALL,
)


def check_format(completion: str) -> float:
  """1.0 if ``completion`` matches the strict think/answer envelope, else 0.0.

  The full string must match (modulo leading/trailing whitespace) and
  both the ``<think>`` and ``<answer>`` blocks must contain at least one
  character (``.+?`` under ``re.DOTALL``).
  """
  if not isinstance(completion, str):
    return 0.0
  return 1.0 if _FORMAT_RE.match(completion) is not None else 0.0


def check_format_batch(completions: List[str]) -> List[float]:
  """Vectorized :func:`check_format` over a list of completions."""
  return [check_format(c) for c in completions]
