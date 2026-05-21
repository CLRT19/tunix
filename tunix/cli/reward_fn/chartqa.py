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
"""ChartQA reward functions.

Mirrors the contract of ``simple_math.py``: each reward function takes
``prompts``, ``completions``, and dataset-driven kwargs, and returns a
``List[float]`` of per-sample rewards.

For ChartQA the ground-truth column is ``label`` (``list[str]``) — a list
of acceptable answers, usually one entry. Reward is 1.0 when the model's
extracted answer matches any entry in ``label`` under either
case-insensitive string equality (after normalization) or numeric
equality with ±1% relative tolerance.

Numeric normalization handles `%`, `$`, thousands separators (`,`), and
unit-suffixed answers (`38%`, `$1,234.5`).
"""

import re
from typing import Callable, List, Sequence
from absl import logging

ExpectedSignature = Callable[..., List[float]]

solution_start = "<answer>"
solution_end = "</answer>"
reasoning_start = "<think>"
reasoning_end = "</think>"

_ANSWER_RE = re.compile(
    rf"{re.escape(solution_start)}(.*?){re.escape(solution_end)}",
    flags=re.MULTILINE | re.DOTALL,
)

_NUMERIC_TOKEN_RE = re.compile(r"-?\d+(?:[,\d]*)(?:\.\d+)?")


def _extract_answer(completion: str) -> str:
  """Pull text from <answer>...</answer>; fall back to the whole completion."""
  m = _ANSWER_RE.search(completion)
  if m is None:
    return completion.strip()
  return m.group(1).strip()


def _normalize_text(s: str) -> str:
  """Lowercase + collapse whitespace + strip trailing punctuation."""
  s = s.strip().lower()
  s = re.sub(r"\s+", " ", s)
  s = s.rstrip(".,!?;:")
  return s


def _parse_number(s: str):
  """Return ``float(s)`` after stripping `%`, `$`, `,`, whitespace. ``None`` on failure."""
  if s is None:
    return None
  cleaned = s.strip().replace(",", "").replace("$", "").replace("%", "").strip()
  if not cleaned:
    return None
  try:
    return float(cleaned)
  except ValueError:
    # Try to recover a leading numeric token (e.g. "about 42.5%").
    m = _NUMERIC_TOKEN_RE.search(cleaned)
    if m is None:
      return None
    try:
      return float(m.group(0).replace(",", ""))
    except ValueError:
      return None


def _numeric_close(a: float, b: float, rel_tol: float = 0.01,
                   abs_tol: float = 1e-6) -> bool:
  if a == b:
    return True
  diff = abs(a - b)
  if diff <= abs_tol:
    return True
  denom = max(abs(a), abs(b))
  return denom > 0 and diff / denom <= rel_tol


def _matches_label(prediction: str, gt: str, rel_tol: float) -> bool:
  if _normalize_text(prediction) == _normalize_text(gt):
    return True
  p_num = _parse_number(prediction)
  g_num = _parse_number(gt)
  if p_num is not None and g_num is not None:
    return _numeric_close(p_num, g_num, rel_tol=rel_tol)
  return False


def _coerce_label(label) -> Sequence[str]:
  """Accept either a single string or a list of acceptable answers."""
  if label is None:
    return ()
  if isinstance(label, str):
    return (label,)
  return tuple(label)


def check_answer(
    prompts,
    completions,
    label,
    r: float = 1.0,
    rel_tol: float = 0.01,
    **kwargs,
):
  """Reward 1.0 if the extracted answer matches any entry in ``label``.

  Args:
    prompts: unused (kept for signature parity).
    completions: list of generated strings, length B.
    label: list of length B; each entry is either a string or a list of
      acceptable strings (per ChartQA's HF schema, ``label`` is ``list[str]``).
    r: positive reward magnitude. Defaults to 1.0.
    rel_tol: relative tolerance for numeric matching. Defaults to 0.01 (±1%).
  """
  del prompts, kwargs
  scores: List[float] = []
  for completion, gts in zip(completions, label):
    prediction = _extract_answer(completion)
    candidates = _coerce_label(gts)
    if not candidates:
      scores.append(0.0)
      continue
    matched = any(
        _matches_label(prediction, gt, rel_tol=rel_tol) for gt in candidates
    )
    scores.append(r if matched else 0.0)
  return scores


def check_format(
    prompts,
    completions,
    r: float = 0.1,
    reasoning_start=reasoning_start,
    reasoning_end=reasoning_end,
    solution_start=solution_start,
    solution_end=solution_end,
    **kwargs,
):
  """Reward r if completion matches the <think>...</think><answer>...</answer> shape."""
  del prompts, kwargs
  match_format = re.compile(
      rf"^[\s]{{0,}}"
      rf"{re.escape(reasoning_start)}.+?{re.escape(reasoning_end)}.*?"
      rf"{re.escape(solution_start)}(.+?){re.escape(solution_end)}"
      rf"[\s]{{0,}}$",
      flags=re.MULTILINE | re.DOTALL,
  )
  return [
      0.0 if match_format.search(c) is None else r for c in completions
  ]
