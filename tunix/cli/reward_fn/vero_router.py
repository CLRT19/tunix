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
"""Vero 5-domain reward router for Qwen3-VL GRPO training.

Ported from ``verl/utils/reward_score/math_verify_reward_type_boxed_strict.py``
(see also sibling helpers ``grounding_reward.py``, ``click_reward.py``,
``instructions.py``, ``text_normalization.py``). The original lives at::

    /home/linrong/repo/tpu-vlm-rl/verl/utils/reward_score/
        math_verify_reward_type_boxed_strict.py

Public API
----------

``score_one(domain, reward_type, ground_truth, completion, extra_info) -> float``
    Return accuracy in [0, 1]. ``reward_type`` drives the dispatch;
    ``domain`` is informational only (kept for parity with the verl-side
    schema). Robust to malformed completions and missing tags: returns
    0.0 on parse failure rather than raising.

``score_batch(domains, reward_types, ground_truths, completions, extra_infos) -> List[float]``
    Vectorized wrapper. Lengths must match (``extra_infos`` may be None).

Supported reward types
----------------------

string_match, numeric, multiple_choice, list_string_match, number_list,
web_action, counting, search, grounding, clicking, instruction_following.

Notes vs. the verl original
---------------------------

* ``acc_reward`` in verl strips ``<answer>...</answer>``, then for the
  "boxed" reward types (string_match, numeric, multiple_choice,
  number_list, web_action, list_string_match, counting, search) also
  unwraps the last ``\\boxed{...}`` payload. We preserve that two-stage
  strip exactly.
* For grounding/clicking, verl additionally enforces strict JSON-array
  parsing on the unwrapped answer; we keep that.
* For ``numeric``: math_verify (sympy-based) is imported lazily so the
  rest of the router stays usable when math_verify is absent. With no
  ``extra_info["tolerance"]``, falls back to the math_verify ``verify``
  path; with a tolerance, runs the pure-python relative-tolerance check.
* For ``instruction_following``: ``verl.utils.reward_score.instructions``
  pulls langdetect + nltk + optional open_instruct. We try to import it
  lazily; if the import fails we emit a one-time warning via ``logging``
  and return 0.0 for every instruction_following row. TODO: port a
  reduced rule-based subset of instructions.py into tunix once the env
  policy on langdetect/nltk is settled.
* For grounding/clicking the task brief specifies that all v2/v3 data
  uses ``normalize_bbox_to_1000=True``; we therefore divide both pred
  and GT coordinates by 1000 before IoU / point-in-box, and we ignore
  ``image_path``/``image_size`` (no PIL dependency).
"""

from __future__ import annotations

import ast
import json
import logging
import math
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from . import vero_click
from . import vero_grounding
from .vero_text_norm import normalize_text_for_match

__all__ = [
    "score_one",
    "score_batch",
    "SUPPORTED_REWARD_TYPES",
]

_LOGGER = logging.getLogger(__name__)

SUPPORTED_REWARD_TYPES = frozenset(
    {
        "string_match",
        "numeric",
        "multiple_choice",
        "list_string_match",
        "number_list",
        "web_action",
        "counting",
        "search",
        "grounding",
        "clicking",
        "instruction_following",
        # Aliases kept for parity with the verl router; treated like
        # instruction_following (no LLM judge here).
        "instruction_following_llm_judge",
    }
)

# Same partition as ``_BOXED_REWARD_TYPES`` in verl: these unwrap the last
# ``\boxed{...}`` payload from the <answer> body before scoring.
_BOXED_REWARD_TYPES = frozenset(
    {
        "string_match",
        "multiple_choice",
        "number_list",
        "web_action",
        "numeric",
        "list_string_match",
        "counting",
        "search",
    }
)

_ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)
_NUMBER_PATTERN = re.compile(r"-?\d+")

_PARSE_TIMEOUT = 3
_VERIFY_TIMEOUT = 3

# One-time guard for missing instruction-checker module warning.
_INSTRUCTIONS_WARNING_EMITTED = False


# ---------------------------------------------------------------------------
# Answer / boxed extraction (port of verl's _extract_answer + _extract_boxed_contents)
# ---------------------------------------------------------------------------


def _extract_answer(predict_str: str) -> str:
  """Pull <answer>...</answer> contents; fall back to the stripped completion."""
  if not isinstance(predict_str, str):
    return ""
  m = _ANSWER_PATTERN.search(predict_str)
  if m is not None:
    return m.group(1).strip()
  # Drop <think> blocks even if there is no <answer> tag, so a malformed
  # completion like "<think>x</think> 42" still scores as "42".
  return _THINK_PATTERN.sub("", predict_str).strip()


def _extract_boxed_contents(text: str) -> List[str]:
  """Return the contents of every well-formed ``\\boxed{...}`` block in ``text``."""
  if not text:
    return []
  contents: List[str] = []
  idx = 0
  needle = r"\boxed"
  while True:
    start = text.find(needle, idx)
    if start == -1:
      break
    cursor = start + len(needle)
    while cursor < len(text) and text[cursor].isspace():
      cursor += 1
    if cursor >= len(text) or text[cursor] != "{":
      idx = cursor
      continue
    cursor += 1
    depth = 1
    content_start = cursor
    while cursor < len(text) and depth > 0:
      ch = text[cursor]
      if ch == "{":
        depth += 1
      elif ch == "}":
        depth -= 1
      cursor += 1
    if depth == 0:
      contents.append(_strip_tex_text_wrapper(text[content_start : cursor - 1].strip()))
      idx = cursor
    else:
      break
  return contents


def _extract_last_boxed_bracket_content(text: str) -> Optional[str]:
  """Recover the payload of a malformed ``\\boxed[...]`` block."""
  if not text:
    return None
  idx = 0
  last: Optional[str] = None
  needle = r"\boxed"
  while True:
    start = text.find(needle, idx)
    if start == -1:
      break
    cursor = start + len(needle)
    while cursor < len(text) and text[cursor].isspace():
      cursor += 1
    if cursor >= len(text):
      break
    if text[cursor] != "[":
      idx = cursor + 1
      continue
    cursor += 1
    depth = 1
    content_start = cursor
    while cursor < len(text) and depth > 0:
      ch = text[cursor]
      if ch == "[":
        depth += 1
      elif ch == "]":
        depth -= 1
      cursor += 1
    if depth == 0:
      last = text[content_start : cursor - 1].strip()
      idx = cursor
    else:
      break
  return last


def _strip_tex_text_wrapper(text: str) -> str:
  """Unwrap a single top-level ``\\text{...}`` around the boxed payload."""
  if not text:
    return text
  stripped = text.lstrip()
  if not stripped.startswith(r"\text"):
    return text
  idx = len(r"\text")
  while idx < len(stripped) and stripped[idx].isspace():
    idx += 1
  if idx >= len(stripped) or stripped[idx] != "{":
    return text
  idx += 1
  depth = 1
  content_start = idx
  while idx < len(stripped) and depth > 0:
    ch = stripped[idx]
    if ch == "{":
      depth += 1
    elif ch == "}":
      depth -= 1
    idx += 1
  if depth != 0:
    return text
  content = stripped[content_start : idx - 1].strip()
  trailing = stripped[idx:].strip()
  if trailing:
    return text
  return content


def _strict_json_array_text(text: str) -> Optional[str]:
  candidate = text.strip()
  if not candidate:
    return None
  try:
    parsed = json.loads(candidate)
  except Exception:
    return None
  if not isinstance(parsed, list):
    return None
  return candidate


def _extract_grounding_clicking_answer(answer_text: str) -> Optional[str]:
  """Mirror verl's strict JSON-array unwrap for grounding/clicking."""
  candidate = answer_text.strip()
  if not candidate:
    return None
  boxed_values = _extract_boxed_contents(candidate)
  source = boxed_values[-1] if boxed_values and boxed_values[-1] else candidate
  return _strict_json_array_text(source)


# ---------------------------------------------------------------------------
# Per-reward-type handlers
# ---------------------------------------------------------------------------


def _string_match_reward(pred: str, truth: str, **_: Any) -> float:
  np_ = normalize_text_for_match(pred)
  nt_ = normalize_text_for_match(truth)
  if not np_ or not nt_:
    return 0.0
  return 1.0 if np_ == nt_ else 0.0


def _multiple_choice_reward(pred: str, truth: str, **_: Any) -> float:
  """A-Z single-letter MCQ match; reduce GT to its first letter token."""
  # Cheap path: if GT is a single A-Z letter (after strip+upper), compare to
  # the first A-Z token of the prediction.
  truth_stripped = truth.strip()
  # Try sympy-free letter extraction first.
  letter_match = re.search(r"\b([A-Z])\b", truth_stripped.upper())
  if letter_match is not None:
    gt_letter = letter_match.group(1)
    pred_letter_match = re.search(r"\b([A-Z])\b", pred.strip().upper())
    if pred_letter_match is None:
      return 0.0
    return 1.0 if pred_letter_match.group(1) == gt_letter else 0.0
  # Fallback to string_match if the GT isn't a single letter.
  return _string_match_reward(pred, truth)


_NUMBER_LIST_DISCOUNT = 0.2


def _parse_number_list(text: str) -> List[int]:
  return [int(val) for val in _NUMBER_PATTERN.findall(text)]


def _number_list_reward(pred: str, truth: str, **_: Any) -> float:
  gold = _parse_number_list(truth)
  pred_list = _parse_number_list(pred)
  if not gold or not pred_list:
    return 0.0
  k = len(gold)
  if len(pred_list) != k or len(set(pred_list)) != k or sorted(pred_list) != sorted(gold):
    return 0.0
  if pred_list == gold:
    return 1.0
  correct = sum(1 for g, p in zip(gold, pred_list) if g == p)
  return _NUMBER_LIST_DISCOUNT * (correct / k)


def _normalize_web_action_payload(payload: Any) -> Dict[str, Any]:
  if not isinstance(payload, dict):
    return {}
  return {str(k).upper(): v for k, v in payload.items()}


def _load_web_action_payload(payload: Any) -> Dict[str, Any]:
  if isinstance(payload, dict):
    return _normalize_web_action_payload(payload)
  if not isinstance(payload, str):
    return {}
  text = payload.strip()
  for parser in (json.loads, ast.literal_eval):
    try:
      return _normalize_web_action_payload(parser(text))
    except Exception:
      continue
  return {}


def _web_action_values_match(gold_value: Any, pred_value: Any) -> bool:
  if isinstance(gold_value, (int, float)) and isinstance(pred_value, str):
    try:
      return float(gold_value) == float(pred_value)
    except Exception:
      pass
  if isinstance(gold_value, str) and isinstance(pred_value, (int, float)):
    try:
      return float(gold_value) == float(pred_value)
    except Exception:
      pass
  return str(gold_value).strip() == str(pred_value).strip()


def _web_action_reward(pred: str, truth: str, **_: Any) -> float:
  gold = _load_web_action_payload(truth)
  pred_payload = _load_web_action_payload(pred)
  if not gold or not pred_payload:
    return 0.0
  considered: List[str] = []
  if "ACTION" in gold:
    considered.append("ACTION")
  mark = gold.get("MARK")
  ignore_mark = mark == -1 or (isinstance(mark, str) and mark.strip() == "-1")
  if "MARK" in gold and not ignore_mark:
    considered.append("MARK")
  value = gold.get("VALUE")
  ignore_value = value is None or value == "None"
  if isinstance(value, str) and value.strip().lower() == "none":
    ignore_value = True
  if "VALUE" in gold and not ignore_value:
    considered.append("VALUE")
  if not considered:
    return 0.0
  weight = 1.0 / len(considered)
  reward = 0.0
  for key in considered:
    if key in pred_payload and _web_action_values_match(
        gold.get(key), pred_payload.get(key)
    ):
      reward += weight
  return reward


def _coerce_truth_to_list(truth: Any) -> List[str]:
  if isinstance(truth, (list, tuple, set)):
    return [str(item) for item in truth]
  if isinstance(truth, str):
    candidate = truth.strip()
    if not candidate:
      return []
    try:
      parsed = ast.literal_eval(candidate)
      if isinstance(parsed, (list, tuple, set)):
        return [str(item) for item in parsed]
    except Exception:
      pass
    return [candidate]
  return [str(truth)]


def _list_string_match_reward(pred: str, truth: str, **_: Any) -> float:
  np_ = normalize_text_for_match(pred)
  if not np_:
    return 0.0
  items = _coerce_truth_to_list(truth)
  if not items:
    return 0.0
  return 1.0 if any(np_ == normalize_text_for_match(it) for it in items) else 0.0


# ---------------------------------------------------------------------------
# Numeric (math_verify-aware with pure-python fallback)
# ---------------------------------------------------------------------------


_NUMERIC_FAST_RE = re.compile(r"-?\d+(?:[,\d]*)(?:\.\d+)?")


def _parse_number_pure(text: str) -> Optional[float]:
  """Pure-python number extraction, mirrors chartqa._parse_number."""
  if text is None:
    return None
  cleaned = (
      text.strip().replace(",", "").replace("$", "").replace("%", "").strip()
  )
  if not cleaned:
    return None
  try:
    return float(cleaned)
  except ValueError:
    m = _NUMERIC_FAST_RE.search(cleaned)
    if m is None:
      return None
    try:
      return float(m.group(0).replace(",", ""))
    except ValueError:
      return None


def _numeric_close(a: float, b: float, tol: float) -> bool:
  diff = abs(a - b)
  if abs(a) <= 1e-12:
    return diff <= tol
  return diff <= tol * abs(a)


def _extract_tolerance(extra_info: Any) -> Optional[float]:
  if not isinstance(extra_info, dict):
    return None
  tol = extra_info.get("tolerance")
  if tol is None:
    return None
  try:
    val = float(tol)
  except (TypeError, ValueError):
    return None
  return val if val >= 0 else None


def _numeric_reward(
    pred: str, truth: str, *, extra_info: Optional[Dict[str, Any]] = None, **_: Any
) -> float:
  tolerance = _extract_tolerance(extra_info)
  # Tolerance path: pure python only — matches the verl behaviour exactly
  # for any "tolerance" present in extra_info (the v2/v3 jsonl always sets it).
  if tolerance is not None:
    g = _parse_number_pure(truth)
    p = _parse_number_pure(pred)
    if g is None or p is None:
      return 0.0
    return 1.0 if _numeric_close(g, p, tolerance) else 0.0

  # No tolerance: try math_verify (sympy) for full LaTeX-aware grading;
  # fall back to a relaxed pure-python equality at 1e-6 if math_verify
  # is absent.
  try:
    from math_verify import parse as mv_parse  # type: ignore
    from math_verify import verify as mv_verify  # type: ignore
    from math_verify.parser import (  # type: ignore
        ExprExtractionConfig,
        LatexExtractionConfig,
    )

    targets = [LatexExtractionConfig(boxed_match_priority=0), ExprExtractionConfig()]
    try:
      gold_parsed = mv_parse(truth, targets, parsing_timeout=_PARSE_TIMEOUT)
      pred_parsed = mv_parse(pred, targets, parsing_timeout=_PARSE_TIMEOUT)
    except Exception:
      gold_parsed = pred_parsed = None
    if not gold_parsed or not pred_parsed:
      # Last-ditch direct float compare.
      g = _parse_number_pure(truth)
      p = _parse_number_pure(pred)
      if g is None or p is None:
        return 0.0
      return 1.0 if _numeric_close(g, p, 1e-6) else 0.0
    try:
      ok = mv_verify(
          gold_parsed,
          pred_parsed,
          float_rounding=6,
          strict=True,
          allow_set_relation_comp=False,
          timeout_seconds=_VERIFY_TIMEOUT,
      )
    except Exception:
      ok = False
    return 1.0 if ok else 0.0
  except Exception:
    g = _parse_number_pure(truth)
    p = _parse_number_pure(pred)
    if g is None or p is None:
      return 0.0
    return 1.0 if _numeric_close(g, p, 1e-6) else 0.0


# ---------------------------------------------------------------------------
# Grounding / clicking
# ---------------------------------------------------------------------------


def _grounding_reward(
    pred: str, truth: str, *, extra_info: Optional[Dict[str, Any]] = None, **_: Any
) -> float:
  # Task brief: v2/v3 always uses normalize_bbox_to_1000=True. We still
  # honour an explicit override in extra_info for forward compat.
  normalize_to_unit = True
  if isinstance(extra_info, dict) and "normalize_bbox_to_1000" in extra_info:
    normalize_to_unit = bool(extra_info.get("normalize_bbox_to_1000"))
  component_weights = vero_grounding.GROUNDING_COMPONENT_WEIGHTS
  iou_threshold = vero_grounding.GROUNDING_IOU_THRESHOLD
  if isinstance(extra_info, dict):
    weights_override = extra_info.get("grounding_component_weights") or extra_info.get(
        "component_weights"
    )
    if isinstance(weights_override, dict):
      numeric_overrides = {
          k: float(v)
          for k, v in weights_override.items()
          if isinstance(v, (int, float))
      }
      if numeric_overrides:
        component_weights = dict(component_weights)
        component_weights.update(numeric_overrides)
    iou_override = extra_info.get("iou_threshold")
    if iou_override is not None:
      try:
        iou_threshold = float(iou_override)
      except (TypeError, ValueError):
        pass
  return vero_grounding.compute_score_accuracy(
      predict_str=pred,
      ground_truth=truth,
      normalize_to_unit=normalize_to_unit,
      component_weights=component_weights,
      iou_threshold=iou_threshold,
  )


def _clicking_reward(
    pred: str, truth: str, *, extra_info: Optional[Dict[str, Any]] = None, **_: Any
) -> float:
  normalize_to_unit = True
  if isinstance(extra_info, dict) and "normalize_bbox_to_1000" in extra_info:
    normalize_to_unit = bool(extra_info.get("normalize_bbox_to_1000"))
  return vero_click.compute_score_accuracy(
      predict_str=pred,
      ground_truth=truth,
      normalize_to_unit=normalize_to_unit,
  )


# ---------------------------------------------------------------------------
# Instruction following (deterministic check via verl.instructions if available)
# ---------------------------------------------------------------------------


def _load_instruction_payload(payload: Any) -> Dict[str, Any]:
  if isinstance(payload, dict):
    return payload
  if not isinstance(payload, str):
    return {}
  text = payload.strip()
  for parser in (json.loads, ast.literal_eval):
    try:
      parsed = parser(text)
    except Exception:
      continue
    if isinstance(parsed, str):
      inner = parsed.strip()
      if inner and inner != text:
        for inner_parser in (json.loads, ast.literal_eval):
          try:
            parsed = inner_parser(inner)
            break
          except Exception:
            continue
    if isinstance(parsed, dict):
      return parsed
  return {}


def _instruction_following_reward(
    pred: str, truth: str, *, extra_info: Optional[Dict[str, Any]] = None, **_: Any
) -> float:
  del extra_info
  payload = _load_instruction_payload(truth)
  instructions_list = payload.get("instructions") if isinstance(payload, dict) else None
  if not isinstance(instructions_list, (list, tuple)) or not instructions_list:
    return 0.0

  try:
    # The verl side keeps the instruction registry in this module; we try
    # to import it lazily. langdetect/nltk only get pulled in when this
    # actually executes — and only when the module is importable.
    from verl.utils.reward_score import instructions as instruction_lib  # type: ignore
  except Exception:
    global _INSTRUCTIONS_WARNING_EMITTED
    if not _INSTRUCTIONS_WARNING_EMITTED:
      _LOGGER.warning(
          "vero_router: verl.utils.reward_score.instructions is not "
          "importable; instruction_following rows will score 0.0. "
          "TODO: port a rule-based subset of instructions.py into tunix "
          "once the langdetect/nltk env policy is settled."
      )
      _INSTRUCTIONS_WARNING_EMITTED = True
    return 0.0

  satisfied = 0
  total = 0
  for item in instructions_list:
    total += 1
    if not isinstance(item, dict):
      continue
    class_name = item.get("class")
    if not isinstance(class_name, str) or not class_name:
      continue
    instruction_cls = getattr(instruction_lib, class_name, None)
    if instruction_cls is None:
      continue
    try:
      instruction = instruction_cls(class_name)
    except Exception:
      try:
        instruction = instruction_cls()
      except Exception:
        continue

    args = item.get("args")
    if not isinstance(args, dict):
      args = {}
    cleaned_args = {k: v for k, v in args.items() if v is not None}
    built = False
    try:
      instruction.build_description(**cleaned_args)
      built = True
    except TypeError:
      try:
        allowed_keys = instruction.get_instruction_args_keys()
      except Exception:
        allowed_keys = None
      if allowed_keys:
        filtered_args = {
            k: v for k, v in cleaned_args.items() if k in set(allowed_keys)
        }
      else:
        filtered_args = {}
      try:
        instruction.build_description(**filtered_args)
        built = True
      except Exception:
        built = False
    except Exception:
      built = False
    if not built:
      continue
    try:
      if pred.strip() and instruction.check_following(pred):
        satisfied += 1
    except Exception:
      continue
  if total == 0:
    return 0.0
  return satisfied / total


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


_HANDLERS = {
    "string_match": _string_match_reward,
    "multiple_choice": _multiple_choice_reward,
    "number_list": _number_list_reward,
    "web_action": _web_action_reward,
    "numeric": _numeric_reward,
    "list_string_match": _list_string_match_reward,
    "counting": _string_match_reward,
    "search": _string_match_reward,
    "grounding": _grounding_reward,
    "clicking": _clicking_reward,
    "instruction_following": _instruction_following_reward,
    "instruction_following_llm_judge": _instruction_following_reward,
}


def _coerce_reward_type(reward_type: Any) -> Optional[str]:
  if not isinstance(reward_type, str):
    return None
  cleaned = reward_type.strip().lower()
  return cleaned if cleaned in SUPPORTED_REWARD_TYPES else None


def _coerce_ground_truth(ground_truth: Any) -> str:
  if isinstance(ground_truth, str):
    return ground_truth
  if isinstance(ground_truth, (int, float, bool)):
    return str(ground_truth)
  # Lists / dicts: stringify via json so downstream parsers can reparse.
  try:
    return json.dumps(ground_truth, ensure_ascii=False)
  except Exception:
    return str(ground_truth)


def score_one(
    domain: str,  # noqa: ARG001 - kept for parity, routing is reward_type-driven
    reward_type: str,
    ground_truth: Any,
    completion: str,
    extra_info: Optional[Dict[str, Any]] = None,
) -> float:
  """Score one row. Returns accuracy in [0, 1]. 0.0 on any parse/handler error."""
  del domain  # informational only
  rt = _coerce_reward_type(reward_type)
  if rt is None:
    return 0.0
  if not isinstance(completion, str) or not completion:
    return 0.0
  truth = _coerce_ground_truth(ground_truth).strip()
  if not truth:
    return 0.0

  # 1. Pull <answer>...</answer> out of the completion (or strip <think>).
  answer_text = _extract_answer(completion)
  if not answer_text:
    return 0.0

  # 2. For boxed-style reward types, unwrap the last \boxed{...} payload.
  if rt in _BOXED_REWARD_TYPES:
    boxed_values = _extract_boxed_contents(answer_text)
    if boxed_values and boxed_values[-1]:
      answer_text = boxed_values[-1].strip()
    elif r"\boxed" in answer_text:
      malformed = _extract_last_boxed_bracket_content(answer_text)
      if malformed:
        answer_text = malformed
  elif rt in {"grounding", "clicking"}:
    strict = _extract_grounding_clicking_answer(answer_text)
    if not strict:
      return 0.0
    answer_text = strict

  if not answer_text:
    return 0.0

  handler = _HANDLERS.get(rt)
  if handler is None:
    return 0.0
  try:
    score = handler(answer_text, truth, extra_info=extra_info)
  except Exception:
    return 0.0
  try:
    score_f = float(score)
  except (TypeError, ValueError):
    return 0.0
  if math.isnan(score_f) or math.isinf(score_f):
    return 0.0
  if score_f < 0.0:
    return 0.0
  if score_f > 1.0:
    return 1.0
  return score_f


def score_batch(
    domains: Sequence[str],
    reward_types: Sequence[str],
    ground_truths: Sequence[Any],
    completions: Sequence[str],
    extra_infos: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
) -> List[float]:
  """Per-row dispatch via :func:`score_one`. Returns ``list[float]`` of length B."""
  n = len(completions)
  if not (len(domains) == len(reward_types) == len(ground_truths) == n):
    raise ValueError(
        "score_batch: length mismatch — "
        f"domains={len(domains)}, reward_types={len(reward_types)}, "
        f"ground_truths={len(ground_truths)}, completions={n}"
    )
  if extra_infos is None:
    extras: Iterable[Optional[Dict[str, Any]]] = [None] * n
  else:
    if len(extra_infos) != n:
      raise ValueError(
          f"score_batch: extra_infos length {len(extra_infos)} != {n}"
      )
    extras = extra_infos
  scores: List[float] = []
  for dom, rt, gt, comp, extra in zip(
      domains, reward_types, ground_truths, completions, extras
  ):
    scores.append(score_one(dom, rt, gt, comp, extra))
  return scores
