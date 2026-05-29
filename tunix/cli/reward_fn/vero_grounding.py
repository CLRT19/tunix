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
"""Grounding (bbox detection) reward — Perception-R1 style.

Ported from ``verl/utils/reward_score/grounding_reward.py``. The original
relies on numpy for IoU and scipy for Hungarian matching; both are
optional here. If they are not present we fall back to a pure-python
greedy matcher that is identical in behaviour when the IoU matrix has
no ties (the common case for VLM bbox tasks).

The 5-domain Vero router assumes Qwen-style 0-1000 normalized
coordinates (``normalize_bbox_to_1000=True``). Per the task brief: divide
bbox coords by 1000 before computing IoU so predictions and GT live in
the same [0, 1] space. ``image_path``/``image_size`` are no longer
required — they were only used for pixel<->1000 conversion under the
original mixed-coordinate logic.

Component formula (preserved 1:1 from upstream):

    score = w_location * mean_iou_tp
          + w_recall   * f1
          - w_penalty  * 0.5 * (fp_rate + fn_rate)

clipped to [0, 1]. ``mean_iou_tp`` is 0 when there are no TPs.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

GROUNDING_COMPONENT_WEIGHTS: Dict[str, float] = {
    "location": 1.0,
    "recall": 0.75,
    "penalty": 0.5,
}
GROUNDING_IOU_THRESHOLD: float = 0.5


_CODE_BLOCK_RE = re.compile(r"```(?:json)?\s*(\[.*?\])\s*```", re.DOTALL)
_BARE_JSON_RE = re.compile(r"\[\s*\{.*?\}\s*\]", re.DOTALL)


def parse_json_from_text(text: str) -> Optional[List[Any]]:
  """Extract a JSON array from free text — supports ```json fenced blocks."""
  if not text:
    return None
  for match in _CODE_BLOCK_RE.findall(text):
    try:
      parsed = json.loads(match)
      if isinstance(parsed, list):
        return parsed
    except json.JSONDecodeError:
      pass
  for match in _BARE_JSON_RE.findall(text):
    try:
      parsed = json.loads(match)
      if isinstance(parsed, list):
        return parsed
    except json.JSONDecodeError:
      pass
  return None


def _coerce_bbox(bbox_coords: Any) -> Optional[Tuple[float, float, float, float]]:
  if not isinstance(bbox_coords, list) or len(bbox_coords) != 4:
    return None
  try:
    coords = tuple(float(x) for x in bbox_coords)
  except (TypeError, ValueError):
    return None
  return coords  # type: ignore[return-value]


def _scale_box(
    box: Tuple[float, float, float, float], scale: float
) -> Tuple[float, float, float, float]:
  return (box[0] * scale, box[1] * scale, box[2] * scale, box[3] * scale)


def parse_prediction_boxes(
    text: str, normalize_to_unit: bool = True
) -> List[Tuple[float, float, float, float]]:
  """Parse pred boxes from text.

  Args:
    text: model completion (already stripped of ``<answer>`` tags).
    normalize_to_unit: divide by 1000 (Qwen normalized) before returning.
  """
  parsed = parse_json_from_text(text)
  if not parsed:
    return []
  scale = 1.0 / 1000.0 if normalize_to_unit else 1.0
  boxes: List[Tuple[float, float, float, float]] = []
  for item in parsed:
    if not isinstance(item, dict):
      continue
    bbox = _coerce_bbox(item.get("bbox_2d"))
    if bbox is None:
      continue
    boxes.append(_scale_box(bbox, scale))
  return boxes


def parse_ground_truth_boxes(
    gt_answer: str, normalize_to_unit: bool = True
) -> List[Tuple[float, float, float, float]]:
  """Parse GT boxes — same schema as predictions (``bbox_2d``)."""
  parsed = parse_json_from_text(gt_answer)
  if not parsed:
    return []
  scale = 1.0 / 1000.0 if normalize_to_unit else 1.0
  boxes: List[Tuple[float, float, float, float]] = []
  for item in parsed:
    if not isinstance(item, dict):
      continue
    bbox = _coerce_bbox(item.get("bbox_2d"))
    if bbox is None:
      continue
    boxes.append(_scale_box(bbox, scale))
  return boxes


def _box_area(b: Tuple[float, float, float, float]) -> float:
  return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _pair_iou(
    a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]
) -> float:
  ix1 = max(a[0], b[0])
  iy1 = max(a[1], b[1])
  ix2 = min(a[2], b[2])
  iy2 = min(a[3], b[3])
  iw = max(0.0, ix2 - ix1)
  ih = max(0.0, iy2 - iy1)
  inter = iw * ih
  union = _box_area(a) + _box_area(b) - inter
  if union <= 1e-12:
    return 0.0
  return inter / union


def iou_matrix(
    pred_boxes: Sequence[Tuple[float, float, float, float]],
    gt_boxes: Sequence[Tuple[float, float, float, float]],
) -> List[List[float]]:
  """Pure-python NxM IoU matrix."""
  return [[_pair_iou(p, g) for g in gt_boxes] for p in pred_boxes]


def _hungarian_assign(
    ious: List[List[float]], iou_thr: float
) -> List[Tuple[int, int, float]]:
  """Hungarian matching via scipy if available, else greedy fallback.

  Greedy fallback iterates pairs in descending IoU order; both fallbacks
  agree on the unique-optimum case (no ties on the cost matrix), which
  is the dominant case for VLM grounding outputs.
  """
  n = len(ious)
  if n == 0:
    return []
  m = len(ious[0]) if ious else 0
  if m == 0:
    return []

  try:
    from scipy.optimize import linear_sum_assignment  # type: ignore
    import numpy as np  # type: ignore

    cost = 1.0 - np.asarray(ious, dtype=float)
    cost[np.asarray(ious) < iou_thr] = 1e6
    row_ind, col_ind = linear_sum_assignment(cost)
    matches: List[Tuple[int, int, float]] = []
    used_pred: set = set()
    used_gt: set = set()
    for r, c in zip(row_ind, col_ind):
      if (
          ious[int(r)][int(c)] >= iou_thr
          and int(r) not in used_pred
          and int(c) not in used_gt
      ):
        matches.append((int(r), int(c), float(ious[int(r)][int(c)])))
        used_pred.add(int(r))
        used_gt.add(int(c))
    return matches
  except Exception:
    pass

  # Greedy fallback: take the highest-IoU pair each round, skip below thr.
  candidates: List[Tuple[float, int, int]] = []
  for r in range(n):
    for c in range(m):
      iou = ious[r][c]
      if iou >= iou_thr:
        candidates.append((iou, r, c))
  candidates.sort(reverse=True)
  used_pred: set = set()
  used_gt: set = set()
  matches: List[Tuple[int, int, float]] = []
  for iou, r, c in candidates:
    if r in used_pred or c in used_gt:
      continue
    matches.append((r, c, iou))
    used_pred.add(r)
    used_gt.add(c)
  return matches


def evaluate_detections(
    pred_boxes: Sequence[Tuple[float, float, float, float]],
    gt_boxes: Sequence[Tuple[float, float, float, float]],
    iou_thr: float = GROUNDING_IOU_THRESHOLD,
) -> Dict[str, Any]:
  """Return TP/FP/FN/precision/recall/f1/mean_iou_tp."""
  ious = iou_matrix(pred_boxes, gt_boxes)
  n = len(pred_boxes)
  m = len(gt_boxes)
  if n == 0 or m == 0:
    return {
        "tp": 0,
        "fp": n,
        "fn": m,
        "mean_iou_tp": None,
        "precision": 0.0 if n > 0 else (1.0 if m == 0 else 0.0),
        "recall": 0.0 if m > 0 else (1.0 if n == 0 else 0.0),
        "f1": 0.0,
    }
  matches = _hungarian_assign(ious, iou_thr)
  tp = len(matches)
  fp = n - tp
  fn = m - tp
  precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if fn == 0 else 0.0)
  recall = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if fp == 0 else 0.0)
  f1 = (
      (2 * precision * recall) / (precision + recall)
      if (precision + recall) > 0
      else 0.0
  )
  mean_iou_tp = (sum(m_[2] for m_ in matches) / tp) if tp > 0 else None
  return {
      "tp": tp,
      "fp": fp,
      "fn": fn,
      "mean_iou_tp": mean_iou_tp,
      "precision": precision,
      "recall": recall,
      "f1": f1,
  }


def grounding_component_score(
    pred_boxes: Sequence[Tuple[float, float, float, float]],
    gt_boxes: Sequence[Tuple[float, float, float, float]],
    weights: Dict[str, float] = GROUNDING_COMPONENT_WEIGHTS,
    iou_threshold: float = GROUNDING_IOU_THRESHOLD,
) -> float:
  if not gt_boxes:
    return 0.0
  results = evaluate_detections(pred_boxes, gt_boxes, iou_thr=iou_threshold)
  mean_iou = results["mean_iou_tp"] or 0.0
  f1 = results["f1"]
  num_pred = len(pred_boxes)
  num_gt = len(gt_boxes)
  tp = results["tp"]
  fp_rate = (num_pred - tp) / num_pred if num_pred > 0 else 0.0
  fn_rate = (num_gt - tp) / num_gt if num_gt > 0 else 0.0
  miss_penalty = 0.5 * (fp_rate + fn_rate)
  score = (
      weights["location"] * mean_iou
      + weights["recall"] * f1
      - weights["penalty"] * miss_penalty
  )
  return float(max(0.0, min(1.0, score)))


def compute_score_accuracy(
    predict_str: str,
    ground_truth: str,
    normalize_to_unit: bool = True,
    component_weights: Optional[Dict[str, float]] = None,
    iou_threshold: float = GROUNDING_IOU_THRESHOLD,
) -> float:
  """Public entry: parse pred/gt JSON arrays, return component score in [0, 1].

  Args:
    predict_str: model output (already stripped to the ``<answer>`` payload).
    ground_truth: JSON-array string with ``[{"bbox_2d": [...], "label": "..."}]``.
    normalize_to_unit: if True (default), assume both pred and GT are in
      Qwen-style 0-1000 space and divide by 1000 before IoU. Set False
      to keep raw coordinates (e.g. absolute pixels).
    component_weights: override default location/recall/penalty weights.
    iou_threshold: IoU >= this counts as a TP.
  """
  weights = (
      dict(component_weights) if component_weights else GROUNDING_COMPONENT_WEIGHTS
  )
  try:
    pred_boxes = parse_prediction_boxes(predict_str, normalize_to_unit=normalize_to_unit)
    gt_boxes = parse_ground_truth_boxes(ground_truth, normalize_to_unit=normalize_to_unit)
    return grounding_component_score(
        pred_boxes, gt_boxes, weights=weights, iou_threshold=iou_threshold
    )
  except Exception:
    return 0.0
