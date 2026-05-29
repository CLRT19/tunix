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
"""Clicking reward — point-in-bbox check.

Ported from ``verl/utils/reward_score/click_reward.py``. Same Qwen-style
0-1000 normalized coordinate assumption as :mod:`vero_grounding`.
Only the FIRST GT bbox is checked (matches upstream behaviour).
"""

from __future__ import annotations

from typing import Optional, Tuple

from . import vero_grounding


def _parse_point_prediction(
    text: str, normalize_to_unit: bool = True
) -> Optional[Tuple[float, float]]:
  """Pull the first ``{"point_2d": [x, y]}`` entry from a JSON array in ``text``."""
  parsed = vero_grounding.parse_json_from_text(text)
  if not parsed:
    return None
  item = parsed[0]
  if not isinstance(item, dict):
    return None
  point_coords = item.get("point_2d")
  if (
      not point_coords
      or not isinstance(point_coords, list)
      or len(point_coords) != 2
  ):
    return None
  try:
    x_raw, y_raw = float(point_coords[0]), float(point_coords[1])
  except (TypeError, ValueError):
    return None
  scale = 1.0 / 1000.0 if normalize_to_unit else 1.0
  return (x_raw * scale, y_raw * scale)


def _point_in_bbox(
    point: Tuple[float, float], bbox: Tuple[float, float, float, float]
) -> bool:
  x, y = point
  x1, y1, x2, y2 = bbox
  return x1 <= x <= x2 and y1 <= y <= y2


def compute_score_accuracy(
    predict_str: str,
    ground_truth: str,
    normalize_to_unit: bool = True,
) -> float:
  """1.0 if predicted point lies inside the first GT bbox, else 0.0.

  Args:
    predict_str: model output, expected to contain a JSON array whose
      first element is ``{"point_2d": [x, y]}``.
    ground_truth: JSON array of ``{"bbox_2d": [x1, y1, x2, y2], ...}``.
      Only the first bbox is used.
    normalize_to_unit: divide all coordinates by 1000 before the
      containment test (Qwen 0-1000 convention).
  """
  try:
    point = _parse_point_prediction(predict_str, normalize_to_unit=normalize_to_unit)
    if point is None:
      return 0.0
    gt_boxes = vero_grounding.parse_ground_truth_boxes(
        ground_truth, normalize_to_unit=normalize_to_unit
    )
    if not gt_boxes:
      return 0.0
    return 1.0 if _point_in_bbox(point, gt_boxes[0]) else 0.0
  except Exception:
    return 0.0
