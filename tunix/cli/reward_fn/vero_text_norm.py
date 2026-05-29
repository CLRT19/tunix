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
"""Text normalization helpers for the Vero reward router.

Ported 1:1 from ``verl/utils/reward_score/text_normalization.py``. Used
by string_match / list_string_match / multiple_choice / counting / search
reward types so equality comparisons are robust to unicode, smart-quotes,
thousands separators, and English articles.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

_NON_ALNUM_PATTERN = re.compile(r"[\W_]+")
_WHITESPACE_PATTERN = re.compile(r"\s+")
_THOUSANDS_SEP_PATTERN = re.compile(r"(?<=\d),(?=\d{3}\b)")
_TRAILING_DECIMAL_PATTERN = re.compile(r"\b(\d+)\.0+\b(?!\.)")

_ARTICLES = {"a", "an", "the"}

_PUNCT_TRANSLATION = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "“": '"',
        "”": '"',
        "–": "-",
        "—": "-",
        "…": "...",
        " ": " ",
    }
)


def _normalize_unicode(text: str) -> str:
  normalized = unicodedata.normalize("NFKC", text)
  normalized = unicodedata.normalize("NFD", normalized)
  return "".join(
      ch for ch in normalized if unicodedata.category(ch) != "Mn"
  )


def _normalize_numbers(text: str) -> str:
  text = _THOUSANDS_SEP_PATTERN.sub("", text)
  return _TRAILING_DECIMAL_PATTERN.sub(r"\1", text)


def _strip_articles(text: str) -> str:
  tokens = text.split()
  if not tokens:
    return text
  kept = [token for token in tokens if token not in _ARTICLES]
  if not kept:
    return text
  # Avoid stripping when it would drop meaningful single-letter tokens.
  if not any(len(token) > 1 for token in kept):
    return text
  return " ".join(kept)


def normalize_text_for_match(text: Any) -> str:
  """Lower + strip unicode marks + smart-quotes + thousands seps + articles."""
  normalized = str(text)
  normalized = _normalize_unicode(normalized)
  normalized = normalized.translate(_PUNCT_TRANSLATION)
  normalized = _normalize_numbers(normalized)
  normalized = normalized.lower()
  normalized = _NON_ALNUM_PATTERN.sub(" ", normalized)
  normalized = _WHITESPACE_PATTERN.sub(" ", normalized).strip()
  return _strip_articles(normalized)
