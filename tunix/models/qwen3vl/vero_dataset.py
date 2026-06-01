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

"""Vero 5-domain JSONL dataset for Qwen3-VL GRPO.

Mirrors the grain idiom from ``grpo_example.py`` (``grain.DataLoader`` +
``grain.IndexSampler`` + ``grain.MapTransform`` + ``worker_count=0``),
but sources rows from a LOCAL ``*.verl.jsonl`` file and joins images
from a LOCAL ``image_root`` (caller pre-stages from GCS).

Each input row follows the verl schema documented in the setup report:

    {
      "data_source": ...,
      "prompt": [{"role": "user", "content": "<image>\\n<question>"}],
      "images": ["images/.../foo.png"],
      "ability": "<domain tag, e.g. chart_ocr>",
      "reward_model": {"style": "rule", "ground_truth": "..."},
      "extra_info": {"reward_type": "numeric", "tolerance": 0.05, ...},
      "avg_pass_rate": 0.16
    }

The mapper yields the dict shape consumed by the 5-domain reward router:

    {
      "image":        PIL.Image (RGB, square-resized to ``max_image_size``),
      "question":     str (leading "<image>\\n" / "<image>" stripped),
      "ground_truth": str (json.dumps if list, else str()),
      "domain":       str (row["ability"]),
      "reward_type":  str (row["extra_info"]["reward_type"], default "string_match"),
      "extra_info":   dict (row["extra_info"] preserved verbatim),
    }
"""

from __future__ import annotations

import json
import logging
import os
import random
import subprocess
from typing import Any

from grain import python as grain
import jax
import PIL.Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Row helpers
# ---------------------------------------------------------------------------


def _strip_image_token(text: str) -> str:
  """Remove a leading ``<image>\\n`` or ``<image>`` from the question."""
  if text.startswith('<image>\n'):
    return text[len('<image>\n'):]
  if text.startswith('<image>'):
    return text[len('<image>'):]
  return text


def _coerce_ground_truth(gt: Any) -> str:
  """List -> ``json.dumps``; everything else -> ``str()``."""
  if isinstance(gt, list):
    return json.dumps(gt)
  return str(gt)


def _extract_question(row: dict[str, Any]) -> str:
  """Pull the user question out of ``row['prompt'][0]['content']``."""
  prompt = row.get('prompt') or []
  if not prompt:
    return ''
  content = prompt[0].get('content', '')
  if isinstance(content, list):
    # Already in chat-template form: pick the first text segment.
    for seg in content:
      if isinstance(seg, dict) and seg.get('type') == 'text':
        return _strip_image_token(str(seg.get('text', '')))
    return ''
  return _strip_image_token(str(content))


def _load_jsonl(jsonl_path: str) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  with open(jsonl_path, 'r', encoding='utf-8') as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      rows.append(json.loads(line))
  return rows


# ---------------------------------------------------------------------------
# Grain source + transforms
# ---------------------------------------------------------------------------


class _VeroRowSource:
  """In-memory random-access source over a list of jsonl rows.

  Grain's ``IndexSampler`` only needs ``__len__`` + ``__getitem__``; we
  hand it a plain list-like adapter so we keep full control over the
  domain filtering / mix-weighted sampling that happens up-front.
  """

  def __init__(self, rows: list[dict[str, Any]]):
    self._rows = rows

  def __len__(self) -> int:
    return len(self._rows)

  def __getitem__(self, idx: int) -> dict[str, Any]:
    return self._rows[idx]


class _PrepareVeroRow(grain.MapTransform):
  """Convert one verl-style jsonl row to the trainer-side dict."""

  def __init__(self, image_root: str, max_image_size: int):
    self._image_root = image_root
    self._max_image_size = int(max_image_size)

  def map(self, element: dict[str, Any]) -> dict[str, Any]:
    images = element.get('images') or []
    if not images:
      raise ValueError(
          f'vero row missing images: id={element.get("extra_info", {}).get("id")}'
      )
    rel = images[0]
    # Defensive: relative paths from verl; never absolute, never gs://.
    image_path = os.path.join(self._image_root, rel)
    image = PIL.Image.open(image_path).convert('RGB')
    # Fixed square resize: keeps the vision-patch count SPMD-uniform on
    # multi-host (same reasoning as _PrepareChartQA in grpo_example.py).
    image = image.resize(
        (self._max_image_size, self._max_image_size),
        PIL.Image.BICUBIC,
    )

    extra_info = element.get('extra_info', {}) or {}
    reward_type = extra_info.get('reward_type') or 'string_match'

    gt = (element.get('reward_model') or {}).get('ground_truth', '')
    ground_truth = _coerce_ground_truth(gt)

    return {
        'image': image,
        'question': _extract_question(element),
        'ground_truth': ground_truth,
        'domain': str(element.get('ability', '')),
        'reward_type': str(reward_type),
        'extra_info': dict(extra_info),
    }


# ---------------------------------------------------------------------------
# Mix-weighted multi-domain source
# ---------------------------------------------------------------------------


class _MixWeightedDomainSource:
  """Random-access source that mixes domains per weight at draw time.

  Grain's ``IndexSampler`` calls ``__getitem__(idx)``; we ignore ``idx``
  for the domain pick (it's used only as a seed so each draw is
  reproducible per epoch+process) and instead:

    1. Sample one domain via the supplied weights.
    2. Sample one row within that domain (shuffled-per-process).

  The resulting stream isn't strictly index-stable, but neither is the
  shuffled HF dataset in ``grpo_example.create_dataset``; in both cases
  the trainer just pulls ``next(data_iter)`` per step.
  """

  def __init__(
      self,
      domain_to_rows: dict[str, list[dict[str, Any]]],
      domains: list[str],
      weights: list[float],
      shuffle_seed: int,
  ):
    if len(domains) != len(weights):
      raise ValueError(
          f'mix_weights length {len(weights)} != domains length {len(domains)}'
      )
    if not domains:
      raise ValueError('mix_weights requires at least one domain')
    self._domains = list(domains)
    total = float(sum(weights))
    if total <= 0:
      raise ValueError(f'mix_weights must sum to > 0; got {weights}')
    self._weights = [float(w) / total for w in weights]
    # Per-domain shuffled index list. Each draw advances a per-domain
    # cursor; we reshuffle when exhausted. Keeps row order stable within
    # a single process for a given seed.
    self._domain_rows = {d: list(domain_to_rows.get(d, [])) for d in domains}
    self._domain_order: dict[str, list[int]] = {}
    self._domain_cursor: dict[str, int] = {}
    base_rng = random.Random(shuffle_seed)
    for d in domains:
      rng = random.Random(base_rng.randrange(1 << 30))
      order = list(range(len(self._domain_rows[d])))
      rng.shuffle(order)
      self._domain_order[d] = order
      self._domain_cursor[d] = 0
    # Total nominal length: sum across domains. IndexSampler treats this
    # as the epoch length; we cycle internally so going past it is fine.
    self._total = sum(len(rows) for rows in self._domain_rows.values())
    if self._total == 0:
      raise ValueError(
          'mix_weighted source has zero rows across all domains'
          f' (domains={domains})'
      )
    # Seeded per-step domain picker. Index passed into __getitem__ is
    # used as an offset so different processes/steps hit different rows.
    self._pick_rng = random.Random(shuffle_seed ^ 0xA5A5)
    # Salt for per-idx seeding in __getitem__ (random.Random rejects
    # tuples, so we mix idx + salt into a single int).
    self._pick_seed_base = self._pick_rng.getrandbits(64)

  def __len__(self) -> int:
    return self._total

  def _next_row_for_domain(self, domain: str) -> dict[str, Any]:
    rows = self._domain_rows[domain]
    if not rows:
      raise RuntimeError(
          f'no rows for domain {domain!r} (filtered out?)'
      )
    order = self._domain_order[domain]
    cursor = self._domain_cursor[domain]
    if cursor >= len(order):
      # Reshuffle and restart the cycle.
      self._pick_rng.shuffle(order)
      cursor = 0
    row_idx = order[cursor]
    self._domain_cursor[domain] = cursor + 1
    return rows[row_idx]

  def __getitem__(self, idx: int) -> dict[str, Any]:
    # Mix idx into the picker so we don't degenerate to one domain per
    # epoch under a deterministic IndexSampler. random.Random only
    # accepts None/int/float/str/bytes — combine idx with a salted int.
    seed = (idx * 0x9E3779B97F4A7C15) ^ self._pick_seed_base
    r = random.Random(seed & 0xFFFFFFFFFFFFFFFF)
    pick = r.random()
    acc = 0.0
    chosen = self._domains[-1]
    for d, w in zip(self._domains, self._weights):
      acc += w
      if pick <= acc:
        chosen = d
        break
    return self._next_row_for_domain(chosen)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_vero_jsonl_dataset(
    jsonl_path: str,
    image_root: str,
    max_image_size: int,
    domains_filter: list[str] | None = None,
    mix_weights: list[float] | None = None,
    shuffle_seed: int = 0,
) -> grain.DataLoader:
  """Build a grain ``DataLoader`` over a verl-style Qwen3-VL jsonl.

  Mirrors the grain idiom in ``grpo_example.create_dataset``:
  ``data_source`` + ``IndexSampler`` + ``operations=[mapper]`` +
  ``worker_count=0`` (the HF / Qwen3-VL ``AutoProcessor`` is not
  fork-safe; multi-process loading is unsafe here too).

  Args:
    jsonl_path: Local path to the ``*.verl.jsonl`` file.
    image_root: Local directory under which ``row['images'][0]`` is
      resolved as a relative path.
    max_image_size: Square resize edge length passed to PIL (BICUBIC).
    domains_filter: If set, drop any row whose ``ability`` is not in
      this list. Order is preserved for use as ``mix_weights`` keys.
    mix_weights: If set AND aligned with ``domains_filter`` (same length,
      same order), weighted-sample one domain per draw and then one row
      within that domain (per-domain shuffled). If unset, rows are
      consumed in jsonl order (shuffled per epoch by IndexSampler).
    shuffle_seed: Seed for per-domain shuffles and the domain picker.
  """
  rows = _load_jsonl(jsonl_path)
  if not rows:
    raise ValueError(f'vero jsonl is empty: {jsonl_path}')

  # Apply domain filter up front so the source length matches the rows
  # we actually sample from.
  if domains_filter is not None:
    keep = set(domains_filter)
    rows = [r for r in rows if str(r.get('ability', '')) in keep]
    if not rows:
      raise ValueError(
          f'no rows match domains_filter={domains_filter}; check ability tags'
      )

  use_mix = (
      mix_weights is not None
      and domains_filter is not None
      and len(mix_weights) == len(domains_filter)
  )
  if use_mix:
    domain_to_rows: dict[str, list[dict[str, Any]]] = {
        d: [] for d in domains_filter
    }
    for r in rows:
      domain_to_rows[str(r['ability'])].append(r)
    data_source: Any = _MixWeightedDomainSource(
        domain_to_rows=domain_to_rows,
        domains=list(domains_filter),
        weights=list(mix_weights),
        shuffle_seed=shuffle_seed,
    )
    logger.info(
        '[vero] mix-weighted source: domains=%s weights=%s total_rows=%d',
        list(domains_filter), list(mix_weights), len(data_source),
    )
  else:
    data_source = _VeroRowSource(rows)
    logger.info(
        '[vero] flat source over %d rows from %s', len(rows), jsonl_path,
    )

  # Sharding policy mirrors grpo_example.create_dataset: under multi-host
  # we shard by JAX process; single host (or any caller test path) gets
  # NoSharding so every process sees the same stream.
  if jax.process_count() == 1:
    shard_options: Any = grain.NoSharding()
  else:
    shard_options = grain.ShardByJaxProcess(drop_remainder=True)

  return grain.DataLoader(
      data_source=data_source,
      sampler=grain.IndexSampler(
          num_records=len(data_source),
          num_epochs=1000,  # effectively infinite for a smoke
          shard_options=shard_options,
          shuffle=True,
          seed=shuffle_seed,
      ),
      operations=[_PrepareVeroRow(image_root, max_image_size)],
      worker_count=0,  # AutoProcessor / PIL fork-safety: keep in-process.
  )


# ---------------------------------------------------------------------------
# Smoke-asset staging
# ---------------------------------------------------------------------------


def _gcloud_cp(src: str, dst: str, recursive: bool = False) -> None:
  """Run ``gcloud storage cp [-r] src dst`` and raise on failure."""
  cmd = ['gcloud', 'storage', 'cp']
  if recursive:
    cmd.append('-r')
  cmd.extend([src, dst])
  subprocess.run(cmd, check=True)


def _collect_image_paths(jsonl_path: str) -> list[str]:
  """Return the unique relative image paths referenced by the jsonl."""
  seen: set[str] = set()
  with open(jsonl_path, 'r', encoding='utf-8') as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      row = json.loads(line)
      for rel in row.get('images') or []:
        if isinstance(rel, str) and rel:
          seen.add(rel)
  return sorted(seen)


def download_smoke_assets(
    gs_jsonl: str,
    gs_image_root: str,
    local_dir: str = '/tmp/qwen3vl_vero_smoke',
) -> tuple[str, str]:
  """Stage the smoke jsonl + referenced images from GCS to ``local_dir``.

  Idempotent: skips files that already exist locally. Only the subset of
  images referenced by the jsonl is downloaded (the full ``test_data``
  tree could be GB). Uses ``gcloud storage cp`` (project convention —
  never ``gsutil``).

  Args:
    gs_jsonl: ``gs://...`` URI of the verl jsonl.
    gs_image_root: ``gs://...`` prefix that the jsonl's relative image
      paths are joined under.
    local_dir: Local staging directory.

  Returns:
    ``(local_jsonl_path, local_image_root)``.
  """
  os.makedirs(local_dir, exist_ok=True)
  local_jsonl = os.path.join(local_dir, os.path.basename(gs_jsonl))
  local_image_root = os.path.join(local_dir, 'images')
  os.makedirs(local_image_root, exist_ok=True)

  # 1. JSONL.
  if not os.path.exists(local_jsonl):
    logger.info('[vero] staging jsonl %s -> %s', gs_jsonl, local_jsonl)
    _gcloud_cp(gs_jsonl, local_jsonl, recursive=False)
  else:
    logger.info('[vero] jsonl already staged at %s', local_jsonl)

  # 2. Images: one recursive copy of the entire image tree. The val 2k
  # tree is ~25 MB / ~100 files; bulk cp finishes in seconds, whereas the
  # per-file loop spends ~10 s on gcloud startup per file (16 min for 100
  # files) and times JAX's coordination barrier out.
  # Path convention: jsonl 'images[0]' is a path that ALREADY starts with
  # 'images/' (e.g. 'images/gs1693/...'), so 'gcloud storage cp -r
  # <gs_root>/images <local_image_root>' produces the doubled
  # '<local_image_root>/images/gs1693/...' layout the row mapper expects.
  gs_root = gs_image_root.rstrip('/')
  src_images = f'{gs_root}/images'
  marker = os.path.join(local_image_root, '.staged_ok')
  if os.path.exists(marker):
    logger.info('[vero] images already staged at %s (marker present)', local_image_root)
  else:
    logger.info('[vero] bulk staging %s -> %s', src_images, local_image_root)
    try:
      _gcloud_cp(src_images, local_image_root, recursive=True)
      with open(marker, 'w') as f:
        f.write('ok')
    except subprocess.CalledProcessError as e:
      logger.warning('[vero] bulk image stage failed: %s', e)
      raise
  logger.info('[vero] image staging complete under %s', local_image_root)

  return local_jsonl, local_image_root
