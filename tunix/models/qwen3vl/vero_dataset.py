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

import glob
import io
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
    # Drop domains that have zero rows after filtering — otherwise the
    # weighted picker can choose an empty bucket and crash. Re-normalize
    # over the surviving domains. Warn so the user sees which got dropped.
    _all_rows = {d: list(domain_to_rows.get(d, [])) for d in domains}
    _empty = [d for d in domains if not _all_rows[d]]
    if _empty:
      logger.warning(
          '[vero] dropping empty domains from mix: %s '
          '(jsonl has no rows tagged with these abilities)', _empty,
      )
    _kept = [(d, w) for d, w in zip(domains, weights) if _all_rows[d]]
    if not _kept:
      raise ValueError(
          'mix_weighted source has zero rows across all requested domains'
          f' (domains={domains})'
      )
    self._domains = [d for d, _ in _kept]
    _kept_weights = [float(w) for _, w in _kept]
    total = float(sum(_kept_weights))
    if total <= 0:
      raise ValueError(f'mix_weights must sum to > 0; got {weights}')
    self._weights = [w / total for w in _kept_weights]
    # Per-domain shuffled index list. Each draw advances a per-domain
    # cursor; we reshuffle when exhausted. Keeps row order stable within
    # a single process for a given seed.
    self._domain_rows = {d: _all_rows[d] for d in self._domains}
    self._domain_order: dict[str, list[int]] = {}
    self._domain_cursor: dict[str, int] = {}
    base_rng = random.Random(shuffle_seed)
    for d in self._domains:
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


# ---------------------------------------------------------------------------
# Parquet (Vero-600k HF snapshot) support
# ---------------------------------------------------------------------------


def _gcloud_ls(uri: str) -> list[str]:
  """Return entries under ``uri`` via ``gcloud storage ls``.

  Lines are stripped; trailing empties dropped. Raises on non-zero exit.
  """
  out = subprocess.run(
      ['gcloud', 'storage', 'ls', uri],
      check=True, capture_output=True, text=True,
  )
  return [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]


def _gcloud_du_bytes(uri: str) -> int:
  """Return the size in bytes of a single object via ``gcloud storage du``.

  ``gcloud storage du`` prints lines of the form ``<bytes> <uri>``; we
  parse the first column. Returns 0 on parse failure to make picking
  smallest-first robust against transient ls/du races.
  """
  try:
    out = subprocess.run(
        ['gcloud', 'storage', 'du', uri],
        check=True, capture_output=True, text=True,
    )
  except subprocess.CalledProcessError:
    return 0
  for line in out.stdout.splitlines():
    parts = line.strip().split()
    if not parts:
      continue
    try:
      return int(parts[0])
    except ValueError:
      continue
  return 0


def stage_parquet_shards(
    gs_snapshot_root: str,
    domains_filter: list[str],
    shards_per_domain: int = 2,
    local_dir: str = '/tmp/qwen3vl_vero_parquet',
) -> str:
  """Stage a small number of train parquet shards per domain from GCS.

  Idempotent. For each ``<domain>-<source>/`` subset directory under
  ``gs_snapshot_root`` whose ``<domain>`` prefix is in ``domains_filter``,
  copy up to ``shards_per_domain`` train shards (smallest-first by
  ``gcloud storage du``) to ``local_dir``. Uses ``gcloud storage cp``
  (project convention — never ``gsutil``). Writes a ``.staged_ok`` marker
  at ``local_dir`` on success; re-launches with the marker present skip
  immediately.

  Args:
    gs_snapshot_root: ``gs://.../snapshots/<sha>/`` root that holds the
      ``<domain>-<source>/`` subset directories.
    domains_filter: List of domain prefixes to keep (e.g.
      ``['chart_ocr', 'stem']``). Subset dirs whose name does not start
      with ``<prefix>-`` are skipped.
    shards_per_domain: Cap on the number of train parquet shards copied
      per subset directory (smallest-first to keep wall-time bounded).
    local_dir: Local staging root. Created if missing.

  Returns:
    ``local_dir`` (absolute or as provided).
  """
  os.makedirs(local_dir, exist_ok=True)
  marker = os.path.join(local_dir, '.staged_ok')
  if os.path.exists(marker):
    logger.info(
        '[vero-parquet] already staged at %s (marker present)', local_dir,
    )
    return local_dir

  root = gs_snapshot_root.rstrip('/') + '/'
  logger.info('[vero-parquet] listing subsets under %s', root)
  entries = _gcloud_ls(root)
  # Subsets are directory entries — gcloud ls reports them with a trailing
  # slash. Filter out non-dir entries (README.md etc.) and any subset whose
  # leading domain token isn't in domains_filter.
  prefixes = tuple(f'{d}-' for d in domains_filter)
  subset_uris: list[str] = []
  for e in entries:
    if not e.endswith('/'):
      continue
    name = e.rstrip('/').rsplit('/', 1)[-1]
    if name.startswith(prefixes):
      subset_uris.append(e)
  if not subset_uris:
    raise ValueError(
        '[vero-parquet] no subset directories matched domains_filter='
        f'{domains_filter} under {root}'
    )
  logger.info(
      '[vero-parquet] %d subset dirs match domains=%s',
      len(subset_uris), domains_filter,
  )

  for subset_uri in subset_uris:
    subset_name = subset_uri.rstrip('/').rsplit('/', 1)[-1]
    local_subset = os.path.join(local_dir, subset_name)
    os.makedirs(local_subset, exist_ok=True)
    # List all train shards in the subset.
    shard_uris: list[str] = []
    for f in _gcloud_ls(subset_uri):
      base = f.rsplit('/', 1)[-1]
      if base.startswith('train-') and base.endswith('.parquet'):
        shard_uris.append(f)
    if not shard_uris:
      logger.warning(
          '[vero-parquet] subset %s has no train-*.parquet — skipping',
          subset_name,
      )
      continue
    # Pick smallest-first to bound wall-time of the staging step.
    sized = [(u, _gcloud_du_bytes(u)) for u in shard_uris]
    sized.sort(key=lambda t: t[1] if t[1] > 0 else 1 << 62)
    picked = [u for u, _ in sized[:max(1, int(shards_per_domain))]]
    logger.info(
        '[vero-parquet] staging %d/%d train shards for %s',
        len(picked), len(shard_uris), subset_name,
    )
    for u in picked:
      base = u.rsplit('/', 1)[-1]
      dst = os.path.join(local_subset, base)
      if os.path.exists(dst) and os.path.getsize(dst) > 0:
        logger.info('[vero-parquet] %s already present, skipping', base)
        continue
      _gcloud_cp(u, dst, recursive=False)

  with open(marker, 'w') as f:
    f.write('ok')
  logger.info('[vero-parquet] staging complete under %s', local_dir)
  return local_dir


# Required columns for the parquet -> trainer-row conversion. Reading
# only what we need avoids pulling the redundant ``extra_info.question``
# /  ``data_source`` for free.
_PARQUET_REQUIRED_COLUMNS = (
    'id',
    'prompt',
    'ability',
    'reward_model',
    'extra_info',
    'image',
)


def _extract_question_parquet(prompt: Any) -> str:
  """Pull the user question from a parquet ``prompt`` struct.

  HF Sequence-of-Struct flattens to ``{'role': [...], 'content': [...]}``
  with parallel lists. Falls back to the jsonl shape
  ``[{'role': ..., 'content': ...}]`` if a caller hands us pre-converted
  data (defensive — keeps the function reusable).
  """
  if isinstance(prompt, dict):
    contents = prompt.get('content') or []
    if isinstance(contents, list) and contents:
      first = contents[0]
      if isinstance(first, str):
        return _strip_image_token(first)
      return _strip_image_token(str(first))
    return ''
  if isinstance(prompt, list) and prompt:
    first = prompt[0]
    if isinstance(first, dict):
      content = first.get('content', '')
      if isinstance(content, str):
        return _strip_image_token(content)
      if isinstance(content, list):
        for seg in content:
          if isinstance(seg, dict) and seg.get('type') == 'text':
            return _strip_image_token(str(seg.get('text', '')))
        return ''
      return _strip_image_token(str(content))
  return ''


def _decode_parquet_image(image_field: Any) -> PIL.Image.Image | None:
  """Decode a parquet ``image`` struct into a PIL RGB image.

  HF ``Image()`` materializes as ``{'bytes': <binary>, 'path': <str>}``.
  We prefer inline ``bytes``; if that's missing and ``path`` resolves to
  a real local file, fall back to opening it. Returns ``None`` on
  failure so the caller can count + drop the row.
  """
  if not isinstance(image_field, dict):
    return None
  b = image_field.get('bytes')
  if b:
    try:
      return PIL.Image.open(io.BytesIO(b)).convert('RGB')
    except Exception as e:  # pylint: disable=broad-except
      logger.debug('[vero-parquet] PIL decode failed: %s', e)
      return None
  p = image_field.get('path')
  if isinstance(p, str) and p and os.path.exists(p):
    try:
      return PIL.Image.open(p).convert('RGB')
    except Exception as e:  # pylint: disable=broad-except
      logger.debug('[vero-parquet] PIL open path failed: %s', e)
      return None
  return None


def _load_parquet_shards(
    local_parquet_root: str,
    domains_filter: list[str] | None,
    max_image_size: int,
) -> dict[str, list[dict[str, Any]]]:
  """Read all staged parquet shards into per-domain row lists.

  Each row is *already* in the trainer-side dict shape emitted by
  ``_PrepareVeroRow.map`` — the parquet path does the decode + resize
  up front (matches the in-memory loading contract called out in the
  task description; downstream grain mapper is a no-op pass-through).

  Returns a ``{domain: [row, ...]}`` mapping. Domains absent from the
  staged data simply yield empty lists (the mix source drops them).
  """
  import pyarrow.parquet as pq  # local import: pyarrow is env-specific.

  size = int(max_image_size)
  domains_set = set(domains_filter) if domains_filter else None
  domain_to_rows: dict[str, list[dict[str, Any]]] = {}
  dropped_missing = 0
  dropped_decode = 0

  # Discover subset dirs matching the domain prefixes.
  if not os.path.isdir(local_parquet_root):
    raise ValueError(
        f'[vero-parquet] local root does not exist: {local_parquet_root}'
    )
  subset_names = sorted(os.listdir(local_parquet_root))
  for subset_name in subset_names:
    subset_dir = os.path.join(local_parquet_root, subset_name)
    if not os.path.isdir(subset_dir):
      continue
    if '-' not in subset_name:
      continue
    domain = subset_name.split('-', 1)[0]
    if domains_set is not None and domain not in domains_set:
      continue
    shard_paths = sorted(glob.glob(os.path.join(subset_dir, 'train-*.parquet')))
    if not shard_paths:
      continue
    for shard_path in shard_paths:
      try:
        pf = pq.ParquetFile(shard_path)
      except Exception as e:  # pylint: disable=broad-except
        logger.warning(
            '[vero-parquet] failed to open %s: %s', shard_path, e,
        )
        continue
      # Project columns we actually use to skip the redundant copies.
      schema_names = set(pf.schema_arrow.names)
      columns = [c for c in _PARQUET_REQUIRED_COLUMNS if c in schema_names]
      try:
        batches = pf.iter_batches(batch_size=128, columns=columns)
      except Exception:  # pylint: disable=broad-except
        # Older pyarrow versions don't accept columns kwarg on iter_batches.
        batches = pf.iter_batches(batch_size=128)
      for batch in batches:
        for raw in batch.to_pylist():
          image_field = raw.get('image')
          prompt = raw.get('prompt')
          reward_model = raw.get('reward_model') or {}
          extra_info = raw.get('extra_info') or {}
          gt = (
              reward_model.get('ground_truth', '')
              if isinstance(reward_model, dict) else ''
          )
          question = _extract_question_parquet(prompt)
          if not question or image_field is None:
            dropped_missing += 1
            if dropped_missing == 1:
              logger.warning(
                  '[vero-parquet] dropping rows with missing question/image '
                  '(first occurrence in %s)', shard_path,
              )
            continue
          img = _decode_parquet_image(image_field)
          if img is None:
            dropped_decode += 1
            if dropped_decode == 1:
              logger.warning(
                  '[vero-parquet] dropping rows whose image failed to '
                  'decode (first occurrence in %s)', shard_path,
              )
            continue
          img = img.resize((size, size), PIL.Image.BICUBIC)
          reward_type = 'string_match'
          if isinstance(extra_info, dict):
            rt = extra_info.get('reward_type')
            if isinstance(rt, str) and rt:
              reward_type = rt.lower()
          ability = str(raw.get('ability') or domain)
          row = {
              'image': img,
              'question': question,
              'ground_truth': _coerce_ground_truth(gt),
              'domain': ability,
              'reward_type': reward_type,
              'extra_info': (
                  dict(extra_info) if isinstance(extra_info, dict) else {}
              ),
          }
          domain_to_rows.setdefault(ability, []).append(row)
      logger.info(
          '[vero-parquet] loaded shard %s (domain=%s, running_total=%d)',
          os.path.basename(shard_path), domain,
          len(domain_to_rows.get(domain, [])),
      )

  if dropped_missing or dropped_decode:
    logger.warning(
        '[vero-parquet] dropped %d missing-field rows + %d decode-failure '
        'rows during load', dropped_missing, dropped_decode,
    )
  return domain_to_rows


class _PrepareVeroParquetRow(grain.MapTransform):
  """Pass-through mapper for parquet rows.

  The decode + resize already happened in ``_load_parquet_shards`` so
  this is a no-op; we still wrap it as a ``MapTransform`` so the grain
  ``DataLoader.operations`` slot stays non-empty (mirrors the jsonl
  pipeline shape).
  """

  def map(self, element: dict[str, Any]) -> dict[str, Any]:
    return element


def build_vero_parquet_dataset(
    local_parquet_root: str,
    max_image_size: int,
    domains_filter: list[str] | None = None,
    mix_weights: list[float] | None = None,
    shuffle_seed: int = 0,
) -> grain.DataLoader:
  """Build a grain ``DataLoader`` over locally-staged Vero-600k parquet.

  Discovers ``<domain>-<source>/*.parquet`` under ``local_parquet_root``,
  filters by domain prefix, decodes inline image bytes -> PIL -> RGB ->
  square-resize(``max_image_size``, BICUBIC), and emits the SAME row
  dict shape as ``build_vero_jsonl_dataset``:

      {image, question, ground_truth, domain, reward_type, extra_info}

  Uses the existing ``_MixWeightedDomainSource`` / mix-weighted draw
  pattern: drops empty domains (with a warning) and renormalizes the
  surviving weights. Shards are read fully into memory at startup (the
  recommended 1-2 shards per kept domain total ~5-10 GB and fit in host
  RAM) rather than streamed per row — this keeps the SPMD-friendly
  ``grain.IndexSampler`` usable on multi-host.

  Args:
    local_parquet_root: Root directory containing ``<domain>-<source>/``
      subset dirs full of ``train-*.parquet`` shards (typically the
      ``stage_parquet_shards`` output dir).
    max_image_size: Square resize edge length passed to PIL (BICUBIC).
    domains_filter: If set, drop subsets whose ``<domain>`` prefix is
      not in this list. Order is preserved for use as ``mix_weights``
      keys.
    mix_weights: If set AND aligned with ``domains_filter`` (same
      length, same order), weighted-sample one domain per draw and then
      one row within that domain (per-domain shuffled). Otherwise rows
      are sampled in the natural concatenation order (shuffled per
      epoch by ``IndexSampler``).
    shuffle_seed: Seed for per-domain shuffles, the domain picker, and
      the ``IndexSampler``.
  """
  domain_to_rows = _load_parquet_shards(
      local_parquet_root, domains_filter, max_image_size,
  )
  total = sum(len(v) for v in domain_to_rows.values())
  if total == 0:
    raise ValueError(
        '[vero-parquet] no rows loaded under '
        f'{local_parquet_root} for domains_filter={domains_filter}'
    )

  use_mix = (
      mix_weights is not None
      and domains_filter is not None
      and len(mix_weights) == len(domains_filter)
  )
  if use_mix:
    # _MixWeightedDomainSource drops empty domains + renormalizes weights.
    bucketed: dict[str, list[dict[str, Any]]] = {
        d: domain_to_rows.get(d, []) for d in domains_filter
    }
    data_source: Any = _MixWeightedDomainSource(
        domain_to_rows=bucketed,
        domains=list(domains_filter),
        weights=list(mix_weights),
        shuffle_seed=shuffle_seed,
    )
    logger.info(
        '[vero-parquet] mix-weighted source: domains=%s weights=%s '
        'total_rows=%d',
        list(domains_filter), list(mix_weights), len(data_source),
    )
  else:
    flat: list[dict[str, Any]] = []
    keys = (
        list(domains_filter)
        if domains_filter else sorted(domain_to_rows.keys())
    )
    for d in keys:
      flat.extend(domain_to_rows.get(d, []))
    data_source = _VeroRowSource(flat)
    logger.info(
        '[vero-parquet] flat source over %d rows from %s',
        len(flat), local_parquet_root,
    )

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
      operations=[_PrepareVeroParquetRow()],
      worker_count=0,  # PIL / processor fork-safety: keep in-process.
  )
