#!/usr/bin/env bash
# Run the standalone Tunix Qwen3 logits/vanilla-sampler probe on the current
# host. Defaults to CPU so it can run from a single SSH'd TPU worker.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$HOME/miniconda3/envs/tunix/bin/python}"
HF_TOKEN_ENV_FILE="${HF_TOKEN_ENV_FILE:-$HOME/.cache/tunix/hf_token_env}"

if [[ -f "$HF_TOKEN_ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$HF_TOKEN_ENV_FILE"
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python env not found: $PYTHON_BIN" >&2
  exit 1
fi

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
export TUNIX_JAX_DISTRIBUTED_AUTO_INIT="${TUNIX_JAX_DISTRIBUTED_AUTO_INIT:-0}"

cd "$REPO_ROOT"
exec "$PYTHON_BIN" "$SCRIPT_DIR/debug_tunix_qwen3_logits.py" "$@"
