#!/usr/bin/env bash
# Direct TPU-VM multi-host vanilla GRPO smoke launcher.
#
# Usage:
#   export HF_TOKEN=...
#   bash /home/gs1693/repo/tunix/examples/rl/grpo/gsm8k/launch_qwen3_simplereward_smoke_multihost_tpu.sh
#
# This wraps launch_qwen3_simplereward_smoke_tpu.sh with explicit multi-host
# defaults. Override any SMOKE_* variable below to scale the run up or down.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_LAUNCHER="$SCRIPT_DIR/launch_qwen3_simplereward_smoke_tpu.sh"

DATE_TIME_STR=$(date -u +%Y%m%dT%H%M%SZ)
export SMOKE_LOG_DIR="${SMOKE_LOG_DIR:-/home/gs1693/repo/tunix/examples/rl/grpo/logs/${DATE_TIME_STR}_qwen3_simplereward_smoke_multihost}"
export SMOKE_LOG_STEM="${SMOKE_LOG_STEM:-qwen3_simplereward_smoke_multihost_${DATE_TIME_STR}}"
export SMOKE_MESH_SHAPE="${SMOKE_MESH_SHAPE:-"(8,4)"}"
export SMOKE_MESH_AXIS_NAMES="${SMOKE_MESH_AXIS_NAMES:-"('fsdp','tp')"}"
export SMOKE_BATCH_SIZE="${SMOKE_BATCH_SIZE:-8}"
export SMOKE_NUM_BATCHES="${SMOKE_NUM_BATCHES:-1}"
export SMOKE_NUM_GENERATIONS="${SMOKE_NUM_GENERATIONS:-4}"
export SMOKE_MAX_STEPS="${SMOKE_MAX_STEPS:-2}"
export SMOKE_TOTAL_GENERATION_STEPS="${SMOKE_TOTAL_GENERATION_STEPS:-128}"
export SMOKE_MAX_PROMPT_LENGTH="${SMOKE_MAX_PROMPT_LENGTH:-256}"

mkdir -p "$SMOKE_LOG_DIR"

echo "[multihost-smoke] Logs will be written under:"
echo "  $SMOKE_LOG_DIR"
echo "[multihost-smoke] After the run, send back matching log paths:"
echo "  $SMOKE_LOG_DIR/${SMOKE_LOG_STEM}_*.log"
echo "[multihost-smoke] Mesh: $SMOKE_MESH_SHAPE $SMOKE_MESH_AXIS_NAMES"
echo "[multihost-smoke] Global batch size: $SMOKE_BATCH_SIZE"

exec bash "$BASE_LAUNCHER"
