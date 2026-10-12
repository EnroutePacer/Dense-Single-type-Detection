#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

: "${DATA_ROOT:?DATA_ROOT must be set}"
: "${INIT_CHECKPOINT:?INIT_CHECKPOINT must be set}"

RUN_ID="${RUN_ID:-RUN-$(date +%Y%m%d-%H%M%S)-s00}"
RUN_DIR="$REPO_ROOT/_local/runs/EXP-TYPE-000/$RUN_ID"
mkdir -p "$RUN_DIR"

# Replace with this experiment's real command before execution.
python train.py \
  --config configs/paper/REPLACE_ME.yml \
  --seed 0 \
  2>&1 | tee "$RUN_DIR/console.log"
