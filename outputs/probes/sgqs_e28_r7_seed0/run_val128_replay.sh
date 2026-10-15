#!/usr/bin/env bash
set -Eeuo pipefail

REPO=/aicamera-mlp/zhuyouwei/code/DEIM
PY=/aicamera-mlp/chenkai/envs/deim/bin/python
RUN="$REPO/outputs/probes/sgqs_e28_r7_seed0"
MANIFEST="$REPO/_local/probe_fixed128_valonly/manifest.json"

cd "$REPO"
test -x "$PY"
test -f "$MANIFEST"

for epoch in 0028 0029 0030 0031 0032 0033 0034; do
    checkpoint="$RUN/checkpoint${epoch}.pth"
    output="$RUN/replay_val128_e${epoch}"
    test -f "$checkpoint"
    if test -e "$output"; then
        echo "Refusing to overwrite existing output: $output" >&2
        exit 1
    fi
    echo "[$(date '+%F %T %z')] replay e${epoch} raw+ema"
    CUDA_VISIBLE_DEVICES=0 "$PY" tools/table_cells/replay_probe.py \
        --config "$RUN/resolved.yml" \
        --checkpoint "$checkpoint" \
        --manifest "$MANIFEST" \
        --device cuda \
        --states raw ema \
        --output "$output"
done

echo "[$(date '+%F %T %z')] all replay checkpoints complete"
