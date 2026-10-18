#!/usr/bin/env bash
# Run sequentially on the same eight GPUs; each arm starts from epoch 27.
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PY="${PY:-python}"
DATA="${DATA:-/aicamera-mlp/chenkai/data/table_det_data/paddlex_l_cell_merged}"
CKPT="${CKPT:-$PWD/output/dfine_hgnetv2_m_table_cell/checkpoint0027.pth}"
RUN_ROOT="${RUN_ROOT:-$PWD/outputs/sgqs_v2}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
if [[ $# -eq 0 ]]; then set -- r0 r1 r2 r3; fi
for arm in "$@"; do
  case "$arm" in r0|r1|r2|r3) ;; *) echo "Unknown arm: $arm" >&2; exit 2;; esac
done
for arm in "$@"; do
  run="$RUN_ROOT/${arm}_e28_r7_seed0"
  extra=()
  if [[ -n "${REV:-}" ]]; then extra+=(--source-revision "$REV"); fi
  "$PY" tools/table_cells/prepare_sgqs_v2_run.py \
    --arm "$arm" --resume "$CKPT" --data-root "$DATA" --output "$run" \
    --workers "${WORKERS:-4}" "${extra[@]}"
  "$PY" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 \
    tools/table_cells/train_sgqs_v2.py -c "$run/resolved.yml" \
    2>&1 | tee "$run/console.log"
  "$PY" tools/table_cells/summarize_probe.py --run "$run" --output "$run/probe_steps.csv"
done
