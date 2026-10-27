#!/usr/bin/env bash
# Continue only the same prepared run; never prepare/overwrite it again.
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
: "${RUN:?Set RUN to the existing prepared run directory}"
: "${ATTEMPT:?Set ATTEMPT to a new positive integer (1, 2, ...)}"
[[ "$ATTEMPT" =~ ^[1-9][0-9]*$ ]] || { echo 'ATTEMPT must be positive' >&2; exit 2; }
PY="${PY:-python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
[[ -f "$RUN/resolved.yml" ]] || { echo 'Missing prepared resolved.yml' >&2; exit 2; }
log="$RUN/console.recovery${ATTEMPT}.log"
[[ ! -e "$log" ]] || { echo "Refusing to overwrite $log" >&2; exit 2; }
"$PY" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 \
  tools/table_cells/train_sgqs_v2.py -c "$RUN/resolved.yml" --recover --attempt "$ATTEMPT" \
  2>&1 | tee "$log"
