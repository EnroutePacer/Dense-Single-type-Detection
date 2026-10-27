#!/usr/bin/env bash
# Generic process launcher. No experiment dispatch, data paths, or training defaults.
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
python_bin=python
nproc=''
port=''
gpus=''
device=''
args=()
while (($#)); do
  case "$1" in
    --python) python_bin="$2"; shift 2;;
    --nproc-per-node) nproc="$2"; shift 2;;
    --master-port) port="$2"; shift 2;;
    --gpus) gpus="$2"; shift 2;;
    --device) device="$2"; args+=("$1" "$2"); shift 2;;
    -h|--help)
      echo 'Launcher: --nproc-per-node N --master-port PORT --device cpu|cuda [--gpus 0,1] [--python PATH]'
      "$python_bin" tools/table_cells/train_sgqs_v2.py --help
      exit 0;;
    *) args+=("$1"); shift;;
  esac
done
[[ "$nproc" =~ ^[1-9][0-9]*$ ]] || { echo 'Explicit positive --nproc-per-node required' >&2; exit 2; }
[[ "$port" =~ ^[0-9]+$ && "$port" -gt 0 && "$port" -le 65535 ]] || { echo 'Explicit valid --master-port required' >&2; exit 2; }
case "$device" in
  cuda)
    [[ -n "$gpus" ]] || { echo 'CUDA launch requires explicit --gpus' >&2; exit 2; }
    IFS=',' read -ra devices <<< "$gpus"
    [[ "${#devices[@]}" -eq "$nproc" ]] || { echo 'GPU list must match process count' >&2; exit 2; }
    declare -A seen=()
    for gpu in "${devices[@]}"; do
      [[ -n "$gpu" && -z "${seen[$gpu]:-}" ]] || { echo 'GPU IDs must be nonempty and unique' >&2; exit 2; }
      seen[$gpu]=1
    done
    export CUDA_VISIBLE_DEVICES="$gpus";;
  cpu)
    [[ -z "$gpus" ]] || { echo 'CPU launch does not accept --gpus' >&2; exit 2; }
    export CUDA_VISIBLE_DEVICES='';;
  *) echo 'Explicit --device cpu|cuda required' >&2; exit 2;;
esac
exec "$python_bin" -m torch.distributed.run --nnodes=1 --nproc_per_node="$nproc" \
  --master_addr=127.0.0.1 --master_port="$port" \
  tools/table_cells/train_sgqs_v2.py "${args[@]}"
