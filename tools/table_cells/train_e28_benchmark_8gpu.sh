#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

EXPERIMENT="${EXPERIMENT:-${1:-}}"
SEED="${SEED:-0}"
DATA_ROOT="${DATA_ROOT:-/aicamera-mlp/chenkai/data/table_det_data/paddlex_l_cell_merged}"
PRETRAINED="${PRETRAINED:-${PROJECT_ROOT}/deim_dfine_hgnetv2_m_coco_90e.pth}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
MASTER_PORT="${MASTER_PORT:-29528}"
NUM_WORKERS="${NUM_WORKERS:-2}"
ALLOW_EXISTING="${ALLOW_EXISTING:-0}"

die() { echo "错误: $*" >&2; exit 2; }

case "${EXPERIMENT}" in
  vfl_clean|mal_clean|vfl_mosaic_only|vfl_mixup_only|vfl_dense_o2o|mal_dense_o2o|deim_official)
    ;;
  *)
    die "EXPERIMENT 必须是: vfl_clean, mal_clean, vfl_mosaic_only, vfl_mixup_only, vfl_dense_o2o, mal_dense_o2o, deim_official"
    ;;
esac

[[ "${SEED}" =~ ^[0-9]+$ ]] || die "SEED 必须为非负整数"
GPU_COUNT="$(awk -F, '{print NF}' <<< "${GPUS}")"
[[ "${GPU_COUNT}" -eq 8 ]] || die "该基准固定使用 8 张 GPU；当前 GPUS=${GPUS}"

CONFIG_FILE="${PROJECT_ROOT}/configs/table_cells/benchmark_e28/${EXPERIMENT}.yml"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/output/benchmark_e28/${EXPERIMENT}_seed${SEED}}"
BASE_LAUNCHER="${PROJECT_ROOT}/tools/table_cells/train_dfine_m_table_cell_multigpu.sh"

[[ -f "${CONFIG_FILE}" ]] || die "缺少配置文件: ${CONFIG_FILE}"
[[ -f "${BASE_LAUNCHER}" ]] || die "缺少基础启动脚本: ${BASE_LAUNCHER}"
[[ -f "${PRETRAINED}" ]] || die "缺少统一预训练权重: ${PRETRAINED}"

if [[ -d "${OUTPUT_DIR}" && -n "$(find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" && "${ALLOW_EXISTING}" != "1" ]]; then
  die "输出目录非空，为防止混合实验已停止: ${OUTPUT_DIR}。如确认继续，显式设置 ALLOW_EXISTING=1。"
fi
mkdir -p "${OUTPUT_DIR}"

{
  echo "created_at=$(date -Is)"
  echo "hostname=$(hostname)"
  echo "experiment=${EXPERIMENT}"
  echo "seed=${SEED}"
  echo "config=${CONFIG_FILE}"
  echo "config_sha256=$(sha256sum "${CONFIG_FILE}" | awk '{print $1}')"
  echo "pretrained=${PRETRAINED}"
  echo "data_root=${DATA_ROOT}"
  echo "gpus=${GPUS}"
  echo "nproc_per_node=8"
  echo "batch_size_per_gpu=2"
  echo "global_batch=16"
  echo "epoches=28"
  echo "epoch_indices=0..27"
  echo "quick_steps=0"
  echo "resume=disabled"
} | tee "${OUTPUT_DIR}/run_manifest.txt"

cp "${CONFIG_FILE}" "${OUTPUT_DIR}/config_source.yml"

CONFIG_FILE="${CONFIG_FILE}" \
PRETRAINED="${PRETRAINED}" \
RESUME="" \
QUICK_STEPS=0 \
QUICK_OUTPUT_DIR="" \
OUTPUT_DIR="${OUTPUT_DIR}" \
DATA_ROOT="${DATA_ROOT}" \
GPUS="${GPUS}" \
NPROC_PER_NODE=8 \
BATCH_SIZE_PER_GPU=2 \
NUM_WORKERS="${NUM_WORKERS}" \
EPOCHS=28 \
NUM_QUERIES=640 \
SEED="${SEED}" \
MASTER_PORT="${MASTER_PORT}" \
bash "${BASE_LAUNCHER}" 2>&1 | tee "${OUTPUT_DIR}/console.log"

