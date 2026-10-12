#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_FILE="${PROJECT_ROOT}/configs/table_cells/dfine_hgnetv2_s_table_cell.yml"
PRETRAINED="${PRETRAINED:-${PROJECT_ROOT}/deim_dfine_hgnetv2_s_coco_120e.pth}"
RUNTIME_ENV="${RUNTIME_ENV:-/aicamera-mlp/chenkai/envs/deim}"

# Override these without editing the script. For the output of
# merge_table_cell_coco.py, DATA_ROOT is the merged root containing images/
# and annotations/. The flat train/val layout remains supported as fallback.
DATA_ROOT="${DATA_ROOT:-}"
TRAIN_IMAGES="${TRAIN_IMAGES:-}"
TRAIN_ANN="${TRAIN_ANN:-}"
VAL_IMAGES="${VAL_IMAGES:-}"
VAL_ANN="${VAL_ANN:-}"
if [[ -n "${DATA_ROOT}" && -d "${DATA_ROOT}/images" && -f "${DATA_ROOT}/annotations/instance_train.json" ]]; then
    TRAIN_IMAGES="${TRAIN_IMAGES:-${DATA_ROOT}/images}"
    TRAIN_ANN="${TRAIN_ANN:-${DATA_ROOT}/annotations/instance_train.json}"
    VAL_IMAGES="${VAL_IMAGES:-${DATA_ROOT}/images}"
    VAL_ANN="${VAL_ANN:-${DATA_ROOT}/annotations/instance_val.json}"
else
    TRAIN_IMAGES="${TRAIN_IMAGES:-${DATA_ROOT}/train}"
    TRAIN_ANN="${TRAIN_ANN:-${DATA_ROOT}/annotations/instances_train.json}"
    VAL_IMAGES="${VAL_IMAGES:-${DATA_ROOT}/val}"
    VAL_ANN="${VAL_ANN:-${DATA_ROOT}/annotations/instances_val.json}"
fi
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/output/dfine_hgnetv2_s_table_cell}"
GPUS="${GPUS:-0,1,2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-$(awk -F, '{print NF}' <<< "${GPUS}")}"
MASTER_PORT="${MASTER_PORT:-29517}"
EPOCHS="${EPOCHS:-120}"
# Batch size is per GPU, matching the PaddleX launcher convention.
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-2}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-0}"
NUM_QUERIES="${NUM_QUERIES:-640}"

die() { echo "错误: $*" >&2; exit 2; }

if [[ ! -d "${RUNTIME_ENV}" ]]; then
    die "Conda 环境不存在: ${RUNTIME_ENV}"
fi
if [[ "${CONDA_PREFIX:-}" != "${RUNTIME_ENV}" ]]; then
    CONDA_CMD="${CONDA_EXE:-$(command -v conda || true)}"
    [[ -n "${CONDA_CMD}" ]] || die "找不到 conda，无法激活环境: ${RUNTIME_ENV}"
    eval "$("${CONDA_CMD}" shell.bash hook)"
    conda activate "${RUNTIME_ENV}"
fi
[[ "${CONDA_PREFIX:-}" == "${RUNTIME_ENV}" ]] || die "Conda 环境激活失败: ${RUNTIME_ENV}"

[[ -f "${CONFIG_FILE}" ]] || die "配置文件不存在: ${CONFIG_FILE}"
[[ -f "${PRETRAINED}" ]] || die "预训练权重不存在: ${PRETRAINED}"
[[ -d "${TRAIN_IMAGES}" ]] || die "训练图片目录不存在: ${TRAIN_IMAGES}"
[[ -f "${TRAIN_ANN}" ]] || die "训练 COCO 标注不存在: ${TRAIN_ANN}"
[[ -d "${VAL_IMAGES}" ]] || die "验证图片目录不存在: ${VAL_IMAGES}"
[[ -f "${VAL_ANN}" ]] || die "验证 COCO 标注不存在: ${VAL_ANN}"
command -v torchrun >/dev/null 2>&1 || die "找不到 torchrun，请先激活包含 PyTorch 的环境"

export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ "${NPROC_PER_NODE}" -lt 1 ]]; then
    die "NPROC_PER_NODE 必须为正整数"
fi
[[ "${BATCH_SIZE_PER_GPU}" -gt 0 ]] || die "BATCH_SIZE_PER_GPU 必须为正整数"
TOTAL_BATCH_SIZE=$((BATCH_SIZE_PER_GPU * NPROC_PER_NODE))

cd "${PROJECT_ROOT}"
echo "启动 D-FINE-S 表格单元格训练: GPUs=${GPUS}, epochs=${EPOCHS}, output=${OUTPUT_DIR}"

CUDA_VISIBLE_DEVICES="${GPUS}" torchrun \
  --master_port="${MASTER_PORT}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  train.py -c "${CONFIG_FILE}" \
  --use-amp --seed="${SEED}" --output-dir "${OUTPUT_DIR}" -t "${PRETRAINED}" \
  -u \
  "num_classes=1" \
  "epoches=${EPOCHS}" \
  "DFINETransformer.num_queries=${NUM_QUERIES}" \
  "PostProcessor.num_top_queries=${NUM_QUERIES}" \
  "train_dataloader.dataset.img_folder=${TRAIN_IMAGES}" \
  "train_dataloader.dataset.ann_file=${TRAIN_ANN}" \
  "val_dataloader.dataset.img_folder=${VAL_IMAGES}" \
  "val_dataloader.dataset.ann_file=${VAL_ANN}" \
  "train_dataloader.total_batch_size=${TOTAL_BATCH_SIZE}" \
  "val_dataloader.total_batch_size=${TOTAL_BATCH_SIZE}" \
  "train_dataloader.num_workers=${NUM_WORKERS}" \
  "val_dataloader.num_workers=${NUM_WORKERS}"
