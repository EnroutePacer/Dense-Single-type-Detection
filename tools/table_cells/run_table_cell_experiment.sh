#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

# Generic, transparent launcher for table-cell experiments.
# Scientific choices live in CONFIG_FILE; runtime overrides are supplied
# explicitly by the calling command.  This script does not dispatch on an
# experiment name and does not enforce a fixed GPU topology or experiment
# contract.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

die() { echo "错误: $*" >&2; exit 2; }
require_env() {
  local name="$1"
  [[ -n "${!name:-}" ]] || die "必须显式设置 ${name}"
}
resolve_project_path() {
  local value="$1"
  if [[ "${value}" == /* ]]; then
    printf '%s\n' "${value}"
  else
    printf '%s\n' "${PROJECT_ROOT}/${value}"
  fi
}
require_uint() {
  local name="$1"
  local value="${!name}"
  [[ "${value}" =~ ^[0-9]+$ ]] || die "${name} 必须为非负整数"
}

for name in \
  RUN_ID CONFIG_FILE OUTPUT_DIR DATA_ROOT RUN_MODE \
  GPUS NPROC_PER_NODE BATCH_SIZE_PER_GPU NUM_WORKERS MASTER_PORT \
  EPOCHS NUM_CLASSES NUM_QUERIES NUM_TOP_QUERIES SEED QUICK_STEPS USE_AMP \
  RUNTIME_ENV PREFLIGHT_ONLY ALLOW_EXISTING; do
  require_env "${name}"
done

CONFIG_FILE="$(resolve_project_path "${CONFIG_FILE}")"
OUTPUT_DIR="$(resolve_project_path "${OUTPUT_DIR}")"
PROVENANCE_FILE="${PROVENANCE_FILE:-}"
RUNTIME_FILES_LIST="${RUNTIME_FILES_LIST:-}"
if [[ -n "${PROVENANCE_FILE}" ]]; then
  PROVENANCE_FILE="$(resolve_project_path "${PROVENANCE_FILE}")"
fi
if [[ -n "${RUNTIME_FILES_LIST}" ]]; then
  RUNTIME_FILES_LIST="$(resolve_project_path "${RUNTIME_FILES_LIST}")"
fi

PYTHON_BIN="${RUNTIME_ENV}/bin/python"
CONFIG_RESOLVER="${PROJECT_ROOT}/tools/table_cells/dump_resolved_training_config.py"
LAUNCHER_PATH="$(resolve_project_path "tools/table_cells/run_table_cell_experiment.sh")"

[[ "${RUN_ID}" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]] || \
  die "RUN_ID 只能包含字母、数字、点、下划线和连字符"
[[ "${RUN_MODE}" == "resume" || "${RUN_MODE}" == "tuning" || "${RUN_MODE}" == "scratch" ]] || \
  die "RUN_MODE 必须是 resume、tuning 或 scratch"
[[ "${GPUS}" =~ ^[0-9]+(,[0-9]+)*$ ]] || die "GPUS 必须是逗号分隔的 GPU 编号"

for name in \
  NPROC_PER_NODE BATCH_SIZE_PER_GPU NUM_WORKERS MASTER_PORT EPOCHS \
  NUM_CLASSES NUM_QUERIES NUM_TOP_QUERIES SEED QUICK_STEPS USE_AMP \
  PREFLIGHT_ONLY ALLOW_EXISTING; do
  require_uint "${name}"
done

[[ "${NPROC_PER_NODE}" -gt 0 ]] || die "NPROC_PER_NODE 必须大于 0"
[[ "${BATCH_SIZE_PER_GPU}" -gt 0 ]] || die "BATCH_SIZE_PER_GPU 必须大于 0"
[[ "${MASTER_PORT}" -gt 0 && "${MASTER_PORT}" -le 65535 ]] || die "MASTER_PORT 必须在 1..65535"
[[ "${EPOCHS}" -gt 0 ]] || die "EPOCHS 必须大于 0"
[[ "${NUM_CLASSES}" -gt 0 ]] || die "NUM_CLASSES 必须大于 0"
[[ "${NUM_QUERIES}" -gt 0 ]] || die "NUM_QUERIES 必须大于 0"
[[ "${NUM_TOP_QUERIES}" -gt 0 ]] || die "NUM_TOP_QUERIES 必须大于 0"
[[ "${USE_AMP}" =~ ^[01]$ ]] || die "USE_AMP 必须是 0 或 1"
[[ "${PREFLIGHT_ONLY}" =~ ^[01]$ ]] || die "PREFLIGHT_ONLY 必须是 0 或 1"
[[ "${ALLOW_EXISTING}" =~ ^[01]$ ]] || die "ALLOW_EXISTING 必须是 0 或 1"

GPU_COUNT="$(awk -F, '{print NF}' <<< "${GPUS}")"
UNIQUE_GPU_COUNT="$(tr ',' '\n' <<< "${GPUS}" | sort -u | wc -l | tr -d ' ')"
[[ "${GPU_COUNT}" -eq "${NPROC_PER_NODE}" ]] || \
  die "GPUS 数量 ${GPU_COUNT} 与 NPROC_PER_NODE=${NPROC_PER_NODE} 不一致"
[[ "${UNIQUE_GPU_COUNT}" -eq "${GPU_COUNT}" ]] || die "GPUS 中存在重复编号: ${GPUS}"

for required_file in "${CONFIG_FILE}" "${CONFIG_RESOLVER}" "${LAUNCHER_PATH}"; do
  [[ -f "${required_file}" ]] || die "缺少文件: ${required_file}"
done
[[ -x "${PYTHON_BIN}" ]] || die "找不到训练环境 Python: ${PYTHON_BIN}"
command -v sha256sum >/dev/null 2>&1 || die "找不到 sha256sum"

if [[ -n "${PROVENANCE_FILE}" ]]; then
  [[ -f "${PROVENANCE_FILE}" ]] || die "方法说明文件不存在: ${PROVENANCE_FILE}"
fi
if [[ -n "${RUNTIME_FILES_LIST}" ]]; then
  [[ -f "${RUNTIME_FILES_LIST}" ]] || die "运行文件清单不存在: ${RUNTIME_FILES_LIST}"
  while IFS= read -r listed_file || [[ -n "${listed_file}" ]]; do
    listed_file="${listed_file%$'\r'}"
    [[ -z "${listed_file}" || "${listed_file}" == \#* ]] && continue
    [[ -f "${PROJECT_ROOT}/${listed_file}" ]] || die "运行文件清单包含不存在的路径: ${listed_file}"
  done < "${RUNTIME_FILES_LIST}"
fi

# Supports merge_table_cell_coco.py output and the legacy split layout.
TRAIN_IMAGES="${TRAIN_IMAGES:-}"
TRAIN_ANN="${TRAIN_ANN:-}"
VAL_IMAGES="${VAL_IMAGES:-}"
VAL_ANN="${VAL_ANN:-}"
if [[ -d "${DATA_ROOT}/images" && -f "${DATA_ROOT}/annotations/instance_train.json" ]]; then
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
[[ -d "${TRAIN_IMAGES}" ]] || die "训练图片目录不存在: ${TRAIN_IMAGES}"
[[ -f "${TRAIN_ANN}" ]] || die "训练标注不存在: ${TRAIN_ANN}"
[[ -d "${VAL_IMAGES}" ]] || die "验证图片目录不存在: ${VAL_IMAGES}"
[[ -f "${VAL_ANN}" ]] || die "验证标注不存在: ${VAL_ANN}"

CHECKPOINT_ARGS=()
SOURCE_CHECKPOINT="none"
CHECKPOINT_LAST_EPOCH="not_applicable"
case "${RUN_MODE}" in
  resume)
    require_env RESUME
    [[ -z "${PRETRAINED:-}" ]] || die "resume 模式不得同时设置 PRETRAINED"
    SOURCE_CHECKPOINT="$(resolve_project_path "${RESUME}")"
    [[ -f "${SOURCE_CHECKPOINT}" ]] || die "恢复训练断点不存在: ${SOURCE_CHECKPOINT}"
    CHECKPOINT_ARGS=(-r "${SOURCE_CHECKPOINT}")
    CHECKPOINT_LAST_EPOCH="$("${PYTHON_BIN}" - "${SOURCE_CHECKPOINT}" <<'PY'
import sys
import torch

try:
    state = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
except TypeError:
    state = torch.load(sys.argv[1], map_location="cpu")
print(state.get("last_epoch", "missing"))
PY
)"
    ;;
  tuning)
    require_env PRETRAINED
    [[ -z "${RESUME:-}" ]] || die "tuning 模式不得同时设置 RESUME"
    SOURCE_CHECKPOINT="$(resolve_project_path "${PRETRAINED}")"
    [[ -f "${SOURCE_CHECKPOINT}" ]] || die "预训练权重不存在: ${SOURCE_CHECKPOINT}"
    CHECKPOINT_ARGS=(-t "${SOURCE_CHECKPOINT}")
    ;;
  scratch)
    [[ -z "${RESUME:-}" && -z "${PRETRAINED:-}" ]] || \
      die "scratch 模式不得设置 RESUME 或 PRETRAINED"
    ;;
esac

if [[ "${QUICK_STEPS}" -gt 0 && "${RUN_MODE}" != "resume" ]]; then
  die "QUICK_STEPS>0 仅支持 resume 模式"
fi

TOTAL_BATCH_SIZE=$((NPROC_PER_NODE * BATCH_SIZE_PER_GPU))
AMP_VALUE="False"
AMP_ARGS=()
if [[ "${USE_AMP}" == "1" ]]; then
  AMP_VALUE="True"
  AMP_ARGS=(--use-amp)
fi

CONFIG_UPDATES=(
  "num_classes=${NUM_CLASSES}"
  "epoches=${EPOCHS}"
  "DFINETransformer.num_queries=${NUM_QUERIES}"
  "PostProcessor.num_top_queries=${NUM_TOP_QUERIES}"
  "train_dataloader.dataset.img_folder=${TRAIN_IMAGES}"
  "train_dataloader.dataset.ann_file=${TRAIN_ANN}"
  "val_dataloader.dataset.img_folder=${VAL_IMAGES}"
  "val_dataloader.dataset.ann_file=${VAL_ANN}"
  "train_dataloader.total_batch_size=${TOTAL_BATCH_SIZE}"
  "val_dataloader.total_batch_size=${TOTAL_BATCH_SIZE}"
  "train_dataloader.num_workers=${NUM_WORKERS}"
  "val_dataloader.num_workers=${NUM_WORKERS}"
)
if [[ "${QUICK_STEPS}" -gt 0 ]]; then
  CONFIG_UPDATES+=("max_train_steps=${QUICK_STEPS}" "quick_probe=True")
else
  CONFIG_UPDATES+=("max_train_steps=None" "quick_probe=False")
fi

PREFLIGHT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/deim-table-cell-preflight.XXXXXX")"
trap 'rm -rf -- "${PREFLIGHT_DIR}"' EXIT
PREFLIGHT_RESOLVED="${PREFLIGHT_DIR}/resolved_config.yml"
RESOLVE_UPDATES=(
  "output_dir=${OUTPUT_DIR}"
  "use_amp=${AMP_VALUE}"
  "seed=${SEED}"
  "${CONFIG_UPDATES[@]}"
)
if [[ "${RUN_MODE}" == "resume" ]]; then
  RESOLVE_UPDATES+=("resume=${SOURCE_CHECKPOINT}" "HGNetv2.pretrained=False")
elif [[ "${RUN_MODE}" == "tuning" ]]; then
  RESOLVE_UPDATES+=("tuning=${SOURCE_CHECKPOINT}" "HGNetv2.pretrained=False")
fi

export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
"${PYTHON_BIN}" "${CONFIG_RESOLVER}" \
  -c "${CONFIG_FILE}" \
  -o "${PREFLIGHT_RESOLVED}" \
  -u "${RESOLVE_UPDATES[@]}"

echo "预检通过: run=${RUN_ID}, mode=${RUN_MODE}, GPUs=${GPUS}, global_batch=${TOTAL_BATCH_SIZE}, epochs=${EPOCHS}, config=${CONFIG_FILE}"
if [[ "${PREFLIGHT_ONLY}" == "1" ]]; then
  echo "PREFLIGHT_ONLY=1：未创建输出目录，未启动 torchrun。"
  exit 0
fi

if [[ -d "${OUTPUT_DIR}" && -n "$(find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" && "${ALLOW_EXISTING}" != "1" ]]; then
  die "输出目录非空，为防止混合实验已停止: ${OUTPUT_DIR}"
fi
mkdir -p "${OUTPUT_DIR}"

CONFIG_SHA256="$(sha256sum "${CONFIG_FILE}" | awk '{print $1}')"
TRAIN_ANN_SHA256="$(sha256sum "${TRAIN_ANN}" | awk '{print $1}')"
VAL_ANN_SHA256="$(sha256sum "${VAL_ANN}" | awk '{print $1}')"
SOURCE_CHECKPOINT_SHA256="none"
if [[ "${SOURCE_CHECKPOINT}" != "none" ]]; then
  SOURCE_CHECKPOINT_SHA256="$(sha256sum "${SOURCE_CHECKPOINT}" | awk '{print $1}')"
fi
if GIT_COMMIT="$(git -C "${PROJECT_ROOT}" rev-parse HEAD 2>/dev/null)"; then
  GIT_DIRTY_ENTRIES="$(git -C "${PROJECT_ROOT}" status --porcelain | wc -l | tr -d ' ')"
else
  GIT_COMMIT="unavailable"
  GIT_DIRTY_ENTRIES="unavailable"
fi

cp "${CONFIG_FILE}" "${OUTPUT_DIR}/config_source.yml"
cp "${PREFLIGHT_RESOLVED}" "${OUTPUT_DIR}/resolved_config.yml"
if [[ -n "${PROVENANCE_FILE}" ]]; then
  cp "${PROVENANCE_FILE}" "${OUTPUT_DIR}/method_provenance.yml"
fi
if [[ -n "${RUNTIME_FILES_LIST}" ]]; then
  cp "${RUNTIME_FILES_LIST}" "${OUTPUT_DIR}/runtime_files.txt"
  (
    cd "${PROJECT_ROOT}"
    while IFS= read -r listed_file || [[ -n "${listed_file}" ]]; do
      listed_file="${listed_file%$'\r'}"
      [[ -z "${listed_file}" || "${listed_file}" == \#* ]] && continue
      sha256sum "${listed_file}"
    done < "${RUNTIME_FILES_LIST}"
  ) > "${OUTPUT_DIR}/runtime_sha256.txt"
fi

{
  echo "created_at=$(date -Is)"
  echo "hostname=$(hostname)"
  echo "run_id=${RUN_ID}"
  echo "run_mode=${RUN_MODE}"
  echo "git_commit=${GIT_COMMIT}"
  echo "git_dirty_entries=${GIT_DIRTY_ENTRIES}"
  echo "launcher=${LAUNCHER_PATH}"
  echo "launcher_sha256=$(sha256sum "${LAUNCHER_PATH}" | awk '{print $1}')"
  echo "config=${CONFIG_FILE}"
  echo "config_sha256=${CONFIG_SHA256}"
  echo "output_dir=${OUTPUT_DIR}"
  echo "data_root=${DATA_ROOT}"
  echo "train_annotation=${TRAIN_ANN}"
  echo "train_annotation_sha256=${TRAIN_ANN_SHA256}"
  echo "val_annotation=${VAL_ANN}"
  echo "val_annotation_sha256=${VAL_ANN_SHA256}"
  echo "source_checkpoint=${SOURCE_CHECKPOINT}"
  echo "source_checkpoint_sha256=${SOURCE_CHECKPOINT_SHA256}"
  echo "source_checkpoint_last_epoch=${CHECKPOINT_LAST_EPOCH}"
  echo "gpus=${GPUS}"
  echo "nproc_per_node=${NPROC_PER_NODE}"
  echo "batch_size_per_gpu=${BATCH_SIZE_PER_GPU}"
  echo "global_batch=${TOTAL_BATCH_SIZE}"
  echo "num_workers=${NUM_WORKERS}"
  echo "master_port=${MASTER_PORT}"
  echo "epochs=${EPOCHS}"
  echo "num_classes=${NUM_CLASSES}"
  echo "num_queries=${NUM_QUERIES}"
  echo "num_top_queries=${NUM_TOP_QUERIES}"
  echo "seed=${SEED}"
  echo "quick_steps=${QUICK_STEPS}"
  echo "use_amp=${USE_AMP}"
  echo "provenance_file=${PROVENANCE_FILE:-none}"
  echo "runtime_files_list=${RUNTIME_FILES_LIST:-none}"
} | tee "${OUTPUT_DIR}/run_manifest.txt"

export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

cd "${PROJECT_ROOT}"
CUDA_VISIBLE_DEVICES="${GPUS}" "${PYTHON_BIN}" -m torch.distributed.run \
  --master_port="${MASTER_PORT}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  train.py -c "${CONFIG_FILE}" \
  "${AMP_ARGS[@]}" --seed="${SEED}" --output-dir "${OUTPUT_DIR}" \
  "${CHECKPOINT_ARGS[@]}" -u "${CONFIG_UPDATES[@]}" \
  2>&1 | tee "${OUTPUT_DIR}/console.log"
