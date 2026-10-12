#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

# Transparent, fail-closed launcher for the fixed-640 e35/r7 matrix.
# It deliberately contains no experiment-to-config lookup table: every
# scientific and runtime parameter must be visible in the calling command.

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

for name in \
  RUN_ID CONTRACT_ID CONFIG_FILE OUTPUT_DIR DATA_ROOT RUN_MODE \
  GPUS NPROC_PER_NODE BATCH_SIZE_PER_GPU NUM_WORKERS MASTER_PORT \
  EPOCHS NUM_QUERIES SEED QUICK_STEPS RUNTIME_ENV \
  METHOD_PROVENANCE FROZEN_INPUTS RUNTIME_MANIFEST VERIFY_FROZEN_INPUTS \
  PREFLIGHT_ONLY ALLOW_EXISTING; do
  require_env "${name}"
done

CONFIG_FILE="$(resolve_project_path "${CONFIG_FILE}")"
OUTPUT_DIR="$(resolve_project_path "${OUTPUT_DIR}")"
METHOD_PROVENANCE="$(resolve_project_path "${METHOD_PROVENANCE}")"
FROZEN_INPUTS="$(resolve_project_path "${FROZEN_INPUTS}")"
RUNTIME_MANIFEST="$(resolve_project_path "${RUNTIME_MANIFEST}")"
BASE_LAUNCHER="${PROJECT_ROOT}/tools/table_cells/train_dfine_m_table_cell_multigpu.sh"
CONTRACT_VALIDATOR="${PROJECT_ROOT}/tools/table_cells/verify_e35_r7_contract.py"
PYTHON_BIN="${RUNTIME_ENV}/bin/python"

[[ "${RUN_ID}" =~ ^[a-z0-9][a-z0-9_-]*$ ]] || die "RUN_ID 只能包含小写字母、数字、_ 和 -"
[[ "${RUN_MODE}" == "resume" || "${RUN_MODE}" == "tuning" ]] || die "RUN_MODE 必须是 resume 或 tuning"
for value_name in NPROC_PER_NODE BATCH_SIZE_PER_GPU NUM_WORKERS MASTER_PORT EPOCHS NUM_QUERIES SEED QUICK_STEPS; do
  value="${!value_name}"
  [[ "${value}" =~ ^[0-9]+$ ]] || die "${value_name} 必须为非负整数"
done
[[ "${NPROC_PER_NODE}" -gt 0 ]] || die "NPROC_PER_NODE 必须大于 0"
[[ "${BATCH_SIZE_PER_GPU}" -gt 0 ]] || die "BATCH_SIZE_PER_GPU 必须大于 0"
[[ "${VERIFY_FROZEN_INPUTS}" =~ ^[01]$ ]] || die "VERIFY_FROZEN_INPUTS 必须是 0 或 1"
[[ "${PREFLIGHT_ONLY}" =~ ^[01]$ ]] || die "PREFLIGHT_ONLY 必须是 0 或 1"
[[ "${ALLOW_EXISTING}" =~ ^[01]$ ]] || die "ALLOW_EXISTING 必须是 0 或 1"

GPU_COUNT="$(awk -F, '{print NF}' <<< "${GPUS}")"
[[ "${GPU_COUNT}" -eq "${NPROC_PER_NODE}" ]] || \
  die "GPUS 数量 ${GPU_COUNT} 与 NPROC_PER_NODE=${NPROC_PER_NODE} 不一致"

for required_file in "${CONFIG_FILE}" "${METHOD_PROVENANCE}" "${FROZEN_INPUTS}" "${RUNTIME_MANIFEST}" \
  "${BASE_LAUNCHER}" "${CONTRACT_VALIDATOR}"; do
  [[ -f "${required_file}" ]] || die "缺少文件: ${required_file}"
done
[[ -x "${PYTHON_BIN}" ]] || die "找不到训练环境 Python: ${PYTHON_BIN}"

(cd "${PROJECT_ROOT}" && sha256sum -c "${RUNTIME_MANIFEST}")

if [[ -d "${DATA_ROOT}/images" && -f "${DATA_ROOT}/annotations/instance_train.json" ]]; then
  TRAIN_IMAGES="${DATA_ROOT}/images"
  VAL_IMAGES="${DATA_ROOT}/images"
  TRAIN_ANN="${DATA_ROOT}/annotations/instance_train.json"
  VAL_ANN="${DATA_ROOT}/annotations/instance_val.json"
else
  TRAIN_IMAGES="${DATA_ROOT}/train"
  VAL_IMAGES="${DATA_ROOT}/val"
  TRAIN_ANN="${DATA_ROOT}/annotations/instances_train.json"
  VAL_ANN="${DATA_ROOT}/annotations/instances_val.json"
fi
[[ -d "${TRAIN_IMAGES}" ]] || die "缺少训练图片目录: ${TRAIN_IMAGES}"
[[ -d "${VAL_IMAGES}" ]] || die "缺少验证图片目录: ${VAL_IMAGES}"
[[ -f "${TRAIN_ANN}" ]] || die "缺少训练标注: ${TRAIN_ANN}"
[[ -f "${VAL_ANN}" ]] || die "缺少验证标注: ${VAL_ANN}"

if [[ "${RUN_MODE}" == "resume" ]]; then
  require_env RESUME
  [[ -z "${PRETRAINED:-}" ]] || die "resume 模式不得同时设置 PRETRAINED"
  SOURCE_CHECKPOINT="${RESUME}"
  CHECKPOINT_LAST_EPOCH="$("${PYTHON_BIN}" - "${SOURCE_CHECKPOINT}" <<'PY'
import sys
import torch

try:
    state = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
except TypeError:
    state = torch.load(sys.argv[1], map_location='cpu')
print(state.get('last_epoch', 'missing'))
PY
)"
else
  require_env PRETRAINED
  [[ -z "${RESUME:-}" ]] || die "tuning 模式不得同时设置 RESUME"
  SOURCE_CHECKPOINT="${PRETRAINED}"
  CHECKPOINT_LAST_EPOCH="not_applicable"
fi
[[ -f "${SOURCE_CHECKPOINT}" ]] || die "缺少初始权重: ${SOURCE_CHECKPOINT}"

SOURCE_CHECKPOINT_SHA256="$(sha256sum "${SOURCE_CHECKPOINT}" | awk '{print $1}')"
TRAIN_ANN_SHA256="$(sha256sum "${TRAIN_ANN}" | awk '{print $1}')"
VAL_ANN_SHA256="$(sha256sum "${VAL_ANN}" | awk '{print $1}')"

PREFLIGHT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/deim-e35-r7-preflight.XXXXXX")"
trap 'rm -rf -- "${PREFLIGHT_DIR}"' EXIT
PREFLIGHT_RESOLVED="${PREFLIGHT_DIR}/resolved_config.yml"

RESOLVE_UPDATES=(
  "output_dir=${OUTPUT_DIR}"
  "use_amp=True"
  "seed=${SEED}"
  "num_classes=1"
  "epoches=${EPOCHS}"
  "HGNetv2.pretrained=False"
  "DFINETransformer.num_queries=${NUM_QUERIES}"
  "PostProcessor.num_top_queries=${NUM_QUERIES}"
  "train_dataloader.dataset.img_folder=${TRAIN_IMAGES}"
  "train_dataloader.dataset.ann_file=${TRAIN_ANN}"
  "val_dataloader.dataset.img_folder=${VAL_IMAGES}"
  "val_dataloader.dataset.ann_file=${VAL_ANN}"
  "train_dataloader.total_batch_size=$((NPROC_PER_NODE * BATCH_SIZE_PER_GPU))"
  "val_dataloader.total_batch_size=$((NPROC_PER_NODE * BATCH_SIZE_PER_GPU))"
  "train_dataloader.num_workers=${NUM_WORKERS}"
  "val_dataloader.num_workers=${NUM_WORKERS}"
)
if [[ "${RUN_MODE}" == "resume" ]]; then
  RESOLVE_UPDATES+=("resume=${SOURCE_CHECKPOINT}")
else
  RESOLVE_UPDATES+=("tuning=${SOURCE_CHECKPOINT}")
fi

"${PYTHON_BIN}" "${PROJECT_ROOT}/tools/table_cells/dump_resolved_training_config.py" \
  -c "${CONFIG_FILE}" \
  -o "${PREFLIGHT_RESOLVED}" \
  -u "${RESOLVE_UPDATES[@]}"

"${PYTHON_BIN}" "${CONTRACT_VALIDATOR}" \
  --project-root "${PROJECT_ROOT}" \
  --provenance "${METHOD_PROVENANCE}" \
  --frozen-inputs "${FROZEN_INPUTS}" \
  --resolved-config "${PREFLIGHT_RESOLVED}" \
  --config-source "${CONFIG_FILE}" \
  --experiment "${CONTRACT_ID}" \
  --run-mode "${RUN_MODE}" \
  --world-size "${NPROC_PER_NODE}" \
  --batch-size-per-gpu "${BATCH_SIZE_PER_GPU}" \
  --num-workers "${NUM_WORKERS}" \
  --quick-steps "${QUICK_STEPS}" \
  --source-checkpoint "${SOURCE_CHECKPOINT}" \
  --source-checkpoint-sha256 "${SOURCE_CHECKPOINT_SHA256}" \
  --source-checkpoint-last-epoch "${CHECKPOINT_LAST_EPOCH}" \
  --train-ann "${TRAIN_ANN}" \
  --train-ann-sha256 "${TRAIN_ANN_SHA256}" \
  --val-ann "${VAL_ANN}" \
  --val-ann-sha256 "${VAL_ANN_SHA256}" \
  --verify-frozen-inputs "${VERIFY_FROZEN_INPUTS}"

if [[ "${PREFLIGHT_ONLY}" == "1" ]]; then
  echo "Preflight 完成：未创建输出目录，未启动 torchrun。"
  exit 0
fi

if [[ -d "${OUTPUT_DIR}" && -n "$(find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" && "${ALLOW_EXISTING}" != "1" ]]; then
  die "输出目录非空，为防止混合实验已停止: ${OUTPUT_DIR}"
fi
mkdir -p "${OUTPUT_DIR}"

if GIT_COMMIT="$(git -C "${PROJECT_ROOT}" rev-parse HEAD 2>/dev/null)"; then
  GIT_DIRTY="$(git -C "${PROJECT_ROOT}" status --porcelain | wc -l | tr -d ' ')"
else
  GIT_COMMIT="unavailable"
  GIT_DIRTY="unavailable"
fi

{
  echo "created_at=$(date -Is)"
  echo "hostname=$(hostname)"
  echo "run_id=${RUN_ID}"
  echo "contract_id=${CONTRACT_ID}"
  echo "run_mode=${RUN_MODE}"
  echo "git_commit=${GIT_COMMIT}"
  echo "git_dirty_entries=${GIT_DIRTY}"
  echo "config=${CONFIG_FILE}"
  echo "config_sha256=$(sha256sum "${CONFIG_FILE}" | awk '{print $1}')"
  echo "output_dir=${OUTPUT_DIR}"
  echo "source_checkpoint=${SOURCE_CHECKPOINT}"
  echo "source_checkpoint_sha256=${SOURCE_CHECKPOINT_SHA256}"
  echo "source_checkpoint_last_epoch=${CHECKPOINT_LAST_EPOCH}"
  echo "train_annotation=${TRAIN_ANN}"
  echo "train_annotation_sha256=${TRAIN_ANN_SHA256}"
  echo "val_annotation=${VAL_ANN}"
  echo "val_annotation_sha256=${VAL_ANN_SHA256}"
  echo "verify_frozen_inputs=${VERIFY_FROZEN_INPUTS}"
  echo "method_reference_commit=6fd2c301bd8c09ff3d49fa077c34eaf75ef35b9a"
  echo "method_provenance=${METHOD_PROVENANCE}"
  echo "method_provenance_sha256=$(sha256sum "${METHOD_PROVENANCE}" | awk '{print $1}')"
  echo "runtime_manifest=${RUNTIME_MANIFEST}"
  echo "runtime_manifest_sha256=$(sha256sum "${RUNTIME_MANIFEST}" | awk '{print $1}')"
  echo "contract_preflight=passed"
  echo "launcher=${BASH_SOURCE[0]}"
  echo "launcher_sha256=$(sha256sum "${BASH_SOURCE[0]}" | awk '{print $1}')"
  echo "gpus=${GPUS}"
  echo "nproc_per_node=${NPROC_PER_NODE}"
  echo "batch_size_per_gpu=${BATCH_SIZE_PER_GPU}"
  echo "global_batch=$((NPROC_PER_NODE * BATCH_SIZE_PER_GPU))"
  echo "num_workers=${NUM_WORKERS}"
  echo "master_port=${MASTER_PORT}"
  echo "target_epoches=${EPOCHS}"
  echo "input_size=640"
  echo "num_queries=${NUM_QUERIES}"
  echo "seed=${SEED}"
  echo "evaluator_max_dets=300"
  echo "quick_steps=${QUICK_STEPS}"
} | tee "${OUTPUT_DIR}/run_manifest.txt"

cp "${CONFIG_FILE}" "${OUTPUT_DIR}/config_source.yml"
cp "${METHOD_PROVENANCE}" "${OUTPUT_DIR}/method_provenance.yml"
cp "${FROZEN_INPUTS}" "${OUTPUT_DIR}/frozen_inputs.yml"
cp "${RUNTIME_MANIFEST}" "${OUTPUT_DIR}/runtime_sha256.txt"
cp "${PREFLIGHT_RESOLVED}" "${OUTPUT_DIR}/resolved_config.yml"

CONFIG_FILE="${CONFIG_FILE}" \
PRETRAINED="${PRETRAINED:-}" \
RESUME="${RESUME:-}" \
QUICK_STEPS="${QUICK_STEPS}" \
QUICK_OUTPUT_DIR="" \
OUTPUT_DIR="${OUTPUT_DIR}" \
DATA_ROOT="${DATA_ROOT}" \
GPUS="${GPUS}" \
NPROC_PER_NODE="${NPROC_PER_NODE}" \
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU}" \
NUM_WORKERS="${NUM_WORKERS}" \
EPOCHS="${EPOCHS}" \
NUM_QUERIES="${NUM_QUERIES}" \
SEED="${SEED}" \
MASTER_PORT="${MASTER_PORT}" \
RUNTIME_ENV="${RUNTIME_ENV}" \
SAVE_RESOLVED_CONFIG=0 \
bash "${BASE_LAUNCHER}" 2>&1 | tee "${OUTPUT_DIR}/console.log"
