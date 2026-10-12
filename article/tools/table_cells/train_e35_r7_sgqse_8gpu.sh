#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

EXPERIMENT="${EXPERIMENT:-${1:-}}"
SEED="${SEED:-0}"
DATA_ROOT="${DATA_ROOT:-/aicamera-mlp/chenkai/data/table_det_data/paddlex_l_cell_merged}"
PRETRAINED="${PRETRAINED:-${PROJECT_ROOT}/deim_dfine_hgnetv2_m_coco_90e.pth}"
ORIGINAL_CHECKPOINT="${RESUME:-${PROJECT_ROOT}/output/dfine_hgnetv2_m_table_cell/checkpoint0027.pth}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
MASTER_PORT="${MASTER_PORT:-29528}"
NUM_WORKERS="${NUM_WORKERS:-2}"
RUNTIME_ENV="${RUNTIME_ENV:-/aicamera-mlp/chenkai/envs/deim}"
ALLOW_EXISTING="${ALLOW_EXISTING:-0}"
ALLOW_CONTROL_RERUN="${ALLOW_CONTROL_RERUN:-0}"
VERIFY_FROZEN_INPUTS="${VERIFY_FROZEN_INPUTS:-1}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
METHOD_PROVENANCE="${PROJECT_ROOT}/configs/e35_r7/method_provenance.yml"
CONTRACT_VALIDATOR="${PROJECT_ROOT}/tools/table_cells/verify_e35_r7_contract.py"

FROZEN_TRAIN_ANN_SHA256="816859c555565c1baed0fa4b99e2f7d502ec3bc9582f34601770101dab6d0e4e"
FROZEN_VAL_ANN_SHA256="a6d107322285595d968f63b4cd6d9ccf2442ac844a6875685a2aa16f3b9dd72b"
FROZEN_RESUME_SHA256="3eba494ac31b049d266af7fc837f72ad79c91830fc47b27f7e611af552094ede"
FROZEN_PRETRAINED_SHA256="2b6cd0582a4aa711f583982057b7fb0f3daebdd98e4dc168824714014c3219bc"

die() { echo "错误: $*" >&2; exit 2; }

case "${EXPERIMENT}" in
  deim_core_e35)
    MODE="fresh_tuning"
    CONFIG_REL="configs/e35_r7/deim_core_e35.yml"
    PLANNED_EPOCHS=35
    RUN_RESUME=""
    ;;
  resume_b0_baseline)
    [[ "${ALLOW_CONTROL_RERUN}" == "1" || "${PREFLIGHT_ONLY}" == "1" ]] || \
      die "B0 baseline 已经在运行；确需重跑时显式设置 ALLOW_CONTROL_RERUN=1"
    MODE="resume_epoch27"
    CONFIG_REL="configs/e35_r7/resume_b0_baseline.yml"
    PLANNED_EPOCHS=7
    RUN_RESUME="${ORIGINAL_CHECKPOINT}"
    ;;
  resume_b1_sgqs)
    MODE="resume_epoch27"
    CONFIG_REL="configs/e35_r7/resume_b1_sgqs.yml"
    PLANNED_EPOCHS=7
    RUN_RESUME="${ORIGINAL_CHECKPOINT}"
    ;;
  resume_b2_efgl)
    MODE="resume_epoch27"
    CONFIG_REL="configs/e35_r7/resume_b2_efgl.yml"
    PLANNED_EPOCHS=7
    RUN_RESUME="${ORIGINAL_CHECKPOINT}"
    ;;
  resume_b3_sgqs_e)
    MODE="resume_epoch27"
    CONFIG_REL="configs/e35_r7/resume_b3_sgqs_e.yml"
    PLANNED_EPOCHS=7
    RUN_RESUME="${ORIGINAL_CHECKPOINT}"
    ;;
  *)
    die "EXPERIMENT 必须是 deim_core_e35、resume_b0_baseline、resume_b1_sgqs、resume_b2_efgl 或 resume_b3_sgqs_e"
    ;;
esac

[[ "${SEED}" =~ ^[0-9]+$ ]] || die "SEED 必须为非负整数"
[[ "${VERIFY_FROZEN_INPUTS}" =~ ^[01]$ ]] || die "VERIFY_FROZEN_INPUTS 必须是 0 或 1"
[[ "${PREFLIGHT_ONLY}" =~ ^[01]$ ]] || die "PREFLIGHT_ONLY 必须是 0 或 1"
GPU_COUNT="$(awk -F, '{print NF}' <<< "${GPUS}")"
[[ "${GPU_COUNT}" -eq 8 ]] || die "固定使用 8 张 GPU；当前 GPUS=${GPUS}"

CONFIG_FILE="${PROJECT_ROOT}/${CONFIG_REL}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/output/e35_r7_sgqse/${EXPERIMENT}_seed${SEED}}"
BASE_LAUNCHER="${PROJECT_ROOT}/tools/table_cells/train_dfine_m_table_cell_multigpu.sh"
TARGET_EPOCHS=35

[[ -f "${CONFIG_FILE}" ]] || die "缺少配置文件: ${CONFIG_FILE}"
[[ -f "${BASE_LAUNCHER}" ]] || die "缺少基础启动脚本: ${BASE_LAUNCHER}"
[[ -f "${METHOD_PROVENANCE}" ]] || die "缺少方法来源清单: ${METHOD_PROVENANCE}"
[[ -f "${CONTRACT_VALIDATOR}" ]] || die "缺少实验契约校验器: ${CONTRACT_VALIDATOR}"
[[ -x "${RUNTIME_ENV}/bin/python" ]] || die "找不到训练环境 Python: ${RUNTIME_ENV}/bin/python"

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

if [[ "${MODE}" == "fresh_tuning" ]]; then
  [[ -f "${PRETRAINED}" ]] || die "缺少统一 COCO 预训练权重: ${PRETRAINED}"
  SOURCE_CHECKPOINT="${PRETRAINED}"
  CHECKPOINT_LAST_EPOCH="not_applicable"
else
  [[ -f "${RUN_RESUME}" ]] || die "缺少 epoch-27 断点: ${RUN_RESUME}"
  [[ "$(basename -- "${RUN_RESUME}")" == "checkpoint0027.pth" ]] || \
    die "短续训必须从 checkpoint0027.pth 开始；当前为 ${RUN_RESUME}"
  CHECKPOINT_LAST_EPOCH="$("${RUNTIME_ENV}/bin/python" - "${RUN_RESUME}" <<'PY'
import sys
import torch

path = sys.argv[1]
try:
    state = torch.load(path, map_location='cpu', weights_only=False)
except TypeError:
    state = torch.load(path, map_location='cpu')
print(state.get('last_epoch', 'missing'))
PY
)"
  [[ "${CHECKPOINT_LAST_EPOCH}" == "27" ]] || \
    die "断点 last_epoch 必须为 27；实际为 ${CHECKPOINT_LAST_EPOCH}"
  SOURCE_CHECKPOINT="${RUN_RESUME}"
fi

SOURCE_CHECKPOINT_SHA256="$(sha256sum "${SOURCE_CHECKPOINT}" | awk '{print $1}')"
TRAIN_ANN_SHA256="$(sha256sum "${TRAIN_ANN}" | awk '{print $1}')"
VAL_ANN_SHA256="$(sha256sum "${VAL_ANN}" | awk '{print $1}')"
if [[ "${VERIFY_FROZEN_INPUTS}" == "1" ]]; then
  EXPECTED_SOURCE_SHA256="${FROZEN_PRETRAINED_SHA256}"
  [[ "${MODE}" == "fresh_tuning" ]] || EXPECTED_SOURCE_SHA256="${FROZEN_RESUME_SHA256}"
  [[ "${SOURCE_CHECKPOINT_SHA256}" == "${EXPECTED_SOURCE_SHA256}" ]] || \
    die "初始权重 SHA-256 与冻结身份不一致"
  [[ "${TRAIN_ANN_SHA256}" == "${FROZEN_TRAIN_ANN_SHA256}" ]] || \
    die "训练标注 SHA-256 与冻结身份不一致"
  [[ "${VAL_ANN_SHA256}" == "${FROZEN_VAL_ANN_SHA256}" ]] || \
    die "验证标注 SHA-256 与冻结身份不一致"
fi

PREFLIGHT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/deim-e35-r7-preflight.XXXXXX")"
trap 'rm -rf -- "${PREFLIGHT_DIR}"' EXIT
PREFLIGHT_RESOLVED="${PREFLIGHT_DIR}/resolved_config.yml"
RESOLVE_UPDATES=(
  "output_dir=${OUTPUT_DIR}"
  "use_amp=True"
  "seed=${SEED}"
  "num_classes=1"
  "epoches=${TARGET_EPOCHS}"
  "HGNetv2.pretrained=False"
  "DFINETransformer.num_queries=640"
  "PostProcessor.num_top_queries=640"
  "train_dataloader.dataset.img_folder=${TRAIN_IMAGES}"
  "train_dataloader.dataset.ann_file=${TRAIN_ANN}"
  "val_dataloader.dataset.img_folder=${VAL_IMAGES}"
  "val_dataloader.dataset.ann_file=${VAL_ANN}"
  "train_dataloader.total_batch_size=64"
  "val_dataloader.total_batch_size=64"
  "train_dataloader.num_workers=${NUM_WORKERS}"
  "val_dataloader.num_workers=${NUM_WORKERS}"
)
if [[ "${MODE}" == "fresh_tuning" ]]; then
  RESOLVE_UPDATES+=("tuning=${PRETRAINED}")
else
  RESOLVE_UPDATES+=("resume=${RUN_RESUME}")
fi
"${RUNTIME_ENV}/bin/python" "${PROJECT_ROOT}/tools/table_cells/dump_resolved_training_config.py" \
  -c "${CONFIG_FILE}" \
  -o "${PREFLIGHT_RESOLVED}" \
  -u "${RESOLVE_UPDATES[@]}"
"${RUNTIME_ENV}/bin/python" "${CONTRACT_VALIDATOR}" \
  --project-root "${PROJECT_ROOT}" \
  --provenance "${METHOD_PROVENANCE}" \
  --resolved-config "${PREFLIGHT_RESOLVED}" \
  --experiment "${EXPERIMENT}"

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
  echo "experiment=${EXPERIMENT}"
  echo "mode=${MODE}"
  echo "seed=${SEED}"
  echo "git_commit=${GIT_COMMIT}"
  echo "git_dirty_entries=${GIT_DIRTY}"
  echo "config=${CONFIG_FILE}"
  echo "config_sha256=$(sha256sum "${CONFIG_FILE}" | awk '{print $1}')"
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
  echo "contract_preflight=passed"
  echo "launcher_sha256=$(sha256sum "${BASH_SOURCE[0]}" | awk '{print $1}')"
  echo "criterion_sha256=$(sha256sum "${PROJECT_ROOT}/engine/deim/deim_criterion.py" | awk '{print $1}')"
  echo "decoder_sha256=$(sha256sum "${PROJECT_ROOT}/engine/deim/dfine_decoder.py" | awk '{print $1}')"
  echo "gpus=${GPUS}"
  echo "batch_size_per_gpu=8"
  echo "global_batch=64"
  echo "target_epoches=${TARGET_EPOCHS}"
  echo "planned_training_epochs=${PLANNED_EPOCHS}"
  echo "input_size=640"
  echo "num_queries=640"
  echo "evaluator_max_dets=300"
  echo "quick_steps=0"
} | tee "${OUTPUT_DIR}/run_manifest.txt"

cp "${CONFIG_FILE}" "${OUTPUT_DIR}/config_source.yml"
cp "${METHOD_PROVENANCE}" "${OUTPUT_DIR}/method_provenance.yml"
cp "${PREFLIGHT_RESOLVED}" "${OUTPUT_DIR}/resolved_config.yml"

CONFIG_FILE="${CONFIG_FILE}" \
PRETRAINED="${PRETRAINED}" \
RESUME="${RUN_RESUME}" \
QUICK_STEPS=0 \
QUICK_OUTPUT_DIR="" \
OUTPUT_DIR="${OUTPUT_DIR}" \
DATA_ROOT="${DATA_ROOT}" \
GPUS="${GPUS}" \
NPROC_PER_NODE=8 \
BATCH_SIZE_PER_GPU=8 \
NUM_WORKERS="${NUM_WORKERS}" \
EPOCHS="${TARGET_EPOCHS}" \
NUM_QUERIES=640 \
SEED="${SEED}" \
MASTER_PORT="${MASTER_PORT}" \
RUNTIME_ENV="${RUNTIME_ENV}" \
SAVE_RESOLVED_CONFIG=0 \
bash "${BASE_LAUNCHER}" 2>&1 | tee "${OUTPUT_DIR}/console.log"
