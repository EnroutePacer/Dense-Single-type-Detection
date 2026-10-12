# e35/r7 server launch commands

Run each block on a separate eight-GPU worker. Every scientific/runtime
parameter is explicit; `run_e35_r7_experiment.sh` contains no experiment-to-
config dispatch table. `PREFLIGHT_ONLY=0` performs the same fail-closed
preflight and then trains. Change it to `1` for a no-output, no-training check.

The completed `resume_b0_baseline` is intentionally absent. Do not extract
`targeted_e28_r7_training_bundle.tar.gz` over the active server tree.

## Controlled DEIM-core, epoch 0--34

```bash
cd /aicamera-mlp/zhuyouwei/code/DEIM || exit 1

RUN_ID=deim_core_e35_seed0 \
CONTRACT_ID=deim_core_e35 \
CONFIG_FILE="$PWD/configs/e35_r7/deim_core_e35.yml" \
OUTPUT_DIR="$PWD/output/e35_r7_sgqse/deim_core_e35_seed0" \
DATA_ROOT=/aicamera-mlp/chenkai/data/table_det_data/paddlex_l_cell_merged \
RUN_MODE=tuning \
PRETRAINED="$PWD/deim_dfine_hgnetv2_m_coco_90e.pth" \
RESUME= \
GPUS=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
BATCH_SIZE_PER_GPU=8 \
NUM_WORKERS=4 \
MASTER_PORT=29531 \
EPOCHS=35 \
NUM_QUERIES=640 \
SEED=0 \
QUICK_STEPS=0 \
RUNTIME_ENV=/aicamera-mlp/chenkai/envs/deim \
METHOD_PROVENANCE="$PWD/configs/e35_r7/method_provenance.yml" \
FROZEN_INPUTS="$PWD/configs/e35_r7/frozen_inputs.yml" \
RUNTIME_MANIFEST="$PWD/tools/table_cells/e35_r7_runtime_sha256.txt" \
VERIFY_FROZEN_INPUTS=1 \
PREFLIGHT_ONLY=0 \
ALLOW_EXISTING=0 \
bash tools/table_cells/run_e35_r7_experiment.sh
```

## B1: SGQS only, resume epoch 28--34

```bash
cd /aicamera-mlp/zhuyouwei/code/DEIM || exit 1

RUN_ID=resume_b1_sgqs_seed0 \
CONTRACT_ID=resume_b1_sgqs \
CONFIG_FILE="$PWD/configs/e35_r7/resume_b1_sgqs.yml" \
OUTPUT_DIR="$PWD/output/e35_r7_sgqse/resume_b1_sgqs_seed0" \
DATA_ROOT=/aicamera-mlp/chenkai/data/table_det_data/paddlex_l_cell_merged \
RUN_MODE=resume \
PRETRAINED= \
RESUME="$PWD/output/dfine_hgnetv2_m_table_cell/checkpoint0027.pth" \
GPUS=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
BATCH_SIZE_PER_GPU=8 \
NUM_WORKERS=4 \
MASTER_PORT=29532 \
EPOCHS=35 \
NUM_QUERIES=640 \
SEED=0 \
QUICK_STEPS=0 \
RUNTIME_ENV=/aicamera-mlp/chenkai/envs/deim \
METHOD_PROVENANCE="$PWD/configs/e35_r7/method_provenance.yml" \
FROZEN_INPUTS="$PWD/configs/e35_r7/frozen_inputs.yml" \
RUNTIME_MANIFEST="$PWD/tools/table_cells/e35_r7_runtime_sha256.txt" \
VERIFY_FROZEN_INPUTS=1 \
PREFLIGHT_ONLY=0 \
ALLOW_EXISTING=0 \
bash tools/table_cells/run_e35_r7_experiment.sh
```

## B2: EFGL only, resume epoch 28--34

```bash
cd /aicamera-mlp/zhuyouwei/code/DEIM || exit 1

RUN_ID=resume_b2_efgl_seed0 \
CONTRACT_ID=resume_b2_efgl \
CONFIG_FILE="$PWD/configs/e35_r7/resume_b2_efgl.yml" \
OUTPUT_DIR="$PWD/output/e35_r7_sgqse/resume_b2_efgl_seed0" \
DATA_ROOT=/aicamera-mlp/chenkai/data/table_det_data/paddlex_l_cell_merged \
RUN_MODE=resume \
PRETRAINED= \
RESUME="$PWD/output/dfine_hgnetv2_m_table_cell/checkpoint0027.pth" \
GPUS=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
BATCH_SIZE_PER_GPU=8 \
NUM_WORKERS=4 \
MASTER_PORT=29533 \
EPOCHS=35 \
NUM_QUERIES=640 \
SEED=0 \
QUICK_STEPS=0 \
RUNTIME_ENV=/aicamera-mlp/chenkai/envs/deim \
METHOD_PROVENANCE="$PWD/configs/e35_r7/method_provenance.yml" \
FROZEN_INPUTS="$PWD/configs/e35_r7/frozen_inputs.yml" \
RUNTIME_MANIFEST="$PWD/tools/table_cells/e35_r7_runtime_sha256.txt" \
VERIFY_FROZEN_INPUTS=1 \
PREFLIGHT_ONLY=0 \
ALLOW_EXISTING=0 \
bash tools/table_cells/run_e35_r7_experiment.sh
```

## B3: SGQS + EFGL, resume epoch 28--34

```bash
cd /aicamera-mlp/zhuyouwei/code/DEIM || exit 1

RUN_ID=resume_b3_sgqs_e_seed0 \
CONTRACT_ID=resume_b3_sgqs_e \
CONFIG_FILE="$PWD/configs/e35_r7/resume_b3_sgqs_e.yml" \
OUTPUT_DIR="$PWD/output/e35_r7_sgqse/resume_b3_sgqs_e_seed0" \
DATA_ROOT=/aicamera-mlp/chenkai/data/table_det_data/paddlex_l_cell_merged \
RUN_MODE=resume \
PRETRAINED= \
RESUME="$PWD/output/dfine_hgnetv2_m_table_cell/checkpoint0027.pth" \
GPUS=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
BATCH_SIZE_PER_GPU=8 \
NUM_WORKERS=4 \
MASTER_PORT=29534 \
EPOCHS=35 \
NUM_QUERIES=640 \
SEED=0 \
QUICK_STEPS=0 \
RUNTIME_ENV=/aicamera-mlp/chenkai/envs/deim \
METHOD_PROVENANCE="$PWD/configs/e35_r7/method_provenance.yml" \
FROZEN_INPUTS="$PWD/configs/e35_r7/frozen_inputs.yml" \
RUNTIME_MANIFEST="$PWD/tools/table_cells/e35_r7_runtime_sha256.txt" \
VERIFY_FROZEN_INPUTS=1 \
PREFLIGHT_ONLY=0 \
ALLOW_EXISTING=0 \
bash tools/table_cells/run_e35_r7_experiment.sh
```

## C1: historical small-reg, resume epoch 28--34

```bash
cd /aicamera-mlp/zhuyouwei/code/DEIM || exit 1

RUN_ID=resume_c1_small_reg_seed0 \
CONTRACT_ID=resume_c1_small_reg \
CONFIG_FILE="$PWD/configs/e35_r7/resume_c1_small_reg.yml" \
OUTPUT_DIR="$PWD/output/e35_r7_sgqse/resume_c1_small_reg_seed0" \
DATA_ROOT=/aicamera-mlp/chenkai/data/table_det_data/paddlex_l_cell_merged \
RUN_MODE=resume \
PRETRAINED= \
RESUME="$PWD/output/dfine_hgnetv2_m_table_cell/checkpoint0027.pth" \
GPUS=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
BATCH_SIZE_PER_GPU=8 \
NUM_WORKERS=4 \
MASTER_PORT=29535 \
EPOCHS=35 \
NUM_QUERIES=640 \
SEED=0 \
QUICK_STEPS=0 \
RUNTIME_ENV=/aicamera-mlp/chenkai/envs/deim \
METHOD_PROVENANCE="$PWD/configs/e35_r7/method_provenance.yml" \
FROZEN_INPUTS="$PWD/configs/e35_r7/frozen_inputs.yml" \
RUNTIME_MANIFEST="$PWD/tools/table_cells/e35_r7_runtime_sha256.txt" \
VERIFY_FROZEN_INPUTS=1 \
PREFLIGHT_ONLY=0 \
ALLOW_EXISTING=0 \
bash tools/table_cells/run_e35_r7_experiment.sh
```

## C2: MAL only, resume epoch 28--34

```bash
cd /aicamera-mlp/zhuyouwei/code/DEIM || exit 1

RUN_ID=resume_c2_mal_only_seed0 \
CONTRACT_ID=resume_c2_mal_only \
CONFIG_FILE="$PWD/configs/e35_r7/resume_c2_mal_only.yml" \
OUTPUT_DIR="$PWD/output/e35_r7_sgqse/resume_c2_mal_only_seed0" \
DATA_ROOT=/aicamera-mlp/chenkai/data/table_det_data/paddlex_l_cell_merged \
RUN_MODE=resume \
PRETRAINED= \
RESUME="$PWD/output/dfine_hgnetv2_m_table_cell/checkpoint0027.pth" \
GPUS=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
BATCH_SIZE_PER_GPU=8 \
NUM_WORKERS=4 \
MASTER_PORT=29536 \
EPOCHS=35 \
NUM_QUERIES=640 \
SEED=0 \
QUICK_STEPS=0 \
RUNTIME_ENV=/aicamera-mlp/chenkai/envs/deim \
METHOD_PROVENANCE="$PWD/configs/e35_r7/method_provenance.yml" \
FROZEN_INPUTS="$PWD/configs/e35_r7/frozen_inputs.yml" \
RUNTIME_MANIFEST="$PWD/tools/table_cells/e35_r7_runtime_sha256.txt" \
VERIFY_FROZEN_INPUTS=1 \
PREFLIGHT_ONLY=0 \
ALLOW_EXISTING=0 \
bash tools/table_cells/run_e35_r7_experiment.sh
```

## C3: Mosaic only, resume epoch 28--34

```bash
cd /aicamera-mlp/zhuyouwei/code/DEIM || exit 1

RUN_ID=resume_c3_mosaic_only_seed0 \
CONTRACT_ID=resume_c3_mosaic_only \
CONFIG_FILE="$PWD/configs/e35_r7/resume_c3_mosaic_only.yml" \
OUTPUT_DIR="$PWD/output/e35_r7_sgqse/resume_c3_mosaic_only_seed0" \
DATA_ROOT=/aicamera-mlp/chenkai/data/table_det_data/paddlex_l_cell_merged \
RUN_MODE=resume \
PRETRAINED= \
RESUME="$PWD/output/dfine_hgnetv2_m_table_cell/checkpoint0027.pth" \
GPUS=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
BATCH_SIZE_PER_GPU=8 \
NUM_WORKERS=4 \
MASTER_PORT=29537 \
EPOCHS=35 \
NUM_QUERIES=640 \
SEED=0 \
QUICK_STEPS=0 \
RUNTIME_ENV=/aicamera-mlp/chenkai/envs/deim \
METHOD_PROVENANCE="$PWD/configs/e35_r7/method_provenance.yml" \
FROZEN_INPUTS="$PWD/configs/e35_r7/frozen_inputs.yml" \
RUNTIME_MANIFEST="$PWD/tools/table_cells/e35_r7_runtime_sha256.txt" \
VERIFY_FROZEN_INPUTS=1 \
PREFLIGHT_ONLY=0 \
ALLOW_EXISTING=0 \
bash tools/table_cells/run_e35_r7_experiment.sh
```

## C4: MAL + Dense O2O retrofit, resume epoch 28--34

```bash
cd /aicamera-mlp/zhuyouwei/code/DEIM || exit 1

RUN_ID=resume_c4_deim_core_seed0 \
CONTRACT_ID=resume_c4_deim_core \
CONFIG_FILE="$PWD/configs/e35_r7/resume_c4_deim_core.yml" \
OUTPUT_DIR="$PWD/output/e35_r7_sgqse/resume_c4_deim_core_seed0" \
DATA_ROOT=/aicamera-mlp/chenkai/data/table_det_data/paddlex_l_cell_merged \
RUN_MODE=resume \
PRETRAINED= \
RESUME="$PWD/output/dfine_hgnetv2_m_table_cell/checkpoint0027.pth" \
GPUS=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
BATCH_SIZE_PER_GPU=8 \
NUM_WORKERS=4 \
MASTER_PORT=29538 \
EPOCHS=35 \
NUM_QUERIES=640 \
SEED=0 \
QUICK_STEPS=0 \
RUNTIME_ENV=/aicamera-mlp/chenkai/envs/deim \
METHOD_PROVENANCE="$PWD/configs/e35_r7/method_provenance.yml" \
FROZEN_INPUTS="$PWD/configs/e35_r7/frozen_inputs.yml" \
RUNTIME_MANIFEST="$PWD/tools/table_cells/e35_r7_runtime_sha256.txt" \
VERIFY_FROZEN_INPUTS=1 \
PREFLIGHT_ONLY=0 \
ALLOW_EXISTING=0 \
bash tools/table_cells/run_e35_r7_experiment.sh
```
