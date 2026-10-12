# D-FINE-S 表格单元格训练

数据集使用 COCO 检测格式，默认目录如下：

```text
DATA_ROOT/
├── images/                # 包含 train/wired、train/wireless、val/... 等路径
└── annotations/
    ├── instance_train.json
    └── instance_val.json
```

两个 JSON 的 `categories` 应只包含一个类别。PaddleX 转换脚本生成的是 `{"id": 0, "name": "cell"}`，因此 DEIM 的标注中 `category_id` 也应为 `0`（当前配置关闭 COCO 类别重映射）。图片文件名由 JSON 的 `file_name` 指定，并相对于对应图片目录解析。

在已安装 PyTorch/torchvision 的 Linux 环境中运行：

```bash
DATA_ROOT=/path/to/merged_table_cell_dataset \
GPUS=0,1,2,3 BATCH_SIZE_PER_GPU=2 \
bash tools/table_cells/train_dfine_s_table_cell_multigpu.sh
```

S 脚本默认加载项目根目录下的 `deim_dfine_hgnetv2_s_coco_120e.pth`；可用 `PRETRAINED=/path/to/other.pth` 覆盖。
脚本默认使用当前已激活的 Python/Conda 环境；如需由脚本激活环境，请显式传入 `RUNTIME_ENV=/path/to/deim`。

D-FINE-M 使用独立入口，默认每卡 batch size 为 1：

```bash
DATA_ROOT=/path/to/merged_table_cell_dataset \
GPUS=0,1,2,3 BATCH_SIZE_PER_GPU=1 \
bash tools/table_cells/train_dfine_m_table_cell_multigpu.sh
```

M 脚本默认加载项目根目录下的 `deim_dfine_hgnetv2_m_coco_90e.pth`；可用 `PRETRAINED=/path/to/other.pth` 覆盖。两个入口都使用 `-t` tuning 模式迁移权重，而不是 `-r` resume 模式。

脚本会自动识别 `merge_table_cell_coco.py` 生成的合并目录；也可以分别设置 `TRAIN_IMAGES`、`TRAIN_ANN`、`VAL_IMAGES`、`VAL_ANN`。脚本会将它们通过 `-u` 注入配置，并把 `BATCH_SIZE_PER_GPU` 换算成 DEIM 所需的全局 batch size。

该任务配置固定 640×640 输入，查询数默认为 640，以覆盖合并脚本允许的最大单表单元格数；仅启用轻度光度扰动和水平翻转，Mosaic、RandomIoUCrop、RandomZoomOut、MixUp 及多尺度训练均关闭，以免破坏表格网格和稀疏小单元格。

## 35 epoch / epoch-27 resume 实验矩阵

正式入口是 `run_e35_r7_experiment.sh`。它是透明的通用执行器，不按实验名
选择配置，也不在内部隐藏 GPU、batch、epoch、checkpoint 或输出路径。调用者
必须显式传入这些参数；脚本只负责解析配置、执行 fail-closed 契约校验、记录
manifest，然后调用底层多卡训练入口。

方法数学以提交
`6fd2c301bd8c09ff3d49fa077c34eaf75ef35b9a` 为唯一基准。
`configs/e35_r7/method_provenance.yml` 冻结方法文件哈希；
`verify_e35_r7_contract.py` 检查最终解析配置中的损失、增强、课程、fixed-640、
query 640、global batch 64、checkpoint 模式、数据/权重身份和
`last_epoch=27`。launcher 还会先按显式 `RUNTIME_MANIFEST` 对整套活动运行
文件逐项验 SHA-256。成功后保存 `run_manifest.txt`、`config_source.yml`、
`resolved_config.yml`、`method_provenance.yml`、`frozen_inputs.yml` 与
`runtime_sha256.txt`。

矩阵分为两层：B0--B3 是论文 SGQS-E 主消融；C1--C4 只是机制诊断对照，
不能替代主消融，也不能把它们的收益归入 SGQS-E。B0 已完成，不重复训练。
旧压缩包中的 `deim_fresh_e28` 被 35-epoch `deim_core_e35` 取代。

| ID | 初始化 | 新增变化 | 状态/用途 |
|---|---|---|---|
| `deim_core_e35` | COCO tuning checkpoint | MAL + Dense O2O，0--34 | 待跑，受控 DEIM 基线 |
| `resume_b0_baseline` | epoch 27 | 无 | 已完成，主消融负对照 |
| `resume_b1_sgqs` | epoch 27 | SGQS | 待跑，论文主消融 |
| `resume_b2_efgl` | epoch 27 | EFGL | 待跑，论文主消融 |
| `resume_b3_sgqs_e` | epoch 27 | SGQS + EFGL | 待跑，论文主方法 |
| `resume_c1_small_reg` | epoch 27 | box 1.30 / local 1.15 | 待跑，历史启发式对照 |
| `resume_c2_mal_only` | epoch 27 | 只把 VFL 换成 MAL | 待跑，分类损失对照 |
| `resume_c3_mosaic_only` | epoch 27 | 只加 Mosaic | 待跑，数据增强对照 |
| `resume_c4_deim_core` | epoch 27 | MAL + Dense O2O | 待跑，晚期 retrofit 诊断 |

八个待执行实验的完整显式命令见
`configs/e35_r7/SERVER_COMMANDS.md`。对 resume 实验，`EPOCHS=35` 是绝对
终点，实际执行 epoch 28--34；不能写成 `EPOCHS=7`。把
`PREFLIGHT_ONLY=1` 可用于只读预检，成功时不会创建输出目录或启动训练。
