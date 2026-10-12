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

## 35 epoch DEIM-core 与 epoch-27 续训屏选

`train_e35_r7_sgqse_8gpu.sh` 固定 8 卡、每卡 batch 8、global batch 64、
fixed-640、query 640、seed 0 和绝对终点 `epoches=35`。对
`checkpoint0027.pth` 使用 resume 时，训练循环执行 epoch 28--34，
正好 7 epoch；不能把 `EPOCHS` 写成 7。配置位于 `configs/e35_r7/`。

启动脚本会检查冻结的数据/权重 SHA-256、resume 文件名和
`last_epoch=27`，拒绝非空输出目录，并保存 `run_manifest.txt`、
`config_source.yml` 和包含命令行覆盖的 `resolved_config.yml`。如果有意更换
数据或初始权重，必须将其视为新实验，并显式设置
`VERIFY_FROZEN_INPUTS=0`。

方法数学以提交
`6fd2c301bd8c09ff3d49fa077c34eaf75ef35b9a` 为唯一基准。
`configs/e35_r7/method_provenance.yml` 冻结方法文件哈希；
`verify_e35_r7_contract.py` 还会检查最终解析配置中的损失系数、消融开关、
课程、固定尺度、query、global batch 和 checkpoint 模式。先做只读预检：

```bash
PREFLIGHT_ONLY=1 bash tools/table_cells/train_e35_r7_sgqse_8gpu.sh resume_b1_sgqs
PREFLIGHT_ONLY=1 bash tools/table_cells/train_e35_r7_sgqse_8gpu.sh resume_b2_efgl
PREFLIGHT_ONLY=1 bash tools/table_cells/train_e35_r7_sgqse_8gpu.sh resume_b3_sgqs_e
```

预检会校验 6 GB 训练标注、验证标注和断点哈希，所以不是瞬时完成；成功时
不会创建输出目录，也不会启动 `torchrun`。B0 可用同样方式预检，但正式
重跑仍需 `ALLOW_CONTROL_RERUN=1`。

在四台独立 8 卡机器上分别执行：

```bash
cd /aicamera-mlp/zhuyouwei/code/DEIM

bash tools/table_cells/train_e35_r7_sgqse_8gpu.sh deim_core_e35
bash tools/table_cells/train_e35_r7_sgqse_8gpu.sh resume_b1_sgqs
bash tools/table_cells/train_e35_r7_sgqse_8gpu.sh resume_b2_efgl
bash tools/table_cells/train_e35_r7_sgqse_8gpu.sh resume_b3_sgqs_e
```

B0 fixed-640 VFL 已在运行，所以默认启动队列不包含
`resume_b0_baseline`，且脚本会主动拒绝该名称。只有确认原 B0 不可用
时，才能使用 `ALLOW_CONTROL_RERUN=1`重跑。若在同一机器并发，
还必须为每个任务设置不同的 `MASTER_PORT`。
