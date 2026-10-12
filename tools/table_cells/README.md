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
