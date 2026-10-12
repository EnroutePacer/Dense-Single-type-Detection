# EXP-TRAIN-002：干净 DEIM Dense O2O/MAL 基线

状态：planned / not_run。

本实验用于回答完整 DEIM 配方在当前密集单类任务上的真实行为；不得用旧目录名或类名代替运行时 `losses`、Mosaic/MixUp 和 scheduler 证据。与 EXP-TRAIN-001 保持模型结构、初始化、global batch、总 epoch 和 seed 可比。

当前冻结入口为 `configs/e35_r7/deim_core_e35.yml`，由
`tools/table_cells/run_e35_r7_experiment.sh` 启动，调用命令显式指定该配置、
GPU、batch、epoch、权重与输出目录。它从统一
COCO detector checkpoint 以 tuning 模式训练 epoch 0--34，固定
640x640、K=640、8×8=global batch 64、seed 0 和 maxDets=300。

为了把结果归因到 DEIM 的两个核心手段，这里只把 VFL/干净增强
换为 MAL + Dense O2O，保留表格基线的 optimizer、ReLU 和 fixed-640。
因此它是“controlled DEIM-core”，不能误写成完整照搬官方 COCO 超参。
当前尚无运行结果。
