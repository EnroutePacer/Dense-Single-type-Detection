# 数据文件说明

本目录只归档足以审计结论的小型派生数据，不含 checkpoint、原始数据集图像、
固定子集 COCO 标注、训练 rank JSONL 或逐 epoch 原始 replay JSONL。

- `full_val_metrics.csv`：服务器完整验证集的 epoch 28--34 COCO 指标；这里的
  `AP_small` 是 COCO 原图面积口径。
- `b0_full_val_metrics.csv`：相同续训窗口的 fixed-640 VFL/B0 EMA 对照曲线，
  只用于图中参照；它不是 SGQS raw 权重。
- `val128_raw_ema_metrics.csv`：固定 val128 上 raw/EMA、`maxDets=300/640` 的
  COCO-small AP/AR 对照。
- `checkpoint_identity.csv`：checkpoint 文件 SHA-256，以及按有序参数名、dtype、
  shape 和 tensor bytes 计算的 raw/EMA state SHA-256。文件容器不同不代表参数不同。
- `val128_replay_analysis.json`：首轮阶段 coverage、rank 和 transition 汇总。
- `val128_ema/`：EMA 精细复核的完整派生表。`transition_annotations.csv` 是逐
  annotation 的跨 epoch 变化表，其余 CSV 是阶段覆盖、转移、rank、滞留位置和
  COCO-small/hybrid-small 交并分组汇总。

`val128_ema/stage_coverage.csv` 等表中的 coverage 是每个 GT 独立询问“是否存在
任意 query 达到 IoU 阈值”。同一 query 可以覆盖多个 GT；它不是 COCO 一对一匹配
下的 TP、FN 或 recall。真 COCO AP/AR 只取自 evaluator 输出。

本地 hybrid-small 定义为直接缩放到 640×640 后
`bbox area < 32² OR bbox short side < 16`；COCO-small 仍是原图 annotation
`area < 32²`。两者不得互换。
