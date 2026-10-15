# EXP-DIAG-006 派生数据

当前目录尚无正式 full-val 结果。CPU 冒烟和固定 val128 门禁产物不复制到这里，也不能
冒充 12,841 图 GPU 回放结果。

当前唯一数据文件 [input_inventory.json](input_inventory.json) 只登记脱敏后的输入数量、
SHA-256、图像存在性、val128 映射门禁范围和 checkpoint 去重矩阵；它不是模型输出。

完成 14 个唯一 raw/EMA 状态后，只归档 README 中列出的脱敏小型派生 CSV/JSON/Markdown
及其 SHA-256 manifest。原始 checkpoint、私有 annotation、图像、memmap、完整逐检测
导出和绝对路径不入库。任何表只有在 `audit.json` 同时证明 12,841 图、1,458,060 GT、
epoch 28--34 × raw/EMA、十个 IoU 阈值与 `maxDets=300/640` 全部齐备后，才可标为正式
结果。

正式分析还会生成两份确定性代表样本索引：

- `representative_annotations.csv`：每个
  `(state, pair, scope, IoU, gain/loss, cause, small partition, domain)` 按 evidence、距
  rank 300 的距离和 annotation id 排序；除 annotation/image id 外，保留 before/after
  TP/FN、formal query/rank/score/IoU 及 dense→Top-640→Decoder-640→Top-300 的 IoU/rank
  证据；
- `representative_images.csv`：按 image 聚合 annotation、cause 与 transition，仅作快速
  索引，不能脱离上一张表解释机制或自行当作 AP 分解。

代表图 renderer 仍须读取本机 canonical replay 的 metadata 与少量 final-decoder
prediction row，并按 `audit.json` 中的逻辑名、bytes 和 SHA-256 复核；这些大型上游文件
不会复制到本目录。最终 contact sheet 的逐 panel 选择、上游 hash、annotation/image/query
和输出图 hash 登记在 figures 目录的 `representative_transition_figure_manifest.csv`。
