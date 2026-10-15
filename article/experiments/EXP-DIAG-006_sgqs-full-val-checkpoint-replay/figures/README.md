# EXP-DIAG-006 图片索引

本目录包含 6 张全量定量图、6 张正式 Decoder 转移代表图，以及 29 张选择性
Encoder/Decoder checkpoint query probe。图像用于阅读已发布 CSV；正式数值以
[data/audit.json](../data/audit.json) 和相应 CSV 为准。

## 全量定量图

- [metrics_trajectory.png](metrics_trajectory.png)：14 个 raw/EMA 状态的
  evaluator AP-small/AR-small @300。EMA 显示明显倒 U，raw 波动范围较小。
- [per_iou_trajectory.png](per_iou_trajectory.png)：EMA 在十个 IoU 阈值上的
  AP-small/AR-small @300，以及 @640-@300 budget gap；用于检查结论是否只由某一个 IoU
  阈值驱动。
- [stage_coverage.png](stage_coverage.png)：evaluator-small 与 local hybrid-small 在
  IoU=.50/.75 的 dense-all、Encoder Top-640、Decoder all-640、Decoder Top-300
  independent best-IoU coverage。**这不是一对一 TP/FN 或 COCO recall。**
- [transition_attribution.png](transition_attribution.png)：EMA evaluator-small 正式
  gain/loss 端点，按十个 IoU 阈值的平均 GT transitions 展示五种直接 cause；不是只画
  IoU=.75，也不是 AP 的可加分解。
- [bucket_recall.png](bucket_recall.png)：EMA evaluator-small、IoU=.75、Top-300 的正式
  一对一 recall，按 domain、每图 GT 密度和 640 输入短边分桶。三个轴相互重叠，不能把
  各轴变化相加。
- [global_pr_score_separation.png](global_pr_score_separation.png)：左侧为 EMA
  e28/e29/e30/e34 在 IoU=.75、Top-300 的 global PR；右侧为 14 状态 TP/FP median score
  separation。左图是十 IoU AP 的一个可解释切片，完整 PR 数据见 CSV。

定量图由 [plot_full_val.py](../scripts/plot_full_val.py) 从正式派生表确定性生成；plotter 会
对状态网格、IoU、budget、population 和对账字段 fail closed。

## 正式 Decoder 转移代表图

前段 gain：

- [representative_ema_e28_e29_gain_iou50.png](representative_ema_e28_e29_gain_iou50.png)
- [representative_ema_e28_e29_gain_iou75.png](representative_ema_e28_e29_gain_iou75.png)
- [representative_ema_e28_e29_gain_iou95.png](representative_ema_e28_e29_gain_iou95.png)

后段 loss：

- [representative_ema_e30_e34_loss_iou50.png](representative_ema_e30_e34_loss_iou50.png)
- [representative_ema_e30_e34_loss_iou75.png](representative_ema_e30_e34_loss_iou75.png)
- [representative_ema_e30_e34_loss_iou95.png](representative_ema_e30_e34_loss_iou95.png)

六页均固定为 EMA、formal `coco_small_eval`、`maxDets=300`。每页按五种直接 cause ×
wired/wireless 排 panel；正式计数为零时显示 `absent`，compact 表不能证明某一 domain
存在或不存在时显示 `domain_not_resolved`，不以空白图冒充证据。

- 洋红：选中的 GT；
- 绿色：正式 TP query；
- 橙色：FN 端可复核的 Decoder query；
- `rank_gt300` 的橙色框仍是有效 Decoder 输出，只是 rank >300。

这些图只画原始 GT 与持久化的**最后一层 Decoder 几何**。`candidate_absence`、
`encoder_selection` 等标签来自逐 GT 阶段表，不能把这些 contact sheet 称为 Encoder
几何输出图。所有 panel 的选择、状态、query/rank/score/IoU、上游 SHA 和图 SHA 见
[representative_transition_figure_manifest.csv](representative_transition_figure_manifest.csv)。

## Checkpoint query probes

[query_probes](query_probes) 下有 29 张
`query_probe_image_<image_id>_ema_e28_e34.png`。每张图固定同一真实图，行是 EMA
e28--e34，三列依次为：

1. **Encoder Top-640 source level**：真实选中的 640 个 dense candidates，按来源
   P3/P4/P5 着色；
2. **Decoder all-640**：Decoder 最后一层全部 640 个输出框；
3. **Decoder local-hybrid-small**：在同一批最终 640 框中，按 640 输入尺度满足
   `area < 32² OR short side <16` 的子集。

### P3/P4/P5 与 Decoder hybrid-small 不是同一量

P3/P4/P5 对应 dense feature strides 8/16/32（80×80、40×40、20×20），表示每个被
Encoder Top-640 选中 query 的**实际来源层**。对任意单图、任意 checkpoint，它们是互斥
分箱，必须满足：

```text
P3 count + P4 count + P5 count = 640
```

单张图也只有 640 个 Decoder query。第三列的 `Decoder hybrid-small count` 是在 Decoder
最后一层输出后，根据最终框宽高事后计算的布尔 mask，取值 0--640；它不是 Encoder
P3/P4/P5，也不是“输入 Decoder 的 query 数”。更不能把根据最终框短边推导的 preferred
scale 称为 Encoder source level。

### 选择与索引

- [query_probe_selection.csv](query_probes/query_probe_selection.csv)：32 个确定性 strata，
  即 evaluator-small/hybrid-only 两条 track × e28→e29 gain/e30→e34 loss ×
  candidate/Encoder/Decoder/rank 四类 cause × wired/wireless；同一图可命中多个 strata，
  去重后为 29 图。
- [query_probe_manifest.csv](query_probes/query_probe_manifest.csv)：每张图×七个 checkpoint
  一行，共 **29×7=203 行**。记录 checkpoint/annotation/replay/runtime/脚本/NPZ/图像哈希、
  P3/P4/P5 数量、Decoder hybrid 数量和精确对账标志。
- [query_probe_data.npz](query_probes/query_probe_data.npz)：按
  `[image, epoch, query]` 保存 Encoder dense index/source level/normalized box，以及 Decoder
  原图 xywh、score、hybrid mask；可离线重画而无需 checkpoint。

正式发布网格已核对：203/203 行的 Encoder selected count=640、P3+P4+P5=640、Decoder
query count=640；Decoder boxes、scores、稳定 rank order、hybrid mask 与 canonical replay
均精确一致，box/score 最大绝对误差为 0。manifest 中冻结的 `plot_full_val.py` SHA-256
也与当前脚本一致。

probe 按机制证据强度选样，是**定性索引**而不是随机样本；不能用 29 张图里某种颜色、
cause 或 hybrid 框的出现频率估计全量比例。总体结论必须引用
[transition_attribution.csv](../data/transition_attribution.csv)、
[match_gt_buckets.csv](../data/match_gt_buckets.csv) 等正式表。

## 范围限制

所有图均来自本地完整验证集的 Windows CPU、batch=1 回放。本地正式 annotation
SHA-256 `f855dd4b9dded934479a54aad70f01b3c678d0ae1fd91e3f8ca1cd6890bd693e`
与服务器训练 annotation SHA-256
`a6d107322285595d968f63b4cd6d9ccf2442ac844a6875685a2aa16f3b9dd72b` 不同，因此图像支持
本地范围内的机制复核，不是服务器 H20 bit-level 可视化复现。
