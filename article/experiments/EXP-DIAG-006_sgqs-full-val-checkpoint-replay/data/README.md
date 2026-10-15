# EXP-DIAG-006 派生数据

## 完整性

本目录是 12,841 张图、1,458,060 个 non-crowd GT、e28--e34 × raw/EMA 共 14 个唯一
参数状态的正式派生数据包。`best_stg1.pth`、`last.pth` 与 e34 参数完全相同，未重复计算。

[audit.json](audit.json) 是当前完成状态的权威入口：`status=complete`，并记录 3,920 项
stage reconciliation、280 项 global-PR reconciliation、10,549,440 行 transition detail、
所有输入逻辑名/SHA-256 及 Windows CPU 运行身份。[artifact_manifest.csv](artifact_manifest.csv)
冻结正式数据文件的 bytes 与 SHA-256。

[input_inventory.json](input_inventory.json) 是实验运行前生成的输入盘点快照，其
`result_status=no_same_backend_gpu_full_validation_results` 只描述当时状态，**不得作为当前
完成状态读取**；正式 CPU 全量结果及其范围以 `audit.json`、本页和实验 YAML 为准。

## 文件索引

### 指标与预算

- [metrics_14_states.csv](metrics_14_states.csv)：每个 epoch/state 一行，保存 COCO
  AP/AP50/AP75、AP-small/medium/large、AR 及 `maxDets=300/640` 两套结果。
- [raw_ema_comparison.csv](raw_ema_comparison.csv)：同 epoch、同 budget 的 raw/EMA 指标
  对齐及 `ema_minus_raw`。
- [coco_small_per_iou.csv](coco_small_per_iou.csv)：正式 evaluator-small 的逐 IoU、逐
  budget AP/AR 切片。
- [coco_small_per_iou_budget_gap.csv](coco_small_per_iou_budget_gap.csv)：同一
  epoch/state/IoU 下 640 与 300 预算的 AP/AR 差。

### 阶段 coverage 与正式匹配

- [stage_coverage.csv](stage_coverage.csv)：dense-all、Encoder Top-640、Decoder all-640、
  Decoder Top-300 的 per-GT independent best-IoU coverage。一个 query 可覆盖多个 GT，
  因此这里的 `covered/coverage` **不是** TP/FN 或 COCO recall。
- [match_aggregates.csv](match_aggregates.csv)：正式 score-ordered 一对一匹配的 detection、
  ground-truth 与 overall 聚合，包括 TP/FP/FN、precision/recall 和截断检测数。
- [match_gt_buckets.csv](match_gt_buckets.csv)：正式 TP/FN 按 cohort、domain、每图 GT
  密度、原图面积、640 输入面积/短边等轴分桶；GT 桶不人为分配 FP。

### 配对转移与直接归因

- [transition_summary.csv](transition_summary.csv)：每个配对、state、scope、cohort、IoU、
  budget 的 stable-FN/gain/loss/stable-TP 和 net-TP。
- [transition_attribution.csv](transition_attribution.csv)：把 gain/loss 的 FN 端按
  `candidate_absence`、`encoder_selection`、`decoder_geometry`、`rank_gt300`、
  `one_to_one_competition` 分解。
- [transition_details.csv.gz](transition_details.csv.gz)：10,549,440 行逐 GT 配对明细，
  包含 before/after 正式状态、query/rank/score/IoU 与四阶段证据；文件使用 Git LFS。

直接 cause 表示“FN 端最后一个仍存活阶段”，不是训练级根因。`rank_gt300` 表示达到当前
IoU 阈值的最终 Decoder query 仍在 640 个 query 中，但图内分数排名 >300。

**重要计数规则：** 单个 IoU 行的 `count` 单位是 GT transition；跨 IoU=.50:.05:.95
求和后，单位必须写成 **GT×IoU transition events**。同一 GT 可在十个阈值重复出现，
十阈值计数不是互异 GT，也不是 AP 的可加分解。

[analysis_summary.md](analysis_summary.md) 的 paired attribution 表固定为 **IoU=.75 单阈值**；
例如 EMA e28→e29 gain 的 1,374 是 IoU=.75 的 GT transitions。十 IoU 合计 13,835
必须从 `transition_attribution.csv` 过滤 `state=ema`、`match_scope=coco_small_eval`、
`cohort=coco_small_eval`、`max_dets=300` 后汇总。

### 全局 PR 与分数排序

- [global_pr_curve.csv](global_pr_curve.csv)：每个 epoch/state/scope/IoU/budget 的 101 点
  interpolated precision，以及达到该 recall anchor 的最低 score。
- [tp_fp_score_separation.csv](tp_fp_score_separation.csv)：全局 score sort 后的 TP/FP
  数量、分数分位数、由 PR 复算的 AP/AR 与 summary 对账差。
- [tp_fp_score_histogram.csv](tp_fp_score_histogram.csv)：TP/FP score histogram，供静态图
  和独立复查使用。

AP-small 的 e28→e29、e28→e30、e30→e34 变化分别有 81.2%、80.1%、88.7% 集中在
recall=.69--.72；该比例来自十 IoU×101 recall anchors 的带符号 precision 差，不来自
transition count。

### 代表样本索引

- [representative_annotations.csv](representative_annotations.csv)：按
  `(state, pair, scope, IoU, transition, cause, small partition, domain)` 确定性排序的逐
  annotation 证据；18.4 MB，使用 Git LFS。
- [representative_images.csv](representative_images.csv)：按 image 聚合 annotation、cause
  与 transition 的快速索引。它不能脱离 annotation 表独立解释机制。

相应 contact sheet 的 panel 选择、上游哈希和输出图哈希位于
[representative_transition_figure_manifest.csv](../figures/representative_transition_figure_manifest.csv)。

## small 口径

| 名称 | 定义 | GT 数 | 允许用途 |
|:---|:---|---:|:---|
| `coco_small_eval` | 原图 annotation `area <= 32²` | 32,427 | 正式 AP-small/AR-small、formal transition |
| `coco_small` | 原图 annotation `area < 32²` | 32,346 | 描述性 cohort |
| `local_hybrid_small` | 640 输入上 `area < 32² OR short side <16` | 710,421 | coverage/recall/transition，不称 AP |
| `hybrid_only` | hybrid-small 且不在 strict coco-small | 684,512 | 本地细长/短边机制诊断 |

`hybrid_only` 占 local hybrid 的 96.353%，所以 hybrid aggregate 不会复现 evaluator-small
的倒 U 也不矛盾。分析 local hybrid 时使用 `scope=all`；解释 COCO AP-small 时必须使用
`scope=coco_small_eval`。密度、短边和 domain 是重叠的交叉轴，不能把各轴贡献相加。

## 可复算的关键数值

正式 evaluator-small、EMA、Top-300，十 IoU 合计：

| 配对 | 方向 | 总 GT×IoU events | `rank_gt300` | 比例 |
|:---|:---|---:|---:|---:|
| e28→e29 | gain | 13,835 | 12,437 | 89.90% |
| e28→e30 | gain | 13,035 | 11,436 | 87.73% |
| e30→e34 | loss | 12,780 | 11,934 | 93.38% |

IoU=.75 单阈值对应 1,374/1,263、1,273/1,149、1,279/1,216，分别为
91.92%、90.26%、95.07%。详细筛选条件与解释见
[analysis/README.md](../analysis/README.md)。

## 身份与发布范围

- 本地正式 merged annotation SHA-256：
  `f855dd4b9dded934479a54aad70f01b3c678d0ae1fd91e3f8ca1cd6890bd693e`。
- 服务器训练 merged annotation SHA-256：
  `a6d107322285595d968f63b4cd6d9ccf2442ac844a6875685a2aa16f3b9dd72b`。
- 推理：Windows CPU、batch=1、PyTorch 2.14.0+cpu；matcher：faster-coco-eval 1.8.0。

两份 annotation SHA 不一致，所以数据包支持“本地完整验证集上的机制诊断”，不支持
“服务器 H20 bit-level 复现”。原始 checkpoint、私有 annotation/图像、memmap 与含
机器绝对路径的上游运行目录不入库；发布的是脱敏派生表、静态图、脚本和哈希身份。
