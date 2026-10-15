# EXP-DIAG-006：SGQS 全验证集 checkpoint 回放

## 当前状态

本实验的协议、回放器、正式一对一匹配导出器和分析脚本已经就绪；**12,841 张图上的
14 个模型状态尚未完成同后端 GPU 回放，因此本页不报告任何新的 full-val 机制结果**。
历史训练日志中的 full-val COCO 曲线仍是已存在的 evaluator 结果，不能替代本实验要求的
逐 annotation、逐阶段回放。

已完成的执行前门禁如下：

- 本地 val 资产共 12,841 张图、1,458,060 个 non-crowd GT；wired 为
  5,847 图 / 627,939 GT，wireless 为 6,994 图 / 830,121 GT，逐条 annotation
  引用的图像均存在。
- 本地 wired / wireless val annotation SHA-256 分别为
  `9ae3c0166315f185805420e0a73b23f2200a924aa87986fcf3b928c8d03fd9cd` 和
  `9ec7c82d45feacaa13142427bb39106bd2c87fa36ef3991e0f0198a07362390f`。
  服务器训练所用 merged val annotation 的已冻结 SHA-256 为
  `a6d107322285595d968f63b4cd6d9ccf2442ac844a6875685a2aa16f3b9dd72b`。
- 本地 split 到 merged 的 ID/path 重排规则已在固定 val128 的全部 128 图、15,446 GT
  上核对一致；这只是一项映射门禁，**不能证明 full-val 结果，也不能替代全量身份审计**。
- `checkpoint0028`--`checkpoint0034` 每轮 raw 与 EMA 状态均不同，共 14 个唯一参数状态。
  `best_stg1.pth` 和 `last.pth` 的 raw/EMA 参数状态都与 `checkpoint0034.pth` 相同，
  因而不重复推理。身份表沿用
  [EXP-DIAG-005/checkpoint_identity.csv](../EXP-DIAG-005_sgqs-top300-ranking-replay/data/checkpoint_identity.csv)。
- e28 raw 已通过真实图 CPU 冒烟，固定 val128 已通过 compact replay、断点恢复与正式
  matcher 链路门禁。CPU 与既有 GPU 输出在总体 COCO 数值上接近，但候选并列排序和个别
  GT 的阶段值并非逐项一致，所以 CPU 门禁只证明链路可运行，不能作为最终机制证据。

当前待完成项是：在一致的 GPU 后端上运行 e28--e34 的 raw/EMA 全量回放、正式匹配、
分析和绘图，再把脱敏的小型派生数据与结论补到本实验目录。

## 问题与比较矩阵

本实验回答：完整验证集上的 AP-small 前段上升与后段回落，分别有多少来自候选不存在、
Encoder Top-640 未选中、Decoder 几何丢失、最终 rank 跌出 300、一对一匹配竞争，或
跨图像的全局 TP/FP 分数排序？

固定比较矩阵为：

- checkpoint：epoch 28、29、30、31、32、33、34；
- 参数状态：每轮 raw、EMA，共 14 个唯一状态；
- 数据：完整 wired + wireless val，共 12,841 图；
- 输入与预算：固定 640×640、Encoder `K=640`、Decoder 640 query，评价
  `maxDets=300/640`；
- IoU：`0.50:0.05:0.95` 十个阈值，不能只用 IoU 0.50 或 0.75 解释由十个阈值平均的
  AP；
- 配对端点：至少 e28→e29、e28→e30、e30→e34，并保留相邻 epoch 的完整轨迹；
- 分桶：wired/wireless、每图 GT 密度、输入 640 后短边、COCO area、COCO/hybrid
  交并子群。

## 两种 small 口径

两种口径必须并列保存，但回答不同问题：

- **COCO evaluator small**：正式 AP-small/AR-small 使用 evaluator 的 small areaRange；
  它用于解释日志里的 COCO AP-small 曲线。此前描述性 `coco_small` cohort 使用原图
  annotation `area < 32²`；精确边界处以 evaluator scope 为准。
- **本地 hybrid-small**：框直接按宽高缩放到 640×640 后，满足
  `area < 32² OR short side < 16`。这是本项目对细长、低像素短边目标的默认机制诊断
  cohort，不得改名为 COCO AP-small，也不得把其 coverage 写成 AP-small。

因此，“解释 AP-small”必须使用 COCO evaluator scope；“按本地标准分析 small”必须
报告 hybrid-small 及其与 COCO-small 的交并。两个数字不能互相替代。

## 统计口径

每个 GT 保存四阶段独立 best-IoU 与 rank：dense-all、Encoder Top-640、Decoder
all-640、Decoder Top-300。这里的独立 coverage 允许同一 query 覆盖多个 GT，**不是**
COCO recall、TP 或 FN。

正式 TP/FP/FN 由仓库同一 `faster_coco_eval` 的按分数排序、一对一匹配产生。即使正式
TP/FN 转移已经分类，计数也不是 AP 的可加分解：AP 仍取决于十个 IoU 阈值以及所有
图像合并后的 TP/FP score 顺序。最终报告必须同时给出逐 IoU AP/AR、Top-300/640 差、
逐 GT 转移和全局 PR/score separation。

## 实现与运行门禁

- compact、可恢复的批量回放：
  [full_val_replay.py](../../../tools/table_cells/full_val_replay.py)
- 正式 COCO 匹配导出：
  [coco_match_npz.py](../../../tools/table_cells/coco_match_npz.py) 与
  [coco_match_export.py](../../../tools/table_cells/coco_match_export.py)
- 配对分析：[analyze_full_val.py](scripts/analyze_full_val.py)
- 确定性绘图：[plot_full_val.py](scripts/plot_full_val.py)
- 针对性测试：
  [test_full_val_replay.py](../../../tests/test_full_val_replay.py)、
  [test_coco_match_export.py](../../../tests/test_coco_match_export.py) 和
  [test_analyze_full_val.py](scripts/test_analyze_full_val.py)

正式执行前还必须满足：合并后的 image/category/annotation ID 唯一；全量图像路径可读；
annotation、配置、checkpoint 和运行环境身份写入审计文件；GPU val128 门禁在同一后端
通过；14 个状态的输出均有完成标记且可断点恢复。任何一项失败都不得生成“完成”报告。

## 最终必须归档的派生数据

预期小型表格至少包括：

- `metrics_14_states.csv`、`raw_ema_comparison.csv`；
- `coco_small_per_iou.csv`、`coco_small_per_iou_budget_gap.csv`；
- `stage_coverage.csv`；
- `match_aggregates.csv`、`match_gt_buckets.csv`；
- `transition_summary.csv`、`transition_attribution.csv`、必要时压缩后的
  `transition_details`；
- `global_pr_curve.csv`、`tp_fp_score_separation.csv`、`tp_fp_score_histogram.csv`；
- `representative_annotations.csv`、`representative_images.csv`；
- `analysis_summary.md`、`audit.json`、`artifact_manifest.csv`。

最终图至少覆盖：14 状态 AP-small/AR-small 轨迹、十阈值与 300/640 差、四阶段
coverage、正式 gain/loss 归因、全局 PR 与 TP/FP 分数分离、domain/density/short-side
分桶，以及由确定性规则选取的代表图。图题必须标明是独立 coverage 还是正式一对一匹配。

原始权重、私有 annotation、图像、memmap、逐检测大表和含机器绝对路径的运行文件不进入
Git；只提交足以复算公开结论的脱敏派生表、静态图、脚本与 SHA-256 manifest。当前
[data/README.md](data/README.md) 只登记待落盘清单，不代表这些结果已经存在。已归档的
[input_inventory.json](data/input_inventory.json) 仅是输入身份清单，不是模型结果。

## 允许与禁止的结论

在 14 状态 GPU 全量回放完成前，只允许写：资产和参数状态已盘点、协议及 CPU/val128
工程门禁已通过、full-val 机制结果待运行。不得把下面任一说法写成事实：

- val128 的 rank-300 机制解释了 full-val AP-small 的多少比例；
- raw/EMA 在 full-val 上具有与 val128 相同的曲线；
- EMA、SGQS、过拟合或分类校准是倒 U 的唯一原因；
- hybrid-small coverage 的变化等价于 COCO AP-small 的变化；
- 某个 checkpoint 在新回放口径下已经优于其他 checkpoint。

固定 val128 的既有结果与边界仍见
[EXP-DIAG-005](../EXP-DIAG-005_sgqs-top300-ranking-replay/README.md)。
