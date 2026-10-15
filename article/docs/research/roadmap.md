# 研究路线图

当前优先路线由 [ADR-003](decisions/ADR-003_training-first-sgqs-e.md) 更新：SGQS-E 已实现，B0--B3/C2 单 seed checkpoint 续训屏选已经完成；[EXP-TRAIN-006](../../experiments/EXP-TRAIN-006_sgqs-supervision-probes/README.md) 的 SGQS 臂已完整训练至 epoch 34，固定 val128 的 e28--e34 raw/EMA 回放也已完成，带同一套探针的 B0 臂尚未运行。当前由 [EXP-DIAG-006](../../experiments/EXP-DIAG-006_sgqs-full-val-checkpoint-replay/README.md) 对 e28--e34 的 raw/EMA 14 个唯一状态做 full-val 逐 annotation、score/rank 和 COCO 一对一匹配回放，再执行 B0 带探针对照；之后才依据证据决定是否修订损失或进入正式从头多 seed 对照。[EXP-DIAG-005](../../experiments/EXP-DIAG-005_sgqs-top300-ranking-replay/README.md) 的 val128 结果是有范围限制的阶段定位，不等于已经完成全数据归因。

## Gate 0：资产与口径

状态：**本地 full-val 资产与协议已盘点，同后端 GPU 输出待落盘**。val128 的确定性
清单、checkpoint 身份、两种 small 定义和派生表已经由
[EXP-DIAG-005](../../experiments/EXP-DIAG-005_sgqs-top300-ranking-replay/README.md)
归档；12,841 图/1,458,060 GT 的完整 val 资产、14 个唯一参数状态与输出 schema
已由 [EXP-DIAG-006](../../experiments/EXP-DIAG-006_sgqs-full-val-checkpoint-replay/README.md)
登记。原始私有 annotation、图像和权重不在仓库内。

- 冻结正式 merged 身份记录，并保存 annotation、checkpoint、配置、环境 SHA-256 与
  许可状态；val128 映射一致不等于全量身份审计完成。
- 固定输入 640、`K=640`、模型权重和后处理。
- 为 full-val 导出每个 annotation 的 dense、Encoder Top-640、Decoder 全 640、最终 score/rank、Top-300 状态与 COCO 一对一匹配；独立 coverage 与正式匹配必须分字段保存。

验收：同一图可以从 dense proposals 一直追踪到 Top-K、每层 decoder 和 output；所有产物含 image/category 映射和配置哈希。

## Gate 1：冻结模型的分阶段归因

状态：**val128 的 SGQS EMA/raw 回放已完成，14 状态 full-val GPU 回放待完成**。
val128 中 e28→e29 的 49 个 Top-300 gain 全部已有 IoU≥0.75 的 Decoder 框但原
rank>300；e30→e34 的 36 个 loss 也仍有合格框但跌出前 300。由于 430 个
COCO-small 只分布在 17 张图、状态变化集中于两张极密图，这一证据只能定位子集上的
直接承载阶段。

- 在一致 GPU 后端回放 e28--e34 full-val raw/EMA，计算独立 GT coverage 与正式
  COCO 一对一匹配结果；不得把前者称为 recall、TP 或 FN。
- 记录最佳 candidate rank、来源层、重复候选、短边与密度分桶。
- 跟踪 Decoder 全 640 的 IoU、score、rank、Top-300 生存与一对一匹配身份；按 COCO-small、hybrid-small 及其交并子群报告。
- 按十个 COCO IoU 阈值与 `maxDets=300/640` 输出 AP/AR，并复算跨全图像的 PR 与
  TP/FP score separation；局部 TP/FN 转移不能作为 AP 的可加分解。
- full-val raw 用于检验 val128 的 raw/EMA 分歧；未经受控干预不得把 EMA 写成唯一
  根因。

验收：full-val 中每个状态变化可追踪到 generation、selection、Decoder 几何、最终 score/rank 截断或一对一匹配竞争；给出这些机制解释完整 AP-small 曲线的数量和比例，未知项单列。val128 结论不能代替该验收。

## Gate 2：selector 诊断对照

状态：**尚未开始干预对照**。固定 val128 已把直接变化定位到最终 Decoder score/rank，但还没有证明应修改 Encoder selector；先完成 Gate 1 full-val 与 B0 带探针对照，再决定是否保留以下比较。

在不训练、同一冻结输出、同 K 下比较：

1. 原始 global Top-K；
2. 简单 per-level calibration；
3. 简单 per-level quota；
4. 去重/coverage-aware selector；
5. GT-aware oracle；
6. K sweep，仅用于判断容量，不作为主方法。

验收：主假设必须相对简单对照和 oracle 定位出可改进空间。

## Gate 3：确定单一方法分支

状态：**等待 full-val 与 B0 证据**。当前不能仅凭 val128 排名跨界就选择 selector、分类校准、EMA 或 SGQS owner/GO 中的任一分支。

- generation 主导：研究 P2/形状先验/表示；
- selection 主导：研究 level-calibrated 或 coverage-aware initialization；
- decoder 主导：研究匹配/质量校准；
- output 主导：修正后处理或评估，不包装成训练创新。

只选择一个由证据支持的主组件，其他作为消融或未来工作。

## Gate 4：训练与论文验证（屏选完成，正式验证待修订）

状态：**单 seed checkpoint 续训屏选完成；正式验证未开始**。EXP-TRAIN-006 的 SGQS 臂和 val128 replay 已完成，但 B0 带同探针对照、参数级单变量干预及从头多 seed 训练均未完成。

相同初始化、global batch、epoch/steps、seed 和 `maxDets=300` 下比较 baseline 与方法。先 seed 0 验证方向，再三 seed；报告早期同 epoch、最终指标、wall-clock、GPU-hours、参数/FLOPs 和固定延迟协议。
