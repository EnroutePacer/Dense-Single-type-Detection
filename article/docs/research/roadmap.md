# 研究路线图

当前优先路线由 [ADR-003](decisions/ADR-003_training-first-sgqs-e.md) 更新：SGQS-E 已实现，B0--B3/C2 单 seed checkpoint 续训屏选已经完成；[EXP-TRAIN-006](../../experiments/EXP-TRAIN-006_sgqs-supervision-probes/README.md) 的 SGQS 与 B0 同探针训练臂均已完整训练至 epoch 34。[EXP-DIAG-006](../../experiments/EXP-DIAG-006_sgqs-full-val-checkpoint-replay/README.md) 又在本地完整 12,841 图上完成 SGQS e28--e34 raw/EMA 14 个唯一状态的逐 annotation、score/rank、全局 PR 和 COCO 一对一匹配回放。当前主线转为两臂详细训练探针配对分析及单变量因果验证；之后才依据证据决定是否补 B0 回放、修订损失或进入正式从头多 seed 对照。

## Gate 0：资产与口径

状态：**本地 full-val 资产、协议与 14 状态 CPU 输出均已完成并审计**。val128 的确定性
清单、checkpoint 身份、两种 small 定义和派生表已经由
[EXP-DIAG-005](../../experiments/EXP-DIAG-005_sgqs-top300-ranking-replay/README.md)
归档；12,841 图/1,458,060 GT 的完整 val 资产、14 个唯一参数状态与输出 schema
已由 [EXP-DIAG-006](../../experiments/EXP-DIAG-006_sgqs-full-val-checkpoint-replay/README.md)
登记。派生表、图与可复算脚本已发布；原始私有 annotation、图像和权重不在仓库内。

- 冻结正式 merged 身份记录，并保存 annotation、checkpoint、配置、环境 SHA-256 与
  许可状态；val128 映射一致不等于全量身份审计完成。
- 固定输入 640、`K=640`、模型权重和后处理。
- 为 full-val 导出每个 annotation 的 dense、Encoder Top-640、Decoder 全 640、最终 score/rank、Top-300 状态与 COCO 一对一匹配；独立 coverage 与正式匹配必须分字段保存。

验收：同一图可以从 dense proposals 一直追踪到 Top-K、每层 decoder 和 output；所有产物含 image/category 映射和配置哈希。

## Gate 1：冻结模型的分阶段归因

状态：**本地完整集 scoped diagnosis 已完成**。EMA e28→29、e28→30 gain 与
e30→34 loss 的十 IoU 正式 evaluator-small 转移事件中，`rank_gt300` 分别占
89.90%、87.73%、93.38%；全局 PR 差异的 80.05%--88.66% 集中在 recall
0.69--0.72。变化人群集中于每图 GT≥301、640 输入短边 8--16 px，两个域同向。

- 已同时保存独立 GT coverage 与正式 COCO 一对一匹配，二者禁止混称。
- 已跟踪 Decoder 全 640 的 IoU、score、rank、Top-300 生存与一对一匹配身份，并按
  domain、密度、短边及 evaluator/hybrid cohort 分桶。
- 已按十个 COCO IoU 阈值与 `maxDets=300/640` 复算 AP/AR 和全局 PR；只有 10×101
  PR precision 差能精确加和为 AP 差，局部 TP/FN cause count 不能。
- raw 的七轮 AP-small 范围约 0.008，EMA 约 0.027；未经受控干预不得把 EMA 写成
  唯一根因。

验收：**已通过（限本地 annotation 与 Windows CPU 后端）**。full-val 中每个状态变化
可追踪到 generation、selection、Decoder 几何、最终 score/rank 截断或一对一匹配
竞争；服务器 H20/训练 annotation 的位级复现不属于本次已通过范围。

## Gate 2：selector 诊断对照

状态：**尚未开始干预对照**。完整集已把直接变化定位到最终 Decoder score/rank，但
还没有证明应修改 Encoder selector；先完成 SGQS/B0 训练探针配对分析，再决定是否
保留以下比较。

在不训练、同一冻结输出、同 K 下比较：

1. 原始 global Top-K；
2. 简单 per-level calibration；
3. 简单 per-level quota；
4. 去重/coverage-aware selector；
5. GT-aware oracle；
6. K sweep，仅用于判断容量，不作为主方法。

验收：主假设必须相对简单对照和 oracle 定位出可改进空间。

## Gate 3：确定单一方法分支

状态：**等待配对探针与单变量干预**。full-val 支持 Top-300 admission 是直接承载阶段，
但不能仅凭排名跨界或两臂终点指标就选择 selector、分类校准、EMA 或 SGQS owner/GO
中的任一分支。

- generation 主导：研究 P2/形状先验/表示；
- selection 主导：研究 level-calibrated 或 coverage-aware initialization；
- decoder 主导：研究匹配/质量校准；
- output 主导：修正后处理或评估，不包装成训练创新。

只选择一个由证据支持的主组件，其他作为消融或未来工作。

## Gate 4：训练与论文验证（屏选完成，正式验证待修订）

状态：**单 seed checkpoint 续训屏选、SGQS/B0 同探针训练与 SGQS full-val 诊断已完成；正式方法验证未开始**。B0 回放、参数级单变量干预及从头多 seed 训练均未完成。

相同初始化、global batch、epoch/steps、seed 和 `maxDets=300` 下比较 baseline 与方法。先 seed 0 验证方向，再三 seed；报告早期同 epoch、最终指标、wall-clock、GPU-hours、参数/FLOPs 和固定延迟协议。
