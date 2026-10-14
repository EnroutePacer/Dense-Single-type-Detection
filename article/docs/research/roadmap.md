# 研究路线图

当前优先路线由 [ADR-003](decisions/ADR-003_training-first-sgqs-e.md) 更新：SGQS-E 已实现，B0--B3/C2 单 seed checkpoint 续训屏选已经完成；当前先执行 [EXP-TRAIN-006](../../experiments/EXP-TRAIN-006_sgqs-supervision-probes/README.md) 定位 DN-DDF、GO/owner、候选排名和 raw/EMA 泛化链路，再依据证据决定最小损失修订与正式从头多 seed 对照。以下 Gate 0–4 保留为诊断路线，其中“必须先完整归因”和“训练暂停”不再是当前门禁。

## Gate 0：资产与口径

- 取得验证 GT、split、checkpoint，保存 SHA-256 与许可状态。
- 固定输入 640、`K=640`、模型权重和后处理。
- 将现有诊断脚本升级为 GT-aware 导出，先写最小合成单测。

验收：同一图可以从 dense proposals 一直追踪到 Top-K、每层 decoder 和 output；所有产物含 image/category 映射和配置哈希。

## Gate 1：冻结模型的分阶段归因

- 计算 dense recall、Top-K recall、SSR 与 unique-GT coverage。
- 记录最佳 candidate rank、来源层、重复候选、短边与密度分桶。
- 跟踪各 decoder layer 的 IoU、score 和匹配身份。

验收：每个漏检可唯一归入 generation、selection、decoder 或 output；未知项单独列出。

## Gate 2：selector 诊断对照

在不训练、同一冻结输出、同 K 下比较：

1. 原始 global Top-K；
2. 简单 per-level calibration；
3. 简单 per-level quota；
4. 去重/coverage-aware selector；
5. GT-aware oracle；
6. K sweep，仅用于判断容量，不作为主方法。

验收：主假设必须相对简单对照和 oracle 定位出可改进空间。

## Gate 3：确定单一方法分支

- generation 主导：研究 P2/形状先验/表示；
- selection 主导：研究 level-calibrated 或 coverage-aware initialization；
- decoder 主导：研究匹配/质量校准；
- output 主导：修正后处理或评估，不包装成训练创新。

只选择一个由证据支持的主组件，其他作为消融或未来工作。

## Gate 4：训练与论文验证（屏选完成，正式验证待修订）

相同初始化、global batch、epoch/steps、seed 和 `maxDets=300` 下比较 baseline 与方法。先 seed 0 验证方向，再三 seed；报告早期同 epoch、最终指标、wall-clock、GPU-hours、参数/FLOPs 和固定延迟协议。
