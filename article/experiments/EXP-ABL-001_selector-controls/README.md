# EXP-ABL-001：固定 K 的 Selector 诊断对照

尚未运行；仅在 EXP-DIAG-003 证明 selection gap 后推进。

同一冻结模型、同一 8,400 dense 输出、同一 `K=640` 下比较原始 global Top-K、简单 per-level score calibration、简单 quota、去重/coverage-aware selector 与 GT-aware oracle。K sweep 只用于估计容量上限，不作为主方法。

主要指标是 unique-GT coverage、SSR、有效 query 利用率和分桶 recall。若简单 calibration/quota 已消除大部分差距，则不应再把复杂 selector 包装成必要贡献。
