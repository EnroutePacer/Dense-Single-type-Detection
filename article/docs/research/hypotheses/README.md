# 可证伪假设注册表

| ID | 假设 | 关键证伪实验 |
|---|---|---|
| H-001 | 跨层 encoder score 不可直接比较，global Top-K 对高分辨率候选产生选择偏置，降低困难 GT 的 unique coverage。 | 带 GT 比较 dense 与 Top-K recall、各层 score-IoU 校准；若没有显著 drop，则否定。 |
| H-002 | 同一目标周围的高分重复候选占用固定 K，造成 coverage starvation。 | 同 K 下做二分图 coverage 和去重/coverage-aware selector；若 unique coverage 不升，则否定。 |
| H-003 | 低分位置不入选且缺少直接 encoder 匹配/回归监督，形成 selection-starvation 闭环。 | 记录选中/未选位置的梯度与最佳 GT candidate rank 随 epoch 的轨迹。 |
| H-004 | 一部分 GT 已被 Top-K 覆盖，但在 decoder 细化或重评分中发生 survival collapse。 | 逐 GT、逐 decoder layer 跟踪 IoU、score 和匹配身份；没有阶段性下降则否定。 |
| H-005 | 部分失败在 Top-K 前已发生，根因是 stride-8 表示、anchor/形状先验或输入分辨率不足。 | 全部 8,400 个 dense proposals 对困难 GT 的 recall 若已高，则否定。 |
| H-006 | VFL 的 IoU 质量目标使短边小、细长框的排序分数更易受微小定位误差影响。 | 按短边/长宽比/层级比较校准，并与 MAL 或独立 objectness 做严格对照。 |

EXP-DIAG-006 的本地完整集证据更新了优先级：前段收益与后段回跌的正式
evaluator-small 转移有约 88%--93% 由 Decoder 合格 query 跨越 Top-300 排名线直接
承载，因此 H-004 的“重评分/输出生存”分支获得 scoped 支持，而“Decoder 几何普遍
退化”不受支持。该结果没有证明 H-001/H-003 是本次曲线的上游原因，也没有区分
H-006、EMA 轨迹与 SGQS 训练项。下一步以训练探针和单变量干预区分这些竞争解释；
在此之前仍不建立“已验证方法”。

EXP-DIAG-007 进一步把三个训练级解释分级：恢复训练 step 0 的 SGQS 新增 loss 为
`5.071082`，其中 dense 为 `4.702855`，所以“无升温启动冲击”获得直接支持；但最终总
AP 高于配对 B0，且没有 train/val 同口径泛化曲线，不能称“几近过拟合或整体性能被
侵蚀”。SGQS owner coverage、IoU、pre-TopK survival 在 e30→34 继续改善，而正式
AP-small 回跌，支持 H-004 的末端代理脱节分支；基础 Encoder/Decoder loss 同期继续
下降，否定“e31 基础 loss 明确回升”的叙事。dense 的确按 P3/P4/P5 层均值固定等权，
P3/P5 kept-owner 数比由约 70× 增至 135×，是高优先级可证伪机制，但尚无分层共享参数
梯度证明它造成回跌。详细边界见
[EXP-DIAG-007](../../../experiments/EXP-DIAG-007_sgqs-loss-hypothesis-audit/README.md)。
