# 科研日志

`article/` 只维护密集单类别检测研究的进展、实验解释、方法推导和证据索引；
不再复制训练代码、配置、测试、权重或派生数据。

## 最新进度

- [2026-10-24：SGQS v2 转向实例覆盖与候选预算](journal/2026-10-24.md)
- [活跃设计 M-003](methods/M-003_sgqs_v2/README.md) / [新协议 EXP-TRAIN-008](experiments/EXP-TRAIN-008_sgqs-v2-coverage-allocation/README.md)

- [2026-10-19：升温/末端干预、本地 small AP 与全验证集矩阵](journal/2026-10-19.md)
- [已取代的历史方案：SGQS v2 单因素续训](experiments/EXP-TRAIN-007_sgqs-v2-single-factor/README.md)
- [已完成：SGQS 损失假说与实例证据审计](experiments/EXP-DIAG-007_sgqs-loss-hypothesis-audit/README.md)
- [已完成（本地全量口径）：SGQS 14 状态全验证集回放](experiments/EXP-DIAG-006_sgqs-full-val-checkpoint-replay/README.md)
- [2026-10-17：启动冲击、监督链路与尺度归一化审计](journal/2026-10-17.md)
- [2026-10-15：SGQS/B0 配对探针完成与 Top-300 排名诊断](journal/2026-10-15.md)
- [2026-10-14：三端校对、探针/C4 中断状态与目录清理](journal/2026-10-14.md)
- [累计科研进展](docs/research/progress_log.md)
- [当前研究状态](docs/research/status.md)
- [研究问题](docs/research/research_question.md)

## 记录入口

- [实验注册表](experiments/registry.csv)
- [候选方法](methods/README.md)
- [证据索引](docs/evidence/evidence_index.md)
- [事实与假设](docs/research/facts/README.md)
- [论文主张—证据矩阵](docs/claims/README.md)

实体文件通常放在仓库根布局：运行代码与配置位于 `engine/`、`configs/`、
`tools/`、`tests/`，原始训练结果与大体积派生数据位于 `output/` 或 `outputs/`。
经明确要求发布且已脱敏的自包含诊断包，可与对应 EXP 记录同置于
`article/experiments/`；普通文件必须低于 10 MiB，确需发布的较大派生表使用精确路径
Git LFS 规则和逐文件 manifest。不得借此复制权重、原始数据或未经筛选的 replay 输出。
历史但仍需保留的非活跃配置和 launcher 位于
`outputs/provenance/retired_article_entities_20261014/`。

每条结论必须指出实验口径、证据位置和可支持/不可支持的主张；中断、无效或
被取代的实验保留记录，但不得包装成方法收益。
