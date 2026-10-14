# 科研日志

`article/` 只维护密集单类别检测研究的进展、实验解释、方法推导和证据索引；
不再复制训练代码、配置、测试、权重或派生数据。

## 最新进度

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

实体文件统一放在仓库根布局：运行代码与配置位于 `engine/`、`configs/`、
`tools/`、`tests/`，训练结果与派生数据位于 `output/` 或 `outputs/`。历史但
仍需保留的非活跃配置和 launcher 位于
`outputs/provenance/retired_article_entities_20261014/`。

每条结论必须指出实验口径、证据位置和可支持/不可支持的主张；中断、无效或
被取代的实验保留记录，但不得包装成方法收益。
