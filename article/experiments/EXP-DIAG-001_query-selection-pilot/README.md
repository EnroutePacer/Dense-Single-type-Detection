# EXP-DIAG-001：无 GT Query Selection 先导诊断

## 目的

观察全部 encoder 候选、Top-640 来源与 decoder 后高置信存活行为，为带 GT 归因提出可证伪假设。

## 观察

- 13 图的 8,320 个 Top-K query 来源为 P3/P4/P5=3,356/4,062/902。
- 1,432 个 `score>0.5` 输出来源为 3/1,429/0。
- predicted-small 与来源层只能作为代理，不能代表 GT 尺度或 TP。

## 结论边界

本实验只支持“selection 与 survival 存在值得研究的不均衡”，不能证明真实小框在 Top-K 或 decoder 中漏检。下一步是 [EXP-DIAG-003](../EXP-DIAG-003_gt-stage-attribution/README.md)。历史解释保存在 [legacy_record.md](legacy_record.md)，其中过强的 decoder 定因已被 canonical 研究问题取代。
