# EXP-EVAL-001：maxDets 100/300 复评

## 问题

密集单元格数据上，COCO 默认 `maxDets=100` 是否系统性低估同一 checkpoint 的指标？

## 结果

| 口径 | mAP@50:95 | 说明 |
|---|---:|---|
| maxDets=100 | 0.566899 | 旧训练日志口径 |
| maxDets=300 | 0.864247 | 修复 evaluator 后复评 |

结论：100 个输出上限会在高密度图上制造结构性截断，论文主评估必须显式写 `maxDets=300`。由于 checkpoint 和标注 JSON 尚未归档，本实验有效性标为 `partial`；理论上限数字也不能作为最终论文事实。

历史记录见 [legacy_record.md](legacy_record.md)，协议见 [P-003](../../docs/research/protocols/P-003_maxdets300.md)。
