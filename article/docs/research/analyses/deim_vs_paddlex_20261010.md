# DEIM 与 PaddleX-L 差异分析

> **NO-GT scope**：本文只能解释输出形态和提出候选机制，不能定位真实漏检阶段。canonical 边界见 [研究问题](../research_question.md)。

## 已排除的解释

没有证据表明 PaddleX 为表格单元格另外设计了专用 loss。落地配置仍是 RT-DETR 常见的 VFL、L1、GIoU、Hungarian matching、auxiliary 和 denoising loss。当前 DEIM/D-FINE 配方甚至额外包含 FGL 与 DDF。

## 更可信的原因

1. PaddleX wired/wireless 使用两份不同权重；DEIM 同一权重覆盖两个域。
2. PaddleX 是面向该任务发布的成熟权重；DEIM 当前 checkpoint 在 epoch 27，且训练链没有启用完整 DEIM 配方。
3. PaddleX-L 有 6 层 decoder；当前 D-FINE-M 表格配置为 4 层。即使忽略参数总量，多两轮框 refinement 也可能改善边界。
4. 两者的训练数据、标注规范和领域预训练不等价。
5. DEIM 输出 640 queries，PaddleX 输出最多 300；同一 0.5 阈值不是相同 operating point，DEIM 更易暴露低校准或重复 query。
6. 当前无 GT query 诊断显示，一部分 predicted-small 代理 query 已进入 decoder，但最终分数较低；这不能证明对应真实小框，也不能判定主要失败发生在 decoder。

## 不能从现有 13 图得出的结论

- 不能计算 mAP 或严格 recall。
- 框多不等于更准确。
- 内部重叠对数只是重复框代理，不是 false-positive 数。
- 不能把视觉优势归因于任一单项配置。

## 严格比较需要

- 同一带 GT 测试集；
- 每个模型按自身正确预处理；
- PR 曲线或阈值扫描，而不是只固定 0.5；
- wired/wireless 分域和合并指标；
- 按 GT 数、面积和结构复杂度分桶；
- 参数量之外同时报告 FLOPs、runtime/backend 和 decoder/query 配置。

详细拼图与逐图统计位于 `../03_evidence/local_inference_benchmark_20261010/contrary/DEIMvsPaddlex/analysis.md`。

