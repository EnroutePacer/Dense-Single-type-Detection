# I008/I009：定性对比与 query 诊断

> **Historical snapshot / superseded**：这是当时合并记录，其中“decoder 是更直接线索”的解释没有 GT 支持。当前只把它视为 NO-GT 先导观察；请以本实验 [README](README.md) 和 [canonical 研究问题](../../docs/research/research_question.md) 为准。

## DEIM vs PaddleX-L

输入为相同 13 张图片，阈值均为 `score > 0.5`。PaddleX wired/wireless 使用各自领域权重，DEIM 使用同一 epoch-27 checkpoint。

主要观察：

- PaddleX 在若干密集表格上框覆盖更完整，尤其 `wired/image (12).png`。
- 以预测框内部两两 IoU≥0.8 作为重复框代理，DEIM 为 24 对，PaddleX 为 2 对。
- 该代理没有 GT，不能证明 precision/recall，只能定位后续需要验证的失败模式。
- PaddleX 配置没有发现边线、拓扑或表格专用 loss；视觉优势不能简单归因于“单元格特化损失”。

## query 诊断

诊断绕过仓库 PostProcessor，保存每张图全部 640 个 decoder query，同时抓取 encoder Top-K 前全部 8,400 个候选。

汇总：

- 13 图共有 109,200 个 dense encoder 候选，选入 8,320 个 decoder query。
- decoder query 中 1,331 个呈面积小于 32² 的 small 形状。
- 其中 739 个分数 >0.01，38 个 >0.10，22 个 >0.50。
- 1,331 个 small query 来源：P3 1,065，P4 205，P5 61。
- small query 的 encoder score 与最终 score Spearman 相关为 0.3429。
- 存在 encoder 高分小框经 decoder 后降至约 0.01 的明确实例。

## 研究含义

当前证据反驳了“640 个名额不够是唯一主因”的强说法。模型能够生成并选入大量小形状 query；更直接的线索是 decoder 分类/匹配学习和最终置信度。下一步必须引入 GT，逐真实框计算：

1. encoder 最佳 IoU 与 rank；
2. 是否进入 Top-K；
3. Hungarian 匹配对象和匹配 IoU；
4. 每层 decoder 的框、类别分数与梯度；
5. 最终漏检属于候选缺失、Top-K 截断、错误匹配还是 decoder 降分。

## 证据

- `../03_evidence/local_inference_benchmark_20261010/contrary/DEIMvsPaddlex/`
- `../03_evidence/local_inference_benchmark_20261010/output/DEIM_query_diagnostic_no_post_topk_640/`

