# EXP-DIAG-003：带 GT 的分阶段归因

当前最高优先级，尚未运行。

2026-10-12 已用用户提供的 GT 可视化完成一次恢复框先导审计，并观察到明显的 dense→Top-K 覆盖下降。由于没有原始 COCO JSON、category/image ID 和精确标注坐标，该先导结果不改变本实验的 `planned` 状态，也不能替代正式运行。

## 研究问题

真实漏检主要发生在 dense generation、global Top-K selection、decoder refinement 还是 output/evaluator？

## 固定条件

- 同一冻结 checkpoint、预处理、640 输入、`K=640`；
- 机制诊断不受 score threshold、postprocess Top-K、NMS、`maxDets` 截断；
- 同时用 per-GT best-IoU 和一对一 unique coverage；
- 按短边、短边/stride、面积、长宽比、密度和域分桶。

## 预注册判定

- dense 低：支持 generation 问题；
- dense 高、Top-K 明显下降：支持 selection starvation；
- Top-K 高、decoder/final 下降：支持 survival collapse；
- 没有配套 GT、坐标映射不一致或候选导出改变模型前向：实验失效。

执行前必须补齐 `experiment.yaml` 中的数据、checkpoint 与 artifact ID。
