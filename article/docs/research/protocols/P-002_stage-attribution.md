# P-002：GT 对齐的分阶段归因协议

版本：1.0（2026-10-10）

## 固定条件

- 冻结 checkpoint、输入预处理和模型参数；
- 输入尺寸 640×640；
- 原始 selector 为 global Top-K，`K=640`；
- 机制统计发生在 score threshold、postprocess Top-K、NMS 和 COCO `maxDets` 之前；
- 所有候选统一映射回同一图像坐标系。

## 输出层级

逐图保存：全部 8,400 dense proposals、Top-K 索引、来源层、encoder score、每层 decoder box/score、最终 postprocess box，以及 GT。

## 匹配与指标

- 在预注册 IoU 阈值集合上计算 dense/Top-K/decoder/output recall；
- 同时报告 per-GT best-IoU 和一对一二分图 unique coverage；
- `SSR_τ=P(C_topK=0 | C_dense=1)`；
- 对同一 GT 的候选冗余、最佳 rank、层级和分数校准进行统计；
- 分桶使用短边像素、短边/stride、面积、长宽比、局部密度、图像 GT 数和域。

不得使用预测面积代理替代 GT-small，也不得把 P3/P4/P5 当作目标尺度标签。
