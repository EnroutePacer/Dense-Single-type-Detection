# 研究状态

> **Historical snapshot / superseded（2026-10-10）**：本文保留当时的研究判断，不属于当前 canonical 结论。当前训练已暂停，优先做带 GT 的 generation→selection→decoder→output 归因。请以 [研究问题](../research_question.md) 和 [当前状态](../status.md) 为准。

更新时间：2026-10-10

## 研究目标

以表格单元格检测为验证基座，研究密集、单类别场景中的训练监督、匹配质量、query 容量、尺度建模和置信度校准，最终形成可复现的论文级方法与实验闭环。

## 已完成

- 固定 CPU 基准协议下完成 PaddleX-L 与 DEIM 的 wired/wireless 推理和可视化。
- 完成 DEIM 512、640、800 单尺度，多尺度、分块和组合推理的输出保存。
- 完成无输出 Top-K 的 640-query 诊断，并抓取 encoder Top-K 前全部 8,400 个候选。
- 把 COCO evaluator 的 `maxDets` 恢复为 `[1,10,300]`，确认 M epoch-27 checkpoint 的正确复评 mAP 为 0.864247。
- 保存服务器源码、配置、日志、TensorBoard 与 COCO evaluator 快照。
- 审计出两个决定性配置问题：未包含完整 DEIM 配方；训练脚本忽略外部 `CONFIG_FILE`。
- 将本地基准输入、输出、模型权重和原始服务器归档收进本目录。

## 当前结论边界

可以下结论：

- `maxDets=100` 严重低估此密集数据集上的 AP，主评估必须使用 300。
- 当前 M 表格训练不是完整 DEIM；它使用 VFL，而不是 MAL。
- 旧的 `small_vfl`/`small_reg` 探针没有实际启用对应配置，不能评价损失加权是否有效。
- 现有无 GT query 诊断表明，小框问题不能简化为“640 个 decoder query 不够”；大量小形状 query 已进入 decoder，但在失败图上最终分类分数极低。
- PaddleX 的视觉优势没有证据表明来自“表格专用损失”；更可能与分域权重、任务数据、训练成熟度、decoder 深度和校准有关。

暂时不能下结论：

- MAL 或 Dense O2O 是否能提升当前表格数据集；还没有正确运行过。
- 小框 VFL/回归加权是否有效；旧实验配置未生效。
- 512/800 多尺度或分块推理是否提高 mAP；当前 13 张图片没有 GT。
- concat4×320 是否有效；尚未完成受控训练，而且可能触发 640-query 容量竞争。

## 资源闭环

当前资源足以支持方法研究的工程闭环：源码可改、四卡训练、COCO 评估、固定 CPU 延迟、推理诊断和完整日志归档均具备。论文级闭环仍缺：

- 数据集 JSON/划分的不可变快照与哈希；
- 训练 checkpoint 和 probe checkpoint；
- Git commit 或完整 diff；
- 环境锁定信息与 GPU/CUDA/PyTorch 版本；
- 至少 3 个随机种子的最终关键实验；
- 带 GT 的错误分解和 query survival 诊断；
- 第二个密集单类别数据集或可公开的合成验证集。

## 下一里程碑

先修正脚本配置注入并建立两个干净基线：D-FINE/VFL 与完整 DEIM。每个实验启动后，必须在第一个 batch 前自动断言解析配置包含预期的 Resize、scales、losses、Mosaic/MixUp 和 scheduler。随后再决定是否投入 concat4×320、小目标回归或 query-budget 方法。

