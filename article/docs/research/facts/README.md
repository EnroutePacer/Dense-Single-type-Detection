# 事实与观察注册表

状态分三类：`FACT` 可由代码或原始产物直接核验；`OBSERVATION_NO_GT` 是没有 GT 的行为观察；`OBSERVATION_RECOVERED_GT` 来自 GT 可视化恢复，可用于立题，但不能替代原始 annotation JSON 的正式评估。

| ID | 状态 | 内容 | 证据边界 |
|---|---|---|---|
| F-001 | FACT | 640×640 输入产生 P3/P4/P5 共 8,400 个位置，当前模型按单类 encoder logit 做 global Top-640。 | 静态代码与诊断导出可核验；不代表所有位置都能形成高质量 GT proposal。 |
| F-002 | FACT | 标准前向只对 Top-K 位置构造 encoder 初始框；诊断脚本额外导出全部位置的预测框。Top-K 索引不可微，decoder 初始化被 detach。 | decoder 仍读取完整 memory，不能说未入选位置完全不可见。 |
| F-003 | OBSERVATION_NO_GT | 13 图的 Top-K 来源为 P3/P4/P5=3,356/4,062/902；1,432 个 `score>0.5` 输出的来源为 3/1,429/0。 | 分数阈值输出不等于 TP 或 recall。 |
| F-004 | OBSERVATION_NO_GT | `wired/image (12)` 的 Top-640 来源为 153/425/62；118 个高置信输出全部来自 P4。 | 示例输入没有配套 GT。 |
| F-005 | OBSERVATION_NO_GT | predicted-area `<32²` 的 11,881 个 dense 代理中，704 个进入 Top-K；Top-K 外存在大量与入选框重叠的代理候选。 | `predicted-small` 不是 GT-small，未入选代理不能直接计作漏检。 |
| F-006 | FACT | 当前 13 张诊断图没有配套 GT JSON，无法确定真实漏检所在阶段。 | 这是全部机制解释的硬边界。 |
| F-007 | FACT | 正式 COCO 主评估采用 `maxDets=[1,10,300]`；机制诊断在 evaluator 截断前完成。 | 评估协议和机制协议必须分开。 |
| F-008 | FACT | 旧 small-object probes 的 shell 脚本覆盖了外部 `CONFIG_FILE`；它们不是有效损失消融。 | 数值可保留为历史 run，不可归因于 small loss。 |
| F-009 | OBSERVATION_RECOVERED_GT | `image (12)` 的 GT 可视化含 322 个框，其中正文 304 个；正文框在 640 输入下短边约 12.6～14.9 px，P4/P5 短边均不足 1 个 feature cell。 | 框由可视化轮廓恢复并配准，最终数字须由原始 COCO JSON 复算。 |
| F-010 | OBSERVATION_RECOVERED_GT | 正文框 IoU≥0.5 的 dense/Top-640 覆盖约为 278/304 与 135/304；P3 对应约为 265/304 与 60/304。 | 支持 selection starvation 立题，不是正式 dataset-level recall。 |
| F-011 | OBSERVATION_RECOVERED_GT | 304 个正文框只有约 7.2% 属于 COCO-small，多数困难框是低像素短边的细长目标。 | AP-small 不能单独代表本文问题，需 short-side/stride 与长宽比分桶。 |
| F-012 | FACT | 标准代码只对全部 8,400 个位置计算 encoder score；bbox head、Hungarian 与 encoder auxiliary loss只作用于 Top-K。 | Top-K 外位置没有直接的 proposal 分类/回归/排名监督，但仍可能从共享特征获得间接梯度。 |
| F-013 | OBSERVATION_NO_GT | 三个声称正确配置的 12,800-step 目录只有 metrics 摘要，没有 console/resolved config；small-reg 未提升 AP-small/AR-small，MAL+Dense O2O 整体退化。 | 数值可记录；在补回运行时配置证据前，不能严格归因于声称方法。 |

新增事实必须给出代码位置、run ID 或 artifact ID；目录名和口述不构成事实证据。
