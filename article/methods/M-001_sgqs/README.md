# M-001 — Scale-aware GT Query Survival

**Status:** `implemented / unvalidated`

SGQS 是面向密集单类别检测的训练期方法。它检验的假设是：Encoder 已经产生部分具有几何能力的候选，但密集、细长、低像素短边目标会因跨层 global Top-K 排名和重复候选占位而失去 Decoder 初始化种子。

## 核心机制

1. 根据增广完成后的 GT 像素短边在线分配 P3/P4/P5。
2. 为每个 GT 分配唯一、有效的 owner point，并处理相邻 GT 冲突。
3. 在 Top-K 前对全部有效 Encoder 分类 logits 施加分层 dense supervision。
4. 只 gather owner memory，复用原 `enc_bbox_head` 做 L1/GIoU 回归。
5. 以第 \(K\) 大选择 logit 为门槛，用带 \(q_{\min}\) 下限的质量权重施加生存约束，使 IoU 为 0 的弱 owner 仍有梯度。
6. 单独记录 owner 覆盖、P3/P4/P5 分布、cutoff gap 和真实 Top-K 生存率，不把诊断指标混入总损失。
7. 推理不需要 GT，且不增加参数、query 或预测头。

完整公式、归一化、边界条件和总损失见 [losses.md](losses.md)；完整调用链、伪代码、梯度、配置、日志、公平对照与风险见 [design.md](design.md)。

## 实现位置

| 文件 | 作用 |
|---|---|
| `engine/sgqs/candidate_supervision.py` | 在线 owner 分配、冲突消解、ignore ring、owner 框和 Top-K 门槛。 |
| `engine/sgqs/criterion.py` | dense classification、owner L1/GIoU、相对质量加权 survival 和总损失组合。 |
| `engine/deim/dfine_decoder.py` | 在 `enc_score_head` 后、global Top-K 前调用可注入的 SGQS supervisor。 |
| `engine/solver/det_engine.py` | 将 SGQS 机制指标写入控制台和 TensorBoard 的 `Metric/` 命名空间。 |
| `configs/sgqs/sgqs_hgnetv2_m_table_cell_fixed640.yml` | clean fixed-640 SGQS 正式配置。 |
| `configs/sgqs/sgqs_hgnetv2_m_table_cell_fixed640_probe.yml` | 从 epoch-27 checkpoint 继续的 probe 配置。 |
| `tests/test_sgqs.py` | 分层、冲突、空 GT、预算溢出、零 IoU 生存梯度、注册、invalid anchor 与推理等价测试。 |

SGQS 通过注册模块注入现有 DEIM/D-FINE 框架：

```text
engine/sgqs/
  SGQSQuerySupervisor
  SGQSCriterion(base_criterion=DEIMCriterion)

configs/sgqs/
  -> DFINETransformer.sgqs_supervisor = SGQSQuerySupervisor
  -> criterion = SGQSCriterion
```

基础 Matcher、Decoder query budget、PostProcessor 和 evaluator 均不修改。SGQS supervisor 无可学习参数，eval 分支不执行；同一代码版本、同一模型权重下，关闭或开启 supervisor 的 eval 输出必须逐元素一致。

当前提交另有一项共享 correctness fix：无效 boundary anchor 的选择 logit 在 global Top-K 前置为 \(-\infty\)。它对 control 和 SGQS 都生效，会改变旧 selector 的结果，不属于 SGQS。所有正式对照必须用同一当前提交重跑，不能直接拿历史 baseline 归因。

## 当前证据边界

实现通过 11 项单元测试，只证明代码链路和梯度符合设计，不证明方法有效。正式结论仍需原始 annotation JSON 的逐 GT 阶段归因、同预算消融、至少三个随机种子和固定 `maxDets=300` 评估。

满足任一情况时不得宣称方法有效：

- selection coverage、最佳 candidate rank 和 starvation 指标没有改善；
- 选择指标改善，但相同步数的 mAP/AP75/AR 没有改善；
- 增益来自更大 query budget、更多 steps、不同增强或 invalid-anchor 修复；
- 只改善局部指标，却显著损伤总 mAP 或中大目标；
- 推理输出路径或固定 CPU latency 发生变化。
