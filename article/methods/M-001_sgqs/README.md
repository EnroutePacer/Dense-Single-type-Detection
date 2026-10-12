# M-001 — SGQS

**Status:** `candidate / unvalidated`  
**Name:** Scale-aware GT Query Survival（尺度感知 GT Query 生存监督）

SGQS 是针对 dense single-type detection 的候选训练方法，不是已验证贡献。它要检验的假设是：小目标的主要瓶颈发生在 decoder 之前——正确的 P3 候选未能进入 global Top-K，后续匹配和回归因而无法直接修正该位置。只有完成带 GT 的 selection 指标、同预算消融和多种子复现后，才能升级其状态。

## 问题与代码链路

固定输入 640 时，P3/P4/P5 分别产生 `80² + 40² + 20² = 8400` 个位置，当前配置仅选择 640 个 query：

1. [HybridEncoder](../../engine/deim/hybrid_encoder.py#L395) 生成 P3/P4/P5；[DFINETransformer._get_encoder_input](../../engine/deim/dfine_decoder.py#L579) 将其展平并拼接。
2. [_generate_anchors](../../engine/deim/dfine_decoder.py#L604) 生成三层固定网格 anchor；[_get_decoder_input](../../engine/deim/dfine_decoder.py#L632) 对全部位置计算 `enc_score_head`。
3. [_select_topk](../../engine/deim/dfine_decoder.py#L681) 以跨层 global Top-K 选出 640 个位置；单类别时排序量就是唯一 objectness logit。
4. `enc_bbox_head` 只作用于已入选位置，且选中的 content 和 reference 在进入 decoder 前被 `detach`。
5. [DEIMCriterion.forward](../../engine/deim/deim_criterion.py#L325) 只对 `enc_aux_outputs` 中的 Top-K 候选做 Hungarian 匹配。现有 small-object VFL、L1/GIoU 和 FGL 加权均发生在匹配之后，不能直接监督落榜位置。

因此，正确但未入选的位置没有位置级分类、回归或生存排名梯度；它只能从共享参数获得间接更新。这是已由代码确认的机制，但“它是当前精度差距的主因”仍是待验证假设。

## 方法

### 1. 在线短边 GT 点分配

点标签必须根据**完成数据增广后的**归一化 GT 在线生成，不能把固定点写死在原始标注中，否则 Mosaic、Crop、Resize、Flip 后会错位。

对 GT `g=(cx, cy, w, h)`，以短边而非面积衡量解析难度：

```text
d_g = min(w_g, h_g)
l_g = argmin_l |log((d_g + eps) / (a_l + eps))|
```

其中 `a_l` 是 P3/P4/P5 的尺度先验；固定 640 下与当前 anchor 对应约为 32、64、128 px。将 GT 中心投影到 `l_g` 的网格中心，为每个 GT 分配一个唯一、有效的 owner point。若多个 GT 冲突，则按短边从小到大优先，在局部邻域选择下一最近空闲点；owner 周围一圈设为 ignore，避免合理邻点被当作强负样本。

### 2. 全候选分数监督

保留全部有效位置的 encoder logits，不先经过 Top-K。owner point 是正样本，ignore ring 不参与负样本损失，其余位置为负样本。正样本质量目标为：

```text
q_g = q0 + (1 - q0) * stopgrad(IoU(b_g, GT_g))
```

建议初始 `q0=0.25`，避免训练早期低 IoU 把正点目标压到零。每层分别计算 VFL/Focal，并对正样本和 hard negatives 独立归一化后再平均三层，防止数量最多的 P3 负样本主导梯度：

```text
L_dense = (1 / 3) * sum_l [mean(VFL_pos_l) + beta * mean(VFL_hard_neg_l)]
```

### 3. Owner point 回归

从 `output_memory` gather owner feature，复用现有 `enc_bbox_head`，不为方法增加新预测头：

```text
L_point_reg = (1 / G) * sum_g [5 * L1(b_g, GT_g) + 2 * (1 - GIoU(b_g, GT_g))]
```

该损失绕过离散 Top-K 和 Hungarian，直接把梯度送到正确候选位置、P3/P4/P5 特征、HybridEncoder 和 backbone。

### 4. Top-K 生存损失

令 `theta_K` 为当前样本全部有效候选中的第 K 大选择 logit，并停止其梯度；`z_g` 为 owner point 的 logit：

```text
theta_K = stopgrad(kth_largest({z_i}, K))
L_survive = (1 / G) * sum_g softplus(theta_K + margin - z_g)
```

它直接优化“每个 GT 至少有一个正确候选越过 Top-K 截止线”。若单图 `G > K`，该目标不可能全部满足，必须单独统计这种样本并提高 query budget 或定义 capped coverage，不能掩盖预算上限。

总损失为：

```text
L = L_DEIM + lambda_dense * L_dense
           + lambda_reg   * L_point_reg
           + lambda_surv  * L_survive
```

首轮建议 `lambda_dense=1.0`、`lambda_reg=0.5`、`lambda_surv=0.25`；`L_survive` 在 warmup 后线性开启，数值仅是待消融起点。

## 无 GT 泄漏与推理成本

- GT 只用于训练期构造监督标签，不作为图像通道、特征输入或推理条件。
- 分支必须由 `self.training and targets is not None` 控制；验证、部署和导出仍使用模型自身的 global Top-K。
- 首版不向 decoder 强制注入 GT query，避免训练/推理选择路径不一致。
- 复用 `enc_score_head` 与 `enc_bbox_head`，预期不增加参数量；训练只额外计算 owner 回归和 dense loss。
- 推理分支应与基线逐输出核对，并用固定 CPU 协议复测 latency。只有输出链路和 latency 均不变，才可声称“推理零额外开销”。

## 计划改动文件

| 文件 | 计划改动 |
|---|---|
| `engine/deim/candidate_supervision.py` | 新增在线层分配、owner/ignore 构造、碰撞消解。 |
| `engine/deim/dfine_decoder.py` | 训练时保留 dense logits、返回 Top-K 索引、gather owner memory 并产生 `candidate_aux_outputs`；eval 路径不变。 |
| `engine/deim/deim_criterion.py` | 新增 `L_dense`、`L_point_reg`、`L_survive`；不进入现有 Hungarian auxiliary 循环。 |
| `configs/table_cells/dfine_hgnetv2_m_table_cell_fixed640_sgqs_probe.yml` | 独立配置方法开关、损失权重和 warmup，不覆盖 canonical baseline。 |
| `tests/test_candidate_supervision.py` | 覆盖层索引、碰撞、空 GT、`G>K`、无效 anchor、eval 等价和落榜 owner 非零梯度。 |

另需先核对 `valid_mask`：当前路径将无效 memory 清零，但没有显式将其选择分数置为 `-inf`。若修复，必须对所有实验组统一应用并单列为 correctness fix，不能计入 SGQS 增益。

## 首轮消融

所有组使用相同初始化、数据顺序、global batch、optimizer steps、增强、query 数和 `maxDets=300`：

| 组别 | Dense score | Point reg | Survival | 目的 |
|---|---:|---:|---:|---|
| A | — | — | — | 严格 baseline |
| B | ✓ | — | — | 检验落榜位置直接分类监督 |
| C | ✓ | ✓ | — | 检验候选生成/定位质量 |
| D | ✓ | ✓ | ✓ | 完整 SGQS |
| E | ✓ | ✓ | ✓ | 用面积分层替换短边分层，验证细长目标假设 |

选择阶段必须报告 small-GT coverage@640、Selection Starvation Rate、每 GT 最佳候选 rank、unique-GT coverage、P3/P4/P5 入选分布；检测阶段报告相同 epoch/steps 的 mAP、AP75、AP_small、AR_small。筛选可做单种子 probe，论文结论必须完整训练、至少 3 个种子并报告均值与方差。

## 失败条件

满足任一项时，不得宣称 SGQS 有效：

1. small-GT coverage@640、最佳 rank 和 starvation 均未改善；说明监督没有作用到选择瓶颈。
2. 选择指标改善但同预算 mAP/AP_small 不改善；说明主要瓶颈位于 decoder、定位或标注，而非 query 生存。
3. 只提升 AP_small，却造成总 mAP、AP75 或中大目标显著退化；需要重新处理损失竞争，不能只报局部指标。
4. 增益只能由 invalid-anchor 修复、更大 query budget、更多训练步数或不同增强解释。
5. 结果只在单个 checkpoint、单一种子或单一表格数据集成立。
6. eval 输出链路或固定 CPU latency 发生额外变化，却仍声称零推理开销。

