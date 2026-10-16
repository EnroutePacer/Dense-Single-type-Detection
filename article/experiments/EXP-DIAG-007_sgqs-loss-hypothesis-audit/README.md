# EXP-DIAG-007：SGQS 损失假说审计

## 结论先行

状态：`completed / valid_with_scope / causal mechanism still open`。

用户提出的三个假说不能整体写成“已经证实”。现有证据支持一个更窄、也更有力的组合解释：

> SGQS 在 epoch-27 checkpoint 上恢复训练时，以没有升温的 dense/owner 目标造成很强的
> 启动冲击，前两轮迅速重排候选；但 SGQS 的直接监督止于 pre-TopK/Encoder，未持续绑定
> 同一 GT owner 到 Decoder 最后一层和最终 Top-300 排名。此后即使 SGQS 自身代理指标继续
> 变好，基础目标与 EMA 轨迹仍可重新分配 300 个输出名额。P3/P4/P5 等权的分层均值又会
> 提高稀少 P5 正样本的单样本杠杆，可能放大竞争，但这最后一环尚未获得梯度级因果证据。

这里的“启动冲击”“监督链路位置”“最终 Top-300 承载阶段”和“等层归一化实现”均有直接
证据；“已经过拟合”“大框 AP 触发基础损失回升”“P5 放大导致后期回跌”仍不能作为事实。

## 三个假说的判定

| 假说 | 判定 | 可以确认 | 不能确认或被反证 |
|:---|:---|:---|:---|
| 1. Encoder 新损失过强，前两轮剧变并近似过拟合、侵蚀总体性能 | **部分支持** | step 0 新增 SGQS loss 为 `5.071082`，其中 dense 为 `4.702855`；dense/owner 没有升温；候选来源和排名前两轮剧变 | loss 标量不是梯度范数；没有同口径 train-set AP 证明泛化间隙；e34 总 AP 反而比 B0 高 `0.0009148`，不能称总体性能被持续侵蚀 |
| 2. 缺少全链路监督；SGQS 变小后基础损失接管并使 small 回跌 | **代理脱节获支持；具体“损失回升/触发”叙事不成立** | SGQS 只直接作用在 pre-TopK/Encoder，而 full-val 倒 U 发生在 Decoder 最终 Top-300；e30→34 owner coverage、survival、owner IoU 都上升，AP-small 却下降 | e30→34 基础 Encoder 与主 Decoder loss 继续下降，没有 e31 明确回升；验证集大框 AP 不参与反向传播，不可能“触发”训练 loss |
| 3. dense 三尺度固定平分，使稀少大尺度 owner 被放大 | **实现机制存在，因果未证实** | dense 正/负项先逐层独立归一化，再三层平均；用整轮平均 owner 数估算，P3/P5 比从约 `69.85×` 增至 `134.80×`；采样探针中的 P5 kept owner 有 `98.91%` 属于增强后 large-input 组 | 没有逐层 positive/negative loss、共享参数梯度范数或 Adam 更新归因；P5 也不等于 COCO-large，因此不能断言它已造成 small 后期回跌 |

结构化判定见 [hypothesis_verdicts.csv](data/hypothesis_verdicts.csv)，详细论证见
[analysis/README.md](analysis/README.md)。

## 1. 启动冲击到底有多大

SGQS 与 B0 使用同一 checkpoint、seed、global batch 和探针入口。在 epoch 28 step 0，
两臂的基础 loss 完全相同，差值恰好是新增 SGQS 项：

| step 0 项目 | 数值 |
|:---|---:|
| B0 总加权 loss | 8.468864 |
| SGQS 总加权 loss | 13.539945 |
| 新增 SGQS 合计 | **5.071082** |
| 其中 dense | **4.702855** |
| owner L1 / GIoU / survival | 0.088774 / 0.279453 / 0.000000 |

新增项相当于当步 B0 总 loss 的 `59.88%`，dense 一项相当于 `55.53%`。survival 因课程
在 step 0 为零，所以启动冲击几乎全部来自未升温的 dense 与 owner 回归。这证明目标尺度
很强，却仍不能把标量比值直接写成“参数梯度占 59.88%”。前 500 step 的配对轨迹见
[early_resume_shock.png](figures/early_resume_shock.png) 和
[early_resume_steps.csv](data/early_resume_steps.csv)。

## 2. 不是“整体性能被侵蚀”，而是早期尺度/预算交换

相对 B0，SGQS 的服务器日志 EMA 指标为：

| epoch | ΔAP | ΔAP-small | ΔAP-large |
|---:|---:|---:|---:|
| 29 | -0.001950 | **+0.024128** | -0.008860 |
| 30 | -0.000625 | **+0.025399** | -0.010534 |
| 34 | **+0.000915** | -0.001539 | -0.000255 |

前段确有明显 small/large 反向交换；但 e34 总 AP 已恢复并略高于 B0，大框也几乎回到
B0。因此最准确的描述是“短暂的排名/预算重分配”，不是“整体能力持续被侵蚀”。完整数值见
[hypothesis_epoch_trajectory.csv](data/hypothesis_epoch_trajectory.csv) 和
[hypothesis_trajectory.png](figures/hypothesis_trajectory.png)。

## 3. 为什么“SGQS 已经变好，AP-small 却回跌”是关键证据

e30→e34：

- AP-small：`0.705554 → 0.682274`，下降 `0.023280`；
- SGQS owner coverage：`0.26361 → 0.45314`；
- pre-TopK survival：`0.93426 → 0.97814`；
- owner IoU：`0.94727 → 0.96349`；
- conflict fraction：`0.73639 → 0.54686`，并非后期过滤越来越多；
- SGQS branch loss：`0.248136 → 0.176962`；
- base Encoder loss：`0.345238 → 0.258981`；
- base main Decoder loss：`0.382164 → 0.346942`。

也就是说，SGQS 自己测量的“owner 找到、进入 Encoder Top-640、框更准”全部继续改善，
基础 loss 也没有在 e31 回升，最终 AP-small 却回落。它直接支持的是**代理目标与最终排名
脱节**，不是“Encoder loss 小到失效后大框 AP 触发原始 loss”。

完整验证集进一步显示：EMA AP-small@640 从 e30 的 `0.935651` 上升到 e34 的
`0.952699`，而 AP-small@300 从 `0.701179` 降到 `0.677757`；e30→34 loss 的十 IoU
转移事件中，`93.38%` 属于合格 Decoder query 跌出 Top-300。对应压缩表见
[full_val_small_budget_trajectory.csv](data/full_val_small_budget_trajectory.csv) 与
[full_val_transition_summary.csv](data/full_val_transition_summary.csv)。

## 4. P3/P4/P5 等权为什么值得警惕

当前 `_dense_loss` 的实际形式是：

\[
L_{dense}=\frac{1}{3}\sum_{l\in\{P3,P4,P5\}}
\left(\frac{S_l^+}{N_l^+}+\lambda_-\frac{S_l^-}{N_l^-}\right).
\]

它不是把所有 owner 放在一起平均，而是让每个层级占一个固定的三分之一槽位。因此一条
P5 正样本与一条 P3 正样本在同一步的显式归一化系数之比为 `N_P3/N_P5`；以整轮平均
owner 数做量级估算，e28 为 `69.85×`，e34 为 `134.80×`。这不是逐 step 系数的精确
平均。采样探针中累计 10,179 个 P5 kept owner，10,068 个属于增强后
large-input 组，占 `98.91%`；见 [owner_level_group_summary.csv](data/owner_level_group_summary.csv)。

这足以把“稀少 P5 单样本高杠杆”列为首要实现级嫌疑，但还不足以证明实际共享参数更新由
P5 主导：不同样本的误差、梯度方向、梯度裁剪和 AdamW 状态都未被这个分母比例测到。

## 5. image 5367 的“Decoder 640 好像漏了很多框”

每个 epoch 实际仍有 640 个 Decoder query。视觉上最突兀的是 e29，而不是持续到 e34：

| epoch | hybrid-small | 整像素取整近似唯一框 | 301--640 与 Top-300 IoU>.9 | score>.5 | rank 300 | rank 500 |
|---:|---:|---:|---:|---:|---:|---:|
| 28 | 621 | 623 | 8 | 613 | 0.9876 | 0.9839 |
| 29 | 611 | **498** | **113** | **445** | 0.9859 | **0.2969** |
| 30 | 618 | 564 | 71 | 523 | 0.9878 | 0.9793 |
| 34 | 625 | 618 | 22 | 612 | 0.9893 | 0.9867 |

e29 的变化发生在低排名尾部：第 300 名分数几乎不变，第 500 名骤降，且尾部与 Top-300
大量重叠。阈值显示或重叠绘图因此看起来少了很多框，但正式 evaluator 使用排序后的
Top-300、没有 `score>0.5` 门槛；e29 的前 300 名仍高分，所以整体 AP-small 反而能升高。
到 e34，尾部唯一性和高分数量基本恢复。实例图见
[image 5367 七轮 query probe](figures/image5367/query_probe_image_5367_ema_e28_e34.png)、
[数值轨迹图](figures/image5367_forensics.png) 与
[逐轮表](data/image5367_forensics.csv)。

在为转移机制定向选择的 29 张 probe 中，e28→29 有 28/29 张近似唯一框减少、27/29 张
尾部重叠增加；但这些图是按 gain/loss cause 选出的机制样本，不能当作全验证集发生率。

## 6. Baseline_test 的 13 张图

已把全部 13 张 e28--e34 contact sheet、aggregate 图和精确 CSV 一并归档到
[figures/baseline_test](figures/baseline_test/README.md)。13 图中 `score>0.5` 总数为
`1421/1414/1416/1415/1404/1406/1422`，基本稳定；预测框的 local hybrid-small 数量却从
`987` 下降到 `521`。这说明框的尺度形态发生了明显变化，但这些旧图没有 GT，不能用
`hybrid-small` 预测框数量替代 evaluator AP-small。

## 最小证伪顺序

下一轮不要同时修改三处。按因果信息量建议：

1. **启动冲击测试**：只给 dense+owner 项增加 1 epoch `0→1` 升温，其他完全不变；若
   e29/e30 的尺度交换明显收敛，才支持“侵略性”因果。
2. **分层归一化测试**：只把三层等均值改为全局 owner-count 归一化，同时记录每层正/负
   loss、共享分类头/Backbone 梯度范数和 clip 前后范数；若后期回跌消失，才支持假说 3。
3. **链路脱节测试**：只做一种干预——e30 后停用 SGQS，或增加最终 Decoder/Top-300 的
   owner identity/rank 约束；不能两者同时做。它用于区分“新增项持续干扰”与“缺少末端
   跟进监督”。
4. 所有分支保持 checkpoint、batch 顺序、optimizer/EMA、步数、seed 与 evaluator 完全一致，
   至少先通过 seed-0 gate，再扩展多 seed。

## 范围与限制

- 训练曲线来自同 seed 的 7-epoch checkpoint continuation，不是从头训练或多 seed 结论。
- probe 每 100 step 采样，e28 前 200 step 每 10 step 加密；采样整轮汇总不是无偏 epoch
  均值，相关表均明确标注。
- full-val 承载机制沿用 EXP-DIAG-006 的本地 annotation/Windows CPU 范围，不等于服务器
  H20 位级复现。
- `loss` 标量、owner 数和分母比例均不能替代共享参数梯度或真实 AdamW 更新归因。
- P3/P4/P5 是 owner 的 dense source level；增强后 input-area group 与 COCO evaluator
  size 也不是同一口径。

输入 SHA-256、生成脚本和全部发布文件分别见 [input_manifest.csv](data/input_manifest.csv)、
[build_hypothesis_audit.py](scripts/build_hypothesis_audit.py) 与
[artifact_manifest.csv](artifact_manifest.csv)。
