# 小目标损失实验审计

> **Historical snapshot / superseded（2026-10-10）**：本文保留旧损失讨论和训练建议，仅用于审计历史思路。当前不据此启动训练；优先顺序以 [roadmap](../roadmap.md) 为准。

## 代码原本设计了什么

服务器快照的 `DEIMCriterion` 新增四个参数：

```text
small_object_weight
small_object_area_threshold
small_object_box_weight
small_object_local_weight
```

### small VFL

`loss_labels_vfl` 在 Hungarian 匹配完成后，根据目标归一化面积判断 small object，并只放大已匹配正 query 的 VFL loss。它不改变：

- encoder query 候选；
- Hungarian cost；
- 负 query 权重；
- decoder 的 Top-K 选择；
- bbox/GIoU/FGL 梯度。

### small regression

`loss_boxes` 对已匹配目标的 L1 与 GIoU 乘平滑面积权重；`loss_local` 对 FGL 乘较温和的面积权重。权重按匹配集合均值归一到 1，从而尽量保持总 loss scale，只重分配大小目标之间的梯度份额。

配置意图：

```yaml
small_object_area_threshold: 0.0025  # 640 下约等于 32×32
small_object_box_weight: 1.30
small_object_local_weight: 1.15
```

## 为什么旧实验没有回答问题

训练脚本忽略了外部配置，resolved cfg 中没有任何 `small_object_*` 键。结果日志仍为普通 VFL。因此旧结果既不能证明该方法有效，也不能证明无效。

## 完整梯度链

```text
图像 -> backbone/encoder 特征
     -> encoder 密集位置得分与初始框
     -> Top-640 query
     -> decoder 输出类别和框
     -> Hungarian cost(class + L1 + GIoU)
     -> 每个 query/GT 的 O2O 匹配
     -> VFL/MAL + bbox/GIoU + FGL/DDF
     -> loss 权重求和
     -> AMP backward
     -> 梯度裁剪
     -> AdamW 更新全部参与参数
```

面积权重位于匹配之后。它只能放大已匹配小 GT 对应路径的梯度，不能创造缺失的高分候选，也不能直接修复 Top-K 前被淘汰的目标。若 GT 数超过 queries，matcher 文档明确匹配数为 `min(num_queries, num_target_boxes)`，超出的 GT 没有 O2O 正匹配。

## 对 concat4×320 的判断

四张 320 图拼成 640 样本，本质上就是当前 DEIM `Mosaic(output_size=320)` 的主要几何形式；代码明确把四张图放进 2×2、最终 640×640 画布。因此不能把它单独包装成全新方法。

它可能有两类相反作用：

- 正面：同一 batch element 中目标更多，单元格被缩小，能训练低分辨率目标并增加 O2O 正监督。
- 负面：当前数据本来就密集，拼接后 GT 总数可能超过 640；Hungarian 只能匹配 640 个，造成真实 GT 无监督、竞争加剧和分布偏移。

所以“向下缩放一定比向上放大正确”并不成立。向上尺度主要改善输入特征分辨率；向下尺度训练主要增强小目标鲁棒性。它们针对不同问题，必须分别评估。

## 合理的下一次消融

先修脚本，然后采用唯一输出目录运行：

1. fixed640 control；
2. fixed640 + small-reg；
3. Mosaic/concat4×320 control，关闭 small-reg；
4. Mosaic/concat4×320 + small-reg。

四组必须相同初始化、global batch、LR、steps、seed 和评估口径；至少记录每个 mosaic 样本 GT 数、超过 640 的比例、实际 matched GT 比例、small/medium/large 的梯度范数和 AP。

只有第 2 对第 1、第 4 对第 3 才能分别估计 small-reg 的净效应；第 3 对第 1 才是 concat/Mosaic 的净效应。

