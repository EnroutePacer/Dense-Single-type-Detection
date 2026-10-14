# 一阶段总结：常规改法的效果有限

## 对照口径

B0、C1、C2、C3 都从同一个 epoch-27 checkpoint 严格恢复，使用 fixed-640、
640 queries、global batch 64、seed 0、epoch 28--34 和 COCO
`maxDets=300`。数值来自各自 `log.txt` 的 epoch-34 精确 JSON；原始日志、
resolved config 和 at-start manifest 均已归档。

| 方法 | mAP@50:95 | 相对 B0 | AP-small | 相对 B0 | AP-large |
|---|---:|---:|---:|---:|---:|
| B0 VFL control | 0.866667089 | 0 | 0.685633491 | 0 | 0.928902582 |
| C1 小框回归加权 | 0.866651559 | -0.000015530 | 0.685224968 | -0.000408523 | 0.928659811 |
| C2 MAL-only | 0.868764798 | +0.002097709 | 0.677576517 | -0.008056974 | 0.933895888 |
| C3 Mosaic-only | 0.865379299 | -0.001287789 | 0.682975777 | -0.002657714 | 0.917625322 |

## 结论

- 小框回归加权几乎复制了 B0，说明仅放大小框已匹配回归项没有带来可测的
  小框收益。
- MAL-only 的总 mAP 增加约 0.21 个百分点，但 AP-small 下降约 0.81 个
  百分点，收益主要不在小框，不能作为密集小框问题的解法。
- 当前 Mosaic-only 同时降低总 mAP、AP-small 和 AP-large，简单增强没有改善
  该续训状态。
- 三个常规改法没有一个超过 B0 的 AP-small。结合 B1--B3 的结果，一阶段
  证据支持把后续研究集中到 Encoder 候选生成、global Top-K 生存和逐边尺度
  监督，而不是继续叠加面积权重、替换分类损失或常规 Mosaic。

这是单 seed、已收敛 checkpoint 的七 epoch 屏选，只能证明这些具体配置在
该协议下缺少正信号；不能证明所有加权、MAL 或 Mosaic 方案普遍无效，也不能
单独证明 Encoder candidate starvation 是唯一原因。C4 最后观察到至少
epoch 28 step 4100/5631，输出仍在变化且没有有效 endpoint，不进入排序或
本次最终 artifact 同步。
