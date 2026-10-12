# Fixed-640 三组 12,800-step Probe 对照分析

记录日期：2026-10-12  
共同起点：`checkpoint0027.pth`，mAP@50:95 = 0.864247558，AP50 = 0.880289  
共同评估：COCO bbox，`maxDets=300`  

三组实验均以相同的 12,800 个优化 step、fixed 640、640 queries 和全局 batch 16 进行对照。除 mAP 和 AP50 的 quick-probe 汇总值外，其他分项指标来自 COCO 输出，只有三位小数，因此小于约 0.001 的分项差异不应被过度解读。

## 1. 核心结果

| 方案 | mAP@50:95 | 相对起点 | AP50 | 相对起点 | 耗时 |
|---|---:|---:|---:|---:|---:|
| 起始 checkpoint0027 | 0.864248 | — | 0.880289 | — | — |
| VFL control | 0.860067 | -0.004180 | 0.871090 | -0.009198 | 1:37:17 |
| VFL + small_reg | **0.860581** | **-0.003666** | **0.876213** | **-0.004076** | 1:38:15 |
| MAL + Dense O2O | 0.850864 | -0.013384 | 0.870690 | -0.009598 | 3:34:55 |

三组续训都没有超过起始 checkpoint。按最终 mAP 排名：`small_reg > VFL control > MAL + Dense O2O`，因此当前最佳权重仍是 epoch 27 的原始 checkpoint，而不是任一 probe 权重。

## 2. 必须使用同周期 control 判断改动效果

VFL control 在没有新方法的情况下，继续训练 12,800 steps 已经下降 0.004180 mAP。这说明 epoch 27 附近已经处于当前训练配方的较优区域，继续按现有学习率和数据分布优化本身会让验证指标回落。因此不能把各 probe 相对起点的全部下降都归因于新方法；方法效果应优先与同周期 VFL control 比较。

| 指标 | small_reg - control | MAL + Dense O2O - control |
|---|---:|---:|
| mAP@50:95 | **+0.000514** | **-0.009203** |
| AP50 | **+0.005123** | -0.000400 |
| AP75 | 约 0.000 | -0.005 |
| AP small | -0.002 | -0.016 |
| AP medium | -0.001 | -0.008 |
| AP large | +0.004 | -0.024 |
| AR@300 | 约 0.000 | -0.009 |
| AR small | -0.002 | -0.013 |
| AR medium | -0.001 | -0.007 |
| AR large | +0.004 | -0.022 |

## 3. small_reg 的判断

配置把已经完成 Hungarian 匹配的小框定位项加权：`small_object_box_weight=1.30`，`small_object_local_weight=1.15`。它相对 control 提高了 0.000514 mAP，并且 AP50 提高了 0.005123，代价只有 58 秒，说明实现没有造成明显吞吐损失。

但它没有实现预期的小框收益：AP small 从 0.682 降到 0.680，AR small 从 0.688 降到 0.686；AP75也没有可见提升。相反，主要正向变化出现在 AP50 和 large 指标。这意味着当前增益更像是对容易匹配或大框的整体优化扰动，而不是更准确地定位、召回小单元格。

原因与代码作用位置一致：small_reg 只在 Hungarian 已经为 GT 找到 query 以后放大该匹配对的 bbox/GIoU/FGL 梯度。没有被正确召回、没有获得合适 query 的小框不会产生这部分回归梯度，所以单纯继续提高回归权重无法解决“小框 query 没匹配上”的问题。另外，训练中的归一化面积阈值与 COCO 按原图像素面积划分的 `AP small` 不一定完全对应。

结论：small_reg 可以暂时保留为 AP50 候选，但目前不能宣称它改善了小目标检测。0.000514 mAP 也很小，需要多 seed 重复后才能判断是否稳定，不建议继续盲目增大权重。

## 4. MAL + Dense O2O 的判断

相对 control，该组合下降 0.009203 mAP，small/medium/large 和对应 recall 全部下降，训练时间则从 1:37:17 增至 3:34:55，约为 **2.21 倍**。这不是单个尺度上的偶然波动。

AP50 仅比 control 低 0.000400，但 AP75 低约 0.005、总体 mAP 低 0.009203，说明模型仍能粗略找到目标，主要损失发生在更严格 IoU 下的定位质量和排序质量。

不过，这个实验同时改变了多项因素：

1. 分类监督由 VFL 改成 MAL，并把 gamma 从 2.0 改为 1.5；
2. 加入 Mosaic、RandomZoomOut、RandomIoUCrop；
3. 加入 MixUp；
4. 从一个此前未使用这些强增强的 epoch-27 checkpoint 直接恢复，而 epoch 27 正处于配置的 Mosaic/MixUp 激活区间。

因此这个结果只能证明“从当前 checkpoint 突然切换到 MAL + Dense O2O，并只适配 12,800 steps”不合适，不能单独证明 MAL 无效。表格单元格高度依赖规则行列结构，Mosaic 的拼接、裁剪及 MixUp 的密集网格叠加会制造大量被截断、重叠或上下文不一致的合成单元格；模型又只有约 0.57 个 epoch 的适配时间，分布突变很可能是主要退化来源。

耗时上升也主要来自 Dense O2O 增强后每批目标数量大幅增加。逐元素 IoU 修复已经消除了 criterion 中不必要的 O(N²) 对角线计算并避免 OOM，但 Hungarian matching 仍必须构造真实的候选-query/GT pairwise cost，目标越密集，训练仍然越慢。

## 5. 下一步实验建议

优先做拆分消融，而不是直接继续当前 MAL + Dense O2O 长训：

1. **MAL-only**：保持 control 的干净 fixed-640 数据管线，只把 VFL 换成 MAL，仍从 checkpoint0027 训练 12,800 steps。这是当前最有信息量的一组，可直接判断 MAL 本身。
2. **Dense-only**：保留 VFL，只加入 Dense O2O 增强，判断退化是否来自 Mosaic/MixUp。
3. 如果要严谨评价完整 Dense O2O 配方，应从预训练初始化或更早 checkpoint 开始，让增强从训练早期按官方阶段进入，而不是在 epoch 27 突然切换。
4. 针对小框，下一步应记录每个尺度的 GT 数量、Hungarian 匹配率、匹配 query 初始 IoU/score 和未匹配原因。只有先确认问题发生在候选/匹配、分类排序还是已匹配后的回归阶段，才能决定修改 matcher、query 选择还是 loss。

当前决策：保留 epoch-27 checkpoint 为最佳；small_reg 仅作为待复现候选；停止直接扩大 small_reg 权重；暂不继续当前 MAL + Dense O2O 组合长训，先做 MAL-only 拆分验证。
