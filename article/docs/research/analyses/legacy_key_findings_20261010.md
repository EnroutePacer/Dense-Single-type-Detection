# 关键发现

> **Historical snapshot / superseded（2026-10-10）**：第 5 节的 decoder 倾向是旧解释，不是已证实机制。无 GT 产物只能提出 selection–survival 假设；请以 [研究问题](../research_question.md) 和 [事实注册表](../facts/README.md) 为准。

## 1. 旧探针不是所声称的配置

服务器快照中的脚本：

```bash
CONFIG_FILE="${PROJECT_ROOT}/configs/table_cells/dfine_hgnetv2_m_table_cell.yml"
```

它没有写成：

```bash
CONFIG_FILE="${CONFIG_FILE:-${PROJECT_ROOT}/configs/table_cells/dfine_hgnetv2_m_table_cell.yml}"
```

因此 shell 命令前设置的 `CONFIG_FILE=...fixed640...yml` 会被脚本第 7 行覆盖。console 的解析配置证明所有相关探针实际仍为：

```text
config = dfine_hgnetv2_m_table_cell.yml
Resize = [800, 800]
scales = [640, 704, 800]
losses = ['vfl', 'boxes', 'local']
```

所以 0.860123 与 0.859120 的差值不是 small-reg 对 control 的有效消融，而是两次同配方恢复训练的波动；二者还复用了同一个输出目录。

## 2. 当前 M 表格模型不是完整 DEIM

实际继承链：

```text
dfine_hgnetv2_m_table_cell.yml
├── custom_detection.yml
├── runtime.yml
├── optimizer.yml
└── dfine_hgnetv2.yml
```

缺少 `base/deim.yml`。所以当前有：

- Hungarian 一对一匹配；
- VFL；
- L1、GIoU、FGL、DDF；
- 640 queries；
- 自定义 640/704/800 尺度路径。

当前没有：

- DEIM 的 Mosaic/MixUp Dense O2O 配方；
- MAL；
- DEIM 的 flat-cosine 参数、增强关闭日程；
- DEIM 配方显式设置的 decoder SiLU/MLP SiLU。

类名仍叫 `DEIMCriterion`，不代表 MAL 已启用；是否启用 MAL 取决于 `losses` 是否包含 `mal`、`weight_dict` 是否包含 `loss_mal`。

## 3. mAP 五十多来自评估截断，不是模型真实退化

旧训练日志 epoch 27 记录 mAP 0.566899；代码当时使用 COCO 默认 `maxDets=100`。当前 evaluator 把上限改为 300，同一 checkpoint 的重新评估为：

```text
mAP@50:95 = 0.864247
AP50        = 0.880287
AP75        = 0.8708 左右
APS         = 0.680 左右
```

数据是超密集场景，单图 GT 可能超过 300，因此 100 个检测上限会系统性制造漏检。旧 0.56 与新 0.864 不能作为训练改动前后的差值。

## 4. 旧损失探针应重新分类

这些结果可以保留为“从 epoch 27 继续训练若干步的重复运行”，但不能作为小目标损失证据：

| 目录标签 | 结果 mAP | 实际配置判定 |
|---|---:|---|
| `small_vfl` 1600-step | 0.856937 | 基础多尺度 VFL |
| `small_vfl` 3200-step | 0.858211 | 基础多尺度 VFL |
| `small_vfl_correct` 12800-step | 0.858137 | 基础多尺度 VFL |
| `small_reg` 6400-step | 0.859313 | 基础多尺度 VFL |
| `small_reg_4gpu` 12800-step | 0.860123 | 基础多尺度 VFL；旧日志记录 |
| `fixed640` 12800-step | 0.859120 | 基础多尺度 VFL；同目录覆盖 console |

共同现象是从 checkpoint 的 0.864247 下降约 0.004～0.007。能支持的结论只是：以当前 checkpoint、当前学习率状态和恢复训练实现继续训练这段步数没有超过起点。不能把下降归因于 small-object loss。

## 5. 推理证据揭示的是“置信度/decoder”线索

13 张无 GT 图的诊断中：

- 109,200 个 encoder 候选中，最终进入 decoder 的 query 共 8,320；
- 其中 1,331 个最终框呈 COCO-small 形状；
- 739 个分数大于 0.01，38 个大于 0.10，只有 22 个大于 0.50；
- 在失败图上存在 encoder 高分小框经 decoder 后显著降分的实例；
- small query 的 encoder 与最终分数 Spearman 相关仅 0.3429。

所以后续应做带 GT 的 decoder 分层轨迹、Hungarian 匹配 IoU 和分类梯度诊断，而不是直接假设“增加 loss 权重”或“增加 query 数”一定解决问题。

