# EXP-AUDIT-002：三组 12,800-step probe 结果审计

## 结论

三份最新目录只保存了 `metrics.txt`，没有 `console.log`、`log.txt`、运行时 `cfg:` 或 resolved config。因此，下面的数值可以作为已报告结果登记，但目录名所声称的 loss、augmentation 和 `CONFIG_FILE` 是否真正生效仍未验证。

这一区分很重要：历史实验已经出现过 launcher 未加载外部 `CONFIG_FILE` 的问题。本次不能仅凭 `small_reg_correct` 或 `mal_dense_o2o` 这样的目录名完成方法归因。

## 证据状态

| Run | 现有证据 | 声称/预期配置 | 方法身份 |
|---|---|---|---|
| `probe12800_fixed640_vfl_control_correct` | 仅 `metrics.txt` | `dfine_hgnetv2_m_table_cell_fixed640_probe.yml` | 未验证 |
| `probe12800_fixed640_small_reg_correct` | 仅 `metrics.txt` | `dfine_hgnetv2_m_table_cell_fixed640_small_reg_probe.yml` | 未验证 |
| `probe12800_fixed640_mal_dense_o2o_bs16_memfix` | 仅 `metrics.txt` | `deim_hgnetv2_m_table_cell_fixed640_mal_dense_o2o_probe.yml` | 未验证 |

摘要共同记录：从 `checkpoint0027.pth` 开始，训练 12,800 steps，评估使用 `maxDets=300`。起点为 mAP@50:95 `0.864247558`、AP50 `0.880289`。fixed 640、640 queries 和 global batch 16 由现有汇总声称，但没有运行时原始日志复核。

## 精确结果

| 方案 | mAP@50:95 | AP50 | AP75 | AP small | AP medium | AP large | AR@300 | AR small | AR medium | AR large | AR50 | AR75 | 时间 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| VFL control | 0.860067 | 0.871090 | 0.871 | 0.682 | 0.857 | 0.913 | 0.866 | 0.688 | 0.861 | 0.919 | 0.880 | 0.874 | 01:37:17 |
| VFL + small-reg | 0.860581 | 0.876213 | 0.871 | 0.680 | 0.856 | 0.917 | 0.866 | 0.686 | 0.860 | 0.923 | 0.880 | 0.875 | 01:38:15 |
| MAL + Dense O2O | 0.850864 | 0.870690 | 0.866 | 0.666 | 0.849 | 0.889 | 0.857 | 0.675 | 0.854 | 0.897 | 0.878 | 0.870 | 03:34:55 |

三组相对起点的 mAP/AP50 变化分别为：

- VFL control：`-0.004180 / -0.009198`
- VFL + small-reg：`-0.003666 / -0.004076`
- MAL + Dense O2O：`-0.013384 / -0.009598`

## 相对 control 差异

| 指标 | small-reg - control | MAL + Dense O2O - control |
|---|---:|---:|
| mAP@50:95 | +0.000514 | -0.009203 |
| AP50 | +0.005123 | -0.000400 |
| AP75 | 0.000 | -0.005 |
| AP small | -0.002 | -0.016 |
| AP medium | -0.001 | -0.008 |
| AP large | +0.004 | -0.024 |
| AR@300 | 0.000 | -0.009 |
| AR small | -0.002 | -0.013 |
| AR medium | -0.001 | -0.007 |
| AR large | +0.004 | -0.022 |
| AR50 | 0.000 | -0.002 |
| AR75 | +0.001 | -0.004 |

按摘要数值，small-reg 的总 mAP 仅比 control 高 `0.000514`，但 AP small 和 AR small 都低 `0.002`，所以当前结果不支持“small-reg 改善了小框”。MAL + Dense O2O 比 control 低 `0.009203` mAP，且所有分尺度 AP/AR 均下降，说明该摘要结果整体退化。

这些结论只描述观测数值，不构成方法因果归因。除 mAP/AP50 外，分项指标只打印到三位小数，约 `0.001` 以内的差异不应过度解释。

## 补证要求

要把记录升级为可用于方法消融的证据，必须从服务器补回三组原始 `console.log` 和 `log.txt`，并逐项确认：

1. 启动行中的配置路径与 `train.py` 打印的运行时 `cfg:` 一致；
2. control 和 small-reg 训练行出现 `loss_vfl`，MAL 训练行出现 `loss_mal`；
3. small-reg 的回归权重确实进入 resolved config；
4. MAL 组的 Mosaic、MixUp 和调度参数确实进入 resolved config；
5. 数据、初始化、batch、seed、steps 和评估器保持可比。

在这些证据补齐前，允许引用精确指标和相对 control 的差值，禁止把差异归因于 small-reg、MAL、Mosaic 或 MixUp。
