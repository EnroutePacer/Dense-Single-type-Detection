# T003～T010：恢复训练探针审计

> **Historical snapshot**：旧路径指向本地私有 legacy 证据，公开追溯请使用 [artifact registry](../../docs/evidence/artifact_registry.csv)。canonical 结论见本实验的 [README](README.md)。

## 原始意图

从 epoch-27 checkpoint 出发，用 1600～12800 个 optimizer steps 快速测试：

- fixed 640、去除双重 Resize；
- 小目标 VFL 正样本权重；
- 小目标 L1/GIoU 与 FGL 平滑权重；
- 单卡与四卡等效总 batch。

## 实际发生的事情

所有通过 `tools/table_cells/train_dfine_m_table_cell_multigpu.sh` 启动的实验都会执行脚本中的硬编码：

```bash
CONFIG_FILE="${PROJECT_ROOT}/configs/table_cells/dfine_hgnetv2_m_table_cell.yml"
```

因此外部传入的 `CONFIG_FILE=...fixed640...yml` 没有进入 `train.py`。console 的完整 cfg 明确显示：

- 数据 transform 仍先 `Resize [800,800]`；
- collate 仍采样 `[640,704,800]`；
- losses 仍为 `['vfl','boxes','local']`；
- 没有 `small_object_weight`、`small_object_box_weight` 或 `small_object_local_weight`；
- 没有 MAL、Mosaic 或 MixUp。

## 结果

| 运行 | steps | GPU/总 batch | mAP | AP50 | APS | 相对起点 mAP |
|---|---:|---|---:|---:|---:|---:|
| 目录名 `small_vfl` 记录 A | 3200 | 1×16 | 0.858211 | 0.871001 | 0.670533 | -0.006037 |
| 目录名 `small_vfl` 记录 B | 1600 | 1×16 | 0.856937 | 0.870976 | 0.684341 | -0.007310 |
| `small_vfl_correct` | 12800 | 1×16 | 0.858137 | 0.871013 | 0.672587 | -0.006113 |
| 目录名 `small_reg` | 6400 | 1×16 | 0.859313 | 0.871000 | 0.676473 | -0.004935 |
| 4-GPU 共享目录记录 A | 12800 | 4×4 | 0.860123 | 0.878005 | 0.679644 | -0.004125 |
| 4-GPU 共享目录记录 B | 12800 | 4×4 | 0.859120 | 0.870922 | 0.678088 | -0.005128 |

## 可保留的结论

- 从这个 checkpoint 继续训练一段步数没有超过起点。
- 相同名义配置的重复运行存在约 0.001 mAP 量级差异，单次小差值不应当作方法收益。
- 当前 quick-probe 对 full recipe 的判断能力有限：它恢复 optimizer/scheduler 状态，并在一个不完整 epoch 后评估 raw model。

## 不能保留的结论

- 不能说 small-reg 比 fixed640 提升 0.001003；两者实际是同一基础配置。
- 不能说 small-object weight 1.5 或 3.0 无效；权重根本没有进入运行时 criterion。
- 不能用这些 probe 判断完整 DEIM 的 Dense O2O 或 MAL；两者从未启用。

## 证据污染

- `probe12800_fixed640_small_reg_4gpu/log.txt` 有两条 epoch-28 记录，`console.log` 只对应后一次。
- `small_vfl_w1p5` 目录同时出现 3200/1600-step 运行，当前 YAML 又写成权重 3.0。
- 早期 200-step 目录有多条记录、无 console，无法恢复严格实验身份。

这些目录必须原样保存，但论文表中不得把它们当作有效消融。

