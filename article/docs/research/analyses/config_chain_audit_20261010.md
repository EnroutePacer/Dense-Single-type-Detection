# DEIM 配置链审计

> **历史服务器快照审计（2026-10-10）**：文中的“当前脚本”指 ART-000001 内的旧服务器版本。根目录活跃 launcher 已在后续同步中修复为尊重外部 CONFIG_FILE；本页不描述 2026-10-12 的活跃代码状态。

## 审计结论

当前服务器 M 表格模型是 D-FINE 架构加普通 VFL 训练配方，不是完整 DEIM 训练配方。此前通过环境变量指定的 fixed640/small-object YAML 又被启动脚本覆盖，因此相关探针也没有测试到目标变量。

## 实际链路

入口：

```text
configs/table_cells/dfine_hgnetv2_m_table_cell.yml
```

include：

```yaml
__include__: [
  '../dataset/custom_detection.yml',
  '../runtime.yml',
  '../base/optimizer.yml',
  '../base/dfine_hgnetv2.yml',
]
```

`base/dfine_hgnetv2.yml` 虽然注册的 criterion 类叫 `DEIMCriterion`，但关键配置是：

```yaml
DEIMCriterion:
  weight_dict: {loss_vfl: 1, loss_bbox: 5, loss_giou: 2,
                loss_fgl: 0.15, loss_ddf: 1.5}
  losses: ['vfl', 'boxes', 'local']
```

所以类名不能当作是否启用 DEIM MAL 的依据。

## 完整 DEIM 链路

官方 M COCO 配置采用：

```yaml
__include__: [
  './dfine_hgnetv2_m_coco.yml',
  '../base/deim.yml'
]
```

`base/deim.yml` 添加四组关键覆盖：

1. Dense O2O 数据配方：Mosaic、MixUp、RandomZoomOut、RandomIoUCrop 及阶段调度；
2. MAL：`losses: ['mal','boxes','local']`，`loss_mal: 1`，`gamma: 1.5`；
3. flat-cosine 与 warmup/no-aug 参数；
4. decoder `activation: silu`、`mlp_act: silu`。

当前表格入口未 include 此文件。

## 启动脚本覆盖外部配置

文件：

```text
tools/table_cells/train_dfine_m_table_cell_multigpu.sh
```

当前第 7 行：

```bash
CONFIG_FILE="${PROJECT_ROOT}/configs/table_cells/dfine_hgnetv2_m_table_cell.yml"
```

因此下面写法不会生效：

```bash
CONFIG_FILE="$PWD/configs/table_cells/dfine_hgnetv2_m_table_cell_fixed640_small_reg_probe.yml" \
bash tools/table_cells/train_dfine_m_table_cell_multigpu.sh
```

脚本应使用带默认值的环境变量：

```bash
CONFIG_FILE="${CONFIG_FILE:-${PROJECT_ROOT}/configs/table_cells/dfine_hgnetv2_m_table_cell.yml}"
```

但仅改这一行还不等于启用 DEIM。还需建立一个明确 include `base/deim.yml` 的 table-cell 配置，并为总 epoch 重设 augmentation/scheduler 阶段。

## 日志中的运行时铁证

以下 console 的 `cfg.yaml_cfg` 均显示：

```text
config = .../dfine_hgnetv2_m_table_cell.yml
Resize [800,800]
scales [640,704,800]
losses ['vfl','boxes','local']
```

- `output/probe3200_fixed640_small_vfl_w1p5/console.log`
- `output/probe3200_fixed640_small_vfl_w1p5_correct/console.log`
- `output/probe6400_fixed640_small_reg_w1p30_l1p15/console.log`
- `output/probe12800_fixed640_small_reg_4gpu/console.log`

相应 `log.txt` 只有 `train_loss_vfl*`，没有 `train_loss_mal*`。

## 第一版 S 为什么不同

第一版 `workspace/config/train.yml` include 的是 `configs/deim_dfine/deim_hgnetv2_s_coco.yml`，该文件会再 include `base/deim.yml`。所以第一版配置链至少在静态 YAML 层面包含 Dense O2O 与 MAL。

但仍有一个需要核验的键位问题：

```yaml
transforms:
  policy:
    mosaic_prob: 0.0
```

官方键位是：

```yaml
transforms:
  mosaic_prob: 0.5
```

因此第一版想关闭 Mosaic 的覆盖可能没有命中。缺少运行时解析 cfg 和日志时，不能确定最终值。

## 正确基线的建立方式

1. 修正 `CONFIG_FILE` 默认值写法。
2. 分别创建 `table_dfine_vfl.yml` 与 `table_deim_mal.yml`，不要通过目录名暗示配置。
3. 启动后把 resolved config 单独保存到输出目录。
4. 在训练前断言：
   - VFL 基线没有 `loss_mal`/Mosaic/MixUp；
   - DEIM 基线包含 `loss_mal`，且增强阶段符合总 epoch；
   - `num_queries=640`、`PostProcessor.num_top_queries=640`；
   - global batch、LR、seed 与对照一致。
5. 完整 DEIM 应从相同初始化重新训练；不能从已训练 27 epoch 的 VFL checkpoint 恢复几千步后声称是 DEIM 基线。

## 证据文件

- `../01_sources/server_DEIM_20261010_182837/DEIM/tools/table_cells/train_dfine_m_table_cell_multigpu.sh`
- `../01_sources/server_DEIM_20261010_182837/DEIM/configs/table_cells/dfine_hgnetv2_m_table_cell.yml`
- `../01_sources/server_DEIM_20261010_182837/DEIM/configs/base/dfine_hgnetv2.yml`
- `../01_sources/server_DEIM_20261010_182837/DEIM/configs/base/deim.yml`
- `../01_sources/server_DEIM_20261010_182837/DEIM/configs/deim_dfine/deim_hgnetv2_m_coco.yml`

