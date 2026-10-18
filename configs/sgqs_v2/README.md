# SGQS v2：单因素续训筛选

对应 [EXP-TRAIN-007](../../article/experiments/EXP-TRAIN-007_sgqs-v2-single-factor/README.md)。
状态：已实现，未运行，收益未验证。旧 `sgqs/`、`sgqs_e/`、`e35_r7/`、`probes/` 保留。

| 分支 | 对照关系 | 唯一干预 |
|---|---|---|
| R0 `r0_original_e28_r7.yml` | 原 SGQS 重跑 | 无，使用原版 SGQSCriterion |
| R1 `r1_baseline_e28_r7.yml` | R0 的整体方法对照 | 关闭 SGQS，使用原 B0 与相同探针 |
| R2 `r2_iou_target_e28_r7.yml` | 相对 R0 | dense 正样本 BCE target 从 q 改为 IoU |
| R3 `r3_global_positive_e28_r7.yml` | 相对 R0 | dense 正样本改为全局 owner-count 均值 |

R1 是整套 SGQS 开/关对照，不用于分离 dense、owner 回归、survival 各自因果。
R2、R3 互不叠加。R2 保留外层 q 权重与 survival 的 q 权重；R3 保留逐层负样本均值。
不加入额外升温。四臂共享 epoch-27 输入、epoch 28--34、8 GPU × 8、seed 0、
640 输入/640 query、原始优化器/EMA/AMP 设置、每 epoch checkpoint 和探针采样。

运行方式见 [SERVER_COMMANDS.md](SERVER_COMMANDS.md)。新损失只由专用入口注册，
直接使用普通 `train.py` 启动 R2/R3 会缺少注册；请使用随附脚本。

`runtime_files.sha256` 固定本次训练依赖的路径、字节数与 SHA-256。
预检和每个训练 rank 都校验它，配置或代码变化必须更新实验版本及清单。
保留原探针清单，避免更改历史复现身份。
