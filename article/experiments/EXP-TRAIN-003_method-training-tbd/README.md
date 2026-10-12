# EXP-TRAIN-003：SGQS 等步数方法训练

状态：planned / gated，不得现在运行。

方法已确定为 M-001 SGQS，代码和配置已实现，但只有 EXP-DIAG-003 与 EXP-ABL-001 完成后才能启动。该实验用于检验 SGQS 能否在相同初始化、global batch、epoch/steps、seed 和 `maxDets=300` 下超越干净 fixed-640 VFL baseline。

正式配置为 `configs/sgqs/sgqs_hgnetv2_m_table_cell_fixed640.yml`，epoch-27 continuation 探针配置为 `configs/sgqs/sgqs_hgnetv2_m_table_cell_fixed640_probe.yml`。由于 source commit `fd0c110` 同时修复了 invalid boundary anchor 选择，control 必须使用同一提交重跑；历史 0.864248 只能作为旧口径参考，不能直接归因。

预注册机制指标为 assignment miss、owner coverage、P3/P4/P5 owner 数、owner IoU、cutoff gap 与 pre-Top-K survival；终局指标保持 maxDets=300 的 mAP/AP50/AP75/AR，并额外报告短边分桶和 evaluator 截断前 coverage。

若正式诊断表明 selection 不是主要瓶颈，本实验取消，ID 不复用。
