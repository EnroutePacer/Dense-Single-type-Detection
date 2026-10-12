# EXP-TRAIN-004：SGQS-E 同预算四组训练消融

状态：planned / not_run。方法见 [M-002](../../methods/M-002_sgqs_e/README.md)，启动方式见 [实现说明](../../methods/M-002_sgqs_e/implementation.md)。本目录没有真实训练结果。

## 设计

| 组 | 训练改动 |
|---|---|
| B0 | 同提交 clean fixed640 baseline |
| B1 | 修订 SGQS：非 owner 竞争门槛与原 Encoder 监督保护 |
| B2 | EFGL：逐边误差与同类邻域补偿，前 30% 训练退火 |
| B3 | 与 B1 相同的 SGQS + 与 B2 相同的 EFGL |

所有组固定数据、初始化、增强、输入 640、K=640、global batch、optimizer steps、seed 和 maxDets=300。首轮 seed 0 筛选；正式结论使用配对 seeds 0/1/2，并报告均值和波动。global batch、步数、数据/初始化哈希须在第一组运行前冻结到 run 元数据，不能看过结果后调整控制组预算。

比较候选存活、unique-GT coverage、逐边误差、AP75/AP90、mAP、AP-small/AR-small，以及短边×密度交叉组。固定步数比较同时报告 wall-clock/GPU-hours；如果各组训练吞吐不同，不能称其为同计算预算结果，需另做等时长比较。

## 判定与后续

只有 B3 跨 seed 稳定超过 B0、B1、B2，并且候选与回归指标分别改善，才能支持互补。若声称正交互，还须报告 B3−B1−B2+B0 的不确定性。内部消融按结果推进：无 survival、无监督保护、统一 IoU 下限、beta=0、无补偿退火；不在首轮同时搜索所有参数。

本轮沿用 clean VFL 控制配方；之后须在完整 DEIM/MAL + Dense O2O 配方上复验。旧 M-001、历史 epoch-27 checkpoint 指标或恢复训练 probe 不能代替这些同提交控制。

## Runs

尚未执行；`experiment.yaml` 中 `runs: []`。数学反例、单元测试与同权重推理等价只证明实现性质，不计入训练收益。
