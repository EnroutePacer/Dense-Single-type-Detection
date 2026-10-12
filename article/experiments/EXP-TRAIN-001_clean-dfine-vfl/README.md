# EXP-TRAIN-001：干净 D-FINE/VFL 基线

状态：planned，训练暂停。

只有在分阶段诊断确认研究方向后才执行。运行时必须验证外部配置实际加载，保存 resolved config，使用唯一 run 目录，并与方法实验保持相同结构、初始化、global batch、epoch/steps、seed 和 `maxDets=300`。

本实验不继承旧 probes 的有效性；旧 probe 只在 EXP-AUDIT-001 中保存。
