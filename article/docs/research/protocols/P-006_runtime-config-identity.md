# P-006：运行时配置身份协议

版本：1.0（2026-10-10）

实验身份以运行时 resolved config 和实际 loss/log 字段为准，不以目录名、YAML 文件名或启动命令为准。

训练开始前必须自动断言并写入 run 目录：

- config 绝对解析结果与配置 SHA-256；
- input resize、collate scales 与增强日程；
- `num_queries`、`num_top_queries`；
- criterion `losses`、`weight_dict` 和方法特定参数；
- optimizer、LR、scheduler、epoch/steps；
- checkpoint 模式是 tuning 还是 resume；
- world size、per-GPU batch 与 global batch；
- evaluator `maxDets`。

任一关键断言失败时立即终止，run 标为 `invalid` 或 `aborted`，不得继续写入与其他实验共用的输出目录。
