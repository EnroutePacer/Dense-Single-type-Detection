# 证据索引

## A：可支持定量结果

- `ART-000006`：epoch-27 原始训练/评估日志；支持历史配置和指标记录。
- `ART-000007`、`ART-000009`～`ART-000014` 及各自同批 raw CSV/config：支持固定 CPU latency。
- `ART-000002`～`ART-000004`：推理实际模型；用于复核模型身份。
- `ART-000001`：服务器代码与日志原始归档，是不可变追溯源。
- `ART-000017`～`ART-000021`：三个新 12,800-step 的指标摘要与三方汇总；缺运行时配置，只支持记录数值和负结果边界。
- `ART-000022`：28-epoch benchmark 的服务器传输代码包；只证明代码包身份，不代表任何训练已执行。
- `ART-000024`～`ART-000027`：EXP-TRAIN-005 的 B0 精确指标与原始日志。
- `ART-000028`、`ART-000029`：B1 SGQS / B2 EFGL 完整 console 附件，字节不变；支持配置、过程和结果记录，但缺历史运行前身份 manifest。
- `ART-000030`～`ART-000035`：B0/B1/B2 [配对诊断、数学说明及可复现解析](../../experiments/EXP-TRAIN-005_epoch27-sgqs-e-resume-screen/diagnostics/20261013_b1_b2/README.md)。支持单 seed 的短续训观察与实现问题定位，不支持稳定收益或 B3 互补性。
- `ART-000036`、`ART-000037`：B1/B2 ingestion provenance，仅证明本轮附件归档身份，不是历史训练 manifest。

## B：支持行为观察，不能替代 GT 指标

- `ART-000005`：13 图无 GT 的 encoder/query 诊断。
- `ART-000008`：DEIM 与 PaddleX-L 同图定性比较。
- `ART-000015` 与 `ART-000016`：dense encoder 全候选和 `image (12)` GT 可视化；支持恢复框先导审计，不替代原始 COCO JSON。
- 多尺度、分块和输出数量记录：只能说明 latency 与输出行为。

## C：待补证据

- 第一版 S 第三 epoch mAP≈0.86：缺原始日志与 checkpoint。
- 数据规模和 GT 密度统计：缺对应 annotation JSON 与复算脚本快照。
- epoch-27 与 probe checkpoint：尚未进入 artifact registry。
- 当前代码的准确上游 commit/remote：同步目录不是 Git 仓库。

完整 locator、bytes、SHA-256、发布状态与关联实验见 [artifact_registry.csv](artifact_registry.csv)。公开论文主张不得直接引用本地目录名，必须引用 ART/RUN/EXP/F/C 链。
