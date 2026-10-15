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
- `ART-000030`～`ART-000035`：B0/B1/B2 [配对诊断、数学说明及可复现解析](../../experiments/EXP-TRAIN-005_epoch27-sgqs-e-resume-screen/diagnostics/20261013_b1_b2/README.md)。这是 B3 到达前的不可变阶段性快照；支持机制问题定位，不代表当前完整矩阵状态。
- `ART-000036`、`ART-000037`：B1/B2 ingestion provenance，仅证明本轮附件归档身份，不是历史训练 manifest。
- `ART-000038`、`ART-000039`：B3 用户附件 console 与 ingestion provenance；与服务器原件分开保留。
- `ART-000040`～`ART-000058`：共同 epoch-27 checkpoint、全输出 SHA-256 manifest、B0/B1/B2/B3/C2 关键服务器日志、at-start manifest 与最终 checkpoint。B1--B3/C2 运行身份可复核；B0 仍明确缺新 launcher 的历史 at-start manifest。
- `ART-000059`、`ART-000060`：[B0--B3/C2 完整跟进](../../experiments/EXP-TRAIN-005_epoch27-sgqs-e-resume-screen/diagnostics/20261013_b0_b3_c2_followup/README.md)与精确 endpoint 表。可计算单 seed 主效应/交互，但不支持稳定互补性或小目标改善。
- `ART-000061`、`ART-000062`：B1/B3 曲线生成脚本与长表。COCO console 值只有三位小数，不能用于证明 B3 相对 B1 的 `+0.000078905` 精确差值。
- `ART-000063`、`ART-000064`、`ART-000066`、`ART-000067`：C1 small-reg 与 C3 Mosaic-only 的完整 console 和结构化日志，支持 epoch-34 精确结果。
- `ART-000068`、`ART-000069`：[一阶段常规方法总结](../../experiments/EXP-TRAIN-005_epoch27-sgqs-e-resume-screen/diagnostics/20261014_phase1_conventional_methods/README.md)及精确指标表。支持“小框加权近似无效、MAL 收益不在小框、当前 Mosaic 为负”的单 seed 屏选结论。
- `ART-000070`、`ART-000071`：C1/C3 的 at-start manifest；与 resolved config 和输出清单共同支持运行身份复核。
- `ART-000081`：EXP-DIAG-006 的脱敏 full-val 输入清单；支持 12,841 图、1,458,060 GT、
  逐域 annotation 哈希、图像存在性及 14 状态去重矩阵，不是模型输出或 full-val 结果。
- `ART-000082`～`ART-000093`：EXP-TRAIN-006 的 SGQS/B0 同探针运行 manifest、
  resolved config、完整 console、结构化 epoch 日志、8-rank 汇总表与 epoch-34
  checkpoint。两个 canonical 运行目录的全部 101 文件另由 `ART-000041` 逐文件登记；
  支持配对运行身份、完整性和终点指标，不替代尚未完成的机制分析。

## B：支持行为观察，不能替代 GT 指标

- `ART-000005`：13 图无 GT 的 encoder/query 诊断。
- `ART-000008`：DEIM 与 PaddleX-L 同图定性比较。
- `ART-000015` 与 `ART-000016`：dense encoder 全候选和 `image (12)` GT 可视化；支持恢复框先导审计，不替代原始 COCO JSON。
- 多尺度、分块和输出数量记录：只能说明 latency 与输出行为。

## C：待补证据

- 第一版 S 第三 epoch mAP≈0.86：缺原始日志与 checkpoint。
- 完整 val 的脱敏规模与 annotation 哈希已由 `ART-000081` 登记；私有 annotation JSON
  与可公开复算许可仍缺，GT 密度分桶须待正式 full-val 派生表落盘。
- 旧 probe checkpoint 尚未完整进入 artifact registry；本轮共同 epoch-27 checkpoint 已登记为 `ART-000040`。
- 当前代码的准确上游 commit/remote：同步目录不是 Git 仓库。

## D：失败/中断证据

- `ART-000065`：旧 DEIM-core 从头训练的中断 console，只支持停止位置记录。
- C4 DEIM-core retrofit 的服务器输出仍在变化，没有稳定 artifact 或有效 endpoint；待其稳定后再归档，不得用于方法排序。

完整 locator、bytes、SHA-256、发布状态与关联实验见 [artifact_registry.csv](artifact_registry.csv)。公开论文主张不得直接引用本地目录名，必须引用 ART/RUN/EXP/F/C 链。
