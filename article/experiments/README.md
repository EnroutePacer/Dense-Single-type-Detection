# 实验系统

实验采用两级标识：

- `EXP-*`：一个预注册的科学问题与受控设计；
- `RUN-YYYYMMDD-HHMMSS-sNN`：一次真实执行。

编号永久保留，标题变化不改编号。`status` 表示运行流程，`validity` 表示科学有效性；`completed + invalid` 是合法且必须保留的状态。

推荐编号：`EXP-TRAIN`、`EXP-EVAL`、`EXP-DIAG`、`EXP-BENCH`、`EXP-ABL`、`EXP-AUDIT`。旧 `T001/I009` 只放在 `legacy_ids`。

每个实验从 `_template/` 创建，并登记到 [registry.csv](registry.csv)。文本和小型证据走普通 Git；开发期需要归档的权重、TensorBoard 等大二进制走 Git LFS，并统一登记为 `ART-*`。

旧实验尚未拥有逐次 `RUN-*` 元数据。注册表分别记录 `legacy_run_count` 与 `registered_run_count`，并将其标为 `run_migration_status=unmapped`；这表示历史结果仍可审计，但 ART→RUN→EXP 链尚未补齐，不能伪装成已完成的新规范记录。

## 当前顺序

2026-10-19：[EXP-TRAIN-007](EXP-TRAIN-007_sgqs-v2-single-factor/README.md) 扩充为
R0--R5 六臂，增加升温与最终 Decoder admission 干预；主指标改为本地 hybrid-small
一对一 AP，完整验证集 84 状态/168 预算行。仍未运行，不宣称因果链已经闭合。

2026-10-18 新增 [EXP-TRAIN-007](EXP-TRAIN-007_sgqs-v2-single-factor/README.md)：
SGQS v2 四臂单因素续训已实现，尚未运行。R0 原 SGQS、R1 基线、R2 仅 BCE 质量目标、
R3 仅正样本全局归一化；从同一个 epoch-27 checkpoint 续训到 epoch 34。

2026-10-17 最新完成项为
[EXP-DIAG-007](EXP-DIAG-007_sgqs-loss-hypothesis-audit/README.md)：已把 SGQS/B0
整轮日志、step-level sampled probe、full-val Top-300 归因、image 5367 与 Baseline_test
13 图放入同一证据链。它确认恢复训练 step 0 存在由 dense 主导的强新增目标冲击，且
SGQS owner/survival 代理继续改善时最终 AP-small 仍回跌，支持缺少末端 Top-300 跟进
监督；当前三层等均值归一化确有稀少 P5 单样本高杠杆，但尚缺梯度级因果证据。

此前完成项
[EXP-DIAG-006](EXP-DIAG-006_sgqs-full-val-checkpoint-replay/README.md)：已在本地完整
12,841 图、1,458,060 non-crowd GT 上完成 e28--e34 raw/EMA 共 14 个唯一状态的
CPU 回放、COCO 一对一匹配和配对归因。EMA AP-small 的前段收益与后段回跌都主要由
最终 Decoder 合格 query 跨越 Top-300 排名边界承载；变化集中于每图 GT≥301 且
640 输入短边 8--16 px 的目标。该结论对本地 annotation/CPU 后端有效，不等于服务器
H20 的位级复现，也没有证明 EMA 或 SGQS 的训练级根因。
[EXP-TRAIN-006](EXP-TRAIN-006_sgqs-supervision-probes/README.md) 的 SGQS 与 B0
详细日志训练臂均已完成；两臂 loss/代理指标配对审计已由 EXP-DIAG-007 完成，B0
checkpoint 回放与固定 train-subset 回放仍未完成。

1. [EXP-TRAIN-005](EXP-TRAIN-005_epoch27-sgqs-e-resume-screen/README.md)：B0/B1/B2/B3 与 C2 核心屏选矩阵及完整证据已归档；
2. [EXP-TRAIN-002](EXP-TRAIN-002_clean-deim-mal/README.md)：运行 fixed-640/query-640/global-batch-64 的 controlled DEIM-core 35e 基线；
3. 只有短续训出现值得复验的正信号时，才启动 [EXP-TRAIN-004](EXP-TRAIN-004_sgqs-e-b0-b3/README.md) 正式从头、多 seed 四组对照；
4. 对有正信号的部分补做机制消融，并在 MAL + Dense O2O 基座上复验。

[ADR-003](../docs/research/decisions/ADR-003_training-first-sgqs-e.md) 已替代旧的“先完成全部归因再训练”门禁。`EXP-DIAG-003` 与 `EXP-ABL-001` 仍可辅助解释结果，但不再是配对训练的前置条件。EXP-TRAIN-005 是明确降格的 checkpoint-continuation screen；它不把旧 probe 当作正式 B0/B1，也不把短续训结果当作 EXP-TRAIN-004 的替代。
