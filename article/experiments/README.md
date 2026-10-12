# 实验系统

实验采用两级标识：

- `EXP-*`：一个预注册的科学问题与受控设计；
- `RUN-YYYYMMDD-HHMMSS-sNN`：一次真实执行。

编号永久保留，标题变化不改编号。`status` 表示运行流程，`validity` 表示科学有效性；`completed + invalid` 是合法且必须保留的状态。

推荐编号：`EXP-TRAIN`、`EXP-EVAL`、`EXP-DIAG`、`EXP-BENCH`、`EXP-ABL`、`EXP-AUDIT`。旧 `T001/I009` 只放在 `legacy_ids`。

每个实验从 `_template/` 创建，并登记到 [registry.csv](registry.csv)。大文件不进 Git，只通过 `ART-*` 引用。

旧实验尚未拥有逐次 `RUN-*` 元数据。注册表分别记录 `legacy_run_count` 与 `registered_run_count`，并将其标为 `run_migration_status=unmapped`；这表示历史结果仍可审计，但 ART→RUN→EXP 链尚未补齐，不能伪装成已完成的新规范记录。

## 当前顺序

1. [EXP-TRAIN-005](EXP-TRAIN-005_epoch27-sgqs-e-resume-screen/README.md)：完成已启动的 B0 与同 checkpoint 的 B1/B2/B3 七 epoch 续训屏选；
2. [EXP-TRAIN-002](EXP-TRAIN-002_clean-deim-mal/README.md)：运行 fixed-640/query-640/global-batch-64 的 controlled DEIM-core 35e 基线；
3. 只有短续训出现值得复验的正信号时，才启动 [EXP-TRAIN-004](EXP-TRAIN-004_sgqs-e-b0-b3/README.md) 正式从头、多 seed 四组对照；
4. 对有正信号的部分补做机制消融，并在 MAL + Dense O2O 基座上复验。

[ADR-003](../docs/research/decisions/ADR-003_training-first-sgqs-e.md) 已替代旧的“先完成全部归因再训练”门禁。`EXP-DIAG-003` 与 `EXP-ABL-001` 仍可辅助解释结果，但不再是配对训练的前置条件。EXP-TRAIN-005 是明确降格的 checkpoint-continuation screen；它不把旧 probe 当作正式 B0/B1，也不把短续训结果当作 EXP-TRAIN-004 的替代。
