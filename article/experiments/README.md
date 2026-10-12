# 实验系统

实验采用两级标识：

- `EXP-*`：一个预注册的科学问题与受控设计；
- `RUN-YYYYMMDD-HHMMSS-sNN`：一次真实执行。

编号永久保留，标题变化不改编号。`status` 表示运行流程，`validity` 表示科学有效性；`completed + invalid` 是合法且必须保留的状态。

推荐编号：`EXP-TRAIN`、`EXP-EVAL`、`EXP-DIAG`、`EXP-BENCH`、`EXP-ABL`、`EXP-AUDIT`。旧 `T001/I009` 只放在 `legacy_ids`。

每个实验从 `_template/` 创建，并登记到 [registry.csv](registry.csv)。大文件不进 Git，只通过 `ART-*` 引用。

旧实验尚未拥有逐次 `RUN-*` 元数据。注册表分别记录 `legacy_run_count` 与 `registered_run_count`，并将其标为 `run_migration_status=unmapped`；这表示历史结果仍可审计，但 ART→RUN→EXP 链尚未补齐，不能伪装成已完成的新规范记录。

## 当前顺序

1. `EXP-DIAG-003`：带 GT 的分阶段归因；
2. `EXP-ABL-001`：同 K、冻结模型的 selector 对照；
3. 根据诊断结果决定方法；
4. `EXP-TRAIN-001/002` 暂不运行。
