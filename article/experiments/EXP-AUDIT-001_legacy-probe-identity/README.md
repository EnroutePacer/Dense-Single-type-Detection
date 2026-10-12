# EXP-AUDIT-001：旧恢复训练探针身份审计

## 问题

旧 `fixed640`、`small_vfl`、`small_reg` 运行是否真正加载了命令中指定的配置？

## 结论

当时的 M launcher 在脚本内部写死基础配置，覆盖外部 `CONFIG_FILE`。console 中的 resolved cfg 与 loss 字段表明，T003～T008 实际仍是基础多尺度 VFL 配方；相关运行不能评价 small-object loss。

审计本身为 `valid`，被审计的目标消融为 `invalid`。它们从 epoch-27 checkpoint 继续训练后 mAP 约下降 0.004～0.007，只能说明这些真实运行没有超过起点，不能把差异归因于目录名中的方法。

当前活跃 launcher 已尊重外部 `CONFIG_FILE`，但这不会追溯性修复旧实验。历史记录见 [legacy_record.md](legacy_record.md)。
