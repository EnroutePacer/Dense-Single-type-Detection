# ADR-002：先做 GT 分阶段归因，再启动训练

状态：superseded by [ADR-003](ADR-003_training-first-sgqs-e.md)
日期：2026-10-12

以下保留历史决策。用户随后明确要求以训练改进密集单类型小框收敛，并授权实现、验证和推送；因此不再以完整失效阶段归因阻止受控方法训练。

## 决策

暂停新的 loss、Mosaic、concat 或 selector 训练。先用冻结模型和配套 GT 完成 generation→selection→decoder→output 的分阶段归因，以及同 K selector 诊断对照。

## 原因

当前 13 图没有 GT。P3 入选少或高置信存活少只能说明跨层行为不均衡，无法证明 selection 是主瓶颈。若直接训练，任何 AP 变化都无法和候选生成、选择、decoder 校准及 evaluator 截断建立因果关系。

## 解除条件

EXP-DIAG-003 完成且有效，并能明确支持至少一个可证伪机制分支；随后 EXP-ABL-001 显示简单对照与 oracle 之间仍有稳定空间。
