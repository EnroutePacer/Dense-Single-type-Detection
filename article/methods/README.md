# Candidate methods

这里存放由诊断证据支持的候选方法设计。当前没有任何方法被标记为已验证贡献。

当前主线为 [M-003 SGQS v2](M-003_sgqs_v2/README.md)：保留几何辅助，通过跨阶段 GT 身份监督候选预算分配；已完成设计、实现与CPU检查，未训练验证。

历史候选保留：

- `M-001_sgqs`：Scale-aware GT Query Survival。代码与配置已实现，并作为 B1 完成一次 epoch-27 单 seed 续训屏选；总 mAP 为弱正信号，但小目标终点低于 B0，尚未正式验证。
- `M-002_sgqs_e`：SGQS-E。组合候选监督与邻域/短边感知 EFGL；代码、四组消融配置与 CPU 验证已完成，B0--B3 单 seed 短续训矩阵也已归档。状态为 `implemented / screened / unvalidated`，尚无稳定收益、small-object 改善或互补性证据。见[评审](M-002_sgqs_e/math_review.md)、[实现](M-002_sgqs_e/implementation.md)与[屏选结果](../experiments/EXP-TRAIN-005_epoch27-sgqs-e-resume-screen/diagnostics/20261013_b0_b3_c2_followup/README.md)。

M-001/M-002 继续保留历史实现与证据。当前主比较由 EXP-TRAIN-008 定义为 b0/support/v2，原 SGQS 可选参照；旧四臂及单因素探针不再是活跃计划。

一个方法目录至少包含 problem link、算法、复杂度、训练/推理差异、实现文件、单测、消融和失败模式。根据用户最新目标，允许直接开展受控训练以验证有效设计；完整阶段归因不是启动前置条件，相关历史决策已由 ADR-003 替代。
