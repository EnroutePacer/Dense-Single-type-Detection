# Candidate methods

这里存放由诊断证据支持的候选方法设计。当前没有任何方法被标记为已验证贡献。

当前候选：

- `M-001_sgqs`：Scale-aware GT Query Survival。由恢复 GT 的先导审计支持，直接监督 Top-K 前的 encoder 候选及其生存排名；代码与配置已实现，尚未训练或验证。
- `M-002_sgqs_e`：SGQS-E。当前最新主线，组合候选监督与邻域/短边感知 EFGL；数学评审后使用非 owner 竞争门槛和基础监督保护。代码、四组消融配置与 CPU 验证已完成，尚无真实训练效果验证。见[评审](M-002_sgqs_e/math_review.md)与[实现](M-002_sgqs_e/implementation.md)。

M-001 必须继续与已实现代码保持一致，M-002 作为新方法单独演进。首轮只比较 baseline、SGQS、EFGL、SGQS+EFGL；邻居约束 DN 暂不并入，避免堆叠第三个变量。

一个方法目录至少包含 problem link、算法、复杂度、训练/推理差异、实现文件、单测、消融和失败模式。根据用户最新目标，允许直接开展受控训练以验证有效设计；完整阶段归因不是启动前置条件，相关历史决策已由 ADR-003 替代。
