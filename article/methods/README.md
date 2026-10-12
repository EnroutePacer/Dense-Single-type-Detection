# Candidate methods

这里存放由诊断证据支持的候选方法设计。当前没有任何方法被标记为已验证贡献。

当前候选：

- `M-001_sgqs`：Scale-aware GT Query Survival。由恢复 GT 的先导审计支持，直接监督 Top-K 前的 encoder 候选及其生存排名；尚未实现、训练或验证。

备选分支仍由正式诊断决定：若 generation、decoder 或 output 才是主瓶颈，则否定或收缩 M-001，而不是堆叠更多组件。

一个方法目录至少包含 problem link、算法、复杂度、训练/推理差异、实现文件、单测、消融和失败模式。不得在诊断完成前预先堆叠多个组件。
