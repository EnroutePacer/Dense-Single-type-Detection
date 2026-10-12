# P-001：实验与证据协议

版本：1.0（2026-10-10）

每个实验使用永久 `EXP-*`，每次真实执行使用唯一 `RUN-YYYYMMDD-HHMMSS-sNN`。实验设计和运行结果分离，失败、终止、配置未生效的 run 不删除。

训练 run 必须保存：

- 原样启动命令；
- resolved config 与关键运行时断言；
- Git commit/dirty diff 或源码 manifest；
- 数据标注、初始化 checkpoint 与关键配置 SHA-256；
- seed、world size、per-GPU/global batch、epoch、optimizer steps；
- 环境、CUDA/PyTorch/GPU 信息；
- 原始日志、结构化 metrics、协议偏离和结论边界。

没有 resolved config 的训练实验不能标为 `valid`。多个 run 不得复用输出目录，也不得覆盖 console、metrics 或 checkpoint。

大文件进入 `_local/` 或对象存储；公开实验目录只登记 artifact ID、SHA-256、字节数、许可和 storage key。
