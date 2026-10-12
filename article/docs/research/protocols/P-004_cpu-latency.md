# P-004：固定 CPU Latency 协议

版本：1.0（2026-10-10）

- 硬件：Intel Core i7-14700K；固定逻辑 CPU 2（P-core）。
- 单线程、batch size 1、FP32。
- PyTorch MKLDNN 与 Paddle oneDNN 均关闭。
- 20 次预热；13 张固定图片各 20 次，共 260 个正式样本。
- inference latency 排除模型加载、磁盘读图、绘图和文件写入。
- pipeline latency 的边界必须在 benchmark config 中明确。
- 报告均值、P50、P95、样本数、CPU affinity、线程、后端、模型哈希和输入哈希。
- PyTorch eager 与 Paddle inference 的差异代表当前部署路径，不解释为纯架构算子差异。

如果输入、权重、线程、后端、预热或重复次数发生变化，必须创建新协议版本，不能与旧表直接拼接。
