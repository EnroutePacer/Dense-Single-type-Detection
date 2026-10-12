# EXP-BENCH-001：固定 CPU 推理延迟

## 协议

Intel Core i7-14700K、逻辑 CPU 2、单线程、batch 1、FP32，关闭 PyTorch MKLDNN 与 Paddle oneDNN；20 次预热，13 图各 20 次，共 260 个正式样本。核心 inference latency 不含模型加载、磁盘读图、绘图和写盘。

## 主要结果

| 配置 | inference mean | pipeline mean |
|---|---:|---:|
| DEIM 640 | 888.091 ms | 894.001 ms |
| PaddleX-L 分域组合 | 1606.122 ms | 1609.734 ms |
| DEIM 512 | 687.173 ms | 692.223 ms |
| DEIM 800 | 1355.416 ms | 1364.638 ms |
| DEIM 512/800 | 2034.097 ms | 2053.569 ms |

这些结果可比较当前部署路径的固定 CPU 延迟，不是纯算子速度；13 图无 GT，输出框数量不能当作精度。完整历史表见 [legacy_record.md](legacy_record.md)。
