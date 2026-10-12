# I001～I007：固定 CPU 推理基准

> **Historical snapshot**：旧路径指向本地私有 legacy 证据，公开追溯请使用 [artifact registry](../../docs/evidence/artifact_registry.csv)。canonical 口径见本实验的 [README](README.md)。

## 固定协议

- 主机：Intel Core i7-14700K。
- affinity：逻辑 CPU 2（P-core）。
- 线程：1；batch size：1；FP32。
- Paddle oneDNN 与 PyTorch MKLDNN 均关闭。
- 阈值：严格 `score > 0.5`。
- 20 次预热；13 张图各重复 20 次，共 260 个正式样本。
- 核心 latency 排除模型加载、磁盘读图、绘图和文件写入。
- PaddleX 的 wired/wireless 使用各自权重；DEIM 两组使用同一 checkpoint。

## 结果

| 配置 | inference mean | pipeline mean | P50 pipeline | P95 pipeline |
|---|---:|---:|---:|---:|
| DEIM 640 | 888.091 ms | 894.001 ms | 874.530 ms | 994.928 ms |
| PaddleX-L 分域组合 | 1606.122 ms | 1609.734 ms | 1585.574 ms | 1703.882 ms |
| DEIM 512 | 687.173 ms | 692.223 ms | 689.428 ms | 725.919 ms |
| DEIM 800 | 1355.416 ms | 1364.638 ms | 1359.591 ms | 1421.284 ms |
| DEIM 512/640/800 | 2983.693 ms | 3014.270 ms | 3005.500 ms | 3119.885 ms |
| DEIM 512/800 | 2034.097 ms | 2053.569 ms | 2044.833 ms | 2136.027 ms |
| DEIM tiled | 2262.428 ms | 2276.226 ms | 1857.192 ms | 5492.989 ms |

PaddleX 分组 inference mean：wired 1665.761 ms，wireless 1579.616 ms。

## 解释边界

- 这些数据可靠比较固定硬件/线程/后端下的 CPU 延迟。
- PyTorch eager 与 Paddle inference 是不同运行时；结果代表当前部署路径，不是纯架构算子速度。
- 多尺度的 forward 次数是 2 或 3，因此延迟接近线性增加。
- tiled 的 tile 数随图像尺寸变化，P95 很高，不能只看总体均值。
- 当前 13 张图没有 GT；检测框数量变化不能当作 mAP 提升。

## 证据

完整原始数据位于 `../03_evidence/local_inference_benchmark_20261010/output/`。每个正式目录均包含 `latency_raw.csv`、`latency_summary.json`、`benchmark_config.json`、`detections.json`、`run.log` 和可视化。

