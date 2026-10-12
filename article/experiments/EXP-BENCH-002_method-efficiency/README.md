# EXP-BENCH-002：方法模型部署效率

状态：planned / gated。

沿用 P-004 的固定硬件、线程、输入、预热和重复协议，分别测 selector、模型 forward 与完整 pipeline。必须固定 decoder query 数并报告参数量、FLOPs；未经实测不得声称“零开销”。
