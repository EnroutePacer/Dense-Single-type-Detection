# 环境记录要求

每个 RUN 至少记录：操作系统、Python、PyTorch/Paddle、CUDA、cuDNN、GPU 型号和数量、驱动、pip/conda freeze 哈希、CPU、线程与 affinity。

公开环境文件使用匿名主机别名，不记录用户名、内网地址、密码、Token 或私有绝对路径。完整私有环境快照放 `_local/raw_evidence` 并登记 artifact。
