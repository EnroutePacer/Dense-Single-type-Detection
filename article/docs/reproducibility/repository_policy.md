# 仓库与证据策略

## 双层结构

公开层包含可运行源码、小型配置、实验设计、结构化指标、哈希和论文材料。本地层 `_local/` 以及暂存的旧编号目录包含权重、数据、原始日志、TensorBoard、完整 CSV、图片和归档。

“已复制到论文文件夹”不等于“可提交 Git”或“可公开再分发”。每项资产必须独立记录来源、许可、字节数和 SHA-256。

## 自动整理新材料

当其他工作目录出现新的训练代码或日志时：

1. 对比活跃源码并保存 commit/diff 或 source manifest；
2. 为研究问题创建/更新 `EXP-*`，为每次执行创建 `RUN-*`；
3. 原始日志复制到 `_local/raw_evidence/<EXP>/<RUN>/`，计算哈希；
4. 将小型、脱敏的 metrics/summary/resolved config 写入实验目录；
5. artifact registry 增加 `ART-*`；
6. 更新 facts、claims 和 paper 时只引用注册 ID；
7. 执行仓库验收脚本，记录失效实验而不是删除它。

## 公开门禁

- Git 索引不得包含权重、原始数据、tar、TensorBoard、缓存或未经审计的 raw log。
- 默认阻止大于 10 MiB 的普通 Git 文件，50 MiB 以上必须硬阻止。
- 不得出现本机/服务器绝对路径、用户名、主机名、凭据或私钥。
- Markdown 相对链接、YAML/JSON/CSV 语法和 source hash 必须通过检查。
- 干净 clone 必须能解析 canonical config，并能运行最小 CPU smoke test。

## 历史错误实验

错误、无效和被新解释取代的实验不会删除；它们用 `validity=invalid` 或 `superseded` 登记，并明确能支持和不能支持的结论。
