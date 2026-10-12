# 仓库与证据策略

## 双层结构

公开层包含可运行源码、小型配置、实验设计、结构化指标、哈希、论文材料，以及开发阶段 `output/`、`outputs/` 中的正式实验记录。文本记录进入普通 Git；checkpoint、evaluator、TensorBoard 和归档由 Git LFS 保存。本地层 `_local/` 只承载临时材料、不可公开的数据及额外备份。

“已复制到论文文件夹”不等于“可提交 Git”或“可公开再分发”。每项资产必须独立记录来源、许可、字节数和 SHA-256。

## 自动整理新材料

当其他工作目录出现新的训练代码或日志时：

1. 对比活跃源码并保存 commit/diff 或 source manifest；
2. 为研究问题创建/更新 `EXP-*`，为每次执行创建 `RUN-*`；
3. 原始日志、metrics、resolved config 和环境信息写入唯一的 `output/<RUN>/`，计算哈希并提交；
4. checkpoint、evaluator、TensorBoard 和归档通过 Git LFS 提交，同时将结构化摘要写入实验目录；
5. artifact registry 增加 `ART-*`；
6. 更新 facts、claims 和 paper 时只引用注册 ID；
7. 执行仓库验收脚本，记录失效实验而不是删除它。

## 公开门禁

- 原始数据、缓存和未经审计的 raw log 不得进入 Git；正式输出提交前必须完成敏感信息检查。
- checkpoint、tar、TensorBoard 等大文件只能以 Git LFS pointer 进入 Git；普通 Git 对象默认不得超过 10 MiB，50 MiB 以上必须硬阻止。
- 不得出现本机/服务器绝对路径、用户名、主机名、凭据或私钥。
- Markdown 相对链接、YAML/JSON/CSV 语法和 source hash 必须通过检查。
- 干净 clone 必须能解析 canonical config，并能运行最小 CPU smoke test。

## 历史错误实验

错误、无效和被新解释取代的实验不会删除；它们用 `validity=invalid` 或 `superseded` 登记，并明确能支持和不能支持的结论。
