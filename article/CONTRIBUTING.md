# Contributing

本仓库以“可复现证据优先”为准则。代码修改和论文表述都必须能追溯到实验、run 与 artifact。

## 新实验

1. 从 `experiments/_template/` 创建新的 `EXP-*` 目录；编号永久不复用。
2. 在运行前写明唯一问题、自变量、不变量、判定标准和失效条件。
3. 每次运行创建唯一 `RUN-YYYYMMDD-HHMMSS-sNN`，不得复用输出目录。
4. 保存真实 `command.sh`、`resolved_config.yaml`、`run.yaml`、环境与哈希。
5. 将结果加入 `experiments/registry.csv`；失败和无效实验同样保留。

## 结论

- `FACT` 必须能由代码或原始产物直接核验。
- 无 GT 定性观察必须标为 `OBSERVATION_NO_GT`。
- 假设必须包含明确证伪条件。
- 论文主张只有满足 `docs/claims/claim_evidence_matrix.yaml` 的准入条件后才能升级为 `supported`。

## 大文件与敏感信息

数据集和临时材料放到 `_local/` 或外部对象存储。正式实验的 `output/`、`outputs/` 不得因整理目录而删除：日志、指标、配置快照和清单进入普通 Git，checkpoint、evaluator、TensorBoard 与归档通过 Git LFS 提交。每项关键产物仍需登记逻辑 ID、SHA-256、字节数与来源；提交前检查绝对路径、用户名、主机名、Token 与密钥。

## 修改活跃代码

论文实现只修改根目录 `engine/`、`configs/`、`tools/`、`scripts/` 和 `tests/`。旧编号目录是不可变历史证据，不作为活跃源码，也不应就地修订。
