# 证据完整性

## 当前状态

2026-10-10 已重新核验以下四项 SHA-256：服务器 tar、DEIM checkpoint、PaddleX wired 参数、PaddleX wireless 参数；均与注册表一致。

旧原始证据仍位于编号目录，未移动、删除或就地编辑。这样保留了现有 20 个 Markdown 相对链接和旧校验脚本的有效性。

## 两套清单

- `private_inventory.sha256`：本地全部原始资产；不得包含在公开仓库或暴露机器绝对路径。
- `public_release_manifest.sha256`：实际公开文件；应在干净 clone 中重新生成和核验。

## 迁移规则

大资产若需要迁移，必须按“复制 -> 文件数/字节数/逐文件哈希比较 -> 更新 locator 与链接 -> 新位置复验 -> 保留旧副本一个备份周期 -> 再决定清理”的顺序。禁止直接剪切后再补证据。

历史源码快照见 [`source_snapshot.sha256`](../reproducibility/source_snapshot.sha256)；
当前运行身份以 Git commit 和对应运行目录内的 manifest/runtime hash 为准。
