# SGQS v2：Support / SGQS epoch28--34 训练证据

本目录保存 2026-10-28 至 2026-10-29 完成的两条受控续训链。每条链因先做
3 epoch 探针、再继续 4 epoch，被保存在两个运行目录中；它们不是四种方法。
服务器端失败的冲突目录已经核验并清除，未混入本目录。

| 方法 | 阶段 | 目录 | 完整 epoch | 状态 |
|---|---|---|---|---|
| Support | 第一段 | `support_native_coverage_e28_e30_seed0_finalfix` | 28--30 | complete |
| Support | 第二段 | `support_native_coverage_e31_e34_seed0_finalfix` | 31--34 | complete |
| SGQS v2 | 第一段 | `v2_native_coverage_e28_e30_seed0_finalfix` | 28--30 | complete |
| SGQS v2 | 第二段 | `v2_native_coverage_e31_e34_seed0_finalfix_final` | 31--34 | complete |

`v2_` 前缀表示启用完整 allocation 的 SGQS v2 实验臂。Support 使用相同几何与
dense 监督，但 `allocation_weight=0`。

## 连续性

- Support epoch30 SHA-256：
  `cdf1fc3e880735444cbd489f561482dd81f38fa1c47679100e7408bd3cc22a53`；
  与第二段 `run_manifest.json` 的初始 checkpoint SHA 完全一致。
- SGQS epoch30 SHA-256：
  `959e9a7993ab0966f2fcf8c997a750dfda0c50690ab2ce1bcb3dc5b86ce0e5f4`；
  与第二段 `run_manifest.json` 的初始 checkpoint SHA 完全一致。
- 14 个 epoch 均有 `epochXXXX.summary.json`，每轮 5631 个训练 step。
- 四个目录均有 `completed.json`，终点分别是 epoch30 或 epoch34。

## 证据分层

普通 Git 文件包括：

- `console.log`；
- `resolved.yml`、`run_manifest.json`、`completed.json`；
- checkpoint 的 JSON sidecar 与 `latest.json`；
- 每轮 epoch summary 和 attempt 起点记录。

第二次大文件推送通过 Git LFS 保存：

- `checkpoint*.pth`；
- `epoch*.steps.jsonl`；
- `epoch*.rank*.samples.jsonl`。

全部 144 个大文件的相对路径、字节数和源文件 SHA-256 见
[`LARGE_ARTIFACTS.csv`](LARGE_ARTIFACTS.csv)。该清单记录真实内容哈希，
不是 Git LFS pointer 的哈希。

训练时显式设置了 `--eval-every 0 --no-eval-initial`，所以这些目录只包含训练
证据，不能从训练日志中虚构验证 AP。完整验证需用 `full_val_replay.py` 对
epoch28--34 EMA 做独立回放；结果生成后再作为新的评价证据登记。
