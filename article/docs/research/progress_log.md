# 科研进展日志

## 2026-10-12：GT 可视化先导归因与 SGQS 立项

- 将 `DEIM_2` 初始化为 Git 仓库并推送到 `EnroutePacer/Dense-Single-type-Detection`；补入新到达的 epoch-28 benchmark 配置后，远端 `main` 与提交 `b275a3e` 一致。权重和 TensorBoard 二进制未进入 Git。
- 归档三个新 12,800-step 指标摘要。small-reg 相对同周期 control 只有 +0.000514 mAP，AP-small/AR-small未提升；MAL+Dense O2O 相对 control 下降 0.009203 mAP。由于缺少新 run 的 console/resolved config，方法归因仍为 partial。
- 从 `image (12)` GT 可视化恢复 322 个框并与 dense encoder 导出配准。304 个正文框在 IoU≥0.5 下从 dense 278 个覆盖降至 Top-640 的 135 个，形成 selection starvation 的强先导证据。
- 明确论文对象应表述为 dense thin / small-short-side detection，而不只是 COCO small-object detection。
- 建立候选方法 M-001 SGQS：短边尺度感知 GT 点分配、Top-K 前全候选监督、GT 点回归和 cutoff-aware survival loss。训练仍暂停，等待原始 COCO JSON 与正式归因。
- 将最新 epoch-28 benchmark 源码同步到活跃树，并保存 `ART-000022` 服务器传输包；保留论文仓库 5 个既有可移植性补丁，未启动这些训练。

## 2026-10-12：仓库与研究主线重构

- 将同步后的 DEIM 代码提升为根目录活跃代码，保留原生运行布局。
- 建立 EXP/RUN/ART/F/H/C/P/ADR 标识和 claim–evidence 链。
- 将核心问题收敛为 query selection–survival gap，并把 generation、selection、decoder、output 设为并列竞争解释。
- 旧训练、推理和审计材料保留字节不变；公开层只保存脱敏摘要、哈希与 locator。
- 参数化训练环境与 benchmark 根目录，活跃公开树的严格机器路径门禁通过。
- 训练继续暂停；下一项是 EXP-DIAG-003 带 GT 分阶段归因。

## 2026-10-10：问题发现与历史审计

- 发现旧 small-loss probes 未加载声称配置，登记为 EXP-AUDIT-001。
- 确认同 checkpoint 的 `maxDets=100/300` 口径差异，登记为 EXP-EVAL-001。
- 完成固定 CPU latency 和 DEIM/PaddleX 定性输出，登记为 EXP-BENCH-001 与 EXP-DIAG-002。
- 完成 13 图无 GT query 先导诊断，登记为 EXP-DIAG-001。
- 明确先导观察不足以判断真实 GT 在哪个阶段丢失。

后续每次自动整理新代码或日志时，在这里追加日期、EXP/RUN/ART ID、得到的新事实以及被否定的假设；不能只写“完成训练”而没有证据引用。
