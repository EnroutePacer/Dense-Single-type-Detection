# Dense Single-Class Detection

这是一个以表格单元格检测为验证基座、研究**密集单类别检测**的论文代码仓库。当前核心问题是：固定 query 预算下，global score-only Top-K 是否造成高分辨率候选的 **query selection–survival gap**。

当前结论必须保持克制：13 张无 GT 图片的先导诊断显示 P3/P4/P5 的 Top-K 入选明显不均衡；进一步从 `image (12)` 的 GT 可视化恢复框后，观察到 dense→Top-K 覆盖大幅下降，selection starvation 已成为主假设。但该 GT 来自图片恢复而非原始 COCO JSON，正式分阶段归因仍是最高优先级；训练暂不推进。

## 当前仓库状态

- 活跃代码来自 2026-10-10 同步后的 `DEIM_2`，保留 DEIM 原生根布局：`train.py`、`engine/`、`configs/`、`tools/`。
- 同步时排除了 `output/`、checkpoint、缓存和字节码；代码快照有独立 SHA-256 清单。
- 历史服务器快照、模型、原始日志、诊断 CSV 和未确认授权的图片仍保留在旧编号目录中，并由 `.gitignore` 隔离；本轮没有移动、删除或改写它们。
- 旧 small-loss probes 已确认没有加载其声称的配置，属于“运行完成但目标消融无效”，不会被包装成论文结果。
- 三个新 12,800-step 摘要已归档：small-reg 没有改善 AP-small/AR-small，MAL+Dense O2O 摘要结果明显退化；由于缺少 console/resolved config，目前只记录数值，不做方法归因。
- 已实现候选方法 [M-001 SGQS](methods/M-001_sgqs/README.md)，在训练期直接监督 Top-K 前的 encoder 候选、owner 回归与生存排名；当前状态为 implemented/unvalidated，尚无训练效果结论。
- 正式 COCO 主评估固定使用 `maxDets=[1,10,300]`；机制诊断在 evaluator 截断前进行。

## 研究入口

1. [研究问题](docs/research/research_question.md)
2. [研究状态与下一步](docs/research/status.md)
3. [科研进展日志](docs/research/progress_log.md)
4. [事实注册表](docs/research/facts/README.md)
5. [假设注册表](docs/research/hypotheses/README.md)
6. [论文主张—证据矩阵](docs/claims/README.md)
7. [实验注册表](experiments/registry.csv)
8. [证据与仓库策略](docs/reproducibility/repository_policy.md)
9. [SGQS 候选方法](methods/M-001_sgqs/README.md)
10. [12,800-step 结果审计](experiments/EXP-AUDIT-002_12800-probe-results/README.md)

## 仓库结构

```text
train.py, engine/, configs/, tools/   可直接运行的 DEIM 代码
engine/sgqs/, configs/sgqs/           SGQS 训练模式与独立配置
scripts/                              训练、评估、诊断、复现与维护入口
tests/                                配置链和 selector 行为测试
docs/research/                        状态、事实、假设、协议、分析与决策
docs/claims/                          论文主张及准入条件
docs/evidence/                        artifact 注册、完整性和证据索引
docs/reproducibility/                 代码来源、环境和仓库政策
experiments/                          EXP/RUN 两级实验记录
methods/                              候选方法；未验证前不得写成贡献
data/, models/                        数据与模型 manifest，不存私有原件
paper/                                论文提纲、图表、正文和补充材料
_local/                               Git 忽略的本地权重、数据、run 和原始证据
```

根目录继续保留 DEIM 原生代码布局，是为了让服务器训练命令、Python import 和 YAML include 不因人为套一层 `src/` 而再次发生路径偏差。

## 实验纪律

每个论文实验必须拥有唯一 `EXP-*`，每次执行拥有唯一 `RUN-*`。最低证据包括实际命令、resolved config、代码/数据/初始化权重哈希、环境、seed、批量大小、完整评估口径、原始指标、日志及结论边界。目录名和启动命令都不能替代运行时配置证据。

大 checkpoint、TensorBoard event、完整诊断 CSV 和原始数据不进入普通 Git；它们保存在 `_local/`、对象存储或 release 中，公开仓库只提交逻辑 ID、大小、SHA-256、许可与获取方式。

## 当前合规表述

> 现有无 GT 诊断显示跨层 Top-K 来源及后续高置信存活存在显著不均衡，由此提出 query selection–survival gap 假设；真实瓶颈位于 generation、selection 还是 decoder，仍需逐 GT 分阶段验证。

在完成带 GT 的归因实验前，不得写成“小目标主要死在 encoder Top-K”或“小目标主要死在 decoder”。

## 历史材料

旧 `00_overview/`、`01_sources/`、`02_experiments/`、`03_evidence/`、`04_analysis/`、`05_paper/`、`99_archive/` 是 2026-10-10 前形成的本地快照和原始证据。它们暂时保留原位以维持哈希与链接链路，但不属于默认公开内容；canonical 文档均位于新结构中。
