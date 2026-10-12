# 科研进展日志

## 2026-10-12 — 将 e35/r7 数学基准钉死到 6fd2c30

- 将提交 `6fd2c301bd8c09ff3d49fa077c34eaf75ef35b9a` 定为 M-002/SGQS-E 的唯一规范数学与实现基准；逐文件 canonical-LF SHA-256 已登记。
- 审计确认 owner、SGQS 四项损失与系数、non-owner survival、监督保护、EFGL 逐边公式、Matcher、GO/DDF、DN 和推理路径没有漂移。
- 唯一 allow-list 差异是在 EFGL 中增加 resume 课程起点与持续时间；默认值严格恢复原公式，r7 仅把同一时间坐标平移到 epoch 28，并用 `7×0.30=2.1` epoch 退火。
- 新增 fail-closed preflight：完整解析 include/CLI 后检查方法哈希、消融开关、损失、课程、fixed-640、query 640、global batch 64、checkpoint last_epoch=27、数据/权重哈希和 maxDets=300；只检模式不创建输出、不启动 torchrun。
- 五份配置契约全部通过；根方法测试 32 项、article 镜像与仓库契约 38 项通过。仍未启动 B1/B2/B3，没有新增 AP 结论。

## 2026-10-12 — 冻结 e35/r7 打靶矩阵

- 确认根目录 `configs/table_cells/` 只保留 M fixed-640 与 S 两份任务基配置是有意精简；新一次性计划集中到 `configs/e35_r7/`，不恢复历史 probe 屎山。
- 预注册 controlled DEIM-core 35e：保留 fixed-640、query 640、表格 optimizer/ReLU，只切换 MAL + Dense O2O；它不是照搬官方 COCO 全部超参。
- 预注册 EXP-TRAIN-005：同一 epoch-27 checkpoint 续训到绝对 epoch 35，B0/B1/B2/B3 分别为 control、SGQS、EFGL、SGQS+EFGL；B0 据报已运行，新入口默认拒绝重启。
- 修复短续训的 EFGL 课程退化：新增显式 `compensation_start_epoch=28` 和 `compensation_duration_epochs=2.1`，默认从头训练语义保持不变。否则 epoch 28 时 EFGL 已归零，B2/B3 会数值退化。
- 冻结共享服务器输入身份：train annotation `816859c5...d0e4e`、val annotation `a6d10732...d72b`、checkpoint0027 `3eba494a...4ede`、DEIM tuning checkpoint `2b6cd058...19bc`。启动脚本默认校验完整 SHA-256。
- 本条只记录配置、课程和复现入口；未启动新 run，没有新 AP 结论。

## 2026-10-12 — 恢复并持久化实验输出

- 撤销“整体忽略 `output/`、`outputs/`”的清理策略；正式实验输出是开发证据，不得因目录整理而删除。
- 从清理前的本地只读归档恢复输出；文本日志、指标、配置快照和清单使用普通 Git，evaluator、TensorBoard、checkpoint 与归档使用 Git LFS。
- `_local/cleanup_20261012/` 原始备份保持不变；恢复目录另生成逐文件 SHA-256 清单，便于核对远端与本地内容。

## 2026-10-12 — SGQS-E 数学评审与实现

- 同步远程 `927b2c6` 的最新方案，保留 M-001 与有效训练设计，不因思想重叠删除模块。
- 复现 K=A 时原 survival 门槛的自追逐反例，改为非 owner 第 K-A+1 名；实际 selector 与真实 cutoff 日志保留。
- 复用原 Encoder 分类及 GO-union 回归索引，保护已有正样本并跳过冲突 owner 的新增监督；不修改 Matcher。
- 实现 EFGL，保持插值 CE、DDF 和 DN 采样；固定 kappa=1/2，仅保留前 30% 的补偿退火。实际输入尺寸由训练循环传入，邻域图每 batch 复用。
- 修复 YAML 顶层加载共用默认字典导致的配置污染，提供独立 B0/B1/B2/B3 配置。
- 完成 29 项方法测试及 4 项仓库契约测试、VFL/MAL + DN 完整前后向、原生训练入口与微型 Decoder trace 检查。完整 HGNetv2-M 的 B0/B3 同权重 640×640 输出逐元素一致。真实数据训练、CUDA/DDP 和目标部署后端尚未验证；没有新增 AP 结论。
- ADR-003 按用户最新授权替代旧的“先完成全部归因再训练”门禁。完整公式、评审、运行命令见 M-002 方法目录。
- 登记 EXP-TRAIN-004 的 B0/B1/B2/B3 配对计划，runs 为空；新增 LF 规范化源码快照，历史字节快照保持不变。

## 2026-10-12 — 公开仓库清理

- 活跃根配置树只保留 fixed-640 M 控制配置与 S 配置；旧 probe、multiscale、small-reg/MAL 和 epoch-28 benchmark 根副本退出活跃入口。
- 根 `output/` 中本地仍存在的 checkpoint、TensorBoard、原始日志和传输包移入 Git 忽略的 `_local/cleanup_20261012/`；公开树只保留 `article/experiments/` 中的结构化审计。
- 移除 4 份无入链且已被 canonical 文档取代的 legacy 分析；保留 M-001 实现、EXP/ART 记录、legacy records 以及受哈希/来源链约束的 `article/` benchmark 快照。
- 本次没有实现 M-002 或启动新训练。

## 2026-10-12 — SGQS-E 最新方案冻结

- 基于当前两个研究问题及其训练差异总结，建立 M-002 SGQS-E；状态为 `designed / unimplemented / unvalidated`。
- 方法仅含两部分：SGQS 在 global Top-K 前提供唯一 owner、dense 分类、owner 回归与 cutoff survival；EFGL 用短边、逐边相对误差和同类邻域歧义替换原 FGL 的四边共享 IoU 权重。
- 完整记录总损失、邻域几何、课程、初值、梯度边界和 B0/B1/B2/B3 最小消融。静态预处理 GT 点被否决，owner 必须在全部数据增强之后在线生成。
- 邻居约束 DN 暂不并入首版；DDF/教师 KL、Matcher、query、预测头与推理路径保持不变，以保证能分别归因候选链和回归链。
- 本条是方法设计记录，不是训练结果；没有新增 AP、收敛或 latency 证据。

## 2026-10-12 — SGQS v1 代码实现

- 将 M-001 从 `candidate` 更新为 `implemented / unvalidated`。
- 新增训练期 `SGQSQuerySupervisor` 与包装式 `SGQSCriterion`，监督位置位于 Encoder 分类头之后、global Top-K 之前。
- 实现分层 dense classification、唯一 owner L1/GIoU 和带质量下限的 Top-K survival 三条损失路径；即使 owner IoU 为 0，survival 仍保留受控梯度。完整数学定义见 `methods/M-001_sgqs/losses.md`。
- 新增 clean fixed-640 正式配置和 epoch-27 continuation probe 配置。
- 训练日志单独记录 assignment miss、owner coverage、P3/P4/P5 owner 数、owner IoU、cutoff gap、真实 pre-Top-K survival 与预算溢出率；这些指标不参与总损失。
- 单元测试覆盖尺度分配、碰撞、空 GT、预算溢出、零 IoU 生存梯度、invalid-anchor Top-K、配置注册、state_dict 和同版本 eval 输出等价，共 11 项。
- 同时修复 invalid boundary anchor 占用 Top-K 的共享问题。该项不是 SGQS 贡献，后续必须在同一提交上重跑 clean control，不能与历史 0.864248 直接做方法归因。
- 本条只记录实现完成，不代表方法获得指标增益；尚未启动训练。

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
