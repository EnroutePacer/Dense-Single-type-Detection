# SGQS v2：实例覆盖驱动的候选分配

训练生命周期已独立：逐步结构化记录、每轮完整 checkpoint、raw/EMA 全量验证与预测留档，以及 epoch 边界恢复，见 [训练框架 README](../../engine/sgqs_v2/README.md)。

2026-10-24 起，本目录覆盖为 [M-003 设计](../../article/methods/M-003_sgqs_v2/README.md) 的训练入口。旧的 r0–r5 单因素筛选停止作为活跃方案，历史配置可在提交 `119bfe2` 查阅；原 SGQS v1 源码与结果保留。

目标是在训练时改善 Encoder 候选对不同实例的覆盖与预算分配。保持训练 query 数 640、原 Matcher、DN/DDF 和推理模块；保留带保护的 owner 几何辅助，以 IoU 为 dense 分类目标，所有新增项在 epoch 28 的一轮内统一升温。新分配项只处理 GT 和 Decoder 反馈共同确认的安全候选关系，不包含旧 survival、EFGL 或 Top-300 admission loss。

| arm | 配置 | 用途 |
|---|---|---|
| `b0` | [b0_e28_r7.yml](b0_e28_r7.yml) | 原生训练对照 |
| `support` | [support_e28_r7.yml](support_e28_r7.yml) | 新 dense、owner、保护和升温，allocation 权重为 0 |
| `v2` | [v2_e28_r7.yml](v2_e28_r7.yml) | 完整新方案，默认启动项 |
| `sgqs` | [sgqs_e28_r7.yml](sgqs_e28_r7.yml) | 可选原 v1 参照，保留其 dense/owner 无升温的历史语义 |

最小主比较为 `b0 / support / v2`。`v2` 对 `support` 才隔离新分配项的作用；`v2` 对原 `sgqs` 同时改变了 dense 目标、归一化、survival 与升温，不能单独归因于 allocation。

`common.yml` 固定训练与评价约定，`base_v2.yml` 固定新方法参数。所有臂均使用冻结的 epoch-27 checkpoint、8 GPU × 每卡 8 图、seed 0，续训绝对 epoch 28–34。单 seed 七轮是筛选，尚无新训练或性能结论。

训练时 PostProcessor 与 CocoEvaluator 均使用 640。完整验证工具按每轮 raw/EMA 输出 COCO 各尺度和本地 hybrid-small 严格一对一 AP/AR；主预算为 640，300 为附加排名诊断。本地 small 为直接缩放到 640 后 `area < 32² OR short side < 16`，不与 COCO-small 混称。高密度场景按固定标注中每图 GT 数 ≥301 定义。

真实计算预算比较使用同一 checkpoint，将 Decoder 输入 K 改为 640/512/384/256 并分别重跑；不对完整 640 输出静态切片，不把升分或过滤数等同于加速。

启动与评估详见 [SERVER_COMMANDS.md](SERVER_COMMANDS.md)。准备工具记录完整源码、配置、输入哈希与运行身份；旧六臂 manifest 不兼容新 schema。新主入口直接使用 criterion 的日志，不再运行原重型 GO/DDF 梯度探针。运行目录已存在即停止，不能混入旧结果。
2026-10-27修订：冗余改为原始网格anchor中心唯一位于GT内 + Decoder IoS>=0.8；可靠代表/救援仍用IoU。有可执行接收者时不设备用数量门槛；整图无配对时，容忍每GT除代表外10个候选，仅抑制溢出中Encoder score>0.8的安全donor。配对和溢出共享allocation权重，support同时关闭二者。新日志增加donor_ios_only、donor_center_ambiguous、overflow_excess/overconfident/penalized与overflow_score。旧5f56027运行不得与此修订混为同一方法状态。
