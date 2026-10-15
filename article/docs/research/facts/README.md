# 事实与观察注册表

状态分三类：`FACT` 可由代码或原始产物直接核验；`OBSERVATION_NO_GT` 是没有 GT 的行为观察；`OBSERVATION_RECOVERED_GT` 来自 GT 可视化恢复，可用于立题，但不能替代原始 annotation JSON 的正式评估。

| ID | 状态 | 内容 | 证据边界 |
|---|---|---|---|
| F-001 | FACT | 640×640 输入产生 P3/P4/P5 共 8,400 个位置，当前模型按单类 encoder logit 做 global Top-640。 | 静态代码与诊断导出可核验；不代表所有位置都能形成高质量 GT proposal。 |
| F-002 | FACT | 标准前向只对 Top-K 位置构造 encoder 初始框；诊断脚本额外导出全部位置的预测框。Top-K 索引不可微，decoder 初始化被 detach。 | decoder 仍读取完整 memory，不能说未入选位置完全不可见。 |
| F-003 | OBSERVATION_NO_GT | 13 图的 Top-K 来源为 P3/P4/P5=3,356/4,062/902；1,432 个 `score>0.5` 输出的来源为 3/1,429/0。 | 分数阈值输出不等于 TP 或 recall。 |
| F-004 | OBSERVATION_NO_GT | `wired/image (12)` 的 Top-640 来源为 153/425/62；118 个高置信输出全部来自 P4。 | 示例输入没有配套 GT。 |
| F-005 | OBSERVATION_NO_GT | predicted-area `<32²` 的 11,881 个 dense 代理中，704 个进入 Top-K；Top-K 外存在大量与入选框重叠的代理候选。 | `predicted-small` 不是 GT-small，未入选代理不能直接计作漏检。 |
| F-006 | FACT | 当前 13 张诊断图没有配套 GT JSON，无法确定真实漏检所在阶段。 | 这是全部机制解释的硬边界。 |
| F-007 | FACT | 正式 COCO 主评估采用 `maxDets=[1,10,300]`；机制诊断在 evaluator 截断前完成。 | 评估协议和机制协议必须分开。 |
| F-008 | FACT | 旧 small-object probes 的 shell 脚本覆盖了外部 `CONFIG_FILE`；它们不是有效损失消融。 | 数值可保留为历史 run，不可归因于 small loss。 |
| F-009 | OBSERVATION_RECOVERED_GT | `image (12)` 的 GT 可视化含 322 个框，其中正文 304 个；正文框在 640 输入下短边约 12.6～14.9 px，P4/P5 短边均不足 1 个 feature cell。 | 框由可视化轮廓恢复并配准，最终数字须由原始 COCO JSON 复算。 |
| F-010 | OBSERVATION_RECOVERED_GT | 正文框 IoU≥0.5 的 dense/Top-640 覆盖约为 278/304 与 135/304；P3 对应约为 265/304 与 60/304。 | 支持 selection starvation 立题，不是正式 dataset-level recall。 |
| F-011 | OBSERVATION_RECOVERED_GT | 304 个正文框只有约 7.2% 属于 COCO-small，多数困难框是低像素短边的细长目标。 | AP-small 不能单独代表本文问题，需 short-side/stride 与长宽比分桶。 |
| F-012 | FACT | 标准代码只对全部 8,400 个位置计算 encoder score；bbox head、Hungarian 与 encoder auxiliary loss只作用于 Top-K。 | Top-K 外位置没有直接的 proposal 分类/回归/排名监督，但仍可能从共享特征获得间接梯度。 |
| F-013 | OBSERVATION_NO_GT | 三个声称正确配置的 12,800-step 目录只有 metrics 摘要，没有 console/resolved config；small-reg 未提升 AP-small/AR-small，MAL+Dense O2O 整体退化。 | 数值可记录；在补回运行时配置证据前，不能严格归因于声称方法。 |
| F-014 | FACT | SGQS 服务器训练日志的 EMA AP-small 在 e28/e29/e30/e34 分别为 `0.679395/0.705142/0.705554/0.682274`，呈现前升后降；服务器日志峰值为 e30。 | evaluator 数值与 checkpoint 身份见 [EXP-DIAG-005](../../../experiments/EXP-DIAG-005_sgqs-top300-ranking-replay/README.md)；服务器 annotation SHA 与后续本地 full-val 不同，该曲线本身也不能定位参数级根因。 |
| F-015 | FACT | 固定 val128 的 EMA 独立 GT coverage 中，e28→e29 的 49 个 Decoder Top-300 gain 在 e28 全部已有 IoU≥0.75 的 Decoder 最终框，只是 rank>300；e30→e34 的 36 个 loss 在 e34 也全部仍有 IoU≥0.75 的 Decoder 最终框，只是 rank 跌到 300 之外。 | 逐 annotation 转移见 [EXP-DIAG-005](../../../experiments/EXP-DIAG-005_sgqs-top300-ranking-replay/README.md)；这是每个 GT 独立询问是否被覆盖，允许同一 query 覆盖多个 GT，**不是** COCO 一对一匹配得到的 recall、TP 或 FN。 |
| F-016 | FACT | 固定 val128 的 hybrid-small Decoder Top-300 独立 coverage 从 e28 到 e34 为 `75.21%→75.02%→75.09%→75.36%→75.46%→75.53%→75.58%`，不呈倒 U；e30→e34 时 `COCO-small∩hybrid-small` 净减 25，而 `hybrid-only` 净增 61，符合有限排名预算在 small 子群间重分配。 | 分桶定义、交集和转移表见 [EXP-DIAG-005](../../../experiments/EXP-DIAG-005_sgqs-top300-ranking-replay/README.md)；净变化不是守恒的一对一 query 流量，也不能替代 hybrid-small custom AP。 |
| F-017 | FACT | 固定 val128 上 raw 权重没有复现 EMA 的 AP-small 倒 U。 | raw/EMA 七轮配对见 [EXP-DIAG-005](../../../experiments/EXP-DIAG-005_sgqs-top300-ranking-replay/README.md)；这只是把 EMA 参数混合、时间滞后或分类校准列为上游线索，尚无受控干预，不能认定 EMA 是唯一根因。 |
| F-018 | FACT | val128 中 430 个 COCO-small GT 只分布在 17 张图，跨 epoch 的 COCO-small Top-300 状态变化又集中在两张分别含 480/320 个 GT 的密集图。 | 代表性审计见 [EXP-DIAG-005](../../../experiments/EXP-DIAG-005_sgqs-top300-ranking-replay/README.md)；该子集足以建立配对机制证据，但不能估计这一机制解释 full-val AP-small 变化的比例。 |
| F-019 | FACT | 本地完整 val 资产含 12,841 图、1,458,060 个 non-crowd GT：wired 5,847 图/627,939 GT，wireless 6,994 图/830,121 GT，annotation 引用的图像全部存在。 | 脱敏输入清单 `ART-000081` 登记 annotation SHA-256、数量与门禁范围；这证明输入资产可用，不是 14 状态 full-val 结果。 |
| F-020 | FACT | e28--e34 每轮 raw/EMA 状态互异，共 14 个唯一参数状态；`best_stg1` 与 `last` 的 raw/EMA 状态均等同 e34，不需重复推理。 | 参数状态哈希属于 `ART-000073` 清单覆盖的 [checkpoint_identity.csv](../../../experiments/EXP-DIAG-005_sgqs-top300-ranking-replay/data/checkpoint_identity.csv)；执行矩阵由 `ART-000081` 冻结。 |
| F-021 | FACT | EXP-TRAIN-006 的同初始化、同预算 SGQS/B0 详细日志训练臂均完成 epoch 28--34；最终 SGQS−B0 为 mAP `+0.000914788489`、AP-small `−0.001539072145`、AR-small `−0.002718051506`。 | `ART-000082`～`ART-000093` 支持运行身份、日志和终点；单 seed checkpoint continuation 不能证明稳定收益，两臂训练探针尚未完成机制分析。 |
| F-022 | FACT | 本地 12,841 图全量回放中，EMA AP-small@300 从 e28 `.674440` 升至 e29/e30 `.701083/.701179`，e34 回落至 `.677757`；同期 AP-small@640 为 `.935530/.897127/.935651/.952699`，不呈同形倒 U。 | `ART-000094` 覆盖 14 状态正式指标与审计；这定位 Top-300 admission/ranking 是输出层直接承载位置，不是训练级根因。结论限本地 annotation/CPU 后端。 |
| F-023 | FACT | 正式 evaluator-small 的 e28→29 gain、e28→30 gain、e30→34 loss 中，十 IoU `GT×IoU transition events` 的 `rank_gt300` 占 `89.90%/87.73%/93.38%`；全局 PR 精确分解将 `80.05%--88.66%` 的 AP 变化定位在 recall `.69--.72`。 | `ART-000094` 的 transition attribution 与 global PR 表支持；cause count 本身不是 AP 可加分解，只有 10×101 precision 差可精确加和为 AP 差。 |
| F-024 | FACT | full-val 净 TP 变化几乎全部集中于每图 GT≥301 与 640 输入短边 8--16 px，wired/wireless 同向；local hybrid-small 的 hybrid-only 约占 96.35%，且后段 Top-300 recall 与 evaluator-small 方向相反。 | `ART-000094` 的正式分桶与 cohort 交并表支持；密度和短边人群重叠，不能相加归因，hybrid-small 也不能称 AP-small。 |

新增事实必须给出代码位置、run ID 或 artifact ID；目录名和口述不构成事实证据。
