# 研究状态

更新时间：2026-10-15

## 当前主线

研究对象是密集单类别场景，不把“表格专用优化”作为最终目标。当前论文问题分为两层：DEIM/D-FINE 的小框 Encoder 候选是否因低分和 global Top-K 得不到足够直接监督；以及进入回归链后，整框 IoU 共享权重是否无法按边表达小框尺度、相对误差与同类邻域歧义。

最新方法为 M-002 SGQS-E：用 SGQS 处理候选监督，用 EFGL 处理邻域/短边感知的逐边定位。数学评审、代码实现和 B0--B3 单 seed checkpoint 续训屏选已完成，状态为 `implemented / screened / unvalidated`；保留有效设计和历史 M-001，不以论文重叠为删除理由。

截至 2026-10-15，[EXP-TRAIN-006](../../experiments/EXP-TRAIN-006_sgqs-supervision-probes/README.md) 的 SGQS 臂已经完整训练至 epoch 34，固定 val128 的 e28--e34 raw/EMA 回放也已完成；带同一套探针的 B0 对照臂尚未运行。本轮回放已在 val128 上定位 AP-small 倒 U 的直接承载阶段，但还没有完成 full-val 逐 annotation 归因，也没有把该现象归因为 SGQS 相对 B0 的特有效应。

[EXP-DIAG-006](../../experiments/EXP-DIAG-006_sgqs-full-val-checkpoint-replay/README.md)
已冻结 e28--e34 raw/EMA 共 14 个唯一状态的完整 val 回放协议。本地 12,841 图、
1,458,060 GT 的资产盘点和 CPU/val128 工程门禁已完成；同后端 GPU 全量输出尚未产生，
因此没有新的 full-val 阶段归因或 raw/EMA 结论。

## 已确认

- 640×640 输入下，P3/P4/P5 共提供 8,400 个 memory 位置，当前模型全局选择 Top-640。
- 13 张无 GT 诊断图中，P3/P4/P5 入选数量和高置信存活数量显著不均衡。
- P3 只表示来源于 stride-8 特征层，不等于 GT-small；预测面积 `<32²` 也不等于 GT-small。
- 旧 13 张定性图没有配套 GT，仍不能据其归因真实漏检；后续固定 val128 诊断使用了独立的原始 annotation，不改变这套旧图的证据等级。
- 旧 small-VFL/small-reg probes 的目标配置未加载，不能用来评价损失改动。
- 同一 M checkpoint 在 `maxDets=300` 下复评约为 mAP 0.864247；旧约 0.5669 是 `maxDets=100` 口径。
- 当前 M 训练脚本已尊重外部 `CONFIG_FILE`；相同口径的 B0--B3 七 epoch 续训 screen 已完成，但正式从头、多 seed 论文训练尚未执行。
- SGQS 完整验证集 EMA AP-small 在 e28/e29/e30/e34 分别为 `0.679395/0.705142/0.705554/0.682274`；峰值在 e30，而 overall AP 峰值在 e34。完整曲线与 checkpoint 身份见 [EXP-DIAG-005](../../experiments/EXP-DIAG-005_sgqs-top300-ranking-replay/README.md)。
- 固定 val128 的 SGQS EMA 回放表明：e28→e29 的 49 个 Decoder Top-300 gain 在 e28 全部已有 IoU≥0.75 的 Decoder 最终框但 rank>300；e30→e34 的 36 个 loss 在 e34 仍全部有 IoU≥0.75 的框，只是跌出前 300。它们是独立 GT coverage，不是 COCO 一对一匹配得到的 recall、TP 或 FN。
- 同一 val128 上 hybrid-small Top-300 coverage 不呈倒 U；e30→e34 的 `COCO-small∩hybrid-small` 净减 25、`hybrid-only` 净增 61，支持有限输出预算在 small 子群间重分配。raw 权重也没有复现 EMA 倒 U，因此 EMA 是待检验线索，而不是已证唯一根因。
- val128 的 430 个 COCO-small GT 只位于 17 张图，状态变化又集中在两张分别含 480/320 个 GT 的密集图；这足以建立配对机制证据，但不能外推它解释 full-val 变化的比例。
- 本地完整 val 含 wired 5,847 图/627,939 GT 与 wireless 6,994 图/830,121 GT，
  合计 12,841 图/1,458,060 GT，annotation 引用图像均存在。e28--e34 的 raw/EMA
  共 14 个唯一状态；`best_stg1`、`last` 与 e34 参数重复，正式回放不重复计算。
- AP-small/AR-small 必须沿用 COCO evaluator areaRange；本地 small 机制分析使用
  640 输入下 `area < 32² OR short side < 16` 的 hybrid-small。两种 cohort 并列
  报告，不能互相替代。
- `image (12)` 的 GT 可视化恢复先导审计显示：304 个正文框在 640 输入下短边中位约 14 px；dense encoder 在 IoU≥0.5 时可覆盖约 278 个，Top-640 后约为 135 个。该结果支持 selection starvation 作为主研究方向，但仍须原始 COCO JSON 复核。
- 三个新 12,800-step 摘要均未超过 epoch-27 起点；small-reg 相对同周期 control 仅 +0.000514 mAP，且 AP-small/AR-small 各低约 0.002；MAL+Dense O2O 相对 control 低 0.009203 mAP。新目录缺少 console/resolved config，因此只能记录数值，暂不能完成方法归因。
- M-001 SGQS 已完成代码实现与 11 项单元测试，并作为 B1 完成一次 epoch-27 单 seed 续训：mAP 相对 B0 `+0.000941367`，但 AP-small/AR-small 分别下降约 `0.003562/0.004795`，尚未正式验证效果与原创性。
- 当前 `loss_local` 将匹配框的同一个 IoU 复制为四条边的 FGL 权重；IoU 为 0 时该 FGL 项为 0，但 L1/GIoU 仍有梯度。历史面积权重只能继续乘这个共享标量，不能提供逐边尺度或邻域歧义权重。
- M-002 SGQS-E 的完整公式、课程、代码和最小消融已落地。B3 相对 B0 为 `+0.001020272` mAP，但 AP-small/AR-small 下降；单 seed、已收敛 checkpoint 的短续训尚不能支持稳定 AP、收敛或小目标改善主张。
- C1 small-reg、C2 MAL-only、C3 Mosaic-only 已完成同口径续训。相对 B0，它们的 mAP 分别变化 `-0.000015530`、`+0.002097709`、`-0.001287789`，AP-small 分别变化 `-0.000408523`、`-0.008056974`、`-0.002657714`；没有一个常规改法提高 AP-small。

## 当前最高优先级

最高优先级是在完全相同的验证预处理与 GPU 后端下，对 e28--e34 的 SGQS
**raw/EMA 14 个唯一状态做 full-val 逐 annotation 回放**。每个状态必须同时保存
Decoder 全 640 个输出的 score、rank、IoU、Top-300 截断状态，并补正式 COCO
一对一匹配，检验 val128 中观察到的排名跨界能解释 full-val AP-small 变化的多大比例。

[EXP-DIAG-005](../../experiments/EXP-DIAG-005_sgqs-top300-ranking-replay/README.md) 已完成 SGQS 固定 val128 的 raw/EMA 七轮回放，证明该子集上的直接承载机制是 IoU 合格 query 跨越 rank 300，而不是框消失或定位普遍退化；但 430 个 COCO-small 只在 17 张图，变化集中于两张极密图，不能把这一结果写成已完成全数据归因。raw 不复现 EMA 峰形只把 EMA 参数混合/滞后与分类校准列为上游线索，尚不能认定唯一根因。

1. 先在同一 GPU 后端完成 e28--e34 full-val raw/EMA；每个状态同时输出独立 GT
   coverage 与 COCO 一对一 TP/FP/FN，禁止混用两个口径。
2. 比较 Top-300/Top-640、十个 IoU 阈值、score/rank、匹配竞争与全局 TP/FP
   score 排序；AP 不能由局部 TP/FN 转移计数直接相加得到。
3. 按 wired/wireless、GT 密度、输入短边及 COCO-small/hybrid-small 交并子群量化
   前段收益与后段回跌来源；raw/EMA 差异仍只是相关性诊断，不是 EMA 因果消融。
4. 完成 full-val 定位后，再运行 EXP-TRAIN-006 的 **B0 带同探针对照**。SGQS 臂与 val128 replay 已完成，B0 尚未运行；历史 B0 没有同一套字段，不能代替。
5. 只有 full-val 与 B0 对照给出稳定方向，才决定是否对 owner/GO、DN-DDF、EMA 或分类校准做单变量短干预；通过 seed 0 gate 后再执行从头至少 3 seeds。C1/C3 已封存，C4 与旧 DEIM-core 保留为中断历史，不复用 partial checkpoint。

## 研究门槛

- 不能用无 GT 可视化宣称 recall 或 mAP 提升。
- 不能把候选来源层直接当作目标尺度。
- 不能把 selector recall 提升直接等同于最终 AP 提升。
- 方法训练必须在相同初始化、global batch、epoch/steps、seed 和 `maxDets=300` 下比较。
- 关键结果最终至少三个 seed，并报告均值、方差、wall-clock 与 GPU-hours。

## 尚缺资产

- 完整 val 的本地 GT、split、逐域 SHA-256 与图像已具备；仍需生成并审计供正式回放
  使用的 merged 身份记录，以及落盘 14 状态同后端 GPU 结果；
- 各旧 probe checkpoint；共同 epoch-27 checkpoint 已按 SHA-256 校验并归档；
- 当前代码准确的上游 commit/remote 与本地 diff；
- 完整环境锁定信息；
- 第二个可公开的密集单类别数据集或合成基准；
- 输入图片、派生可视化与模型权重的再发布许可结论。
- 三个旧 12,800-step probe 的原始 `console.log`、`log.txt`、resolved config 与 checkpoint；当前仍只有文本指标摘要。B0--B3/C1--C3 的本轮服务器日志证据已归档，C1/C3 最终 checkpoint 尚未同步。

详细步骤见 [roadmap.md](roadmap.md)，结论边界见 [facts](facts/README.md) 与 [hypotheses](hypotheses/README.md)。
