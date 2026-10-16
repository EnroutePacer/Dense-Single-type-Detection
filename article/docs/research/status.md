# 研究状态

更新时间：2026-10-17

## 当前主线

研究对象是密集单类别场景，不把“表格专用优化”作为最终目标。当前论文问题分为两层：DEIM/D-FINE 的小框 Encoder 候选是否因低分和 global Top-K 得不到足够直接监督；以及进入回归链后，整框 IoU 共享权重是否无法按边表达小框尺度、相对误差与同类邻域歧义。

最新方法为 M-002 SGQS-E：用 SGQS 处理候选监督，用 EFGL 处理邻域/短边感知的逐边定位。数学评审、代码实现和 B0--B3 单 seed checkpoint 续训屏选已完成，状态为 `implemented / screened / unvalidated`；保留有效设计和历史 M-001，不以论文重叠为删除理由。

截至 2026-10-17，[EXP-TRAIN-006](../../experiments/EXP-TRAIN-006_sgqs-supervision-probes/README.md) 的 SGQS 与 B0 详细日志训练臂均已完整训练至 epoch 34；SGQS 固定 val128 与本地完整 val 的 e28--e34 raw/EMA 回放均已完成，B0 回放尚未执行。SGQS 相对 B0 的最终 mAP 为 `+0.000914788489`，但 AP-small/AR-small 分别为 `-0.001539072145/-0.002718051506`。[EXP-DIAG-007](../../experiments/EXP-DIAG-007_sgqs-loss-hypothesis-audit/README.md) 已完成两臂 loss/代理指标配对审计：确认无升温启动冲击与末端代理脱节，排除 e31 基础 loss 明确回升；尚未用单变量干预判定唯一训练根因。

[EXP-DIAG-006](../../experiments/EXP-DIAG-006_sgqs-full-val-checkpoint-replay/README.md)
已在本地 12,841 图、1,458,060 non-crowd GT 上完成 e28--e34 raw/EMA 共 14 个唯一
状态的 CPU 回放、十 IoU 正式一对一匹配、全局 PR 复算和逐 GT 转移归因。每状态处理
8,218,240 个 Decoder query。`best_stg1`、`last` 与 e34 参数状态相同，未重复推理。
结论限定于本地 annotation SHA-256 `f855dd4b...bd693e` 和 Windows CPU/PyTorch
2.14.0 后端；服务器训练 annotation 哈希不同，不能把它写成 H20 位级复现。

## 已确认

- 640×640 输入下，P3/P4/P5 共提供 8,400 个 memory 位置，当前模型全局选择 Top-640。
- 13 张无 GT 诊断图中，P3/P4/P5 入选数量和高置信存活数量显著不均衡。
- P3 只表示来源于 stride-8 特征层，不等于 GT-small；预测面积 `<32²` 也不等于 GT-small。
- 旧 13 张定性图没有配套 GT，仍不能据其归因真实漏检；后续固定 val128 诊断使用了独立的原始 annotation，不改变这套旧图的证据等级。
- 旧 small-VFL/small-reg probes 的目标配置未加载，不能用来评价损失改动。
- 同一 M checkpoint 在 `maxDets=300` 下复评约为 mAP 0.864247；旧约 0.5669 是 `maxDets=100` 口径。
- 当前 M 训练脚本已尊重外部 `CONFIG_FILE`；相同口径的 B0--B3 七 epoch 续训 screen 已完成，但正式从头、多 seed 论文训练尚未执行。
- EXP-TRAIN-006 的 SGQS/B0 同探针单 seed 续训均已完成。最终总 AP 几乎持平，SGQS
  的 AP-small 与 AR-small 反而较低；这支持继续分析监督链路，不支持宣称小目标收益。
- 服务器训练日志的 EMA AP-small 在 e28/e29/e30/e34 分别为 `0.679395/0.705142/0.705554/0.682274`；本地完整集重放的对应值为 `0.674440/0.701083/0.701179/0.677757`，曲线形状与峰值 epoch 一致。两者 annotation SHA 不同，约 `0.0041--0.0050` 的差额不能单独归因于 CPU/GPU。
- 固定 val128 的 SGQS EMA 回放表明：e28→e29 的 49 个 Decoder Top-300 gain 在 e28 全部已有 IoU≥0.75 的 Decoder 最终框但 rank>300；e30→e34 的 36 个 loss 在 e34 仍全部有 IoU≥0.75 的框，只是跌出前 300。它们是独立 GT coverage，不是 COCO 一对一匹配得到的 recall、TP 或 FN。
- 同一 val128 上 hybrid-small Top-300 coverage 不呈倒 U；e30→e34 的 `COCO-small∩hybrid-small` 净减 25、`hybrid-only` 净增 61，支持有限输出预算在 small 子群间重分配。raw 权重也没有复现 EMA 倒 U，因此 EMA 是待检验线索，而不是已证唯一根因。
- val128 的 430 个 COCO-small GT 只位于 17 张图，状态变化又集中在两张分别含 480/320 个 GT 的密集图；这足以建立配对机制证据，但不能外推它解释 full-val 变化的比例。
- 本地完整 val 含 wired 5,847 图/627,939 GT 与 wireless 6,994 图/830,121 GT，
  合计 12,841 图/1,458,060 GT，annotation 引用图像均存在。e28--e34 的 raw/EMA
  共 14 个唯一状态；`best_stg1`、`last` 与 e34 参数重复，正式回放不重复计算。
- 本地完整集正式 evaluator-small 一对一转移表明，EMA e28→29 gain、e28→30 gain、
  e30→34 loss 的十 IoU `GT×IoU transition events` 中，`rank_gt300` 分别占
  `89.90%/87.73%/93.38%`；IoU=.75 时分别为 `91.92%/90.26%/95.07%`。
  这表示合格框通常仍存在于 Decoder 全 640 query，只是是否进入 Top-300 翻转。
- 全局 COCO PR 的精确 10×101 网格分解把 `80.05%--88.66%` 的 AP 变化定位在
  recall `0.69--0.72` 的开启/关闭；净 TP 变化几乎全部集中于每图 GT≥301 与输入
  短边 8--16 px 的目标，wired/wireless 两域同向。密度与短边分桶高度重叠，贡献
  不能相加，局部 TP/FN cause count 也不是 AP 的可加分解。
- 本地 hybrid-small 含大量非 evaluator-small 目标，hybrid-only 占约 96.35%，其
  后段 Top-300 coverage 反而上升，足以淹没 evaluator-small 回跌。hybrid-small
  只能报告 coverage/recall/transition，不能称 AP-small。
- raw 的 AP-small 七轮范围约 `0.008`，EMA 约 `0.027`；可写成 EMA-specific
  transient/rank trajectory，但没有受控证据把 EMA decay、resume 初始化或 SGQS
  某个损失项认定为训练级根因。
- SGQS/B0 step-0 基础 loss 完全相同，新增 SGQS 项为 `5.071082`，其中 dense
  `4.702855`；dense/owner 没有升温，确认强启动目标冲击。但 e34 总 AP 高于 B0，
  且没有 train-set checkpoint replay，不能称已证过拟合或整体性能被侵蚀。
- e30→34 SGQS owner coverage、IoU、pre-TopK survival 全部继续改善，基础 Encoder、
  main Decoder 与 pre loss 同期继续下降，AP-small 却回跌；这支持 SGQS 代理止于
  pre-TopK/Encoder、未约束最终 Top-300 的链路缺口，不支持“e31 基础 loss 回升”。
- dense positive/negative 逐层归一化后固定三层等权；P3/P5 kept-owner 数比由约
  `69.85×` 增至 `134.80×`。该实现会提高稀少 P5 单样本显式系数，但没有逐层共享
  参数梯度，尚不能登记为后期回跌原因。
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

最高优先级从“确认倒 U 在哪里发生”转为“解释为什么最终 score/rank 会这样变化”。
本地 full-val 已把前段收益和后段回跌都定位到 Decoder Top-300 admission，并确定最敏感
人群是极密图中的 8--16 px 短边 evaluator-small。配对训练探针已完成相关性审计，
现阶段必须进入受控单变量干预，不能再从更多分桶相关性直接跳到损失因果。

1. 做启动冲击单变量：只给 dense+owner 增加 1 epoch `0→1` 升温，保持数据顺序、
   optimizer、EMA 和步数完全相同；同步记录共享参数梯度与 train/val AP。
2. 做归一化单变量：只把 P3/P4/P5 等层均值改为全局 owner-count 归一化，记录逐层
   positive/negative loss、clip 前后共享参数梯度；不同时改 owner 分配。
3. 做链路单变量：e30 后停用 SGQS，或增加最终 Decoder/Top-300 owner/rank 约束，
   二者只选其一；另补 B0 同口径 checkpoint replay 作为模型特异性复核。
4. 若要消除后端/annotation 身份疑点，再以服务器训练 annotation 和同一 H20 环境复跑
   关键 e28/e30/e34；当前本地结果本身已构成完整集 scoped diagnosis。
5. 通过 seed 0 gate 后再执行从头至少 3 seeds。C1/C3 已封存，C4 与旧 DEIM-core
   保留为中断历史，不复用 partial checkpoint。

## 研究门槛

- 不能用无 GT 可视化宣称 recall 或 mAP 提升。
- 不能把候选来源层直接当作目标尺度。
- 不能把 selector recall 提升直接等同于最终 AP 提升。
- 方法训练必须在相同初始化、global batch、epoch/steps、seed 和 `maxDets=300` 下比较。
- 关键结果最终至少三个 seed，并报告均值、方差、wall-clock 与 GPU-hours。

## 尚缺资产

- 本地完整 val 回放、正式 matcher 与派生分析已具备；如需与服务器日志做位级对齐，
  仍缺服务器训练 annotation 的可复跑副本与同 H20 环境；
- 各旧 probe checkpoint；共同 epoch-27 checkpoint 已按 SHA-256 校验并归档；
- 当前代码准确的上游 commit/remote 与本地 diff；
- 完整环境锁定信息；
- 第二个可公开的密集单类别数据集或合成基准；
- 输入图片、派生可视化与模型权重的再发布许可结论。
- 三个旧 12,800-step probe 的原始 `console.log`、`log.txt`、resolved config 与 checkpoint；当前仍只有文本指标摘要。B0--B3/C1--C3 的本轮服务器日志证据已归档，C1/C3 最终 checkpoint 尚未同步。

详细步骤见 [roadmap.md](roadmap.md)，结论边界见 [facts](facts/README.md) 与 [hypotheses](hypotheses/README.md)。
