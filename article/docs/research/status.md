# 研究状态

更新时间：2026-10-14

## 当前主线

研究对象是密集单类别场景，不把“表格专用优化”作为最终目标。当前论文问题分为两层：DEIM/D-FINE 的小框 Encoder 候选是否因低分和 global Top-K 得不到足够直接监督；以及进入回归链后，整框 IoU 共享权重是否无法按边表达小框尺度、相对误差与同类邻域歧义。

最新方法为 M-002 SGQS-E：用 SGQS 处理候选监督，用 EFGL 处理邻域/短边感知的逐边定位。数学评审、代码实现和 B0--B3 单 seed checkpoint 续训屏选已完成，状态为 `implemented / screened / unvalidated`；保留有效设计和历史 M-001，不以论文重叠为删除理由。

## 已确认

- 640×640 输入下，P3/P4/P5 共提供 8,400 个 memory 位置，当前模型全局选择 Top-640。
- 13 张无 GT 诊断图中，P3/P4/P5 入选数量和高置信存活数量显著不均衡。
- P3 只表示来源于 stride-8 特征层，不等于 GT-small；预测面积 `<32²` 也不等于 GT-small。
- 这 13 张图没有配套 GT，因此目前无法给真实漏检归因。
- 旧 small-VFL/small-reg probes 的目标配置未加载，不能用来评价损失改动。
- 同一 M checkpoint 在 `maxDets=300` 下复评约为 mAP 0.864247；旧约 0.5669 是 `maxDets=100` 口径。
- 当前 M 训练脚本已尊重外部 `CONFIG_FILE`；相同口径的 B0--B3 七 epoch 续训 screen 已完成，但正式从头、多 seed 论文训练尚未执行。
- `image (12)` 的 GT 可视化恢复先导审计显示：304 个正文框在 640 输入下短边中位约 14 px；dense encoder 在 IoU≥0.5 时可覆盖约 278 个，Top-640 后约为 135 个。该结果支持 selection starvation 作为主研究方向，但仍须原始 COCO JSON 复核。
- 三个新 12,800-step 摘要均未超过 epoch-27 起点；small-reg 相对同周期 control 仅 +0.000514 mAP，且 AP-small/AR-small 各低约 0.002；MAL+Dense O2O 相对 control 低 0.009203 mAP。新目录缺少 console/resolved config，因此只能记录数值，暂不能完成方法归因。
- M-001 SGQS 已完成代码实现与 11 项单元测试，并作为 B1 完成一次 epoch-27 单 seed 续训：mAP 相对 B0 `+0.000941367`，但 AP-small/AR-small 分别下降约 `0.003562/0.004795`，尚未正式验证效果与原创性。
- 当前 `loss_local` 将匹配框的同一个 IoU 复制为四条边的 FGL 权重；IoU 为 0 时该 FGL 项为 0，但 L1/GIoU 仍有梯度。历史面积权重只能继续乘这个共享标量，不能提供逐边尺度或邻域歧义权重。
- M-002 SGQS-E 的完整公式、课程、代码和最小消融已落地。B3 相对 B0 为 `+0.001020272` mAP，但 AP-small/AR-small 下降；单 seed、已收敛 checkpoint 的短续训尚不能支持稳定 AP、收敛或小目标改善主张。

## 当前最高优先级

优先运行[EXP-TRAIN-006](../../experiments/EXP-TRAIN-006_sgqs-supervision-probes/README.md)的B1/B0观测对照。
远程B3日志已到达并复现主要走势；首要排查为浅层DN-DDF与GT监督不同步、
GO/guard的选择性遗漏和候选重排。现有统计不支持“后期owner过滤越来越多”，
也未证明小框过拟合或大框梯度压制。新探针代码已实现，真实运行尚未开始。

2026-10-13 新证据：[B0--B3/C2 完整跟进](../../experiments/EXP-TRAIN-005_epoch27-sgqs-e-resume-screen/diagnostics/20261013_b0_b3_c2_followup/README.md)
显示 B3 总 mAP 的单 seed 增量只有 `+0.001020272`，且小目标终点下降；
C2 MAL-only 总 mAP 最高但 AP-small 最差。此前
[B1/B2 诊断](../../experiments/EXP-TRAIN-005_epoch27-sgqs-e-resume-screen/diagnostics/20261013_b1_b2/README.md)
发现 SGQS 整组冲突过滤末期撤销约 55.08% owner 的新增监督，EFGL 主分支
平均补偿仅约 1e-4 且过早结束。这些只是候选原因，必须先由 EXP-TRAIN-006
定位监督来源、候选排名和 raw/EMA 泛化链路，再决定是否采用
[数学说明](../../experiments/EXP-TRAIN-005_epoch27-sgqs-e-resume-screen/diagnostics/20261013_b1_b2/math_diagnosis.md)
中的最小修订。

1. 先执行 EXP-TRAIN-006 的冻结输入 B1/B0 观测探针，直接记录全 GT unique-owner coverage、owner rank、Top-K survival、DN 监督与 raw/EMA 固定子集回放；该阶段不修改 loss。
2. 依据探针再决定是否分离 SGQS 的分类保护与 owner 回归保护、是否去除 EFGL 的重复低误差衰减；这些目前只是候选干预，不作为既定结论。
3. 只有修订后出现明确正信号，才用同一代码提交从头执行 B0/B1/B2/B3，先做 seed 0 gate，再对通过项补至少 3 seeds。
4. 同时报告 AP75/AP90、按短边分桶的 recall 和逐边相对误差；总 mAP 不能替代机制证据。
5. C1/C3/DEIM-core 中断目录只作为失败记录；在查明调度终止原因前不复用其 partial checkpoint。

## 研究门槛

- 不能用无 GT 可视化宣称 recall 或 mAP 提升。
- 不能把候选来源层直接当作目标尺度。
- 不能把 selector recall 提升直接等同于最终 AP 提升。
- 方法训练必须在相同初始化、global batch、epoch/steps、seed 和 `maxDets=300` 下比较。
- 关键结果最终至少三个 seed，并报告均值、方差、wall-clock 与 GPU-hours。

## 尚缺资产

- 配套验证 GT JSON、split 与 SHA-256；
- 各旧 probe checkpoint；共同 epoch-27 checkpoint 已按 SHA-256 校验并归档；
- 当前代码准确的上游 commit/remote 与本地 diff；
- 完整环境锁定信息；
- 第二个可公开的密集单类别数据集或合成基准；
- 输入图片、派生可视化与模型权重的再发布许可结论。
- 三个旧 12,800-step probe 的原始 `console.log`、`log.txt`、resolved config 与 checkpoint；当前仍只有文本指标摘要。B0--B3/C2 的本轮服务器证据已归档。

详细步骤见 [roadmap.md](roadmap.md)，结论边界见 [facts](facts/README.md) 与 [hypotheses](hypotheses/README.md)。
