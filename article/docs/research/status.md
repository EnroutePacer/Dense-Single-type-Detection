# 研究状态

更新时间：2026-10-12

## 当前主线

研究对象是密集单类别场景，不把“表格专用优化”作为最终目标。当前论文问题是：DEIM/D-FINE 在固定 query 数下使用 global score-only Top-K 初始化 decoder，是否在密集、细长或低像素短边目标上产生 selection gap；以及已经进入 decoder 的 query 是否继续发生 survival gap。

当前阶段只做问题诊断与证据建设，训练暂放一边。

## 已确认

- 640×640 输入下，P3/P4/P5 共提供 8,400 个 memory 位置，当前模型全局选择 Top-640。
- 13 张无 GT 诊断图中，P3/P4/P5 入选数量和高置信存活数量显著不均衡。
- P3 只表示来源于 stride-8 特征层，不等于 GT-small；预测面积 `<32²` 也不等于 GT-small。
- 这 13 张图没有配套 GT，因此目前无法给真实漏检归因。
- 旧 small-VFL/small-reg probes 的目标配置未加载，不能用来评价损失改动。
- 同一 M checkpoint 在 `maxDets=300` 下复评约为 mAP 0.864247；旧约 0.5669 是 `maxDets=100` 口径。
- 当前同步后的 M 训练脚本已经改为尊重外部 `CONFIG_FILE`，但尚未以论文协议重新训练。
- `image (12)` 的 GT 可视化恢复先导审计显示：304 个正文框在 640 输入下短边中位约 14 px；dense encoder 在 IoU≥0.5 时可覆盖约 278 个，Top-640 后约为 135 个。该结果支持 selection starvation 作为主研究方向，但仍须原始 COCO JSON 复核。
- 三个新 12,800-step 摘要均未超过 epoch-27 起点；small-reg 相对同周期 control 仅 +0.000514 mAP，且 AP-small/AR-small 各低约 0.002；MAL+Dense O2O 相对 control 低 0.009203 mAP。新目录缺少 console/resolved config，因此只能记录数值，暂不能完成方法归因。
- M-001 SGQS 已完成代码实现与 11 项单元测试：训练时对 Top-K 前的 Encoder 位置施加短边尺度感知 owner、分层 dense 分类、owner 回归与生存排名监督；尚未训练、尚未验证效果与原创性。

## 当前最高优先级

1. 取得与 checkpoint/诊断输入对应的验证集 COCO JSON，并登记哈希与许可；复算 GT 可视化恢复先导结果。
2. 冻结模型，导出全部 dense proposals、Top-K、每层 decoder 与最终输出，完成正式 generation/selection/decoder/output 归因。
3. 在相同冻结输出上比较 global Top-K、简单分层校准/配额、去重或 coverage-aware selector、GT-aware oracle。
4. 已在当前代码中屏蔽 invalid anchor score；用同一提交重跑 clean control，单独量化该 correctness fix，不能混入 SGQS 增益。
5. 正式诊断继续支持 selection gap 后，用已实现的 M-001 SGQS 进入等 steps、同 seed 消融；在此之前不启动训练。

## 研究门槛

- 不能用无 GT 可视化宣称 recall 或 mAP 提升。
- 不能把候选来源层直接当作目标尺度。
- 不能把 selector recall 提升直接等同于最终 AP 提升。
- 方法训练必须在相同初始化、global batch、epoch/steps、seed 和 `maxDets=300` 下比较。
- 关键结果最终至少三个 seed，并报告均值、方差、wall-clock 与 GPU-hours。

## 尚缺资产

- 配套验证 GT JSON、split 与 SHA-256；
- epoch-27 训练 checkpoint 和各 probe checkpoint；
- 当前代码准确的上游 commit/remote 与本地 diff；
- 完整环境锁定信息；
- 第二个可公开的密集单类别数据集或合成基准；
- 输入图片、派生可视化与模型权重的再发布许可结论。
- 三个新 12,800-step run 的原始 `console.log`、`log.txt`、resolved config 与 checkpoint；目前只有文本指标摘要。

详细步骤见 [roadmap.md](roadmap.md)，结论边界见 [facts](facts/README.md) 与 [hypotheses](hypotheses/README.md)。
