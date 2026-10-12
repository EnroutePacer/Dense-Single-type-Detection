# 研究状态

更新时间：2026-10-12

## 当前主线

研究对象是密集单类别场景，不把“表格专用优化”作为最终目标。当前论文问题分为两层：DEIM/D-FINE 的小框 Encoder 候选是否因低分和 global Top-K 得不到足够直接监督；以及进入回归链后，整框 IoU 共享权重是否无法按边表达小框尺度、相对误差与同类邻域歧义。

最新方法为 M-002 SGQS-E：用 SGQS 处理候选监督，用 EFGL 处理邻域/短边感知的逐边定位。数学评审与代码实现已完成，状态为 implemented/unvalidated；已修正生存自追逐、基础监督冲突和重复课程，并提供 B0/B1/B2/B3 配置。保留有效设计和历史 M-001，不以论文重叠为删除理由。

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
- 当前 `loss_local` 将匹配框的同一个 IoU 复制为四条边的 FGL 权重；IoU 为 0 时该 FGL 项为 0，但 L1/GIoU 仍有梯度。历史面积权重只能继续乘这个共享标量，不能提供逐边尺度或邻域歧义权重。
- M-002 SGQS-E 的完整公式、课程、代码和最小消融已落地。CPU 检查包含零 IoU 梯度、同权重 DN/eval 等价、训练入口与 trace；尚不能声称 AP 或收敛已有提升。

## 当前最高优先级

1. 为配对训练固定数据 split/COCO JSON、初始化 checkpoint、配置和预算，并登记哈希；已有先导归因可并行复核，不阻止方法对照。
2. 用当前同一提交的 B0 作 clean control，不能把共享 invalid-anchor correctness fix 混入 SGQS 增益。
3. 按已登记的 EXP-TRAIN-004 运行 B0 baseline、B1 SGQS、B2 EFGL、B3 SGQS+EFGL；先做单 seed 相同步数筛选，再对正信号做至少 3 seeds。
4. 同时记录候选链的 owner rank/unique-GT coverage 与回归链的逐边误差/AP75/AP90；只看总 mAP 不能证明设计机制。
5. 邻居约束 DN 暂不进入首版。只有 B1/B2 已分别建立正信号后，才把它作为独立候选研究。

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
