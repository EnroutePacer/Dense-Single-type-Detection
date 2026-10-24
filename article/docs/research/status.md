# 研究状态

更新时间：2026-10-24。

## 当前主线

[M-003 SGQS v2](../../methods/M-003_sgqs_v2/README.md)：实例覆盖驱动的候选预算分配。完整设计先于实现冻结，活跃代码覆盖旧SGQS v2。新训练尚未启动，检测收益、困难实例泛化和真实GPU加速均未验证。

[EXP-TRAIN-008](../../experiments/EXP-TRAIN-008_sgqs-v2-coverage-allocation/README.md) 采用 b0/support/v2，原sgqs可选参照。support与v2隔离分配项；同冻结epoch27恢复、8x8、七轮seed0。原 [EXP-TRAIN-007](../../experiments/EXP-TRAIN-007_sgqs-v2-single-factor/README.md) 已被取代，不再扩展旧六臂探针。

## 已有证据

- [完整验证回放](../../experiments/EXP-DIAG-006_sgqs-full-val-checkpoint-replay/README.md)：SGQS EMA的AP-small@640在e30到e34由.935651升至.952699，@300倒U主要由最终排名截断承载，未证明训练级唯一根因。
- [配对loss诊断](../../experiments/EXP-DIAG-007_sgqs-loss-hypothesis-audit/README.md)：新增dense/owner无升温造成明显启动冲击；等层正样本均值提高稀有层单样本系数；不存在“验证大框AP触发训练loss”的因果链。
- [候选复算](../../../outputs/diagnostics/sgqs_v2_reframe_20261024/README.md)：13张无GT图支持Encoder升分，未证明更适合裁剪；高Encoder/低Decoder重叠候选增多，但须用GT鉴别真实冗余。
- 原M-001/M-002仍为已实现、经过单seed屏选、未验证稳定收益；历史源码和实验不删除。

## 当前实施边界

保留几何辅助，修正dense目标/全局正样本归一化、统一升温；用严格跨阶段GT归属形成rescue/prefix配对。保护分类正标签与跨GT回归目标；同GT GO回归正不自动当作分类正而过度过滤。无安全配对不强造负标签。

训练CocoEvaluator和PostProcessor均明确640；完整val使用严格local-small AP和COCO各尺度，报告固定高密度场景。真实K sweep会重新forward，输出maxDets与Encoder输入预算分开。

## 尚未完成的科学验证

新方案真实训练、完整val比较、CUDA多卡运行和目标部署后端性能；多seed/完整训练；固定独立困难实例集与第二个密集单类别场景。CPU合成检查仅验证实现，不填补这些实验结论。

## 解释约束

不得将无GT图解释成AP/FP收益，不将候选来源层等同GT尺度，不将prefix配对数当新增Top640覆盖，不将静态过滤当真实推理节省。方法比较必须保持相同初始化、数据身份、实际K、输出预算和后端。
