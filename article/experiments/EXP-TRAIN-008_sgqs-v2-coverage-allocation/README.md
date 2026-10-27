# EXP-TRAIN-008：SGQS v2 实例覆盖与候选预算分配

状态：已实现并通过CPU检查；真实训练未运行、方法收益未验证。初版设计于2026-10-24，2026-10-27按中心/IoS与无接收者溢出规则修订，完整算法见 [M-003](../../methods/M-003_sgqs_v2/README.md)。本实验替代已暂停的 [EXP-TRAIN-007](../EXP-TRAIN-007_sgqs-v2-single-factor/README.md)，不继续旧六臂根因探针。

## 目标与对照

在原训练 K=640 下改善不同实例的候选排序，并以真实较小 K 重跑检验性能—计算量取舍。不能以 Encoder 均分、可视化规整或静态裁剪结果宣称方法有效。

| arm | 内容 | 用途 |
|---|---|---|
| b0 | 原 base；新评价预算640 | 同预算检测基线 |
| support | v2 几何+dense辅助与统一升温，allocation=0 | 新分配项的直接控制组 |
| v2 | support + 跨阶段实例归属约束的 allocation | 完整方法 |
| sgqs | 原 v1 SGQS，保留原升温语义 | 可选历史方法参照 |

默认仅启动 v2；最小受控比较为 b0/support/v2。support 与 v2 仅分配项及其必要的分类冲突mask不同；b0/v1不用于单独归因新分配项。

## 冻结训练条件

- 同一冻结 epoch27 checkpoint，完整恢复 model/EMA/optimizer/scaler；身份沿用 `configs/e35_r7/frozen_inputs.yml`。
- 8 GPU × batch8，global64，seed0，固定输入640，训练 epochs28–34，每轮保留checkpoint。
- 新辅助项统一在epoch28的一个epoch内升温；恢复使用绝对epoch，不重置学习率/EMA时钟。
- DN、DDF、Matcher、FGL、网络模块及训练query数量均不变。训练-only dense框解码与门控有额外开销，需要记录训练时间，不能称训练提速。
- 输入、源码、环境、配置和权重身份记录到run_manifest，已有输出目录不能覆盖。

## 预注册指标

主终点：epoch34 EMA、实际K=640、完整验证集、maxDets=640；raw与epochs28–34全部报告，避免事后挑最佳点。primary为固定高密度场景中的local_hybrid_small AP@[.50:.05:.95]；全体local-small作为辅助结果，整体COCO AP为非退化约束。所有数值用0–1尺度，预注册整体AP允许下降上限0.001仅作筛选容差，不是统计显著性阈值。

local-small必须按输入640后 `area<32² OR short_side<16` 采用严格一对一 evaluator；COCO-small独立报告。全部预测先经过预算再分组，不能先过滤small或做score threshold。

固定场景子群：annotation中的 non-crowd GT数>=301 的高密度图片；在该完整图片子群中报告COCO与local-small AP/AR、AP75/AP95。这个群组不等同于所有困难实例；无此类图片时报告undefined而不是0。

@300仅作为输出排序诊断。训练日志的CocoEvaluator与PostProcessor均设640。所有臂必须在同annotation、后端与推理state下比较，不能把既有不同annotation的本地回放与服务器结果直接相减。

完整矩阵：主三臂 ×7epoch ×raw/EMA=42状态，常规实际K=640；每状态输出@640和@300。可先跑单臂再补齐，不默认强制84状态旧矩阵。

## 实际计算预算评价

对预注册epoch34、raw/EMA，真正设置Decoder输入K=640/512/384/256，每个K重新forward；同checkpoint strict load，同一model，输出评价上限固定640。每条结果的身份包含K，禁止用640预测静态截断冒充小K前向。

先看相同K的AP，再报告在相对各自K640整体AP与高密度场景local-small AP均下降<=0.001的最小K。数据不足或所有K失败就如实报告。同步模型forward时间排除I/O并预热；这只是当前后端的模型延迟，不等于部署端吞吐或端到端加速。

本轮不部署动态阈值。排序损失不能保证概率校准；阈值策略需以后独立验证。

## 机制反馈与否定条件

两类配对必须分别记录：outside coverage救援与inside prefix分配。prefix不新增actual640覆盖；outside有合格几何入口也不保证最终Decoder检出。

记录有效配对数、门控拒绝原因、teacher可靠代表、selected覆盖、missing GT、无候选GT、分数差与升温。若有效配对和overflow处罚均长期近零，应判定方法未形成足够干预，不能把零loss当作学习成功。GT>=K时全部query被匹配、无安全donor，是明确边界；不绕过保护凑样本。

成功需要v2对support产生同预算性能或预算曲线改善，同时不违反整体AP约束；改善必须伴随一致的机制变化。七轮单seed只能筛选，正式结论还需多seed/完整训练与固定独立难例集。

## 运行

[配置入口](../../../configs/sgqs_v2/README.md)；[服务器命令](../../../configs/sgqs_v2/SERVER_COMMANDS.md)。本轮提交代码与协议，不自动启动服务器训练。

2026-10-27修订不扩大训练矩阵。记录方法revision=center_ios_overflow_20261027；support关闭配对与overflow，v2启用两者。新增加的overflow只在整图无配对时触发，不能记为跨实例救援。启动命令不变；使用新运行目录并核对runtime manifest，不能将5f56027结果混入本修订。
