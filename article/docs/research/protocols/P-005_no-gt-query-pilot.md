# P-005：无 GT Query 先导诊断

版本：1.0（2026-10-10）

该协议只用于观察候选数量、来源层、encoder/decoder 分数和输出形态，并生成后续可证伪假设。

允许报告：候选数、Top-K 来源、预测框代理尺寸、score 阈值数量、候选重叠、encoder 与 decoder 分数关系。

禁止报告或暗示：GT recall、TP/FP、真实 small-object recall、selection starvation rate、某阶段是主要漏检原因、方法精度提升。

所有图表必须注明 `NO GT`，并说明 P3/P4/P5 是来源层，不是目标尺度类别。
