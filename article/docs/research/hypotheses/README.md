# 可证伪假设注册表

| ID | 假设 | 关键证伪实验 |
|---|---|---|
| H-001 | 跨层 encoder score 不可直接比较，global Top-K 对高分辨率候选产生选择偏置，降低困难 GT 的 unique coverage。 | 带 GT 比较 dense 与 Top-K recall、各层 score-IoU 校准；若没有显著 drop，则否定。 |
| H-002 | 同一目标周围的高分重复候选占用固定 K，造成 coverage starvation。 | 同 K 下做二分图 coverage 和去重/coverage-aware selector；若 unique coverage 不升，则否定。 |
| H-003 | 低分位置不入选且缺少直接 encoder 匹配/回归监督，形成 selection-starvation 闭环。 | 记录选中/未选位置的梯度与最佳 GT candidate rank 随 epoch 的轨迹。 |
| H-004 | 一部分 GT 已被 Top-K 覆盖，但在 decoder 细化或重评分中发生 survival collapse。 | 逐 GT、逐 decoder layer 跟踪 IoU、score 和匹配身份；没有阶段性下降则否定。 |
| H-005 | 部分失败在 Top-K 前已发生，根因是 stride-8 表示、anchor/形状先验或输入分辨率不足。 | 全部 8,400 个 dense proposals 对困难 GT 的 recall 若已高，则否定。 |
| H-006 | VFL 的 IoU 质量目标使短边小、细长框的排序分数更易受微小定位误差影响。 | 按短边/长宽比/层级比较校准，并与 MAL 或独立 objectness 做严格对照。 |

H-001 是当前主假设，H-002～H-006 是竞争或补充解释。未完成带 GT 分阶段诊断前，不建立“已验证方法”。
