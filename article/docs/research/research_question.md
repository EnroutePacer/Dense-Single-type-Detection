# 研究问题：Query Selection–Survival Gap

## 问题定义

在固定 query 预算下，D-FINE/DEIM 使用全局分类分数 Top-K 初始化 decoder。对密集、单类别、细长或低像素短边目标，高分重复候选与跨层分数失衡可能占用 query 预算，使 dense encoder 本可表示的 GT 在 Top-K 后失去唯一覆盖；入选 query 还可能在 decoder 细化或重评分过程中丧失几何覆盖或置信度。

这是**待验证、可证伪的假设**，不是已经成立的结论。

## 当前代码链路

```text
P3/P4/P5 memory（640 输入共 8,400 个位置）
  -> encoder 单类分类分数
  -> global Top-K=640
  -> 入选位置生成初始 reference/query
  -> decoder 读取完整 memory 并逐层细化
  -> postprocess / evaluator
```

关键实现：

- [`engine/deim/dfine_decoder.py`](../../engine/deim/dfine_decoder.py)：encoder score、global Top-K、query/reference 初始化与 decoder。
- [`engine/deim/deim_criterion.py`](../../engine/deim/deim_criterion.py)：匹配和损失监督。
- [`scripts/diagnose/diagnose_deim_query_selection.py`](../../scripts/diagnose/diagnose_deim_query_selection.py)：当前无 GT 先导诊断。
- [`engine/data/dataset/coco_eval.py`](../../engine/data/dataset/coco_eval.py)：正式 COCO 评估口径。

Top-K 索引不可微，query content/reference 在进入 decoder 前被 detach；但 decoder 仍以完整 memory 作为 cross-attention value。因此不能说未选位置“完全不可见”，只能说它无法作为新的独立 decoder 初始化种子。

## 先导观察（无 GT）

640 输入的候选池：P3 6,400、P4 1,600、P5 400，共 8,400。

| 范围 | P3 入选 | P4 入选 | P5 入选 | `score>0.5` 来源 |
|---|---:|---:|---:|---|
| `wired/image (12)` | 153 | 425 | 62 | 118 个全部来自 P4 |
| 13 张图合计 | 3,356 | 4,062 | 902 | P3/P4/P5 = 3/1,429/0 |

这些数字只说明来源和分数行为不均衡，不能证明 GT recall 下降。示例图一是模型预测可视化，不是 GT；P3/P4/P5 是来源层，不是 small/medium/large 标签。

## GT 可视化恢复先导证据（非正式评估）

用户提供了 `wired/image (12)` 的 GT 可视化，但尚未取得原始 COCO JSON。由可视化恢复出 322 个矩形，其中正文为 38×8=304 个密集单元格。正文框缩放到 640 后的短边约 12.6～14.9 px：在 P3 上只有 1.57～1.87 个 cell，在 P4/P5 上均不足 1 个 cell。

把恢复框与已导出的 dense encoder boxes 配准后，得到以下探索性结果：

| IoU | Dense 可覆盖 | Top-640 可覆盖 | selection-starved |
|---:|---:|---:|---:|
| 0.30 | 303/304 | 236/304 | 67 |
| 0.50 | 278/304 | 135/304 | 143 |
| 0.75 | 137/304 | 78/304 | 59 |

在 IoU≥0.5 下，P3 dense 可覆盖 265/304，入选后仅覆盖 60/304；每个 GT 的最佳几何候选 rank 中位数约 3814，79.6% 位于 rank 640 之后。这为 selection starvation 提供了强烈先导证据，但坐标来自 GT 图片恢复，不能替代使用原始 annotation JSON 的 EXP-DIAG-003 正式结论。

另一个重要边界是：这些正文框只有 22/304 按 COCO 面积属于 small，多数实际难点是**短边很小且细长**。论文应报告 short-side/stride 与 aspect-ratio 分桶，不能只用 COCO AP-small 描述问题。

## 正式阶段定义

对每个 GT `g` 和 IoU 阈值 `τ`：

```text
C_dense(g,τ) = 1[max IoU(b_i,g) >= τ, i 属于全部 dense proposals]
C_topK(g,τ)  = 1[max IoU(b_i,g) >= τ, i 属于入选 Top-K]
SSR_τ        = P(C_topK=0 | C_dense=1)
```

逐 GT 的失败分为：

1. **Generation failure**：全部 dense proposals 都不能覆盖 GT。
2. **Selection failure**：dense 可覆盖，Top-K 后失去覆盖。
3. **Decoder failure**：Top-K 已覆盖，decoder 后失去几何覆盖、匹配身份或置信度。
4. **Output failure**：decoder 已产生有效框，但被阈值、输出 Top-K、NMS 或 evaluator 截断。

## 必须报告

- dense proposal recall、Top-K proposal recall、SSR；
- unique-GT coverage 与有效 query 利用率；
- 每个 GT 的最佳 dense candidate rank；
- 每个 GT 周围重复候选数；
- encoder score 与 GT IoU 的分层校准；
- decoder 每层 IoU、score 与匹配身份轨迹；
- 按 GT 数、短边像素、`min(w,h)/stride`、面积、长宽比、局部密度、边界和域分桶。

## 判定逻辑

- dense recall 已低：优先研究表示、stride、输入或形状先验；否定“Top-K 是主瓶颈”。
- dense recall 高而 Top-K recall 明显下降：支持 selection starvation。
- Top-K recall 高而 decoder/final recall 下降：支持 survival collapse。
- 同 K 的 coverage-aware selector 改善 unique coverage，并经训练转化为相同 epoch 更高 AP/AR，才可能成为论文贡献。

当前唯一合规总述是：**无 GT 先导诊断揭示了值得验证的 selection–survival 不均衡，但尚未定位真实漏检阶段。**
