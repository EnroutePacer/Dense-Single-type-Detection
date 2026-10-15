# EXP-DIAG-006：SGQS 全验证集 checkpoint 回放

## 状态与结论

`completed / valid_with_scope`。本实验已在本地完整验证集上完成 epoch 28--34 的 raw/EMA
回放、正式一对一匹配、阶段归因、全局 PR 对账和代表样本复核。

直接结论是：**EMA AP-small 的倒 U 主要由 IoU 合格的最终 Decoder query 在每图
Top-300 排名边界内外迁移承载，而不是 640 个 Decoder query 的整体几何能力先升后降。**
这回答了收益和回落“直接发生在哪里”，但不把 score/rank 重排进一步归因为 SGQS、EMA
更新公式、过拟合或某一个训练损失。

## 完整验证范围

- 数据：wired 5,847 图 / 627,939 non-crowd GT，wireless 6,994 图 / 830,121
  non-crowd GT，合计 **12,841 图 / 1,458,060 GT**；所有 annotation 引用的图像存在。
- checkpoint：`checkpoint0028.pth`--`checkpoint0034.pth`，每轮 raw、EMA，共 **14 个唯一
  参数状态**。`best_stg1.pth` 和 `last.pth` 的 raw/EMA 参数均与 `checkpoint0034.pth`
  相同，因此按参数哈希去重，没有重复推理。
- 每个状态：固定 640×640 输入、Encoder Top-640、Decoder 640 query，即每状态
  8,218,240 个 Decoder query；正式评价同时保留 `maxDets=300/640` 与
  IoU=.50:.05:.95 十个阈值。
- 正式分析完成 3,920 项阶段对账、280 项 global-PR 对账和 10,549,440 行配对转移明细；
  [audit.json](data/audit.json) 的状态为 `complete`。

## EMA AP-small 轨迹

下表的 AP-small 均为正式 evaluator-small，`APs@300` 是日志指标的同义口径；
`APs@640` 只改变每图检测预算。

| epoch | APs@300 | APs@640 |
|---:|---:|---:|
| 28 | 0.674440 | 0.935530 |
| 29 | 0.701083 | 0.897127 |
| 30 | **0.701179** | 0.935651 |
| 31 | 0.685404 | 0.944499 |
| 32 | 0.678901 | 0.949416 |
| 33 | 0.677411 | 0.950384 |
| 34 | 0.677757 | **0.952699** |

Top-300 在 e29/e30 达峰后回落，而 640-query AP-small 在后段持续提高。尤其 e29 的
Top-300 上升与 APs@640 下探同时发生；因此倒 U 不是“Decoder 的 640 个输出整体先变好再
变坏”，而是最终分数排序与 300 个 admission 名额发生了重分配。

## 前段收益与后段回落的直接载体

正式 `coco_small_eval`、EMA、`maxDets=300` 下，把十个 IoU 阈值的转移相加时，单位是
**GT×IoU transition events**，不是互异 GT 数，也不是 AP 的可加分解：

| 配对 | 方向 | 总事件 | `rank_gt300` | 占比 |
|:---|:---|---:|---:|---:|
| e28→e29 | gain | 13,835 | 12,437 | 89.90% |
| e28→e30 | gain | 13,035 | 11,436 | 87.73% |
| e30→e34 | loss | 12,780 | 11,934 | 93.38% |

`rank_gt300` 表示该 GT 在 640 个最终 Decoder query 中仍有达到当前 IoU 阈值的 query，
但该 query 的图内分数排名大于 300。也就是说，前段收益主要是合格 query 进入 Top-300，
后段回落主要是合格 query 又跌出 Top-300。生成的
[analysis_summary.md](data/analysis_summary.md) 中 paired attribution 表只展示
**IoU=.75 单阈值**；十 IoU 合计必须从
[transition_attribution.csv](data/transition_attribution.csv) 汇总，不能把两者混为一谈。

全局 101 点 PR 对账给出同一结论的另一视角：e28→e29、e28→e30、e30→e34 的 AP
变化分别有 81.2%、80.1%、88.7% 集中在 recall=.69--.72 四个采样点的打开或关闭。
因此变化不是均匀分布在整条 PR 曲线上。

## 信号出现在哪里

在 IoU=.75、EMA、evaluator-small、`maxDets=300` 下，变化几乎全部定位到：

- 每图 non-crowd GT 数量 **≥301**：recall 为
  e28/e29/e30/e34 = 0.26800/0.33514/0.33407/0.26915；较低密度桶基本稳定。
- 直接缩放到 640×640 后 GT 短边 **8--16 px**：recall 为
  0.40510/0.46762/0.46365/0.40373；其他短边桶没有同幅度倒 U。
- wired 与 wireless 同向：wired 为
  0.66302/0.69084/0.69551/0.66662，wireless 为
  0.71013/0.73624/0.73194/0.70749，所以不是单一 domain artifact。

密度 ≥301 与短边 8--16 大量重叠，是两条交叉观察轴，**不能把两者的变化量相加**。

## evaluator-small 与本地 hybrid-small

本实验同时保留三种明确命名的集合：

- `coco_small_eval`：原图 annotation `area <= 32²`，32,427 GT；这是正式 AP-small/AR-small
  的 evaluator population。
- `coco_small`：描述性严格边界 `area < 32²`，32,346 GT。
- `local_hybrid_small`：框直接缩放到 640×640 后满足
  `area < 32² OR short side < 16`，710,421 GT。

其中 `hybrid_only` 有 684,512 GT，占 local hybrid 的 96.353%。在 IoU=.75 的 all-area
正式匹配下，hybrid-only Top-300 recall 从 e30 的 0.75945 上升到 e34 的 0.76275；这一
巨大集合会淹没 evaluator-small 的后段下降。因此 hybrid-small 只能报告
recall、coverage 或 transition，**不能称为 AP-small，也不能用来替代 evaluator-small
解释日志曲线**。

## 图、代表样本与 query probe

- [analysis/README.md](analysis/README.md)：详细结论、数值分解和可/不可推出的因果范围。
- [data/README.md](data/README.md)：正式 CSV/JSON、字段语义、计数单位与复算入口。
- [figures/README.md](figures/README.md)：6 张定量图、6 张 Decoder 代表图和 29 张
  checkpoint query probe 的索引与读图规则。
- [query_probe_selection.csv](figures/query_probes/query_probe_selection.csv) 从
  evaluator-small 与 hybrid-only 两条 track、前段 gain/后段 loss、四类直接 cause、
  wired/wireless 共 32 个 strata 确定性选样；因复用同图，共 29 张唯一图。
- [query_probe_manifest.csv](figures/query_probes/query_probe_manifest.csv) 有
  29×7=203 行，覆盖 EMA e28--e34。每行均验证 Encoder source-level
  P3+P4+P5=640、Decoder query=640，且 Decoder boxes/scores/rank/hybrid mask 与正式回放
  完全一致。

probe 中的 P3/P4/P5 是 **Encoder Top-640 选中 query 的 dense source level**，对应
stride 8/16/32，单图单 checkpoint 互斥且合计 640；`Decoder hybrid-small` 是对最终
Decoder box 按本地宽高规则事后分组。两者不是同一个统计，不能混称；probe 是定性机制
示例，不是总体占比估计。

## 有效范围与限制

本实验是完整的**本地数据、本地后端 scoped diagnosis**，不是服务器 H20 的 bit-level
复现：

- 正式本地 merged annotation SHA-256 为
  `f855dd4b9dded934479a54aad70f01b3c678d0ae1fd91e3f8ca1cd6890bd693e`；服务器训练日志
  冻结的 merged annotation SHA-256 为
  `a6d107322285595d968f63b4cd6d9ccf2442ac844a6875685a2aa16f3b9dd72b`，二者不同。
- 推理后端为 Windows CPU、PyTorch 2.14.0+cpu、batch=1；matcher 为
  faster-coco-eval 1.8.0。本地 EMA AP-small 比服务器日志低约 0.0041--0.0050，但由于
  annotation 身份也不同，不能把差额单独归因于 CPU/GPU。
- raw 的 `APs@300` 七轮范围仅 0.00798，而 EMA 范围为 0.02674；可以描述为
  “EMA-specific transient/rank trajectory”，但没有受控实验支持“EMA decay 是根因”。
- 这里证明的是 AP 变化的直接承载阶段和集中子群；为什么训练使这些 query 的分类分数
  穿越第 300 名，仍需训练级受控干预回答。

固定 val128 的先导结果见
[EXP-DIAG-005](../EXP-DIAG-005_sgqs-top300-ranking-replay/README.md)。
