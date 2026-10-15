# EXP-DIAG-006 详细分析

## 结论先行

在本地完整验证集的正式 COCO 一对一匹配下，EMA AP-small 的前段上升与后段回落都主要
发生在 **Decoder 最终输出之后的图内 score/rank Top-300 admission**：IoU 合格框通常
仍在 640 个 Decoder query 中，只是排名穿越了第 300 名。完整 640-query 的 AP-small
没有形成同样的倒 U，后段反而提高。

这个结论定位了直接承载机制，但不是训练级根因。现有证据不能区分 SGQS 权重、EMA
更新、分类头校准、训练样本顺序或其他优化动力学中的哪一个导致 score 重排。

## 1. 分析对象与对账

- 完整 wired+wireless val：12,841 图、1,458,060 non-crowd GT。
- e28--e34 × raw/EMA：14 个唯一状态；`best_stg1.pth`、`last.pth` 的两个参数分支都
  与 e34 相同，已去重。
- 每状态固定 640×640，Encoder Top-640，Decoder 640 query，`maxDets=300/640`，
  IoU=.50:.05:.95。
- 每状态共有 8,218,240 个 Decoder query；全实验正式分析包含 10,549,440 行 GT
  transition detail。
- 3,920 项阶段汇总与 280 项 global-PR 汇总全部对账；正式状态见
  [audit.json](../data/audit.json)。

必须区分两套统计：

1. `stage_coverage` 是每个 GT 独立寻找 best IoU；一个 query 可覆盖多个 GT，所以它不是
   TP/FN、COCO recall 或 AP。
2. `match_*`、`transition_*` 与 global PR 来自 score-ordered COCO 一对一匹配，才可用于
   解释 evaluator AP-small。但 transition count 仍不是 AP 的可加分解，因为 AP 还取决于
   全局 TP/FP 分数顺序和 101 点插值。

## 2. EMA 曲线：Top-300 与全部 640 query 分离

| epoch | APs@300 | APs@640 | ARs@300 | ARs@640 |
|---:|---:|---:|---:|---:|
| 28 | 0.674440 | 0.935530 | 0.680017 | 0.946751 |
| 29 | 0.701083 | 0.897127 | **0.706599** | 0.907151 |
| 30 | **0.701179** | 0.935651 | 0.706266 | 0.944944 |
| 31 | 0.685404 | 0.944499 | 0.690354 | 0.953610 |
| 32 | 0.678901 | 0.949416 | 0.683347 | 0.957153 |
| 33 | 0.677411 | 0.950384 | 0.681343 | 0.958279 |
| 34 | 0.677757 | **0.952699** | 0.681571 | **0.959972** |

e30→e34 时，APs@300 下降 0.023423，而 APs@640 上升 0.017049；ARs@300 下降，
ARs@640 也上升。由此可排除“全部 640 query 的 small 检出/几何能力同步退化”作为倒 U
的直接解释。e29 更强地显示预算重排：APs@300 已跃升，但 APs@640 暂时下探。

## 3. 配对归因：前段 gain 与后段 loss 是同一排名机制

直接缺失原因按 FN 端“最深仍存活阶段”定义：

- `candidate_absence`：dense candidates 中没有达到当前 IoU 阈值的框；
- `encoder_selection`：dense 中有合格框，但未进入 Encoder Top-640；
- `decoder_geometry`：Encoder Top-640 有合格框，Decoder all-640 中已不存在；
- `rank_gt300`：Decoder all-640 仍有合格框，但其最终分数排名 >300；
- `one_to_one_competition`：Top-300 独立 coverage 存在，但正式贪心一对一匹配仍为 FN。

### 十 IoU 合计

下表把 IoU=.50:.05:.95 相加，单位严格为 **GT×IoU transition events**。同一 GT 最多可
在十个阈值各贡献一次，因此“13,835”不能读成 13,835 个互异 GT。

| 配对/方向 | candidate | Encoder | Decoder | rank >300 | competition | 总事件 | rank 占比 |
|:---|---:|---:|---:|---:|---:|---:|---:|
| e28→e29 gain | 488 | 780 | 130 | **12,437** | 0 | 13,835 | **89.90%** |
| e28→e30 gain | 603 | 827 | 169 | **11,436** | 0 | 13,035 | **87.73%** |
| e30→e34 loss | 510 | 159 | 177 | **11,934** | 0 | 12,780 | **93.38%** |

所以前段并非主要由“新产生更多 dense candidate”或“Decoder 几何突然变好”带来；后段也
并非主要由这些框消失。两个方向都由合格最终 query 进入/离开 Top-300 承载。

### IoU=.75 单阈值复核

| 配对/方向 | 总转移 | rank >300 | rank 占比 |
|:---|---:|---:|---:|
| e28→e29 gain | 1,374 | 1,263 | 91.92% |
| e28→e30 gain | 1,273 | 1,149 | 90.26% |
| e30→e34 loss | 1,279 | 1,216 | 95.07% |

[analysis_summary.md](../data/analysis_summary.md) 的 “Paired one-to-one transition
attribution” 表就是这个 **IoU=.75 单阈值**切片。它没有 IoU 列，不能把其中的 1,374、
1,273、1,279 误写成十 IoU 合计。十 IoU 数值来自
[transition_attribution.csv](../data/transition_attribution.csv)。

## 4. Global PR：变化集中在 recall=.69--.72

[global_pr_curve.csv](../data/global_pr_curve.csv) 对每个 epoch/state/scope/IoU/budget
保存 101 个 recall anchor 的插值 precision。对 evaluator-small、EMA、Top-300 的十个
IoU 切片逐点相减后：

| 配对 | AP 方向 | recall=.69--.72 对带符号 AP 变化的占比 |
|:---|:---|---:|
| e28→e29 | gain | 81.2% |
| e28→e30 | gain | 80.1% |
| e30→e34 | loss | 88.7% |

这四个 recall anchor 的 precision 从接近 0 到接近 1（或反向）打开/关闭，形成 AP 变化
的主体；其余 recall 区域只有小幅差异。该结果与 rank-300 归因互相印证：变化首先影响
“是否能达到约 0.7 recall”，而不是在整个 PR 区间均匀降低 precision。

## 5. 变化集中于高密度、8--16 px 短边目标

以下均为 evaluator-small、EMA、IoU=.75、`maxDets=300` 的正式一对一 recall。

### 每图 GT 密度

| 每图 non-crowd GT | evaluator-small GT | e28 | e29 | e30 | e34 |
|:---|---:|---:|---:|---:|---:|
| 1--32 | 2,067 | 0.98452 | 0.98645 | 0.98597 | 0.98936 |
| 33--128 | 11,024 | 0.98140 | 0.98086 | 0.97968 | 0.97778 |
| 129--300 | 6,213 | 0.97216 | 0.97055 | 0.97151 | 0.97393 |
| **≥301** | **13,123** | **0.26800** | **0.33514** | **0.33407** | **0.26915** |

低于 301 的三组一直约为 0.97--0.99；倒 U 几乎全部来自单图对象数已经超过 Top-300
预算的图。这与每图 rank 截断机制在结构上吻合。

### 640 输入下 GT 短边

| 短边桶 | evaluator-small GT | e28 | e29 | e30 | e34 |
|:---|---:|---:|---:|---:|---:|
| <8 px | 829 | 0.38601 | 0.38842 | 0.38601 | 0.39324 |
| **8--16 px** | **13,851** | **0.40510** | **0.46762** | **0.46365** | **0.40373** |
| 16--32 px | 15,226 | 0.91836 | 0.91803 | 0.92033 | 0.91830 |
| 32--64 px | 2,299 | 0.99174 | 0.99478 | 0.99391 | 0.99435 |
| ≥64 px | 222 | 0.97748 | 0.97297 | 0.98198 | 0.99550 |

倒 U 主要出现在 8--16 px；最小的 <8 px 组并没有同步大幅上升/回落。因此这里不是一个
“尺寸越小越严重”的单调效应，而是高密度表格中一段特定短边范围的优先级变化。

### domain 复核

| domain | evaluator-small GT | e28 | e29 | e30 | e34 |
|:---|---:|---:|---:|---:|---:|
| wired | 13,084 | 0.66302 | 0.69084 | 0.69551 | 0.66662 |
| wireless | 19,343 | 0.71013 | 0.73624 | 0.73194 | 0.70749 |

两域前升后降同向，排除单一 domain artifact。密度 ≥301、短边 8--16 与 domain 是交叉
分组而非互斥贡献分解；尤其前两组大量重叠，不能把它们的 gain/loss 相加。

## 6. 为什么本地 hybrid-small 看不到同一条曲线

集合规模如下：

| 集合 | 定义 | GT |
|:---|:---|---:|
| evaluator-small | 原图 `area <= 32²` | 32,427 |
| descriptive `coco_small` | 原图 `area < 32²` | 32,346 |
| local hybrid-small | 640 输入上 `area < 32² OR short side <16` | 710,421 |
| hybrid-only | local hybrid-small 且不是 strict coco-small | 684,512 |

hybrid-only 占 hybrid-small 的 **96.353%**，因此 local hybrid 的整体行为几乎由它决定。
在 IoU=.75、all-area 正式匹配、Top-300 下：

| cohort | e28 | e29 | e30 | e34 |
|:---|---:|---:|---:|---:|
| local hybrid-small recall | 0.75573 | 0.75371 | 0.75541 | 0.75741 |
| hybrid-only recall | 0.76101 | 0.75766 | 0.75945 | 0.76275 |

e30→e34 的 hybrid-only recall 上升，足以掩盖 evaluator-small 的下降。两种口径回答不同
问题：hybrid-small 适合分析本项目关心的细长/短边对象，但它没有独立 COCO area range，
本实验也没有为它计算 AP；只能称 recall、coverage 或 transition。

## 7. raw/EMA 比较的边界

raw `APs@300` 为 0.682396/0.678731/0.686712/0.683483/0.685895/0.684099/
0.678740，七轮范围 0.00798；EMA 范围 0.02674。倒 U 在 EMA 路径显著、在 raw 路径很弱，
所以可以写成 **EMA-specific transient/rank trajectory**。

但这不是“EMA decay 导致倒 U”的受控因果证据：raw 与 EMA 是不同参数轨迹，不是只改变
一个变量的干预；resume 初始化、SGQS、数据顺序等也未在本实验中独立控制。

## 8. Query probe 对结论的作用

29 张 probe 从 evaluator-small 与 hybrid-only、e28→e29 gain 与 e30→e34 loss、四种
非竞争直接 cause、wired/wireless 的 32 个 strata 中确定性选取。每张图固定比较 EMA
e28--e34；[query_probe_manifest.csv](../figures/query_probes/query_probe_manifest.csv)
共 203 行并冻结 checkpoint、annotation、回放、脚本、图像和数值数组哈希。

每个图×epoch 都满足：

- Encoder 选中 640 个 query；其 dense 来源 P3/P4/P5 对应 stride 8/16/32，互斥分箱且
  `P3+P4+P5=640`；
- Decoder 最后一层有且只有 640 个 query；`decoder_hybrid_small_count` 是按最终框在
  640 输入尺度上的面积/短边规则事后计数，与 Encoder 来源层无关；
- Decoder boxes、scores、rank order、hybrid mask 与正式 replay 完全一致，最大绝对误差
  都为 0。

probe 提供“同一真实图跨 checkpoint 如何变化”的可视检查，但样本按机制强度选取，不能
用于估计各机制在总体中的发生率；总体比例只能引用正式 CSV。

## 9. 适用范围与未决问题

- 本地正式 annotation SHA-256：
  `f855dd4b9dded934479a54aad70f01b3c678d0ae1fd91e3f8ca1cd6890bd693e`。
- 服务器训练 annotation SHA-256：
  `a6d107322285595d968f63b4cd6d9ccf2442ac844a6875685a2aa16f3b9dd72b`。
- 执行环境：Windows CPU、batch=1、PyTorch 2.14.0+cpu、faster-coco-eval 1.8.0。

两份 annotation 身份不同，本地结果比服务器日志的 EMA AP-small 低约 0.0041--0.0050，
所以这是本地完整集范围内的机制诊断，不是服务器 H20 bit-level 复现，也不能把差额单独
归因于 CPU/GPU。下一步若要回答“为什么 score 会重排”，需要在同一 annotation、同一
后端上做冻结权重分支或训练级受控干预，而不是继续从相关性指标中猜根因。
