# EXP-DIAG-005：SGQS Top-300 排名回放诊断

状态：`completed / valid_with_scope`；论文角色：`diagnosis`。

本实验归档 SGQS epoch 28--34 checkpoint 的完整验证集曲线、固定 val128
raw/EMA 回放、逐 GT 阶段覆盖和跨 epoch 排名转移。当前最直接的结论是：固定
val128 上复现的 COCO AP-small 倒 U，主要由最终 Decoder query 的分类分数排名跨过
`maxDets=300` 边界承载；它不是这批样本中候选框消失或定位 IoU 普遍退化的结果。
这个结论定位了**指标变化发生在哪一阶段**，但尚未证明**是哪项训练机制导致了分数
重排**，也不能用 128 图子集估计该机制解释完整验证集变化的比例。

## 问题与实验范围

SGQS 的完整验证集 EMA AP-small 在 epoch 29/30 突然升高，随后回落，而常见的
Encoder/Decoder 汇总量没有给出同样形状的直接对应指标。本次回放因此不再比较
“一共画出了多少小框”，而是对同一批 GT 逐轮追踪：合格框是否存在、是否进入实际
Encoder Top-640、是否被 Decoder 保留、以及最终分类分数排名是否进入前 300。

完整验证集指标来自服务器逐 epoch evaluator 输出；机制定位使用由 SHA-256 规则
固定抽取、与模型输出无关的 128 张 val 图，共 15,446 个 non-crowd GT。回放关闭
增强，使用与验证一致的 640×640 输入，对 epoch 28--34 的 raw 与 EMA 状态分别执行。
精细的逐 GT 转移分析只读取 EMA，不把 raw 行混入。

本实验同时保留两种 small 定义，二者用途不同：

- **COCO-small**：原图 annotation `area < 32²`，val128 中有 430 个，只分布在
  17 张图；这是 COCO AP-small/AR-small 的正式分桶口径。
- **本地 hybrid-small**：将框直接缩放到 640×640 后，满足
  `bbox area < 32² OR bbox short side < 16`，val128 中有 7,314 个；这是结合面积
  与短边的任务诊断口径，不等于 COCO-small，也没有被当作 COCO AP-small。

两组交集为 323 个 GT，COCO-only 为 107 个，hybrid-only 为 6,991 个。正式
AP-small 曲线必须按 COCO 定义阅读；hybrid-small 只用于检查本地关心的细长/短边
目标是否呈现同样变化。

## 阶段口径与重要限制

每张图的阶段顺序为：

1. `dense-all`：8084 个有效 dense candidate；
2. `dense Top-640`：Encoder 选择器实际取出的 640 个候选；
3. `decoder-all640`：Decoder 最后一层输出的全部 640 个 query；
4. `decoder Top-300`：按最终分类分数排序后保留的前 300 个输出。

这里的 `decoder-all640` 和 `decoder Top-300` 都是 **Decoder 最终输出**；Top-300
不是输入 Decoder 的量。阶段 coverage 对每个 GT 独立询问“该阶段是否至少存在一个
IoU 达到阈值的框”。因此，同一个 query 可以同时覆盖多个相邻 GT，表中的 gain、
loss 和 coverage **不是 COCO 一对一匹配得到的 TP、FN 或 recall**，也不计重复预测
造成的 FP。真 AP/AR 只取 evaluator 输出。这个区别对高密度表格尤其重要。

## 完整验证集：EMA AP-small 的倒 U

| epoch | overall AP | AP-small |
|---:|---:|---:|
| 28 | 0.865362 | 0.679395 |
| 29 | 0.864145 | 0.705142 |
| 30 | 0.865513 | **0.705554** |
| 31 | 0.866372 | 0.690087 |
| 32 | 0.866606 | 0.683792 |
| 33 | 0.867318 | 0.681879 |
| 34 | **0.867590** | 0.682274 |

epoch 29 和 30 都超过 0.69，且两轮 raw/EMA 参数哈希均不同；完整验证集峰值是
epoch 30，但它只比 epoch 29 高约 0.00041。随后 overall AP 继续上升，AP-small
却回到接近 epoch 28 的水平，说明“保存 overall 最优”与“保存 AP-small 峰值”不是
同一个选择问题。完整数值见 [full_val_metrics.csv](data/full_val_metrics.csv)，曲线见
[trajectory_overview.png](figures/trajectory_overview.png)。

## 固定 val128：倒 U 首先出现在最终 300 名预算

固定子集也出现低—高—低，但 EMA 峰值在 epoch 29，不是完整验证集的 epoch 30；
这一区别本身就是小样本方差的警告。下表中的 AP/AR 是 val128 上真正的 COCO
evaluator 结果，而不是独立 GT coverage。

| epoch | EMA APs@300 | EMA ARs@300 | EMA ARs@640 | raw APs@300 |
|---:|---:|---:|---:|---:|
| 28 | 0.609321 | 0.614884 | 0.976047 | 0.626702 |
| 29 | **0.687327** | **0.688605** | 0.890698 | 0.607533 |
| 30 | 0.656896 | 0.663488 | 0.948837 | 0.597251 |
| 31 | 0.607165 | 0.609767 | 0.967209 | 0.618387 |
| 32 | 0.605198 | 0.607907 | 0.967674 | 0.581170 |
| 33 | 0.602446 | 0.602326 | 0.968140 | 0.609214 |
| 34 | 0.602340 | 0.602093 | 0.967907 | 0.617864 |

EMA 的 @300 指标在 epoch 29 抬升时，@640 AR 反而下降；raw APs@300 也没有
EMA 的倒 U。两点都不支持“所有小目标候选和定位能力在中期普遍增强”的解释。
raw/EMA 明细见 [val128_raw_ema_metrics.csv](data/val128_raw_ema_metrics.csv)。

进一步看 IoU≥0.75 的 COCO-small 独立 GT coverage（分母均为 430）：

| epoch | dense-all | dense Top-640 | decoder-all640 | decoder Top-300 |
|---:|---:|---:|---:|---:|
| 28 | 417 | 417 | 426 | 266 |
| 29 | 417 | 368 | 390 | **298** |
| 30 | 419 | 404 | 417 | 288 |
| 34 | 417 | 417 | 426 | 263 |

Decoder Top-300 的覆盖在 IoU≥0.50 与 IoU≥0.75 下七轮完全相同。换言之，这批
跨过 300 名边界的框一旦进入 Top-300，通常已经达到 0.75 IoU；峰形首先对应的是
分数/rank，而不是 IoU 在 0.50 与 0.75 之间摆动。完整七轮阶段表见
[stage_coverage.csv](data/val128_ema/stage_coverage.csv)，图见
[stage_coverage.png](figures/stage_coverage.png)。

## 前段收益：epoch 28 → 29

COCO-small Decoder Top-300 在 IoU≥0.75 下有 49 个 gain、17 个 loss，净增加
32 个 GT，即 266→298：

- 49 个 gain 在 epoch 28 的 `decoder-all640` 中已经全部拥有 IoU≥0.75 的框，
  只是排名位于 301--640；其合格框 rank 中位数从 359 上升到 259。
- 17 个 loss 中，15 个在 epoch 29 仍有合格 dense candidate，但未进入实际
  dense Top-640；另 2 个仍在 `decoder-all640`，但跌到 Top-300 之外。
- 同期 dense Top-640 和 decoder-all640 的覆盖分别从 417 降至 368、从 426
  降至 390；因此最终
  Top-300 的净收益不能解释为上游广覆盖整体变好。

这里最明确的正向机制是：一批原本已有高 IoU 框的 COCO-small GT，其对应 query
在最终分类排序中从 300 名外升到 300 名内。17 个 loss 表明 epoch 28→29 并非每个
个体都只在最终排名阶段变化，但 49 个 gain 的来源全部是最终 rank。

## 后段回跌：epoch 30 → 34

同一口径下有 11 个 gain、36 个 loss，净减少 25 个 GT，即 288→263：

- 36 个 loss 在 epoch 34 的 `decoder-all640` 中仍然全部存在 IoU≥0.75 的框，
  只是全部跌至 rank>300；其 rank 中位数从 271.5 变为 338.5。
- 同期 dense Top-640 和 decoder-all640 在 IoU≥0.75 下分别净增加 13 和 9 个
  覆盖（IoU≥0.50 下均净增加 9 个），所以上游候选生存和 Decoder 全 640 几何
  覆盖没有随 AP-small 一起退化。
- 11 个 gain 中，5 个在 epoch 30 已进入 Decoder 但 rank>300，另 6 个当时有
  dense 合格候选却未入 dense Top-640；这些反向变化不改变 36 个 loss 的直接
  去向均为最终 rank 的事实。

因此，val128 后段回跌的直接承载位置可以明确到最终 score/rank 截断，而不能写成
“Decoder 不再输出这些小框”或“定位框消失”。逐 GT 转移见
[transition_annotations.csv](data/val128_ema/transition_annotations.csv)，rank 分布见
[transition_rank_shifts.csv](data/val128_ema/transition_rank_shifts.csv) 和
[rank_transition.png](figures/rank_transition.png)。

### 24 个最干净的同一 GT 轨迹

有 24 个 COCO-small GT 在四个关键时点呈现完全一致的 Top-300 状态
`0 → 1 → 1 → 0`（epoch 28、29、30、34），且全部来自 `image_id=11570`：

| epoch | 28 | 29 | 30 | 34 |
|---:|---:|---:|---:|---:|
| Top-300 状态 | 0 | 1 | 1 | 0 |
| 合格框 rank 中位数 | 356.5 | 238 | 276 | 351 |
| Decoder-all 最佳 IoU 中位数 | 0.9806 | 0.9843 | 0.9858 | 0.9866 |

几何质量不仅没有随 rank 回落，最佳 IoU 中位数还保持在约 0.98。这个配对轨迹是
“同一个 GT 的合格 query 反复跨过 Top-300 边界”的直接证据；它比比较四轮互不
对应的框数量更有解释力，但因为全部集中在一张极密图，不能代表普通图像的频率。

## 两种 small 定义揭示的预算重分配

hybrid-small 的 Decoder Top-300 独立 coverage 从 epoch 28 到 34 依次为
`75.21%、75.02%、75.09%、75.36%、75.46%、75.53%、75.58%`，没有 COCO
AP-small 的倒 U。发生变化的 COCO-small GT 全部位于 323 个
`COCO-small ∩ hybrid-small` 中；107 个 COCO-only GT 七轮始终 100% 被
Top-300 覆盖。

IoU≥0.50 的分组净变化为：

| 区间 | COCO∩hybrid | COCO-only | hybrid-only | hybrid 合计 |
|---|---:|---:|---:|---:|
| e28→e29 | +32 | 0 | -46 | -14 |
| e28→e30 | +22 | 0 | -31 | -9 |
| e30→e34 | -25 | 0 | +61 | +36 |

这与有限 Top-300 预算在 small 子群之间重分配相一致：中期更偏向
COCO∩hybrid，后期又偏回数量更大的 hybrid-only，而不是本地定义下所有 small-like
GT 一起变差。但 coverage 允许同一个 query 覆盖多个 GT，所以上表不是守恒流量，
不能机械解释成 25 个 query 被 61 个 query 一一替换，更不能替代 custom hybrid
AP。交并分组原表见
[exclusive_top300_coverage.csv](data/val128_ema/exclusive_top300_coverage.csv) 和
[exclusive_top300_transitions.csv](data/val128_ema/exclusive_top300_transitions.csv)。

## raw/EMA 与 checkpoint 身份

val128 上只有 EMA 明显复现中期峰值，raw 七轮没有同形状的倒 U。这使 EMA 的参数
混合、时间滞后或与分类校准/排序的相互作用成为上游强线索；但现有证据没有 full-val
raw 配对曲线，也没有改变 EMA 的受控干预，因此不能断言“EMA 导致了 AP-small
倒 U”，更不能据此证明训练过拟合。

`checkpoint0034.pth`、`last.pth` 和 `best_stg1.pth` 的容器文件 SHA-256 不同，
但按参数名、dtype、shape 与 tensor bytes 计算后，三者的 raw state SHA-256 完全
相同，EMA state SHA-256 也完全相同。因此重复回放这三个文件不会产生新的模型输出。
`best_stg1.pth` 的保存依据是 overall AP，而不是 AP-small；完整验证集 overall AP
在 epoch 34 最高，所以它没有保存 epoch 29/30 的 small 峰值。逐 checkpoint 身份见
[checkpoint_identity.csv](data/checkpoint_identity.csv)。

## 代表性与定性图片边界

val128 适合严格的跨 epoch 配对，但不适合估计完整验证集中的机制占比：430 个
COCO-small 只出现在 17 张图，所有跨 epoch 的 COCO-small Top-300 状态变化又只
发生在 `image_id=11570` 和 `image_id=12461`。两张图分别含 480 和 320 个总 GT，
恰好处于或超过 `maxDets=300` 的强容量约束区间。子集峰值在 epoch 29、且 e29
相对 e30 高约 0.0304 AP；完整验证集则由 epoch 30 略高约 0.00041。这些差异都
要求克制外推。

另有一套相同 13 图、无 GT 的 Encoder/Decoder 可视化。它可以定性展示框密度、
尺度分布和 query 形态，但无法判断 TP/FN、独立 GT coverage 或 AP 变化，也不能为
上述阶段归因增加定量证据。归档的代表性 contact sheet 为
[wired image 12](figures/qualitative_wired_image12.png) 和
[wireless image 5](figures/qualitative_wireless_image5.png)；它们只应作为视觉索引，
不应被引用为“小框召回上升/下降”的证明。

## 已证、未证与下一步

本实验已经直接支持：

- full-val EMA AP-small 在 epoch 29/30 达峰后回落，且峰值 checkpoint 确实存在、
  参数状态互不相同；
- val128 EMA 的前段收益以已有高 IoU query 升入 Top-300 为主要正向来源，后段
  36 个 loss 则全部是已有高 IoU query 跌出 Top-300；
- 24 个同一 GT 的配对轨迹在定位 IoU 稳定时完成 `0→1→1→0`，直接排除了这些
  个例中的“框消失/定位崩塌”解释；
- 本地 hybrid-small coverage 不呈倒 U，COCO∩hybrid 与 hybrid-only 的净变化方向
  相反，支持排序预算在 small 子群间重分配；
- `best_stg1`、`last` 与 `checkpoint0034` 的 raw/EMA 模型状态相同，重复评估无益。

本实验尚未证明：

- full-val AP-small 倒 U 有多大比例由同一排名机制解释；
- SGQS 的 owner/GO、DN 蒸馏、分类校准、样本密度、EMA 或其他优化动态中哪一项
  是 score/rank 重排的根因；
- coverage 净变化是一对一 query 预算交换，或 hybrid-small 的 custom AP 同步提升；
- raw 与 EMA 的差异等价于过拟合，或 13 图定性现象具有统计代表性。

下一步已登记为
[EXP-DIAG-006](../EXP-DIAG-006_sgqs-full-val-checkpoint-replay/README.md)：在完全相同的
验证预处理与 GPU 后端下，对 epoch 28--34 的 raw/EMA 共 14 个唯一状态做 full-val
逐 GT 回放，同时保存 Decoder 全 640 的 score/rank、Top-300 截断、COCO 一对一匹配、
十个 IoU 阈值、`maxDets=300/640` 与全局 PR/score 排序。只有 full-val 配对结果确认
同样的 rank 转移占主导后，才应进一步用冻结 checkpoint/batch 对 EMA、SGQS 监督来源
或分类校准做单变量干预。

## 归档入口与未入库材料

- 详细中文分析：[val128 EMA 报告](analysis/val128_ema_report.md)；首轮机器汇总：
  [initial replay report](analysis/initial_replay_report.md)。
- 小型派生数据及字段说明：[data/README.md](data/README.md)；精细表目录：
  [data/val128_ema/](data/val128_ema/)。
- 复算脚本及输入约定：[scripts/README.md](scripts/README.md)、
  [analyze_val128_ema.py](scripts/analyze_val128_ema.py) 和
  [analyze_val128.py](scripts/analyze_val128.py)。
- 统计图：[trajectory overview](figures/trajectory_overview.png)、
  [stage coverage](figures/stage_coverage.png)、
  [rank transition](figures/rank_transition.png)。
- 全包逐文件大小与 SHA-256：[artifact_manifest.csv](artifact_manifest.csv)。

仓库没有提交原始 `.pth` 权重；其文件哈希和 raw/EMA state 哈希已进入
`checkpoint_identity.csv`。七轮 raw+EMA 原始 replay JSONL 合计约 120 MiB，也未
入库，当前只归档可审计结论所需的派生 CSV/JSON。固定 val128 的私有 COCO 标注、
原始数据集图像及服务器绝对路径同样不公开；重新生成逐 GT 结果需要在获授权的服务
器环境中提供这些材料，并按 [scripts/README.md](scripts/README.md) 的接口复算。
