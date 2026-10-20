# DEIM / D-FINE-M 实际链路与 `multiscale_nms070` 输出审计

本文只针对以下实际文件与输出作结论：

- 推理配置：`D:\Baseline_test\models\DEIM\configs\table_cells\dfine_hgnetv2_m_table_cell.yml`
- 推理权重：`D:\Baseline_test\models\DEIM\outputs\last.pth`
- 多尺度脚本：`D:\Baseline_test\tools\benchmark_deim_tta_cpu.py`
- 待解释结果：`D:\Baseline_test\output\_smoke_DEIM_multiscale_nms070`
- D-FINE 论文：[D-FINE: Redefine Regression Task in DETRs as Fine-grained Distribution Refinement](https://arxiv.org/abs/2410.13842)
- DEIM 论文：[DEIM: DETR with Improved Matching for Fast Convergence](https://arxiv.org/abs/2412.04234)
- RT-DETRv2 论文：[RT-DETRv2: Improved Baseline with Bag-of-Freebies for Real-Time Detection Transformer](https://arxiv.org/abs/2407.17140)

## 1. 先给出校对结论

原来的描述：

```text
三层密集特征
→ 每个特征位置生成参考框
→ encoder 给所有位置打分
→ 选出 Top-640
→ 作为 640 个 decoder query
→ 4 层 D-FINE decoder 反复细化
→ 输出 640 个框和分数
→ 按分数取 Top-640，不做 NMS
```

作为**单次前向推理的简写基本正确**，但需要进行以下四处严格修正。

1. 这主要是 **RT-DETR 风格的查询选择 + D-FINE 解码器链路**，不是 DEIM 论文另外发明的一套候选框生成链路。DEIM 论文的核心贡献主要是训练阶段的 Dense O2O 和 Matchability-Aware Loss（MAL）。
2. “encoder 给所有位置打分”不够精确。实际是 HGNetv2 与 HybridEncoder 先生成三层融合特征，随后 `DFINETransformer` 将其拉平，并用 `enc_output + enc_score_head` 对每个位置计算类别 logit。
3. Top-640 不只是取 640 个位置：代码还使用对应位置的特征作为 query content，并用该位置的确定性网格参考框加上 encoder 回归残差，形成 decoder 的初始 reference box。
4. 当前是单类别、`num_queries=640`、`num_top_queries=640`。因此模型原生后处理中的 Top-640 实际上不会减少框数，只会按分数重排 640 个结果；它既不是阈值过滤，也不是 NMS。

更准确的完整推理链路是：

```text
方形缩放后的图像
→ HGNetv2-B2 backbone
→ HybridEncoder：三层投影、最高层注意力、FPN/PAN 融合
→ P3/P4/P5 三层特征（stride 8/16/32，通道数 256）
→ 拉平并拼成密集 memory
→ 每个空间位置生成一个确定性参考框
→ enc_output + enc_score_head 给所有位置计算类别 logit
→ 按类别 logit 选择 Top-640 个位置
→ 选中位置的 memory 作为 query content
→ 选中参考框 + encoder box residual 作为初始 reference box
→ 4 层 D-FINE decoder：query self-attention、MS-Deformable cross-attention、门控融合、FFN、FDR 分布细化
→ 最后一层输出 [B, 640, 1] 类别 logits 与 [B, 640, 4] 归一化框
→ sigmoid + cxcywh→xyxy + 映射回原图 + Top-640
→ 单次模型原生链路结束：没有分数阈值，也没有 NMS
```

训练时还存在两条未写进原简图的支路：

- 训练专用 contrastive denoising queries；
- Hungarian matching、VFL、L1、GIoU、FGL、DDF 以及各 decoder 层/encoder 输出的辅助损失。

## 2. 本次实际模型配置

配置继承关系是：

```text
configs/table_cells/dfine_hgnetv2_m_table_cell.yml
└── configs/base/dfine_hgnetv2.yml
```

解析后的关键参数如下。

| 项目 | 实际值 | 证据 |
|---|---:|---|
| 类别数 | 1（`cell`） | table-cell 配置 `num_classes: 1` |
| Backbone | HGNetv2-B2 | table-cell 配置 `HGNetv2.name: B2` |
| Backbone 输出层 | `[1,2,3]` | `return_idx: [1,2,3]` |
| HybridEncoder 输入通道 | `[384,768,1536]` | table-cell 配置覆盖值 |
| 融合后通道 | 256 | `hidden_dim: 256` |
| 特征步长 | `[8,16,32]` | base D-FINE 配置 |
| decoder 层数 | 4 | table-cell 配置 `num_layers: 4` |
| matching queries | 640 | `num_queries: 640` |
| 部署输出层 | 最后一层，即索引 3 | `eval_idx: -1` |
| 每层多尺度采样点 | `[3,6,3]` | base D-FINE 配置 |
| FDR 离散上界 | `reg_max=32`，即每条边 33 个 bin | base D-FINE 配置 |
| 原生后处理数量 | 640 | `PostProcessor.num_top_queries: 640` |
| 训练时 DN 预算 | 100 | `num_denoising: 100` |
| 分类损失 | VFL，不是 MAL | `losses: ['vfl','boxes','local']` |
| 回归/定位损失权重 | L1=5，GIoU=2，FGL=0.15，DDF=1.5 | `DEIMCriterion.weight_dict` |

模型的最外层类名虽然叫 `DEIM`，但它只负责按顺序调用：

$$
\operatorname{Model}(I)=
\operatorname{DFINETransformer}
\left(
\operatorname{HybridEncoder}
\left(
\operatorname{HGNetv2}(I)
\right)
\right).
$$

对应代码为 `engine/deim/deim.py:22-28`。

## 3. 每一步的实际算法、公式和训练/推理属性

### 3.1 HGNetv2 与三层密集特征

设输入张量为：

$$
I\in\mathbb{R}^{B\times3\times S\times S}.
$$

HGNetv2-B2 输出三层 backbone 特征，HybridEncoder 首先用 `1×1 Conv + BatchNorm` 将各层投影到 256 通道：

$$
X_l=\operatorname{BN}\left(\operatorname{Conv}_{1\times1}^{(l)}(F_l)\right),
\qquad l\in\{0,1,2\}.
$$

只有索引 2 的最高层进入一层 Transformer encoder。忽略多头拆分时，其自注意力可写为：

$$
Q=K=X_2+PE,\qquad V=X_2,
$$

$$
\operatorname{Attention}(Q,K,V)
=\operatorname{Softmax}\left(\frac{QK^\top}{\sqrt{d_h}}\right)V.
$$

随后通过自顶向下 FPN 和自底向上 PAN 进行跨尺度融合，得到：

$$
P_l\in\mathbb{R}^{B\times256\times H_l\times W_l},
\qquad (H_l,W_l)=\left(\frac{S}{s_l},\frac{S}{s_l}\right),
$$

其中：

$$
(s_0,s_1,s_2)=(8,16,32).
$$

对应实现：

- `engine/deim/hybrid_encoder.py:315-336`：通道投影与 Transformer encoder；
- `engine/deim/hybrid_encoder.py:395-433`：最高层注意力以及 FPN/PAN 融合。

**阶段属性：训练和推理都执行。** 训练时参数接收梯度；推理时只做前向计算。

### 3.2 拉平三层特征形成密集 memory

每一层从 `[B,C,H_l,W_l]` 转成 `[B,H_lW_l,C]`，然后沿空间维拼接：

$$
M=\operatorname{Concat}_{l=0}^{2}\left(
\operatorname{Flatten}(P_l)^\top
\right)
\in\mathbb{R}^{B\times N_S\times256},
$$

$$
N_S=\sum_{l=0}^{2}H_lW_l.
$$

本次多尺度推理中的实际数量为：

| 输入尺度 | 三层尺寸 | 密集位置数 $N_S$ | 最终选入 decoder |
|---:|---|---:|---:|
| 512 | `64×64, 32×32, 16×16` | 5,376 | 640 |
| 640 | `80×80, 40×40, 20×20` | 8,400 | 640 |
| 800 | `100×100, 50×50, 25×25` | 13,125 | 640 |

对应实现：`engine/deim/dfine_decoder.py:579-602`。

**阶段属性：训练和推理都执行。**

### 3.3 每个密集位置生成确定性参考框

对第 $l$ 层的第 $(i,j)$ 个位置，代码生成一个归一化 `cxcywh` 参考框：

$$
a_{l,i,j}=
\left[
\frac{j+0.5}{W_l},
\frac{i+0.5}{H_l},
g\,2^l,
g\,2^l
\right],
\qquad g=0.05.
$$

因此三层初始宽高分别为 `0.05、0.10、0.20`。随后转换到 inverse-sigmoid 空间：

$$
\widetilde a=\operatorname{logit}(a)
=\log\frac{a}{1-a}.
$$

只有满足每个分量都处于 $(\epsilon,1-\epsilon)$ 的参考框才有效，当前 $\epsilon=0.01$。无效位置的参考框被置为无穷，并用 mask 屏蔽 memory。

这里代码将其命名为 `anchors`，但它不同于传统 YOLO 的多比例/多宽高 anchor 模板：每个特征位置只有一个确定性方形 reference box，主要用于初始化 DETR query 的空间参考。

对应实现：`engine/deim/dfine_decoder.py:604-629`。

**阶段属性：训练和推理都执行。** 固定 640 推理时可预生成；本次动态尺度脚本将 `eval_spatial_size=None`，所以每个尺度会根据实际特征尺寸重新生成。

### 3.4 encoder 位置打分和初始框回归

密集 memory 先经过线性投影与 LayerNorm：

$$
E_i=\operatorname{LN}(W_eM_i+b_e).
$$

分类头对每个位置产生类别 logit：

$$
z_{i,c}=W_cE_i+b_c.
$$

当前只有一个类别，所以每个位置只有 $z_{i,0}$。与此同时，encoder box head 预测相对于参考框的残差：

$$
\widetilde b_i^{enc}=\operatorname{MLP}_{box}(E_i)+\widetilde a_i,
$$

$$
b_i^{enc}=\sigma(\widetilde b_i^{enc}).
$$

对应实现：

- `engine/deim/dfine_decoder.py:487-497`：encoder 分类头和框头；
- `engine/deim/dfine_decoder.py:647-663`：实际前向计算。

**阶段属性：训练和推理都执行。** 训练时 `enc_topk_logits` 和 `enc_topk_bboxes` 还会作为 encoder auxiliary outputs 参与损失；推理时只用于选择和初始化 query。

### 3.5 Top-640 查询选择与 query 初始化

默认查询选择策略为：

$$
s_i^{enc}=\max_c z_{i,c},
$$

$$
\mathcal I=\operatorname{TopK}\left(\{s_i^{enc}\}_{i=1}^{N_S},K=640\right).
$$

注意这里用的是 raw logit；对于单类别来说，对 raw logit 或 sigmoid 后概率排序结果相同。

代码默认 `learn_query_content=False`，因此 decoder query content 不是一套与图像无关的 learned embedding，而是选中位置的 encoder memory：

$$
Q^0=\operatorname{Detach}(E_{\mathcal I}).
$$

初始 reference box 为选中位置的 encoder 回归结果：

$$
R^0=\operatorname{Detach}
\left(
\sigma\left(
\operatorname{MLP}_{box}(E_{\mathcal I})+\operatorname{logit}(a_{\mathcal I})
\right)
\right).
$$

Top-K 索引选择本身不可微，且代码显式 `detach` 了 decoder 输入的 query content 和 reference box。因此 decoder 后续损失不会穿过这条初始化路径反传；encoder 选择/回归头主要由自己的辅助输出损失训练。

对应实现：`engine/deim/dfine_decoder.py:654-703`。

**阶段属性：训练和推理都执行。**

### 3.6 训练专用 contrastive denoising queries

训练时若 `num_denoising>0`，代码会从 GT 构造带标签噪声和框噪声的重建 queries，并拼在 640 个 matching queries 前面：

$$
Q_{decoder}=[Q_{DN};Q^0],\qquad
R_{decoder}=[R_{DN};R^0].
$$

设一个 batch 中最大 GT 数为 $G_{max}$，配置预算为 100，则组数和实际 DN query 数为：

$$
n_{group}=\max\left(1,\left\lfloor\frac{100}{G_{max}}\right\rfloor\right),
$$

$$
n_{DN}=2G_{max}n_{group}.
$$

其中每组包含正、负两套 noisy GT。attention mask 阻止正常 matching queries 读取 DN queries，也隔离不同 DN 组。

对应实现：`engine/deim/denoising.py:23-100`。

**阶段属性：仅训练执行。** 正常推理没有 DN queries，decoder 始终只处理 640 个 matching queries。

### 3.7 每层 D-FINE decoder 的注意力与门控更新

当前共有 4 层，索引为 0、1、2、3。第 $l$ 层首先从 reference box 生成 query positional embedding：

$$
p_q^l=\operatorname{MLP}_{pos}(R_q^{l-1}).
$$

#### Query self-attention

忽略多头符号，可写为：

$$
\operatorname{SA}(Q^l)=
\operatorname{Softmax}
\left(
\frac{(Q^l+p^l)W_Q((Q^l+p^l)W_K)^\top}{\sqrt{d_h}}
\right)
Q^lW_V.
$$

该步骤让 640 个 queries 相互交换信息，是抑制重复预测、建立集合关系的一部分。

#### Multi-Scale Deformable cross-attention

对 query $q$、注意力头 $m$、特征层 $l$ 和采样点 $k$，可概括为：

$$
\operatorname{MSDeformAttn}(q)=
\sum_m W_m
\left[
\sum_{l=0}^{2}\sum_{k=1}^{K_l}
A_{m,l,q,k}\,
P_l\left(\phi_l(R_q)+\Delta p_{m,l,q,k}\right)
\right],
$$

其中：

$$
(K_0,K_1,K_2)=(3,6,3),
\qquad \sum_lK_l=12
$$

是每个 head 在三层上的采样点数量；$A$ 经 softmax 归一化，$P_l(\cdot)$ 表示在特征图上的插值采样。

代码中的四维 reference box 情况使用：

$$
\operatorname{sampleLoc}
=R_{xy}
+\Delta p\cdot\frac{1}{K_l}\cdot R_{wh}\cdot0.5.
$$

#### Target gating 与 FFN

D-FINE 不直接将 cross-attention 结果普通残差相加，而是学习两个门：

$$
[g_1,g_2]=\sigma\left(W_g[Q,CA]+b_g\right),
$$

$$
Q'=\operatorname{LN}(g_1\odot Q+g_2\odot CA).
$$

之后经过 FFN：

$$
Q^{l+1}=\operatorname{LN}
\left(
Q'+W_2\,\rho(W_1Q')
\right).
$$

对应实现：

- `engine/deim/dfine_decoder.py:105-146`：MS-Deformable Attention；
- `engine/deim/dfine_decoder.py:198-225`：self-attention、cross-attention、gate、FFN；
- `engine/deim/dfine_decoder.py:230-243`：gate 的精确实现。

**阶段属性：训练和推理都执行。** 训练会保留各层辅助输出；部署推理仍执行 4 层，但只物化最后一层的检测输出。

### 3.8 D-FINE 的 Fine-grained Distribution Refinement（FDR）

“4 层反复细化”不能简单理解为每层直接预测一个 `Δcxcywh`。D-FINE 实际对四条边分别预测离散分布，并跨层累加 distribution logits。

第一层先根据 encoder reference box 得到 preliminary box：

$$
b^0=\sigma\left(
\operatorname{MLP}_{pre}(Q^1)+\operatorname{logit}(R^0)
\right).
$$

这个 $b^0=(x,y,w,h)$ 被 detach 后作为 FDR 的固定初始参考框。每层为左、上、右、下四条边各预测 $N+1=33$ 个 logits：

$$
L_e^l(n)=L_e^{l-1}(n)+\Delta L_e^l(n),
\qquad e\in\{left,top,right,bottom\},
$$

$$
P_e^l(n)=\operatorname{Softmax}_n(L_e^l(n)).
$$

这对应论文公式：

$$
\mathbf{Pr}^{l}(n)=
\operatorname{Softmax}
\left(
\Delta\operatorname{logits}^{l}(n)+
\operatorname{logits}^{l-1}(n)
\right).
$$

每条边的连续修正量为非均匀 bin 权重的期望：

$$
\delta_e^l=\sum_{n=0}^{32}W(n)P_e^l(n).
$$

当前代码设置 `up=0.5`、`reg_scale=4`。令：

$$
r=\left(|up|\,|reg\_scale|+1\right)^{\frac{2}{N-2}}
=3^{1/15},
$$

则代码实际构造的权重满足：

$$
W(0)=-4,\quad W(16)=0,\quad W(32)=4,
$$

中心附近间隔较细，两端变化较大。该非均匀函数使已经较准的框能够做细粒度修正，同时仍允许较大的边界移动。

对初始框 $b^0=(x,y,w,h)$，代码的 `distance2bbox` 精确解码式为：

$$
x_1=x-\left(\frac{s}{2}+\delta_{left}\right)\frac{w}{s},
$$

$$
y_1=y-\left(\frac{s}{2}+\delta_{top}\right)\frac{h}{s},
$$

$$
x_2=x+\left(\frac{s}{2}+\delta_{right}\right)\frac{w}{s},
$$

$$
y_2=y+\left(\frac{s}{2}+\delta_{bottom}\right)\frac{h}{s},
\qquad s=4.
$$

然后再转回 `cxcywh`。每层解出的框会 detach，成为下一层 deformable attention 的 reference box；但 FDR 的边分布 logits 按残差方式跨层累加。

最后，LQE（Localization Quality Estimator）从每条边分布中取概率最高的 4 个 bin 及其均值，经 MLP 产生质量修正量，并加到类别 logit：

$$
z_q^l=h_{cls}(Q_q^l)+h_{LQE}(P_q^l).
$$

对应实现：

- `engine/deim/dfine_utils.py:7-39`：非均匀 $W(n)$；
- `engine/deim/dfine_utils.py:88-113`：`distance2bbox`；
- `engine/deim/dfine_decoder.py:259-285`：Integral 与 LQE；
- `engine/deim/dfine_decoder.py:355-394`：4 层 FDR 细化主循环。

**阶段属性：FDR 解码在训练和推理都执行。** FGL/DDF 则是训练专用损失，见第 4 节。

### 3.9 模型原始输出

推理模式只返回：

$$
Z\in\mathbb{R}^{B\times640\times1},
\qquad
B\in\mathbb{R}^{B\times640\times4}.
$$

即 640 个 query 各自的一类 logit 和一个归一化 `cxcywh` 框。

训练模式还会返回：

- `pred_corners`：四条边的离散分布 logits；
- `ref_points`；
- 中间 decoder 层的 `aux_outputs`；
- encoder top-k 的 `enc_aux_outputs`；
- 第一层 preliminary head 的 `pre_outputs`；
- DN 分支输出。

对应实现：`engine/deim/dfine_decoder.py:754-771`。

**阶段属性：最后一层 boxes/logits 在训练和推理都有；其余辅助结构主要仅训练返回。**

### 3.10 仓库原生 PostProcessor：Top-K，但无 NMS

由于配置 `use_focal_loss=True`，原生后处理先计算：

$$
p_{q,c}=\sigma(z_{q,c}).
$$

随后对 query×class 拉平后取：

$$
\mathcal J=\operatorname{TopK}
\left(
\{p_{q,c}\},K=640
\right).
$$

当前 $Q=640,C=1$，总共恰好只有 640 个分数，所以 Top-640 保留全部结果。之后将框从 `cxcywh` 转为 `xyxy`：

$$
(x_1,y_1,x_2,y_2)
=\left(c_x-\frac{w}{2},c_y-\frac{h}{2},c_x+\frac{w}{2},c_y+\frac{h}{2}\right),
$$

再乘原图宽高：

$$
b_{pixel}=b_{norm}\odot(W,H,W,H).
$$

实现位于 `engine/deim/postprocessor.py:50-64`。该文件没有调用 NMS，也没有计算任意两框之间的 IoU。

**阶段属性：不参与训练损失。** 它用于验证/推理阶段把网络张量转换为检测结果。仓库原生 PostProcessor 本身也没有 `score > threshold` 过滤。

## 4. 训练链路：必须区分 D-FINE 与 DEIM 论文贡献

### 4.1 Hungarian 一对一匹配

对预测 $i$ 与 GT $j$，当前 matcher 使用：

$$
C_{ij}
=2C_{cls}(i,j)
+5\lVert b_i-g_j\rVert_1
+2C_{giou}(i,j),
$$

其中：

$$
C_{giou}(i,j)=-\operatorname{GIoU}(b_i,g_j).
$$

focal classification matching cost 为：

$$
C_{cls}
=\alpha(1-p)^\gamma(-\log p)
-(1-\alpha)p^\gamma(-\log(1-p)),
$$

当前 matcher 的 $\alpha=0.25,\gamma=2$。Hungarian 算法求：

$$
\hat\sigma
=\arg\min_{\sigma}
\sum_j C_{\sigma(j),j},
$$

从而在每次匹配内，每个 GT 最多对应一个 query，每个 query 最多对应一个 GT。

对应实现：`engine/deim/matcher.py:72-117`。

**阶段属性：仅训练。** 推理不会运行 Hungarian matching。

### 4.2 当前配置实际使用的训练损失

当前配置的总损失族为：

$$
\mathcal L
=1\mathcal L_{VFL}
+5\mathcal L_{L1}
+2\mathcal L_{GIoU}
+0.15\mathcal L_{FGL}
+1.5\mathcal L_{DDF},
$$

并对主输出、辅助 decoder 输出、encoder 输出、pre-output 与 DN 输出按代码规则重复计算。

#### VFL

对匹配正样本，以预测框与 GT 的 IoU $q$ 作为 soft target；对负样本 target 为 0。论文形式为：

$$
\operatorname{VFL}(p,q)=
\begin{cases}
-q\left[q\log p+(1-q)\log(1-p)\right], & q>0,\\
-\alpha p^\gamma\log(1-p), & q=0.
\end{cases}
$$

当前 criterion 的 $\alpha=0.75,\gamma=2$。实现位于 `engine/deim/deim_criterion.py:82-109`。

#### L1 与 GIoU

$$
\mathcal L_{L1}=\frac{1}{N_+}\sum_{(i,j)}\lVert b_i-g_j\rVert_1,
$$

$$
\mathcal L_{GIoU}=\frac{1}{N_+}\sum_{(i,j)}
\left(1-\operatorname{GIoU}(b_i,g_j)\right).
$$

实现位于 `engine/deim/deim_criterion.py:145-161`。

#### FGL

D-FINE 将 GT 相对边偏移 $\phi$ 定位到相邻的两个非均匀 bin $n_{\leftarrow},n_{\rightarrow}$，插值权重为：

$$
\omega_{\leftarrow}=
\frac{|\phi-W(n_{\rightarrow})|}
{|W(n_{\leftarrow})-W(n_{\rightarrow})|},
$$

$$
\omega_{\rightarrow}=
\frac{|\phi-W(n_{\leftarrow})|}
{|W(n_{\leftarrow})-W(n_{\rightarrow})|}.
$$

FGL 为：

$$
\mathcal L_{FGL}
=\sum_l\sum_k\operatorname{IoU}_k
\left[
\omega_{\leftarrow}CE(P_k^l,n_{\leftarrow})
+\omega_{\rightarrow}CE(P_k^l,n_{\rightarrow})
\right].
$$

#### GO-LSD 与 DDF

D-FINE 对各 decoder 层、pre-output 和 encoder output 分别做 Hungarian matching，再合并不同层匹配得到 localization union set。分类仍使用各层的一对一匹配；union set 主要用于 boxes/localization loss。

最终层边分布作为 teacher，浅层作为 student。论文中的 DDF 为：

$$
\mathcal L_{DDF}=T^2\sum_{l=1}^{L-1}
\left[
\sum_{k=1}^{K_m}\alpha_k KL(P_k^l,P_k^L)
+\sum_{k=1}^{K_u}\beta_k KL(P_k^l,P_k^L)
\right],
$$

$$
\alpha_k=\operatorname{IoU}_k
\frac{\sqrt{K_m}}{\sqrt{K_m}+\sqrt{K_u}},
\qquad
\beta_k=\operatorname{Conf}_k
\frac{\sqrt{K_u}}{\sqrt{K_m}+\sqrt{K_u}}.
$$

当前代码温度 $T=5$。实现位于：

- `engine/deim/deim_criterion.py:165-213`：FGL 与 DDF；
- `engine/deim/deim_criterion.py:229-248`：跨层匹配 union set；
- `engine/deim/deim_criterion.py:275-318`：各层匹配与损失使用方式。

以上全部是**训练专用**，推理时不存在 criterion、GT、Hungarian、FGL 或 DDF。

### 4.3 DEIM 论文的 Dense O2O 不是 D-FINE 的跨层 union set

这两个概念很容易混淆，但实际不同。

DEIM 论文将一般训练监督写为：

$$
\operatorname{loss}
=\sum_{i=1}^{N}\sum_{j=1}^{M_i}
f(\hat y_{ij},y_i).
$$

普通 O2O 中每个目标 $M_i=1$。DEIM 的 Dense O2O 仍保持 $M_i=1$，但通过 Mosaic、MixUp 等增强让单张训练合成图中的目标总数 $N$ 增加，从而在不改成 O2M 的情况下增加正样本监督。

D-FINE 的 GO-LSD union set 则是在同一批 targets 上聚合不同 decoder/encoder 层的匹配索引，用于定位优化与蒸馏。它不是 DEIM 论文所说的 Dense O2O 数据增强。

### 4.4 当前表格配置并未启用论文完整的 DEIM 训练方法

当前 table-cell 配置明确覆盖为：

```yaml
train_dataloader:
  dataset:
    transforms:
      ops:
        - RandomPhotometricDistort
        - RandomHorizontalFlip
        - Resize: [640, 640]
        - SanitizeBoundingBoxes
        - ConvertPILImage
        - ConvertBoxes
  collate_fn:
    mixup_prob: 0.0
```

其中没有 Mosaic，且 MixUp 概率为 0，因此按此配置**没有执行 DEIM 论文的 Dense O2O 增强**。

此外，它继承的是 `configs/base/dfine_hgnetv2.yml`，其中：

```yaml
losses: ['vfl', 'boxes', 'local']
weight_dict:
  loss_vfl: 1
  loss_bbox: 5
  loss_giou: 2
  loss_fgl: 0.15
  loss_ddf: 1.5
```

因此按该文件执行的是 **VFL，而不是 DEIM 论文的 MAL**。MAL 的论文公式为：

$$
\operatorname{MAL}(p,q,y)=
\begin{cases}
-q^\gamma\log p-(1-q^\gamma)\log(1-p), & y=1,\\
-p^\gamma\log(1-p), & y=0.
\end{cases}
$$

仓库确实实现了 MAL，也在 `configs/base/deim.yml` 中配置了 Mosaic、MixUp 和 `loss_mal`；但当前 table-cell 配置没有 include 那个文件。

严谨结论是：

> 当前可验证的模型结构是 HGNetv2-B2 + HybridEncoder + D-FINE Transformer，运行在 DEIM 仓库/`DEIM` wrapper 中；当前选中的配置是 D-FINE 的 VFL + FGL + GO-LSD/DDF 训练方案，不是论文完整的 Dense O2O + MAL 方案。

输出目录没有保存原始训练命令。如果训练 `last.pth` 时通过命令行 `-u` 临时覆盖过损失或增强，单靠推理产物无法反证；但当前用于重建模型的实际配置文件本身就是上述内容。

## 5. `_smoke_DEIM_multiscale_nms070` 是怎样产生的

### 5.1 产物与当前文件可以一一对应

输出中的 `benchmark_config.json` 记录了 SHA-256。重新核对当前文件后全部一致：

| 对象 | SHA-256 |
|---|---|
| `benchmark_deim_tta_cpu.py` | `353760CDD41B376A358A20EA8317A247E5042D9560B1A0654A3FC0AEE675AB8B` |
| table-cell 配置 | `9DB2845DEB9443A24AC75ED16DEDB3A1038FC9318FBD13784C16163AE3807F56` |
| 权重注册表 `path.txt` | `AFA5B793F4F7CD7906676360FBDFBFF067F39EF16C48565A4A554E0F5AA6F2B9` |
| `last.pth` | `81413C4638EB2CE76F175207AD01A844C6D88FDC62BC10D5E68EA49B2893FAAC` |

因此以下解释对应的就是生成该目录时的脚本和权重，而不是事后修改过的另一版本。

### 5.2 加载的权重和执行模式

脚本从注册表解析到：

```text
D:\Baseline_test\models\DEIM\outputs\last.pth
```

权重信息：

- checkpoint 的 `last_epoch=27`；
- 实际加载 `ema.module`；
- `strict=True`，不存在 missing/unexpected keys；
- 随后执行 `model.deploy().eval().cpu()`；
- 参数量 19,187,359；
- PyTorch eager、CPU、FP32、batch size 1；
- 单线程，CPU affinity 固定为逻辑 CPU 2；
- MKLDNN 关闭，确定性算法开启。

加载逻辑位于 `tools/benchmark_deim_cpu.py:279-344`。

### 5.3 这次是全图多尺度，不是分块推理

运行模式记录为：

```text
mode = multiscale
scales = [512, 640, 800]
groups = [wired]
```

因此每张原图只生成三个 full-image view：

```text
原图 → Resize(512×512)
原图 → Resize(640×640)
原图 → Resize(800×800)
```

没有 tile/crop。`benchmark_config.json` 中虽然仍记录了 tile size 和 overlap 默认参数，但在 `mode=multiscale` 分支中完全不会使用。对应代码为 `tools/benchmark_deim_tta_cpu.py:193-210`。

预处理为 Pillow RGB 图像经过：

$$
I_s=\operatorname{ToTensor}
\left(
\operatorname{Resize}_{bilinear}(I,(s,s))
\right),
\quad s\in\{512,640,800\}.
$$

`keep_ratio=false`，也就是每个尺度都会直接拉伸为方形，而不是 letterbox。

训练配置固定使用 640×640，但该脚本在加载/部署模型后显式设置：

```python
wrapper.detector.encoder.eval_spatial_size = None
wrapper.detector.decoder.eval_spatial_size = None
```

所以 512、640、800 三次前向会分别重新计算位置编码和参考框，而不是错误复用 640 的预计算网格。对应代码为 `tools/benchmark_deim_tta_cpu.py:532-538`。

### 5.4 每个尺度内部使用模型原生无 NMS 输出

每个尺度分别执行：

```python
output = wrapper(image_tensor, original_size)
```

其中 wrapper 等价于：

```python
PostProcessor(DEIM(image_tensor), original_size)
```

每次调用都会返回 640 个 `(label, box, score)`，所以一张原图三次前向共有：

$$
3\times640=1920
$$

个 raw query results。

这一步仍然没有 NMS。模型原生 PostProcessor 只做 sigmoid、Top-640、坐标转换和原图尺寸缩放。

### 5.5 自定义脚本先做严格分数阈值

每个尺度输出被转换到 NumPy 后执行：

```python
keep = scores_np > 0.5
```

注意是严格大于，不是大于等于。随后将框裁剪到原图边界，并删除宽或高不为正的框。对应代码为 `tools/benchmark_deim_cpu.py:347-383`。

因为本次都是 full-image view，不存在 tile offset，所以各尺度框直接位于同一原图坐标系。

### 5.6 额外执行的是“跨尺度 NMS”，不是模型原生 NMS

三次推理的阈值后候选框被合并，再执行脚本自定义的 deterministic class-wise cross-source greedy NMS。

两个框 $A,B$ 的 IoU 为：

$$
\operatorname{IoU}(A,B)=
\frac{|A\cap B|}{|A|+|B|-|A\cap B|}.
$$

算法过程为：

1. 按类别分组；当前只有 `cell` 一类。
2. 按 score 降序稳定排序。
3. 逐个检查候选框与已经保留的框。
4. **若两个框来自同一个 source view，则跳过比较，不抑制。**
5. 若来自不同 source view，且：

   $$
   \operatorname{IoU}(A,B)\ge0.70,
   $$

   则抑制当前分数较低的框。
6. 最终结果再次按 score 降序排列。

本次 source view 由 `(full, input_size, tile_index=None, crop=整图)` 标识，所以：

- 512 与 640 的重复框可以互相抑制；
- 512 与 800 的重复框可以互相抑制；
- 640 与 800 的重复框可以互相抑制；
- **同一个 512 前向内部的两个框永不互相抑制；同理 640、800 内部也不抑制。**

因此它不是通常意义上对全部候选框执行的标准全局 NMS，更准确的名字是：

> 同类别、跨尺度来源的贪心去重；保留每个单尺度 DETR 预测集合的内部结构。

对应实现：`tools/benchmark_deim_tta_cpu.py:299-356`。

这个 NMS 完全是**自定义推理/融合组件**：

- 不属于 DEIM/D-FINE 模型原生 PostProcessor；
- 不参与训练；
- 不改变 checkpoint；
- 只用于合并三个独立尺度的预测结果。

### 5.7 四张图的实际数量变化

本目录只处理了 `wired` 组的 4 张图，没有处理 `wireless`。

| 原图 | 原图尺寸 | 三尺度 raw query | `score>0.5` 后、NMS 前 | NMS 后 | 跨尺度抑制数 | 最终框来源（512 / 640 / 800） |
|---|---:|---:|---:|---:|---:|---:|
| `image (10).png` | 777×620 | 1,920 | 151 | 53 | 98 | 7 / 32 / 14 |
| `image (11).png` | 1111×1086 | 1,920 | 145 | 49 | 96 | 4 / 45 / 0 |
| `image (12).png` | 891×677 | 1,920 | 534 | 366 | 168 | 32 / 43 / 291 |
| `image (9).png` | 812×865 | 1,920 | 372 | 126 | 246 | 2 / 76 / 48 |

最终框被画到：

```text
D:\Baseline_test\output\_smoke_DEIM_multiscale_nms070\wired\*_vis.png
```

完整坐标、分数和来源尺度被写入 `detections.json`。

### 5.8 这里没有 `maxDets=300`

该可视化脚本没有在三尺度融合后再截断为 300 个框。因此 `image (12).png` 最终保留了 366 个框。

所以该目录与 COCO 评估中的：

```text
maxDets = 300
```

不是同一个后处理协议。它只是可视化/延迟实验，不能从这里直接推导 `mAP@50:95`，也不能把 366 个可视化框等价为 COCO evaluator 实际会使用的 300 个检测。

### 5.9 延迟数字代表三次前向之和

每张原图需要 3 次模型调用，记录的平均值为：

| 指标 | 平均值 |
|---|---:|
| 三次模型+原生 PostProcessor forward 总和 | 3028.774 ms / 原图 |
| 完整无磁盘 I/O pipeline | 3060.569 ms / 原图 |

测量样本数为 8，即 4 张原图 × 2 次 repeat。每个尺度只 warm-up 1 次。这是一个 smoke run，适合验证链路和相对行为，但 warm-up/repeat 数量不足以当作最终高置信度性能基准。

### 5.10 可等价复现的命令

原始命令行文本没有被写入产物，但 `benchmark_config.json` 已保存全部有效参数。以下命令在当前脚本下与记录参数等价：

```powershell
python "D:\Baseline_test\tools\benchmark_deim_tta_cpu.py" `
  --mode multiscale `
  --groups wired `
  --scales 512 640 800 `
  --threshold 0.5 `
  --merge-iou 0.70 `
  --warmup 1 `
  --repeats 2 `
  --threads 1 `
  --affinity 2 `
  --seed 20260929 `
  --output-dir "D:\Baseline_test\output\_smoke_DEIM_multiscale_nms070"
```

## 6. 最终应如何理解两条链路

### 单尺度、模型原生链路

```text
三层特征
→ 密集参考位置
→ encoder 位置分类/回归
→ Top-640 查询选择
→ 4 层 D-FINE/FDR decoder
→ 640 个 boxes/logits
→ sigmoid + Top-640 + 坐标映射
→ 无阈值、无 NMS
```

### `_smoke_DEIM_multiscale_nms070` 外部推理链路

```text
同一原图
├── 512×512 → 单尺度原生链路 → 640 个结果 → score>0.5
├── 640×640 → 单尺度原生链路 → 640 个结果 → score>0.5
└── 800×800 → 单尺度原生链路 → 640 个结果 → score>0.5

三尺度候选映射回同一原图坐标系
→ 同类别、仅跨尺度来源、IoU≥0.70 的贪心 NMS
→ 不做 maxDets=300 截断
→ 写 detections.json 并绘制 *_vis.png
```

最关键的边界是：

> **模型单次前向是 NMS-free；`multiscale_nms070` 为了融合三次彼此独立的前向结果，在模型外部额外加入了跨尺度 NMS。两者并不矛盾。**
