# M-002 — SGQS-E：候选生存与逐边定位

**状态：** `designed / unimplemented / unvalidated`  
**目标：** 在固定输入、固定 query 数和固定推理图下，同时解决两个训练层问题：

1. 小框 Encoder 候选分数偏低，经 global Top-K 后缺少进入 Decoder 的种子；
2. 密集同类目标只用整框 IoU 给四条边共享权重，无法按边表达短边尺度、相对误差和邻域歧义。

SGQS-E 只保留两个与问题一一对应的模块：

- **SGQS**：在 Top-K 之前监督每个 GT 的唯一 owner 候选及其生存排名；
- **EFGL**：把原 FGL 的整框共享权重替换为短边、逐边误差和同类邻域共同决定的有界逐边权重。

```text
增广后 GT ──> 唯一 owner ──> dense / owner box / survival ──> 原 global Top-K ──> 原 Decoder
      └──────> 同类近邻图 ────────────────────────────────> EFGL 逐边定位监督
```

首版不修改 DN 采样、Matcher、query 数、预测头或后处理，也不增加新的推理分支。附件总结中的“邻居约束 DN”保留为后续候选；它不并入首版，以免同时改变 Top-K、FGL 和 DN 三条链而无法归因。

## 1. 记号与在线 GT owner

第 \(b\) 张图在**全部训练增强之后**的尺寸为 \(H_b\times W_b\)。归一化 cxcywh GT 记为 \(\bar b_g\)，\(T(\cdot)\) 表示从 cxcywh 到 xyxy 的转换。Encoder 在 P3/P4/P5 上产生候选 \(i\in\mathcal C_b\)，其原始单类别 logit 为 \(z_{b,i}\)，global Top-K 预算为 \(K\)。

每个 GT \(g\) 按增广后的像素短边分配到合适层，并在 GT 中心附近取得唯一有效位置 \(o_g\)。冲突时小短边 GT 优先，再扩大局部半径或回退到其他层。owner 必须在线生成，不能离线预画后直接复用，否则 resize、crop、mosaic 等增强后坐标会失配。具体分层与冲突规则沿用 [M-001 SGQS](../M-001_sgqs/losses.md)。

owner 框复用原 Encoder 框头：

\[
\hat{\bar b}_g^{o}=\sigma\!\left(h_{\mathrm{box}}(m_{o_g})+a_{o_g}^{\mathrm{unact}}\right).
\]

定义停止梯度的 owner 质量与带下限软目标：

\[
u_g=\operatorname{sg}\!\left[
\operatorname{IoU}\!\left(T(\hat{\bar b}_g^{o}),T(\bar b_g)\right)
\right],
\qquad
q_g=q_{\min}+(1-q_{\min})u_g,
\qquad q_{\min}=0.25.
\]

## 2. SGQS：Top-K 前候选生存监督

### 2.1 分层 dense 分类

令 \(\mathcal P_l\) 为第 \(l\) 层 owner 正点，\(\mathcal N_l\) 为去掉 owner 邻域 ignore ring 后的有效负点。记 \(x\vee1=\max(x,1)\)，则

\[
L_{\mathrm{dense}}
=\frac1{|\mathcal L|}\sum_{l\in\mathcal L}
\left[
\frac{\sum_{g:o_g\in\mathcal P_l}q_g\,\operatorname{BCEWithLogits}(z_{o_g},q_g)}{|\mathcal P_l|\vee1}
+\frac{\sum_{i\in\mathcal N_l}\alpha\,\sigma(z_i)^\gamma\operatorname{BCEWithLogits}(z_i,0)}{|\mathcal N_l|\vee1}
\right],
\]

其中 \(\alpha=0.75,\gamma=2\)。逐层平均防止候选数最多的 P3 仅因数量主导梯度。

### 2.2 owner 框回归

令 \(N_A=\sum_b|\mathcal A_b|\)。若全局没有 owner，两项均定义为 0；非空时使用第 6 节的 DDP 分母。

\[
L_{\mathrm{owner\text{-}L1}}
=\frac1{N_A\vee1}\sum_b\sum_{g\in\mathcal A_b}
\|\hat{\bar b}_g^{o}-\bar b_g\|_1,
\]

\[
L_{\mathrm{owner\text{-}GIoU}}
=\frac1{N_A\vee1}\sum_b\sum_{g\in\mathcal A_b}
\left[1-\operatorname{GIoU}\!\left(T(\hat{\bar b}_g^{o}),T(\bar b_g)\right)\right],
\]

\[
L_{\mathrm{owner}}
=5L_{\mathrm{owner\text{-}L1}}+2L_{\mathrm{owner\text{-}GIoU}}.
\]

这条梯度直接进入原 `enc_bbox_head`、Encoder memory、HybridEncoder 和 backbone，不等待候选先进入 Top-K。

### 2.3 相对 Top-K 截止线的生存损失

令 \(\mathcal V_b\subseteq\mathcal C_b\) 为 invalid-anchor mask 后的有效元素，\(M_b=|\mathcal C_b|\)，\(K_b=\min(K,M_b)\)。真实 selector 分数记为

\[
s_{b,i}=
\begin{cases}
z_{b,i}, & i\in\mathcal V_b,\\
-\infty, & i\notin\mathcal V_b.
\end{cases}
\]

dense 分类使用原始有效 logit \(z\)；生存门槛必须使用与真实 selector 完全相同的 mask 后分数 \(s\)。实现必须断言 \(K_b\le|\mathcal V_b|\)，防止无效位置以 \(-\infty\) 占用预算：

\[
\theta_b=\operatorname{sg}
\left[\operatorname{kthlargest}_{i\in\mathcal C_b}(s_{b,i},K_b)\right].
\]

令

\[
\chi_b=\mathbf 1[0<G_b\le K_b<M_b].
\]

对成功分配的 owner 集合 \(\mathcal A_b\)，令

\[
W=\sum_b\sum_{g\in\mathcal A_b}\chi_bq_g^\eta,
\]

\[
L_{\mathrm{survive}}
=
\begin{cases}
\dfrac{
\sum_b\sum_{g\in\mathcal A_b}\chi_bq_g^\eta\,
\tau\log\!\left(1+\exp\dfrac{\theta_b+m-z_{o_g}}{\tau}\right)
}{W},
&W>0,\\[12pt]
0,&W=0.
\end{cases}
\]

其中 \(m=0.2,\tau=0.1,\eta=1\)。它要求 owner 相对**本图当前**第 \(K_b\) 名取得 margin，而不是设置不可迁移的绝对置信度阈值。若 \(G_b>K_b\)，该图只计算 dense 与 owner 回归，不计算 survival。

## 3. EFGL：邻域与短边感知的逐边 FGL

当前 `engine/deim/deim_criterion.py::loss_local` 先计算匹配框 IoU，再把同一个 IoU 复制给左、上、右、下四条边。clean 基线可写成

\[
L_{\mathrm{FGL}}
=\frac1N\sum_g I_g\sum_{e\in\{l,t,r,b\}}
\operatorname{CE}_{\mathrm{interp}}(p_{g,e},y_{g,e}),
\]

其中 \(\hat{\bar b}_g^d\) 是匹配后的归一化 cxcywh Decoder 预测，

\[
I_g=\operatorname{IoU}\!\left(T(\hat{\bar b}_g^d),T(\bar b_g)\right),
\]

\(\operatorname{CE}_{\mathrm{interp}}\) 是 D-FINE 原有相邻 bin 插值交叉熵，\(N\) 是原 criterion 的跨卡匹配框归一化数；全局无匹配框时该项为 0。M-002 首版关闭历史 `small_object_local_weight`，只替换这个权重，避免重复加权。

### 3.1 短边与逐边相对误差

令 \(P_b(\cdot)\) 把归一化 xyxy 的横坐标乘 \(W_b\)、纵坐标乘 \(H_b\)。EFGL 的几何量统一使用增广后实际输入的像素 xyxy：

\[
\tilde b_g=P_b(T(\bar b_g)),\qquad
\hat{\tilde b}_g=P_b(T(\hat{\bar b}_g^d)),
\]

\[
\tilde b_g=[x_g^l,y_g^t,x_g^r,y_g^b],\qquad
d_g=[\tilde w_g,\tilde h_g,\tilde w_g,\tilde h_g],\qquad
s_g=\min(\tilde w_g,\tilde h_g),
\]

\[
r_{g,e}=
\frac{|\hat{\tilde b}_{g,e}-\tilde b_{g,e}|}
{\max(d_{g,e},\epsilon_{\mathrm{pix}})},
\qquad
a(s_g)=e^{-s_g/s_0},
\qquad
h(r)=\frac{r}{1+r}.
\]

建议初值为 \(s_0=16\) px、\(\epsilon_{\mathrm{pix}}=2\) px。这样同为 2 px 的误差，在 10 px 短边上会比在 100 px 长边上得到更强补偿。

### 3.2 同类邻域歧义

对任意框 \(u,v\)，定义

\[
D_g(u,v)=
\sum_{e\in\{l,t,r,b\}}
\frac{|u_e-v_e|}{\max(d_{g,e},\epsilon_{\mathrm{pix}})}.
\]

\[
\delta_{gj}=D_g(\tilde b_g,\tilde b_j),
\]

每个 GT 从满足 \(0<\delta_{gj}\le\delta_{\max}\) 的同类 GT 中保留距离最近的 4 个，首轮取 \(\delta_{\max}=4\)。对邻居 \(j\in\mathcal N(g)\)：

\[
\Delta_{gj}(\hat{\tilde b}_g)
=D_g(\hat{\tilde b}_g,\tilde b_j)-D_g(\hat{\tilde b}_g,\tilde b_g).
\]

当预测落在自身 GT 上时 \(\Delta_{gj}=\delta_{gj}\)；接近两个实例的几何分界时 \(\Delta_{gj}\approx0\)。定义有界歧义：

\[
U_{gj}=
\operatorname{clip}_{[0,1]}
\left(
\frac{\kappa(t)\delta_{gj}-\Delta_{gj}(\hat{\tilde b}_g)}
{\kappa(t)\delta_{gj}+\epsilon_0}
\right).
\]

再把歧义分配给真正负责区分两个实例的边：

\[
v_{gj,e}=
\frac{
|\tilde b_{g,e}-\tilde b_{j,e}|/\max(d_{g,e},\epsilon_{\mathrm{pix}})
}{\delta_{gj}+\epsilon_0},
\qquad
U_{g,e}=\max_{j\in\mathcal N(g)}v_{gj,e}U_{gj}.
\]

例如两个上下相邻且左右边对齐的细长框，邻域项主要加强上、下边，不会无差别推动左右边。\(\delta_{gj}\le\epsilon_0\) 的重复框配对跳过；没有有效邻居时 \(U_{g,e}=0\)。这里的 \(\epsilon_0=10^{-6}\) 是无量纲数值稳定项，不能与像素单位的 \(\epsilon_{\mathrm{pix}}\) 混用。

### 3.3 有界逐边权重

最终权重为

\[
\boxed{
\omega_{g,e}
=\operatorname{sg}\!\left[
I_g+
\lambda c(t)a(s_g)(1-I_g)
\min\{1,h(r_{g,e})+\beta U_{g,e}\}
\right]
}.
\]

因此

\[
\boxed{
L_{\mathrm{EFGL}}
=\frac1N\sum_g\sum_e
\omega_{g,e}\,
\operatorname{CE}_{\mathrm{interp}}(p_{g,e},y_{g,e})
}.
\]

当 \(0\le\lambda\le1\) 时，

\[
I_g\le\omega_{g,e}\le1.
\]

所以 EFGL 只在训练早期补回低 IoU 小框的逐边分布监督，不会产生无上限的 \(1/\text{area}\) 放大。权重整体停止梯度，模型不能通过操纵权重本身降低损失。

EFGL 替换普通匹配与正 DN 分支中的直接 FGL 权重，但不改变 DN 样本构造；DDF/教师 KL 保持原样。普通分支与 DN 分支分别开关是机制消融，不是首版新增模块。

## 4. 总损失

令 \(L_{\mathrm{base}}^{\mathrm{FGL}\rightarrow\mathrm{EFGL}}\) 表示保留 DEIM/D-FINE 全部原损失，只把原 FGL 换成 EFGL，则

\[
\boxed{
L_{\mathrm{total}}
=L_{\mathrm{base}}^{\mathrm{FGL}\rightarrow\mathrm{EFGL}}
+L_{\mathrm{dense}}
+0.5L_{\mathrm{owner}}
+0.25r_s(t)L_{\mathrm{survive}}
}.
\]

展开为实际四项 SGQS 权重：

\[
\boxed{
L_{\mathrm{total}}
=L_{\mathrm{base}}^{\mathrm{FGL}\rightarrow\mathrm{EFGL}}
+1.0L_{\mathrm{dense}}
+2.5L_{\mathrm{owner\text{-}L1}}
+1.0L_{\mathrm{owner\text{-}GIoU}}
+0.25r_s(t)L_{\mathrm{survive}}
}.
\]

这里 EFGL 是**替换项**，不是在原 FGL 之外再叠加一份 CE。

## 5. 首轮课程与初值

设总训练进度 \(p=t/T\in[0,1]\)：

\[
r_s(t)=\operatorname{clip}\!\left(\frac{t}{T_{1\mathrm{ep}}},0,1\right),
\]

\[
c(p)=\max\left(0,1-\frac{p}{0.30}\right),
\qquad
\kappa(p)=0.5-0.4\min\left(\frac{p}{0.30},1\right).
\]

- dense 与 owner 回归从 step 0 完整启用；
- survival 用 1 epoch 线性升温；
- EFGL 补偿只在前 30% 训练存在，随后 \(c=0\)，精确退回原 IoU-FGL；
- 初值：\(\lambda=0.5,\beta=0.5,s_0=16\) px、\(\epsilon_{\mathrm{pix}}=2\) px、\(\epsilon_0=10^{-6}\)、\(\delta_{\max}=4\)、最多 4 个邻居；
- 首轮不叠加历史 small-reg/small-VFL，也不同时搜索多组超参数。

## 6. 梯度与实现边界

所有按样本计数的均值沿用 M-001/DETR 的 DDP 归一化。world size 为 \(R\)、第 \(r\) 个 rank 的计数为 \(n^{(r)}\) 时，

\[
\bar n=\max\!\left(\epsilon_0,\frac1R\sum_{r=1}^{R}n^{(r)}\right).
\]

各 rank 用本地 loss sum 除以 \(\bar n\)，再由 DDP 平均梯度；全局计数为 0 时损失定义为 0。survival 的 \(W\) 也按同样方式跨卡汇总连续权重，不能把小于 1 的非零权重截断到 1。

- owner 分配、邻域索引、\(q_g\)、\(\theta_b\) 与 \(\omega_{g,e}\) 全部停止梯度；
- SGQS 分类梯度进入 `enc_score_head`，owner 回归进入原 `enc_bbox_head`；两者继续传到 Encoder 与 backbone；
- EFGL 只改变原 FGL 每条边 CE 的系数，目标 bin、FDR、L1、GIoU、MAL/VFL、GO-LSD 与 Matcher 不变；
- 实现时只在原 `loss_fgl` 内替换 `weight_targets`，保留原外层 loss 系数和 `num_boxes` 归一化；禁止先计算原 FGL 再额外相加 EFGL，survival ramp 也只能乘一次；
- 不新增参数、head、query 或推理后处理；所谓“推理零新增计算”仍须导出与固定 CPU latency 实测；
- 当前 M-001 代码保持原样。M-002 是新设计，尚未实现训练代码，不能把 M-001 的单测或任何历史 AP 当作 M-002 证据。

## 7. 最小实验矩阵

主消融只跑四组：

| ID | 训练改动 | 回答的问题 |
|---|---|---|
| B0 | clean baseline | 同提交基线 |
| B1 | 仅 SGQS | Top-K 前监督是否有效 |
| B2 | 仅 EFGL | 逐边邻域补偿是否有效 |
| B3 | SGQS + EFGL | 两个阶段是否互补 |

若 B2 有正信号，再做：统一 IoU 下限、四边共享权重、\(\beta=0\)、\(c(t)=1\) 和历史面积 small-reg。若 B1 有正信号，再去掉 survival，判断收益来自 dense/owner 还是 cutoff margin。

所有比较固定初始化、输入 640、\(K=640\)、增强、global batch、optimizer steps、seed、Evaluator 和 `maxDets=300`；正式结果至少 3 seeds。必须同时报告：

- **候选链：** owner rank、pre-Top-K survival、unique-GT coverage、重复占位率、短边分桶；
- **回归链：** AP75/AP90、四边相对误差、短边分桶、密集/稀疏 × 小/大交叉组；
- **最终指标：** 相同 epoch/steps 的 mAP、AP-small/AR-small、中大框退化、wall-clock 与 GPU-hours。

只有 B3 在相同训练预算下稳定超过 B0、B1、B2，且两条机制指标分别改善，才能称两个阶段“互补”。若要进一步声称正交互，必须使

\[
\Delta_{\mathrm{int}}
=(B3-B0)-(B1-B0)-(B2-B0)>0
\]

并跨 seed 给出不确定性。当前只有设计和代码事实，没有训练收益证据。
