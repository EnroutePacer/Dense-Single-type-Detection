# SGQS v1：损失函数与总目标

本文给出 Scale-aware GT Query Survival（SGQS）v1 与当前实现完全一致的数学定义。SGQS 只在训练期监督 Encoder 全局 Top-K 之前的候选；不新增可学习参数，不改变 Decoder query 数、Matcher 或 PostProcessor。

## 1. 记号与候选排序

第 \(b\) 张完成全部数据增强后的输入大小为 \(H_b\times W_b\)。特征层集合为

\[
\mathcal L=\{3,4,5\},\qquad s_l\in\{8,16,32\},
\]

其中第 \(l\) 层大小为 \(H_l\times W_l\)，展平偏移为

\[
O_l=\sum_{j<l}H_jW_j.
\]

候选位置 \(i=(l,r,c)\) 的展平索引和归一化中心为

\[
\operatorname{idx}(i)=O_l+rW_l+c,\qquad
\pi_i=\left(\frac{c+0.5}{W_l},\frac{r+0.5}{H_l}\right).
\]

\(\mathcal C_b\) 表示全部 Encoder 候选，\(\mathcal V_b\subseteq\mathcal C_b\) 表示 anchor 有效的候选。分类头输出 logit \(z_{b,i,k}\)，单类别任务简写为 \(z_{b,i}\)。

实际 Top-K 排序分数与 DEIM selector 一致：

\[
s_{b,i}=
\begin{cases}
\max_k z_{b,i,k}, & \text{default},\\
z_{b,i,1}, & \text{agnostic},\\
z_{b,i,k}\text{ 展平后的元素}, & \text{one2many}.
\end{cases}
\]

无效 anchor 的排序分数被置为 \(-\infty\)，因此不会消耗有限的 query 预算。这个修复同时作用于 control 与 SGQS，不能计入 SGQS 方法增益。

## 2. 尺度感知的唯一 owner

GT \(g=(c_x^g,c_y^g,w_g,h_g)\) 的像素短边为

\[
d_g=\min(w_gW_b,\ h_gH_b).
\]

为 P3/P4/P5 设置像素尺度先验 \(a_l\)。固定 640 的 v1 配置为

\[
(a_3,a_4,a_5)=(32,64,128).
\]

尺度距离和首选层定义为

\[
\Delta_{g,l}=
\left|\log\frac{d_g+\varepsilon}{a_l+\varepsilon}\right|,
\qquad
l_g^\star=\arg\min_{l\in\mathcal L}\Delta_{g,l}.
\]

距离相同时优先选择 stride 更小的层。GT 在层 \(l\) 上的连续网格坐标为

\[
\xi_{g,l}=c_x^gW_l-0.5,\qquad
\eta_{g,l}=c_y^gH_l-0.5,
\]

候选中心距离为

\[
D(g,i)=(c_i-\xi_{g,l_i})^2+(r_i-\eta_{g,l_i})^2.
\]

GT 按 \((d_g,\text{GT index})\) 从小到大处理。对每个 GT，先在首选层半径 \(R=1\) 的邻域内寻找距离最近、有效且未占用的候选；冲突时逐步扩大到 \(R_{\max}=4\)，再按 \(\Delta_{g,l}\) 从小到大尝试其他层。设已占用集合为 \(\mathcal O_b\)，则

\[
o_g=
\arg\min_{i\in\mathcal V_b\setminus\mathcal O_b}D(g,i),
\qquad
\mathcal O_b\leftarrow\mathcal O_b\cup\{o_g\}.
\]

成功分配的 GT 集合记为 \(\mathcal A_b\)。无法分配的 GT 记入 assignment miss，不参与 owner 正点与回归损失。

Owner 周围同层 Chebyshev 半径 \(r_{\mathrm{ign}}=1\) 的有效位置组成 ignore ring：

\[
\mathcal I_b=
\left(
\bigcup_{g\in\mathcal A_b}
\left\{
i\in\mathcal V_b:
l_i=l_{o_g},\
\left\|(r_i,c_i)-(r_{o_g},c_{o_g})\right\|_\infty
\le r_{\mathrm{ign}}
\right\}
\right)\setminus\mathcal O_b.
\]

Owner 的优先级高于 ignore。

## 3. Owner 框与质量目标

SGQS 只 gather owner 位置的 memory，并复用原有 Encoder 框头：

\[
\hat b_g=
\sigma\!\left(h_{\mathrm{box}}(m_{o_g})+a_{o_g}^{\mathrm{unact}}\right).
\]

定位质量不向框分支反向传播：

\[
u_g=
\operatorname{stopgrad}
\left[
\operatorname{IoU}(\hat b_g,g)
\right].
\]

为避免训练早期细长框 IoU 很低而使分类正梯度消失，正样本质量目标为

\[
q_g=q_{\min}+(1-q_{\min})u_g,\qquad q_{\min}=0.25.
\]

## 4. 分层 dense 分类损失

第 \(l\) 层 owner 正点集合为

\[
\mathcal P_l=
\{(o_g,y_g):g\in\mathcal A,\ l_{o_g}=l\}.
\]

有效负样本集合为

\[
\mathcal N_l=
\left\{(i,k):i\in\mathcal V_l\setminus\mathcal I_l\right\}
\setminus\mathcal P_l.
\]

Owner 正点采用质量感知 BCE：

\[
\ell_{g}^{+}
=
q_g\,
\operatorname{BCEWithLogits}(z_{o_g,y_g},q_g).
\]

负候选采用 VFL/Focal 形式：

\[
\ell_{i,k}^{-}
=
\alpha\,\sigma(z_{i,k})^\gamma
\operatorname{BCEWithLogits}(z_{i,k},0),
\]

其中

\[
\alpha=0.75,\qquad \gamma=2,\qquad \beta_{\mathrm{neg}}=1.
\]

每个特征层分别归一化：

\[
L_l^{+}
=
\frac{\sum_{(o_g,y_g)\in\mathcal P_l}\ell_g^{+}}
{\max(1,|\mathcal P_l|)},
\qquad
L_l^{-}
=
\frac{\sum_{(i,k)\in\mathcal N_l}\ell_{i,k}^{-}}
{\max(1,|\mathcal N_l|)}.
\]

最终 dense 分类损失为

\[
\boxed{
L_{\mathrm{dense}}
=
\frac{1}{|\mathcal L|}
\sum_{l\in\mathcal L}
\left(
L_l^{+}+\beta_{\mathrm{neg}}L_l^{-}
\right)
}.
\]

分层平均避免 P3 仅因候选数量更多而主导梯度。v1 使用全部有效负样本，不使用 OHEM。

## 5. Owner 回归损失

成功分配的 owner 数量为

\[
N_A=\sum_b|\mathcal A_b|.
\]

L1 与 GIoU 损失分别为

\[
\boxed{
L_{\mathrm{point\text{-}L1}}
=
\frac{1}{\max(1,N_A)}
\sum_b\sum_{g\in\mathcal A_b}
\|\hat b_g-g\|_1
}
\]

和

\[
\boxed{
L_{\mathrm{point\text{-}GIoU}}
=
\frac{1}{\max(1,N_A)}
\sum_b\sum_{g\in\mathcal A_b}
\left[
1-\operatorname{GIoU}(\hat b_g,g)
\right]
}.
\]

沿用 DEIM 框损失的 \(5:2\) 比例：

\[
\boxed{
L_{\mathrm{point\text{-}reg}}
=
5L_{\mathrm{point\text{-}L1}}
+2L_{\mathrm{point\text{-}GIoU}}
}.
\]

SGQS 首轮不得叠加 historical small-object reweighting，否则无法区分增益来自 owner 机制还是普通重加权。

## 6. Top-K 生存损失

令排序池大小为 \(M_b\)，有效预算为

\[
K_b=\min(K,M_b).
\]

第 \(K_b\) 大排序分数构成当前 Top-K 截止线：

\[
\theta_b=
\operatorname{stopgrad}
\left[
\operatorname{kthlargest}
\left(\{s_{b,i}\}_{i\in\mathcal C_b},K_b\right)
\right].
\]

对 owner 使用平滑 margin：

\[
\ell_{\mathrm{surv}}(g)
=
\tau\log\left[
1+\exp\left(
\frac{\theta_b+m-z_{o_g,y_g}}{\tau}
\right)
\right],
\]

其中

\[
m=0.2,\qquad \tau=0.1.
\]

为让几何质量较高的 owner 获得更大相对权重，同时确保 \(u_g=0\) 的弱小框仍有生存梯度，使用带下限的相对质量权重：

\[
\omega_g=\chi_bq_g^\eta,\qquad \eta=1,
\]

\[
\chi_b=
\mathbf 1[0<G_b\le K_b<M_b].
\]

未乘 ramp 的生存损失为

\[
\boxed{
L_{\mathrm{survive}}
=
\begin{cases}
\dfrac{
\sum_b\sum_{g\in\mathcal A_b}
\omega_g\ell_{\mathrm{surv}}(g)
}{
\sum_b\sum_{g\in\mathcal A_b}\omega_g
},
&
\sum\omega_g>0,\\[10pt]
0,&\text{otherwise}.
\end{cases}
}
\]

当 \(G_b>K_b\) 时，所有 GT 同时进入 Top-K 在数学上不可能，因此该图继续计算 dense 与 owner regression，但不计算 survival。

令训练进度为

\[
t=e+\frac{s}{S_e},
\]

其中 \(e\) 为 epoch，\(s\) 为 epoch 内 step，\(S_e\) 为每个 epoch 的 step 数。生存项 ramp 为

\[
\boxed{
r(t)=
\operatorname{clip}
\left(
\frac{t-t_0}{T_r},0,1
\right)
}.
\]

从 epoch-27 checkpoint 继续的 probe 配置取 \(t_0=28,\ T_r=1\)。

## 7. 总损失

原有 DEIM/D-FINE criterion 完全保留，记为

\[
L_{\mathrm{base}}
=
L_{\mathrm{final}}
+L_{\mathrm{decoder\text{-}aux}}
+L_{\mathrm{pre}}
+L_{\mathrm{encoder\text{-}TopK}}
+L_{\mathrm{DN}}.
\]

SGQS 不替换 Matcher，不把新增损失复制到 Decoder auxiliary 层。总损失为

\[
\boxed{
L_{\mathrm{total}}
=
L_{\mathrm{base}}
+\lambda_dL_{\mathrm{dense}}
+\lambda_rL_{\mathrm{point\text{-}reg}}
+\lambda_sr(t)L_{\mathrm{survive}}
}.
\]

v1 起始权重为

\[
\lambda_d=1,\qquad
\lambda_r=0.5,\qquad
\lambda_s=0.25.
\]

展开成代码实际求和的四项：

\[
\boxed{
L_{\mathrm{total}}
=
L_{\mathrm{base}}
+1.0L_{\mathrm{dense}}
+2.5L_{\mathrm{point\text{-}L1}}
+1.0L_{\mathrm{point\text{-}GIoU}}
+0.25r(t)L_{\mathrm{survive}}
}.
\]

对应损失键为：

- loss_sgqs_dense
- loss_sgqs_bbox
- loss_sgqs_giou
- loss_sgqs_survival

## 8. DDP 归一化

设分布式 world size 为 \(R\)，第 \(r\) 个 rank 上某项计数为 \(n^{(r)}\)。实现使用 DEIM/DETR 的平均 rank 分母：

\[
\bar n=
\max\left(
\varepsilon,\frac{1}{R}\sum_{r=1}^{R}n^{(r)}
\right).
\]

每个 rank 用本地 loss sum 除以 \(\bar n\)，随后 DDP 对梯度求平均；只要全局计数非零，所得梯度就等价于用全局总和除以全局计数。这里不能把平均 rank 分母截断到 1：例如 8 卡全局只有 1 个 P5 owner 时，正确分母是 \(1/8\)，否则该正样本梯度会被额外缩小 8 倍。dense 正负数和 owner 数都使用该规则；全局计数为 0 时分子也为 0。

survival 的连续质量权重使用

\[
\bar W=
\max\left(
\varepsilon,
\frac{1}{R}\sum_{r=1}^{R}\sum_g\omega_g^{(r)}
\right),
\]

避免把小但非零的质量权重错误截断到 1。

## 9. 非损失诊断指标

以下量会单独写入训练日志和 TensorBoard 的 Metric 命名空间，绝不会进入 \(L_{\mathrm{total}}\)：

\[
\mathrm{miss\_rate}
=
\frac{\sum_b\mathrm{assignment\_miss}_b}
{\max(1,\sum_bG_b)},
\]

\[
\mathrm{owner\_coverage}
=
\frac{\sum_b|\mathcal A_b|}
{\max(1,\sum_bG_b)},
\qquad
\mathrm{owner\_count}
=
\frac{\sum_b|\mathcal A_b|}
{\max(1,B)},
\]

\[
\mathrm{budget\_overflow\_rate}
=
\frac{
\sum_b\mathbf 1[G_b>K_b]
}{
\max(1,\sum_b\mathbf 1[G_b>0])
}.
\]

此外记录 owner mean IoU、真实 pre-Top-K survival rate、owner logit 与 cutoff 的均值差，以及 P3/P4/P5 owner 数。日志键统一使用 sgqs_ 前缀。

## 10. 边界条件与梯度路径

- \(G_b=0\)：owner regression 与 survival 为 0；dense loss 仍训练有效背景候选。
- 某层无正点或无负点：该层对应均值项定义为 0。
- \(G_b>K_b\)：只关闭该图的 survival，不关闭 dense 或回归。
- owner 分配、\(u_g\)、\(q_g\) 和 \(\theta_b\) 均停止梯度。
- \(L_{\mathrm{dense}}\) 与 \(L_{\mathrm{survive}}\) 的梯度进入 enc_score_head、owner memory、HybridEncoder 与 backbone。
- \(L_{\mathrm{point\text{-}L1}}\) 与 \(L_{\mathrm{point\text{-}GIoU}}\) 的梯度进入原 enc_bbox_head、owner memory、HybridEncoder 与 backbone。
- GT 只在训练期生成监督；推理不需要 GT，SGQS supervisor 无参数且 eval 分支不执行。

## 11. 首轮可证伪实验

首轮唯一合法对照是 clean fixed-640 VFL control。初始化 checkpoint、数据顺序、global batch、optimizer steps、增强、query 数 \(K=640\)、seed 与 maxDets=300 必须完全相同。由于当前代码另含 invalid-anchor selector 修复，control 与 SGQS 必须都用当前同一 commit 重跑；不得直接拿历史 0.864248 作为 SGQS 对照。

依次比较：

1. baseline；
2. 仅 \(L_{\mathrm{dense}}\)；
3. \(L_{\mathrm{dense}}+L_{\mathrm{point\text{-}reg}}\)；
4. 完整 SGQS；
5. 将短边分层替换为面积分层。

先看逐 GT 的 P3 入选覆盖、owner rank 和 Top-K survival，再看同一步数的 mAP、AP75、AR 与短边分桶指标。论文主表保持 maxDets=300；另报告 maxDets=640 或 evaluator 截断前的逐 GT coverage，避免最密图中超过 300 个 GT 的机制收益被主评估上限隐藏，但该辅助口径不能替代主表。若机制指标不改善，应否定当前损失设计；若机制指标改善而 AP 不改善，瓶颈应继续定位到 Decoder 或后处理，而不能宣称 SGQS 已有效。
