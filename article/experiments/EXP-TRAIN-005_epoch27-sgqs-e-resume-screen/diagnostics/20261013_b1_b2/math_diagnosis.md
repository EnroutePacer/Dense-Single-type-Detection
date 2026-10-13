# SGQS-E：原设计、实际损失与当前问题

本文解释本次 B1/B2 七 epoch 续训的数学与实现，不宣布替代方案已验证。
数值、原始日志和证据限制见[诊断报告](README.md)。方法规范见
[M-002](../../../../methods/M-002_sgqs_e/README.md)。

## 1. 原设计

保持原有 Encoder、Decoder、query 数和推理 Top-K 不变，改变两处训练监督：

1. **SGQS：** 每个增强后的 GT 在线指定唯一 owner，在 Top-K 前监督分类、框回归和入选排名。
2. **EFGL：** Decoder 匹配框的四条边分别加权，补偿小框的相对定位误差与同类实例归属歧义。

记 GT 为归一化 cxcywh 框 \(b_g\)，owner 预测为 \(\hat b_g^o\)，其单类别
logit 为 \(z_{o_g}\)。\(\operatorname{sg}\) 表示停止梯度。\(K=640\)，
invalid-anchor mask 后的有效候选集合为 \(\mathcal V_b\)，真实入选集合为
\(\mathcal S_b=\operatorname{TopK}_{i\in\mathcal V_b}z_i\)。

### 1.1 SGQS 分类与回归

owner 的质量目标有非零下限：

\[
q_g=0.25+0.75\operatorname{sg}[\operatorname{IoU}(\hat b_g^o,b_g)].
\]

令 \(\mathcal A_b\) 为实际参与新增监督的 GT 集合，\(\mathcal P_l\) 为其
第 \(l\) 层 owner，\(\mathcal N_l\) 为排除 ignore ring、owner 和基础正样本
后的有效负点。简写 \(B(z,q)=\operatorname{BCEWithLogits}(z,q)\)：

\[
L_{\rm dense}=\frac13\sum_{l\in\{P3,P4,P5\}}
\left[
\frac{\sum_{g:o_g\in\mathcal P_l}q_g B(z_{o_g},q_g)}{|\mathcal P_l|\vee1}
+\frac{\sum_{i\in\mathcal N_l}0.75\sigma(z_i)^2 B(z_i,0)}{|\mathcal N_l|\vee1}
\right].
\]

\[
L_1^o=\frac{\sum_{b,g\in\mathcal A_b}\|\hat b_g^o-b_g\|_1}{N_A\vee1},
\qquad
L_G^o=\frac{\sum_{b,g\in\mathcal A_b}[1-\operatorname{GIoU}(\hat b_g^o,b_g)]}{N_A\vee1},
\qquad N_A=\sum_b|\mathcal A_b|.
\]

回归中的 IoU/GIoU 使用对应 xyxy 转换。这里用单进程记号写均值；实现使用
跨卡平均计数归一化，空集合定义为零。

### 1.2 Survival：区分最初方案与本次实际门槛

最初截取方案采用真实第 K 名：

\[
\theta_b^{\rm old}=\operatorname{sg}[\operatorname{kthlargest}_{i\in\mathcal V_b}(z_i,K)].
\]

但若 \(A_b=K\) 且全部 owner 已入选，最低 owner 本身就是第 K 名，要求
\(z_{o_g}>\theta_b^{\rm old}+m\) 会形成自追逐。修订后的 B1 使用：

\[
A_b=|\mathcal A_b|,\quad \mathcal O_b=\{o_g:g\in\mathcal A_b\},\quad
\theta_b=\operatorname{sg}\left[
\operatorname{kthlargest}_{i\in\mathcal V_b\setminus\mathcal O_b}(z_i,K-A_b+1)
\right].
\]

全部有效 owner 严格超过这个门槛，是它们都能进入原 Top-K 的充分条件；
该条件只覆盖当前有效 owner，不能保证全部 GT 被检测。

\[
L_s=\frac{\sum_{b,g\in\mathcal A_b}\chi_bq_g\,
0.1\log\left(1+\exp\frac{\theta_b+0.2-z_{o_g}}{0.1}\right)}
{\sum_{b,g\in\mathcal A_b}\chi_bq_g}.
\]

\(\chi_b\) 限制为非空、GT 数不超过 K、存在有效 owner 且候选未全选的图。
分母为零时损失为零。课程写在总损失外，只乘一次。

### 1.3 EFGL：逐边有界补偿

\(\tilde b_g\)、\(\hat{\tilde b}_g\) 为增强后输入像素 xyxy；
\(d_g=(w_g,h_g,w_g,h_g)\)，\(s_g=\min(w_g,h_g)\)，\(e\) 表示四条边。

\[
r_{g,e}=\frac{|\hat{\tilde b}_{g,e}-\tilde b_{g,e}|}{\max(d_{g,e},2\ {\rm px})},
\quad a(s)=e^{-s/16},\quad h(r)=\frac r{1+r}.
\]

同类邻域使用相对框距离，取距离不超过 4 的最近最多 4 个 GT：

\[
D_g(u,v)=\sum_e\frac{|u_e-v_e|}{\max(d_{g,e},2\ {\rm px})},\quad
\delta_{gj}=D_g(\tilde b_g,\tilde b_j),
\]

\[
\Delta_{gj}=D_g(\hat{\tilde b}_g,\tilde b_j)-D_g(\hat{\tilde b}_g,\tilde b_g),
\quad U_{gj}=\operatorname{clip}_{[0,1]}\left(1-2\frac{\Delta_{gj}}{\delta_{gj}}\right),
\]

\[
v_{gj,e}=\frac{|\tilde b_{g,e}-\tilde b_{j,e}|/\max(d_{g,e},2\ {\rm px})}{\delta_{gj}},
\quad U_{g,e}=\max_{j\in\mathcal N(g)}v_{gj,e}U_{gj}.
\]

只对 \(\delta_{gj}>10^{-6}\) 的有效配对计算；无有效邻居时 \(U_{g,e}=0\)。
实际版本固定 \(\kappa=1/2\)，没有第二套邻域 margin 课程。

令 \(I_g\) 为 Decoder 匹配框的 IoU，则

\[
\boxed{\omega_{g,e}=\operatorname{sg}\left[
I_g+0.5c(t)e^{-s_g/16}(1-I_g)
\min\{1,h(r_{g,e})+0.5U_{g,e}\}\right]},\qquad I_g\le\omega_{g,e}\le1.
\]

\[
L_{\rm FGL}=\frac1N\sum_{g,e}I_g\,\operatorname{CE}_{\rm interp}(p_{g,e},y_{g,e}),
\qquad
L_{\rm EFGL}=\frac1N\sum_{g,e}\omega_{g,e}\,\operatorname{CE}_{\rm interp}(p_{g,e},y_{g,e}).
\]

\(\operatorname{CE}_{\rm interp}\) 为原相邻 bin 插值 CE：令
\(k=\lfloor y\rfloor,\xi=y-k\)，有效 bin 区间内
\(\operatorname{CE}_{\rm interp}(p,y)=-(1-\xi)\log p_k-\xi\log p_{k+1}\)。
EFGL 替换原 FGL，保留其外层系数和普通/正 DN 分支；不另加一份 CE。

## 2. 实际总损失

令 \(L_{\rm rest}\) 汇总基础 VFL、L1、GIoU、DDF 及其原辅助、Encoder、DN
分支；本次基础 FGL 系数为 0.15。用 \(\Sigma L_{\rm FGL}\) 表示原实现各
FGL 分支项之和，则

\[
L_{B0}=L_{\rm rest}+0.15\Sigma L_{\rm FGL},
\]

\[
L_{B1}=L_{B0}+L_{\rm dense}+2.5L_1^o+L_G^o+0.25r_s(t)L_s,
\]

\[
L_{B2}=L_{\rm rest}+0.15\Sigma L_{\rm EFGL},
\]

\[
\boxed{L_{B3}=L_{B2}+L_{\rm dense}+2.5L_1^o+L_G^o+0.25r_s(t)L_s}.
\]

本次 \(t\) 为绝对 epoch 进度（含 batch 小数部分）：

\[
r_s(t)=\operatorname{clip}_{[0,1]}(t-28),\qquad
c(t)=\max\left(0,1-\frac{t-28}{2.1}\right)\quad(t\ge28).
\]

dense/owner 无升温；survival 在 1 epoch 内开启；EFGL 在 epoch 30.1 归零。
原从头方案的 EFGL 持续前 30% 训练，本次是平移起点后的 7 epoch 屏选。
B3 没有到达日志，这里只解释其设计。

## 3. 当前问题的数学位置

### 3.1 SGQS 的有效集合依赖基础匹配，监督衔接不完整

记原分配集合为 \(\mathcal A_b^0\)，基础分类给候选分配的 GT 为
\(c_b(i)\)，未匹配背景记为 \(\varnothing\)；基础 GO 回归分配记为 \(r_b(i)\)。
当前保留掩码为

\[
M_g=\mathbf1\left[
o_g\notin\mathcal S_b\ \lor\
\left(c_b(o_g)=g\ \land\ [r_b(o_g)\text{ 未分配}\ \lor\ r_b(o_g)=g]\right)
\right],\quad
\mathcal A_b=\{g\in\mathcal A_b^0:M_g=1\}.
\]

同一个 \(M_g\) 同时决定新增分类、回归和 survival 是否存在。
owner 未入选时可被抬分，入选后若基础匹配仍选其他候选，就失去全部新增
正监督。基础损失仍在；这里说的是新增 owner 监督被撤销。

**逻辑过粗之处：** 分类为背景与回归指向另一 GT 是两类不同冲突，当前却
共用整组屏蔽。密集单类别中，分类匹配另一 GT 也不自动意味着类别标签相反，
还须检查具体质量软目标；不同实例身份本身并不等于不同类别。

定义 batch 内的覆盖与条件生存：

\[
C=\frac{|\mathcal A|}{G},\quad
S=\frac{|\{g\in\mathcal A:o_g\in\mathcal S\}|}{|\mathcal A|},\quad
E=\frac{|\{g\in\mathcal A:o_g\in\mathcal S\}|}{G}=CS.
\]

在无分配失败、无预算溢出的条件下，epoch 34 日志均值为
\(\bar C=0.4492,\bar S=0.9778\)。因此 97.78% 是条件生存率，不能当作全 GT
有效覆盖。\(\bar C\bar S\approx0.4392\) 只是量级估算；均值乘积不等于乘积均值。
由 \(0\le C,S\le1\) 可得，忽略打印舍入，

\[
0.4270=\max(0,\bar C+\bar S-1)\le\overline{CS}\le\min(\bar C,\bar S)=0.4492.
\]

所有被 guard 丢弃的 owner 都已入选，故同条件下原始 owner 生存率为
\((1-C)+CS\)。高冲突率不能被误读为“排序没有学会”；真正暴露的是入选后
监督相容性与覆盖。以上均不是最终检测 recall。

### 3.2 新目标的启动和最终目标偏离

\[
L_{B0}^{\rm first}=8.4689,\quad L_{B1}^{\rm first}=13.5399,\quad
L_{\rm dense}^{\rm first}=4.7029,\quad r_s(28)=0.
\]

初始扰动来自立即启用的 dense/owner，而非尚未开启的 survival。
损失幅值不能代替梯度范数，故不据此声称梯度爆炸。

\[
\mathrm{AP}_{small}^{B1}:\quad0.706\ (e30)\longrightarrow0.682\ (e34),
\qquad \mathrm{mAP}^{B1}\text{ 同期上升}.
\]

这个结果说明小目标收益未保持。epoch 34 的 DDF 合计 B1 约 3.81、B0 约
1.90，提示基础优化也受扰动；现有汇总不能确定回落原因及梯度方向。

### 3.3 EFGL 在已较准确框上可能被重复误差衰减

当邻域歧义 \(U\approx0\)，且预测处于接近正确框的常见小扰动区域时，
令 \(\varepsilon\) 表示归一化几何扰动量级，可有
\(1-I=O(\varepsilon)\)、\(r_{g,e}=O(\varepsilon)\) 及
\(h(r)=r+O(r^2)\)，于是

\[
\Delta\omega=\omega-I=0.5c(t)a(s)(1-I)h(r)=O(\varepsilon^2).
\]

例如 \(I=0.95,r=0.03,s=14\text{ px},c=1\)，

\[
\Delta\omega=0.5e^{-14/16}\cdot0.05\cdot\frac{0.03}{1.03}
\approx3.04\times10^{-4}.
\]

这是数学例子，不是日志样本。实际主普通分支平均 bonus 在 e28/e29 约为
\(2\times10^{-4}\)、\(10^{-4}\)，之后课程很快归零。基础 CE 系数本来已经
是 IoU 接近 1 的值，因此这类微小补偿很难产生明显变化。
由于权重停止梯度，\(\nabla\ell=\omega\nabla\operatorname{CE}\)，
这里直接限制了对原 CE 梯度的干预强度。全局均值不代表困难小框尾部，
还需按短边、密度、层与 DN 分组。

## 4. 待验证的最小修订

1. 将 \(M_g\) 拆为分类、回归相容性掩码；保留可行回归或选择相容备用 owner。
   不直接关闭全部保护，避免重新引入矛盾监督。
2. 对 dense/owner 做独立短升温对照；该措施不替代长期覆盖问题的修复。
3. 固定课程与其余参数，独立比较去掉重复 \((1-I)\) 的有界候选式：

\[
\omega'_{g,e}=\operatorname{sg}\left[\min\left\{1,
I_g+0.5c(t)a(s_g)\min(1,h(r_{g,e})+0.5U_{g,e})\right\}\right].
\]

该式仍有 \(I_g\le\omega'_{g,e}\le1\)，但这是候选实验，不是已验证改善。
本次只有单 seed 的 B1/B2；COCO small 是原图面积分桶，不等于增强后短边小。
需要匹配我们任务定义的分组指标与 B3、多 seed 才能判断最终价值和互补性。
