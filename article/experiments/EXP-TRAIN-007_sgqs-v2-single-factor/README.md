# EXP-TRAIN-007：SGQS v2 单因素分支

状态：`implemented / not_run / unvalidated`。本次只实现续训筛选，没有训练结果。
依据 [EXP-DIAG-007](../EXP-DIAG-007_sgqs-loss-hypothesis-audit/README.md)
与全验证集排名诊断，检验目标不一致、等层均值两项机制。
它们是可检验的原因假说；排名归因和 loss 数值尚不等于共享参数梯度因果证据。

代码位于 [engine/sgqs_v2](../../../engine/sgqs_v2/criterion.py)，
配置与命令位于 [configs/sgqs_v2](../../../configs/sgqs_v2/README.md)。
沿用 `engine/`、`configs/`、`tools/table_cells/`、`tests/`、`outputs/` 原文件树，
不在论文目录复制运行代码，也不覆盖原 SGQS 文件。

## 数学与四臂

令 \(I_g=\operatorname{sg}[\mathrm{IoU}(\hat b_g,b_g)]\)，
\(q_g=0.25+0.75I_g\)，\(P_l\) 是**冲突过滤后**的第 \(l\) 层 owner。
定义 \(S_l(t)=\sum_{g\in P_l}q_g\mathrm{BCE}(z_g,t_g)\)，
\(B_l=\sum_{i\in N_l}0.75\sigma(z_i)^2\mathrm{BCE}(z_i,0)\)。
下面按整个分布式 batch 的有效计数写，空集合项为 0；代码用全局计数除 world size
作为各 rank 分母，再由 DDP 平均梯度，等价于全局均值。

\[
L_D^{R0}=\frac13\sum_l\left(\frac{S_l(q)}{|P_l|\vee1}
                       +\frac{B_l}{|N_l|\vee1}\right),
\]
\[
L_D^{R2}=\frac13\sum_l\left(\frac{S_l(I)}{|P_l|\vee1}
                       +\frac{B_l}{|N_l|\vee1}\right),
\]
\[
L_D^{R3}=\frac{\sum_l S_l(q)}{\sum_l|P_l|\vee1}
                       +\frac13\sum_l\frac{B_l}{|N_l|\vee1}.
\]

R0/R2/R3 都使用：
\[
L=L_{base}+L_D+2.5L_{owner,L1}+L_{owner,GIoU}
                         +0.25r_s(t)L_{survive}.
\]
R1 为 \(L=L_{base}\)，关闭 SGQS supervisor，使用原 B0 路径。
R1 是整体 SGQS 对照，不能单独证明 owner 回归或 survival 的因果。
EFGL 四臂均关闭；owner 匹配、冲突过滤、非 owner 生存阈值、DN/DDF、Matcher、
query 数和推理路径不变。

R2 将正样本对 logit 的导数由 \(q(\sigma(z)-q)\) 改为
\(q(\sigma(z)-I)\)，仅消除 dense BCE 的质量目标抬升；它不消除仍存在的
survival 与质量分数之间的张力。不能把 `q_min=0` 当作 R2，那会同时改权重与 survival。
零 IoU owner 在 R2 中会受到分类下压，这是该因素本身的后果，必须观察是否损伤早期候选。

R3 取消稀少层 owner 的 \(1/(3|P_l|)\) 高系数，改为每个 owner 相同的
\(1/\sum_l|P_l|\)。全局正样本均值**不再除 3**；负样本仍等层平均。
当某层没有 owner 时，R3 相对 R0 也会改变正样本总系数，不能把结果描述为严格保总幅度的
尺度重分配；这是明确登记的归一化干预，不在本轮另加幅度补偿。

## 运行控制与判读

四臂均从冻结的 epoch-27 checkpoint 进行绝对 epoch 28--34 续训，8×8、seed 0、AMP、
优化器/EMA/scaler 状态一致。dense/owner 自恢复 step 0 生效，survival 保留 28→29 的原升温。
不增加 dense/owner 升温，不合并 R2+R3，不改为多 owner。R0/R1 必须同期重跑，
历史曲线只作参考；旧 checkpoint 未显式保存完整 RNG 状态，不能宣称与历史轨迹逐步相同。

保留原 8-rank 训练探针：前 200 steps 每 10 steps，之后每 100 steps 及 epoch 最后一步。
观察 owner 过滤/保留、尺度组监督、weighted loss、DN-DDF 分解与局部 corner 梯度。
局部 corner 梯度不代表共享 encoder 参数梯度，layer count 也不能直接冒充梯度份额。
代码/输入/实际配置的哈希写入独立 run manifest，训练入口拒绝配置覆盖与重复启动。

预先固定比较：R2−R0、R3−R0 为单因素效应，R0−R1 为整体 SGQS 效应。
主要报告 epoch 34 的 EMA AP-small@300、七轮逐轮曲线和 epoch 30→34 变化；
同时报告相同预测预算 @640、raw/EMA、AP-medium/large，避免用不同预算比较不同臂。
固定 train/val 子集回放用于检查泛化差距，完整验证集用于最终指标。
补充关注输入短边 <8、8--16 px 及严格 IoU .95；它们需要相应的一对一分组评估，
不能从现有独立覆盖探针直接推导。

若仅 @300 改善而 @640 或严格定位明显下降，应解释为排序/预算取舍；
若仅代理改善而末端 AP-small 不改善，不宣布该修复有效。
单 seed 七 epoch 只筛选候选方案，胜出后再做独立 seeds 与完整训练；没有人为设定的
统计显著性结论，也不以峰值挑选替代预先固定的末轮比较。

## 实现验证

2026-10-18：70 项相关单元/CPU 集成测试通过（SGQS、SGQS-E、v2、探针、部署入口、
仓库契约）；另通过 Bash 语法和新增文档相对链接检查。
覆盖两项干预的解析梯度、未改动损失一致性、空 owner、非均衡 rank 计数的模拟 DDP
梯度与合并 batch 一致性、四臂实际注册构造及配置单因素差异。
没有进行真实八卡训练或真实多进程 GPU 验证，不能据此宣称方法有效。
