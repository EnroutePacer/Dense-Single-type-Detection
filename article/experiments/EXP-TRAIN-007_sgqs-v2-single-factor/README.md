# EXP-TRAIN-007：SGQS v2 单因素分支

状态：`implemented / not_run / unvalidated`。本次只实现续训筛选，没有训练结果。
依据 [EXP-DIAG-007](../EXP-DIAG-007_sgqs-loss-hypothesis-audit/README.md)
与全验证集排名诊断，检验目标不一致、等层均值、resume 升温及末端 admission 四项机制。
它们是可检验的原因假说；排名归因和 loss 数值尚不等于共享参数梯度因果证据。

代码位于 [engine/sgqs_v2](../../../engine/sgqs_v2/criterion.py)，
配置与命令位于 [configs/sgqs_v2](../../../configs/sgqs_v2/README.md)。
沿用 `engine/`、`configs/`、`tools/table_cells/`、`tests/`、`outputs/` 原文件树，
不在论文目录复制运行代码，也不覆盖原 SGQS 文件。

2026-10-19 修订：原 R0--R3 保留，新增独立 R4/R5；主指标改为本地 hybrid-small
严格一对一 AP，完整验证集评估从执行要求落实为 84 状态矩阵。
旧四臂协议只覆盖两项假说，不能闭合历史 COCO AP-small 倒 U 的因果链。

## 数学与六臂

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
EFGL 六臂均关闭；owner 匹配、冲突过滤、非 owner 生存阈值、DN/DDF、Matcher、
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

六臂均从冻结的 epoch-27 checkpoint 进行绝对 epoch 28--34 续训，8×8、seed 0、AMP、
优化器/EMA/scaler 状态一致。除 R4 外 dense/owner 自恢复 step 0 生效，
survival 六臂均保留 28→29 的原升温。
不叠加干预，不改为多 owner。R0/R1 必须同期重跑，
历史曲线只作参考；旧 checkpoint 未显式保存完整 RNG 状态，不能宣称与历史轨迹逐步相同。

保留原 8-rank 训练探针：前 200 steps 每 10 steps，之后每 100 steps 及 epoch 最后一步。
观察 owner 过滤/保留、尺度组监督、weighted loss、DN-DDF 分解与局部 corner 梯度。
原探针的 `small_input` 是输入面积分组，仍保持原语义；不能把它冒充新增 hybrid-small
的监督份额。新增本地集合的正式结论来自明确命名的全量 evaluator。
局部 corner 梯度不代表共享 encoder 参数梯度，layer count 也不能直接冒充梯度份额。
代码/输入/实际配置的哈希写入独立 run manifest，训练入口拒绝配置覆盖与重复启动。

预先固定比较：R2−R0、R3−R0、R4−R0、R5−R0 为单因素效应，R0−R1 为整体 SGQS 效应。
主指标为 epoch 34 的 EMA **local_hybrid_small AP@[.50:.95]@300**，
同时必报同集合 @640、raw/EMA、七轮曲线和 epoch 30→34 变化。
历史 COCO AP-small@300、COCO medium/large 为单独命名的副指标，避免不同集合和预算混比。
固定 train/val 子集回放用于检查泛化差距，完整验证集用于最终指标。
补充关注输入短边 <8、8--16 px 及严格 IoU .95；它们需要相应的一对一分组评估，
不能从现有独立覆盖探针直接推导。

若仅 @300 改善而 @640 或严格定位明显下降，应解释为排序/预算取舍；
若仅代理改善而末端 AP-small 不改善，不宣布该修复有效。
单 seed 七 epoch 只筛选候选方案，胜出后再做独立 seeds 与完整训练；没有人为设定的
统计显著性结论，也不以峰值挑选替代预先固定的末轮比较。

## R4：dense/owner 的启动升温

令 \(t=e+\mathrm{step}/S_e\)，\(r_d(t)=\operatorname{clip}(t-28,0,1)\)：
\[
L^{R4}=L_{base}+r_d(t)\big[L_D^{R0}+2.5L_{owner,L1}+L_{owner,GIoU}\big]
                  +0.25r_s(t)L_{survive}.
\]
只改变这三项新增目标的启动日程，survival 自己的原升温不变。
记录 `sgqs_dense_owner_ramp`、升温前已乘各自系数的 dense/bbox/giou 数值，
有效损失仍进入原逐步探针。对比 e28 初期冲击与后期 local/COCO 指标。
**R4 同时减少首 epoch 的累计 SGQS 剂量**；即使有效，也只能支持该日程有帮助，
不能单独区分平滑启动与少监督半个 epoch 的作用。后续需等累计系数的常量对照，
以及共享参数梯度/更新测量，才能收紧“resume shock 导致倒 U”的因果主张。

## R5：只测试末端预算边界监督

使用已有最终 Decoder Hungarian 一对一匹配，记录每个 query 的 detached 匹配 IoU
\(u_i\)，未匹配者取 0。只考虑原始最终层，不使用 GO union、aux 或 DN。
对最终分数排序的 Top-300 集合 \(A\)，从外部选本地 small 且 \(u_i\ge0.5\) 的 query，
按 IoU 降序排列；内部 query 按 IoU 升序排列，同质量优先靠近排名边界者。
两列逐个配对，只保留 \(u_i-u_j\ge0.05\)。得到不复用 query 的集合 \(E\)：
\[
L_{adm}=\frac{1}{|E|\vee1}\sum_{(i,j)\in E}
   0.1\operatorname{softplus}\left(\frac{0.2+z_j-z_i}{0.1}\right),\qquad
L^{R5}=L^{R0}+0.25L_{adm}.
\]
成员选择、匹配、质量和几何均 stop-gradient；梯度只经最终分类 logits 回传。
一进一出避免要求超过预算的 GT 同时进入；每步记录有效配对数量和已加权 loss。
无配对时为连图的零损失，不能把该步视为有效干预。固定阈值和系数只是预注册的
机制探针设置，不代表已成熟的 SGQS v2 最终设计。

它不保证 AP 单调提升：Hungarian 质量代理不等于 COCO 贪心匹配，较低质量的内部
query 也可能贡献真阳性。只有 R5 的完整一对一结果改善，才支持这项训练干预；
R5 失败也不能排除所有 admission 方法。现有 DN/DDF 与 inference graph 均不改。

另提供固定框/固定 score 多重集的 GT 辅助置换诊断：仅交换上述配对的分数后重新
完整评估。离线配对使用纯 IoU Hungarian，不等同训练 matcher；它是排名路径干预，
不是可部署结果、最优上界或训练根因证明，指标与正式模型结果分栏保存。

## 本地 small 的 AP 口径与全量评估

\[
G_{local}=\{g:\ w_g^{640}h_g^{640}<1024\ \lor\ \min(w_g^{640},h_g^{640})<16\}.
\]
两个条件均为严格小于，宽高来自原始 bbox 直接缩放至 640×640，不从 annotation area
推算。AP 在 IoU=.50:.05:.95 上用 101 点召回插值，单类别下也是该集合的 mAP。
使用 COCO 非 crowd 一对一匹配、重复预测惩罚和全数据集 score 排序累积；
**先对所有检测应用 maxDets，再做集合处理**，不能先筛 small 后补满预算。

集合外 GT 作为 ignored GT 保留，正常匹配的集合外目标不成为 FP；未匹配预测按
自身 640 bbox 判断，集合外预测忽略、集合内预测计 FP，与 COCO area-slice 约定一致。
crowd 沿用 COCO 规则；集合无 GT 返回 undefined/null，有 GT 无预测返回 AP=0。
实现用临时二值 area 标签复用 evaluator 的 ignore 机制，bbox/IoU 不变，原 GT area 恢复。
不把 local AP 命名为原生 COCO AP-small；两套指标都输出且绝不混写。

历史 local hybrid-only recall 与 COCO-small 倒 U 并不同向，新的 local AP 是否存在倒 U
必须重算。不能更换主集合后声称解释了旧曲线，旧 COCO-small 仍单独追踪。

[全验证集执行器](../../../tools/table_cells/evaluate_sgqs_v2_matrix.py) 默认六臂 ×
七轮 × raw/EMA = **84 状态**，每状态同时 300/640 两预算，共 **168 行**；
`--arms r0 r1 r2 r3` 则是完整旧四臂 56 状态/112 行。
同一份预测复用两个预算，记录标注、图像内容、checkpoint、配置、运行代码和后端身份。
缺失状态直接报错，完成文件只在请求矩阵全部成功后产生。128 图不进入主结果。
执行命令见 [SERVER_COMMANDS.md](../../../configs/sgqs_v2/SERVER_COMMANDS.md)。

六臂仍是 seed 0、七轮 continuation screen；既不证明普适收益，也不闭合全部因果链。
通过筛选后，冻结胜出配置，至少做多个独立 seeds 的完整训练并配对报告均值/方差。

## 实现验证

2026-10-19：87 项相关测试通过，覆盖新升温/末端监督分支的实际 Detector + DN +
探针反向传播、本地 AP 的一对一/重复框/预算/边界/空集语义，以及小型 CPU 数据集上
raw/EMA × 双 epoch × 双预算的完整矩阵、缓存复用和输入变化拒绝。
真实全验证集矩阵和八卡训练尚未执行。

2026-10-18：70 项相关单元/CPU 集成测试通过（SGQS、SGQS-E、v2、探针、部署入口、
仓库契约）；另通过 Bash 语法和新增文档相对链接检查。
覆盖两项干预的解析梯度、未改动损失一致性、空 owner、非均衡 rank 计数的模拟 DDP
梯度与合并 batch 一致性、四臂实际注册构造及配置单因素差异。
没有进行真实八卡训练或真实多进程 GPU 验证，不能据此宣称方法有效。
