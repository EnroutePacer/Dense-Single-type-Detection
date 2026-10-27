# M-003 — SGQS v2：实例覆盖驱动的候选分配

训练生命周期已独立：逐步结构化记录、每轮完整 checkpoint、raw/EMA 全量验证与预测留档，以及 epoch 边界恢复，见 [训练框架 README](../../../engine/sgqs_v2/README.md)。

初版设计：2026-10-24；当前修订：2026-10-27（center_ios_overflow）；依据仓库 119bfe2、EXP-DIAG-006/007 与 baseline_test 的现有导出。状态：implemented / cpu_checked / unvalidated。完整方案先于训练实现写入，随后完成代码与CPU检查；本轮没有运行真实训练。

## 目标与证据边界

在保持主干、Encoder/Decoder 模块、DN、DDF、Matcher、训练 query 数 640 不变的前提下，通过训练改善候选排序：减少简单目标的可确认冗余，让不同实例获得预算。先检验同 K 下的覆盖与预定义困难子群，再以真实较小 K 重跑验证性能—计算量取舍。

完整 val 的 SGQS EMA AP-small@640 在 e30→34 从 .935651 增至 .952699，而 @300 从 .701179 降到 .677757；这定位输出排名而不证明训练 loss 的主导权反转。baseline_test 的 13 张无 GT 图上，Encoder 均分 .1921→.3281，但 Top256 对原 Decoder 高分候选的静态保留率 99.52%→96.34%。高分/低分跨阶段重叠候选 39→768 是研究动机，不是已确认假阳性或可安全删除标签。原“双方重叠都约 .6%”尚无一致口径复现。

因此不将升分、规则排列、独立 IoU coverage 或静态截断称作 AP 改善或计算节省。局部 owner 诊断不是训练 owner，119bfe2 的条件化替代救援率也不证明真实删 query 无损。

## 1. 保留几何辅助，解除唯一 owner 生存要求

继承 v1 增强后在线生成的几何 owner，保留其 L1/GIoU。它仅是辅助几何学习采样点，不再必须成为进入 Top-K 的唯一候选。沿用 Encoder 分类与 GO 回归匹配冲突保护；不改原 Matcher、base targets、base losses。

对 guard 后 owner，记 detached IoU 为 I_g，q_g=.25+.75 I_g。辅助 dense 正项使用 q_g BCE(z_g,I_g)，不再将 q_g 当 BCE target；q_g 仅控制样本权重。正项按全局有效 owner 数归一化，避免等层正样本均值给稀有层过大权重。负项保留 v1 focal 形式与各层负样本均值：

$$
L_{\mathrm{dense}}
=\frac{\sum_g q_g\,\mathrm{BCEWithLogits}(z_g,I_g)}{N_{\mathrm{pos}}\vee1}
+\frac1L\sum_l\frac{\sum_{i\in\mathcal N_l}0.75\,\sigma(z_i)^2\,\mathrm{BCEWithLogits}(z_i,0)}{|\mathcal N_l|\vee1}.
$$

上述 max 表示全局计数的空集保护；DDP 实现使用全局分子对应的 count/world_size，非空稀疏全局计数不能被本地 clamp 到 1。

唯一 owner survival 项退出 v2；v1 源码与历史结果保留。

## 2. 来源中心与 Decoder IoS 判断冗余

记 S 为真实 invalid-anchor mask 后 Encoder Top-K，K=640；z_i 为 Encoder logit，E_i、D_i 为同一 query 的 Encoder/最终 Decoder 框。selected_indices 连接身份并校验分数一致。c_i 是原始网格 anchor 的中心，即 sigmoid(anchors_unact)[..., :2]，不是回归后的 E_i 中心；它和 GT 都在增强后的归一化坐标系内。

来源归属 C(i)=g 要求 c_i 位于且仅位于一个 GT 的闭矩形内；落在多个 GT 内（含共享边界）则弃权。只在训练时传递 detached 中心，不改变推理。

可靠代表和 rescue 接收者仍沿用 IoU 质量检查：最佳 IoU>=.5 且与次佳相差>=.05。可靠代表 k_g 必须是现有 final Hungarian 对 g 的正匹配，Decoder score>=.5，Encoder/Decoder 的 IoU 归属均为 g，且 C(k_g)=g。因此 IoS 不会把小碎框升级为可靠代表。

对待判冗余的 Decoder 框改用：

$$
\operatorname{IoS}(D_i,b_g)=\frac{|D_i\cap b_g|}{\min(|D_i|,|b_g|)}.
$$

最佳 IoS>=rho_s=.8、与次佳差>=.05，且最佳 GT 必须等于 C(i)。这同时覆盖 GT 内小碎框和包含 GT 的大框；旧版 IoU>=.5 会遗漏许多这种候选。这里不要求 E_i 与本 GT 的 IoU>=.5。

安全 donor i 还须：i属于S；Decoder score<.1；该GT已有可靠代表；不是任何 final Hungarian 正样本、Encoder 分类正样本或 guard 后有效owner；GO回归目标不存在或与C(i)一致。保留原跨GT IoU保护，且另一个GT的Decoder IoS>=.8也排除；模糊的跨GT IoU归属保守排除。一个大框包住多个GT时不会因为IoS高而获得冗余资格。

这些门槛只控制新增监督。原背景分类、有效owner回归及可能适用的dense背景监督仍然存在，不能把旧版专项门控遗漏说成“完全无监督”。仅凭低分和包含关系不能证明删query后Decoder其他输出不变。

## 3. 两种接收者，共用一个排序项

优先处理 coverage rescue：某 GT 在当前 S 中不存在任何 Encoder IoU>=rho 的候选。对其 Top-K 外有效 dense 候选，取明确归属该 GT 且分数最高的一项 j。每GT至多一个，不要求所有GT都能塞进K，GT>K不跳过整图。该类改善的是入口几何覆盖的潜力，不预先宣称最终 Decoder 会成功。

再处理 prefix allocation：从已有可靠 final 代表中选择当前 Encoder 排名较弱的其他GT代表 j，要求它是同GT的 Encoder 分类正匹配，且已有 GO box/owner 目标没有跨GT冲突。这一类已经在640内，只改善较小候选前缀的分配，不记为新增Top640覆盖。

receiver按 rescue优先、组内分数从高到低，donor按分数从低到高确定性配对；跳过同GT与不满足 m+z_i>z_j 的配对。每donor、receiver只用一次，每GT至多一个receiver；缺少可靠配对时进入下述溢出检查，不降低配对门槛凑样本。

对 rescue receiver，当步从辅助 dense 的正、负分类监督中排除，但保留其 owner 几何回归；它在S外，本来无 base Encoder 分类项。避免新排序正梯度与旧 dense 背景/软目标冲突。对 prefix receiver，不新增正梯度，只作为 detached 参照；其已有正监督正常工作。

定义 t_j=z_j（rescue）或 stopgrad(z_j)（prefix）：

$$
L_{\mathrm{allocate}}
=\frac{\sum_{(i,j)\in\mathcal P}[m+z_i-t_j]_+
+\sum_{i\in\mathcal O}[z_i-\operatorname{logit}(u)]_+}
{(|\mathcal P|+|\mathcal O|)\vee1},
\qquad m=0.2,
\qquad t_j=\begin{cases}z_j&\text{rescue},\\\operatorname{sg}(z_j)&\text{prefix}.\end{cases}
$$

rescue 在激活区的梯度为 (+1,-1)，prefix为(+1,0)，除以全局有效配对与溢出样本的总计数。无温度、额外质量权重、邻域强制均匀项或可学习门控。单独记录两类配对与溢出样本，不能用prefix或overflow数量宣称困难召回提升。

### 无可执行接收者时的备用数量与高置信度检查

逐图先寻找所有可执行的跨GT配对（必须满足m+z_i>z_j）；一张图只要有一对，整图不执行额外溢出惩罚，未配上的donor也不走此分支。它不是“配对用完后的第二轮处罚”。

若整图无配对，对有可靠代表的GT统计n_g：TopK内来源中心唯一落在g内的候选总数减1。计数包括受保护候选，不把TopK外dense anchor当作Decoder开销。容忍T=10个备用，e_g=max(n_g-T,0)。先在额外候选中保留至多T个备用：非安全donor（有正监督或尚不能确认冗余者）优先，其余按Decoder对该GT的IoU降序、Decoder分数降序、来源索引升序确定。备用选择不依赖Encoder分数，避免降分后轮换处罚全部备用。仅对备用集合之外的安全donor检查s_i>u=.8，再按Encoder分数降序处理；进入O的数量自然不超过e_g，不够就少罚，任何受保护候选即使超出T也不处罚。

溢出项只给Encoder logit正梯度，降到logit(.8)后该项停止。没有接收者升分、没有新增框回归或teacher反传。P、O在同一图片上互斥；全局可以由不同图片同时贡献，两者共同取均值并共享原allocation权重与升温，避免另加一份独立归一化的损失。T、u、rho_s是显式配置、未验证的首轮参数，u不是已经部署的推理阈值。

## 4. 总损失与升温

$$
L_{\mathrm{v2}}=L_{\mathrm{base}}+r(t)\left[
L_{\mathrm{dense}}+2.5L_{\mathrm{owner-L1}}+L_{\mathrm{owner-GIoU}}+0.25L_{\mathrm{allocate}}\right].
$$

r(t)=clip((t-28)/1,0,1)，t=epoch+step/epoch_step，绝对epoch，恢复训练不重置时钟。所有新增分支统一升温，处理已观察到的 resume shock；几何/分类/排序都参与该升温。新方案不包含 EFGL 或旧 Decoder Top300 admission 项。

support 控制组采用完全相同的 v2 几何、dense、guard与升温，仅 allocation_weight=0且不应用allocation mask。它用于识别新分配项的净作用。

## 5. 实现边界与诊断

engine/sgqs_v2 下只增加训练 supervisor、候选配对函数及新criterion；覆盖旧单因素criterion。训练supervisor复用v1 payload，额外提供detached dense boxes。final匹配通过一次base forward的已有matcher结果读取，不再跑teacher网络。

固定次数全局归约：空GT、空pair、某rank无pair都不能改变DDP collective顺序。所有辅助loss的分母遵从全局总计数；指标提供全局总计数/图均值和比率所需分子分母。只记录已存在的分数/几何与门控统计，不继续运行旧GO/DDF逐项梯度探针。

必报：GT/selected几何覆盖、missing-GT、rescue可用GT、无几何候选GT、可靠代表、双阶段不一致、受保护donor、可用donor、rescue/prefix pair数量、有效pair比例、pair logit gap、donor/receiver/keeper分数、owner冲突与辅助ramp。记录未加权和加权损失可区分升温与学习。

GT>=K时全部query可能被Hungarian标正，从而没有安全donor；不为产生配对绕过正标签保护。无pair且无overflow/长期coverage rescue为0是可解释失败反馈，不能自动放松门控。teacher高低分阈值是保守首轮设置，不代表验证过最优。

## 6. 训练与评价协议

沿用文件树 engine/sgqs_v2、configs/sgqs_v2、tools/table_cells/*sgqs_v2*。新实验 EXP-TRAIN-008；旧 EXP-TRAIN-007 标记已被新方向取代，保留历史文档。旧六臂配置退出活跃入口，历史通过旧提交可追溯。

活跃 arms：b0（base）、support（v2无allocation）、v2（完整）；sgqs（原v1）为可选参照。默认只运行v2，最小科学比较为 b0/support/v2。全部同一冻结epoch27 checkpoint（model、EMA、optimizer、scaler）、8x8 global64、epoch28..34。七轮单seed仅筛选，不作为最终论文稳定结论。可选原sgqs保留原来无dense/owner升温的语义，不能称公平隔离新allocation。

训练主预算640。专用框架自动执行每臂×7epoch×raw/EMA完整val，并保存epoch27起点；输出严格一对一COCO各尺度及本地small：输入640后area<32^2 OR shortside<16，@640为主，@300为附加排名诊断。同次前向的候选与最终预测留档，离线矩阵工具用于旧运行或独立复核。旧实验默认300保持。

预注册困难场景：GT数>=301的高密度图片及其中local-small；分别报告全体/该子群AP、AP75/AP95和recall，不把所有small或高密度样本一概宣称困难。按固定annotation决定群组，与模型预测无关。真正“困难实例”泛化结论还需独立固定难例集，当前不把它作为已建立事实。

计算评价：同checkpoint实际设置Decoder输入K=640/512/384/256，每个K重新forward，固定输出maxDets=640；不对640输出静态切片，不动态阈值部署。拒绝learned-query-content模式的不兼容修改。缓存身份必须包含K、权重/图像/配置/源码哈希。记录同步后的模型forward耗时，排除I/O并说明不等于部署端完整延迟；使用预热，报告实际query数。没有实测不宣称速度收益。

成功标准：v2相对support在同K下增加正确的不同GT覆盖或困难子群性能且整体不退化；在较小K下优于b0/support/v1参照的性能—计算量取舍。均分、规则排列或过滤数不作为成功标准。报告置信区间/多seed前只称screening。

## 7. 必需实现检查

配对梯度方向与teacher detach；prefix receiver零新梯度；rescue分类mask保留回归；跨GT漂移/共享覆盖/现有正标签保护；空GT、没有候选、GT>K、tie、无pair、嵌套框IoS、原始中心与回归中心不同、跨GT包含、10/11备用边界及无接收者溢出；不均匀DDP计数与pooled梯度一致；真实微型DFINE训练前后向含DN；同权重同K eval逐元素一致；实际改变K在Decoder前生效；严格local-small边界与一对一重复惩罚；CocoEvaluator cleanup保持640；入口预检、resume时钟、runtime manifest、全val矩阵K缓存。

## 8. 相关工作与贡献边界

DDQ（CVPR2023，https://arxiv.org/abs/2303.12776）已研究dense/distinct queries；DQ-DETR（ECCV2024，https://arxiv.org/abs/2404.03507）已有动态query数。hinge ranking本身是标准工具。此处拟研究的具体组合是用GT约束的跨阶段实例身份监督候选分配，保留原几何辅助并避免正负冲突；不以论文想法重合删除好设计，也不把通用去重/动态K称为独有创新。

## 实现与验证入口

- [训练supervisor](../../../engine/sgqs_v2/candidate_supervision.py)：只添加no-grad dense几何检查，无推理参数。
- [归属和配对](../../../engine/sgqs_v2/allocation.py)：门控、唯一配对与两类梯度。
- [总损失](../../../engine/sgqs_v2/criterion.py)：旧base匹配复用、分类冲突mask、归一化、统一升温与全局日志。
- [配置/命令](../../../configs/sgqs_v2/SERVER_COMMANDS.md)、[实验协议](../../experiments/EXP-TRAIN-008_sgqs-v2-coverage-allocation/README.md)、[实际K全val评价](../../../tools/table_cells/evaluate_sgqs_v2_matrix.py)。
- [候选证据](../../../outputs/diagnostics/sgqs_v2_reframe_20261024/README.md)、[CPU检查记录](../../../outputs/diagnostics/sgqs_v2_reframe_20261024/validation.json)。

门控的几何计算为O(NG+KG)，IoU峰值临时空间O(chunk_size×G)，不保存N×G反传图；路由排序后用CPU整数身份确定性配对，批量传输避免每候选一次CUDA同步。训练增加dense框解码和GT门控成本；正常推理无该分支。默认全局K640推理路径与同权重base逐元素一致已在真实微型DFINE检查；GPU与生产数据未验证。

2026-10-24初版检查完成：137项仓库测试、两进程CPU/Gloo的稀疏/全空配对梯度与SGD一致性、Bash语法、运行文件校验、Markdown链接。检查记录保留环境与范围，不代表模型AP或加速收益。排序margin在代码评审中明确为配对条件m+z_i>z_j，避免只改变loss数值但不起实际作用；其余模型目标按上述冻结设计实现。

2026-10-27修订检查与旧门控覆盖分析见 [修订记录](../../../outputs/diagnostics/sgqs_v2_ios_overflow_20261027/README.md)。
