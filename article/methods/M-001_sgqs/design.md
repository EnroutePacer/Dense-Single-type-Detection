# SGQS v1：实现设计说明

状态：implemented / unvalidated  
源码基线：fd0c1108ec0289237c27156edbda36aa90940b97  
适用问题：密集、单类别、细长或低像素短边目标在 Encoder global Top-K 前发生候选饥饿。

本文记录 SGQS v1 的设计动机、数据流、精确算法、梯度边界、配置、日志、实验约束和已知风险。所有损失公式见 [losses.md](losses.md)；本文重点解释代码为什么这样组织以及每一步如何执行。

## 1. 要解决的具体问题

D-FINE/DEIM 的 Decoder 不直接读取全部 Encoder memory，而是先按分类分数从 P3/P4/P5 的全部候选中选出固定数量 \(K\)：

\[
\{m_i\}_{i=1}^{N}
\xrightarrow{h_{\mathrm{score}}}
\{z_i\}_{i=1}^{N}
\xrightarrow{\operatorname{global\ TopK}}
\{m_{j_k}\}_{k=1}^{K}
\xrightarrow{\mathrm{Decoder}}
\text{detections}.
\]

如果某个 GT 周围没有候选进入 Top-K，后续 Decoder 只能细化其他 query，普通 Decoder 匹配损失很难直接修复这个更早的选择错误。

现有 12,800-step small-reg 实验没有改善 AP-small/AR-small，说明只放大已经参与匹配的框回归梯度不足以解决候选进入 Decoder 之前的召回问题。SGQS 因此把 GT 监督前移到 enc_score_head 后、global Top-K 前。

SGQS v1 检验三个可证伪子假设：

1. 每个 GT 都需要一个尺度适配且唯一的 Encoder owner。
2. Top-K 前的所有有效候选需要显式分类监督，而不能只监督已入选的 query。
3. owner 不只需要得到正分类梯度，还需要相对当前第 \(K\) 名分数获得生存 margin。

## 2. 非目标

SGQS v1 不做以下改动：

- 不增加 Decoder query 数；
- 不替换 Hungarian Matcher；
- 不修改 Decoder、PostProcessor 或 COCO evaluator；
- 不增加预测头或可学习参数；
- 不把损失复制到每层 Decoder auxiliary output；
- 不与 MAL、Mosaic、MixUp 或 historical small-object reweighting 同时启用；
- 不把 maxDets=640 作为论文主指标，主表仍固定 maxDets=300。

## 3. 模块结构

实现保持 DEIM 的注册与 YAML 注入方式：

    engine/sgqs/
      __init__.py
      candidate_supervision.py
        assign_scale_aware_owners
        SGQSQuerySupervisor
      criterion.py
        SGQSCriterion

    configs/sgqs/
      sgqs_hgnetv2_m_table_cell_fixed640.yml
      sgqs_hgnetv2_m_table_cell_fixed640_probe.yml

调用关系：

    train_one_epoch
      -> DEIM.forward(samples, targets)
      -> backbone + HybridEncoder
      -> DFINETransformer.forward(feats, targets)
      -> DFINETransformer._get_decoder_input

    engine/__init__.py
      -> import engine.sgqs
      -> 注册 SGQSQuerySupervisor 与 SGQSCriterion

    DFINETransformer._get_decoder_input
      -> enc_output(memory)
      -> enc_score_head(output_memory)
      -> invalid-anchor mask
      -> SGQSQuerySupervisor（仅 self.training）
      -> 原 global Top-K
      -> 原 Decoder

    SGQSCriterion
      -> 先调用未修改的 DEIMCriterion
      -> 再计算一次 pre-Top-K SGQS 损失
      -> 返回四个已经乘权重的 SGQS loss

## 4. Tensor 接口

Supervisor 的主要输入为：

| 名称 | 形状 | 含义 |
|---|---|---|
| output_memory | \([B,N,D]\) | enc_output 后的全部 Encoder memory |
| dense_logits | \([B,N,C]\) | 未 mask 的 enc_score_head logits，用于 dense/survival 梯度 |
| selection_logits | \([B,N,C]\) | invalid anchor 已置为 \(-\infty\) 的真实排序 logits |
| anchors_unact | \([B,N,4]\) 或 \([1,N,4]\) | inverse-sigmoid anchors |
| valid_mask | \([B,N,1]\) 或 \([1,N,1]\) | anchor 有效性 |
| spatial_shapes | \(L\times2\) | P3/P4/P5 的 \(H_l,W_l\) |
| targets | 长度 \(B\) 的字典列表 | 增强完成后的归一化 cxcywh GT |

固定 640、stride 8/16/32 时：

\[
N=80^2+40^2+20^2=8400.
\]

SGQS payload 只在训练输出中出现，键名为 sgqs_aux_outputs；eval 输出不包含该键。

Payload 输出契约：

| 字段 | 类型/形状 | device | detach |
|---|---|---|---|
| dense_logits | Tensor \([B,N,C]\) | 模型 device | 否 |
| owner_boxes | 长度 \(B\) 的 Tensor 列表，每项 \([A_b,4]\) | 模型 device | 否 |
| owner_indices | 长度 \(B\) 的 LongTensor 列表 \([A_b]\) | 模型 device | 离散索引 |
| target_indices | 长度 \(B\) 的 LongTensor 列表 \([A_b]\) | 模型 device | 离散索引 |
| owner_levels | 长度 \(B\) 的 LongTensor 列表 \([A_b]\) | 模型 device | 离散索引 |
| ignore_mask | BoolTensor \([B,N]\) | 模型 device | 无梯度 |
| valid_mask | BoolTensor \([B,N]\) | 模型 device | 无梯度 |
| assignment_misses | LongTensor \([B]\) | 模型 device | 无梯度 |
| spatial_shapes | LongTensor \([L,2]\) | 模型 device | 无梯度 |
| topk_cutoff | Tensor \([B]\) | 模型 device | 是 |
| selected_indices | LongTensor \([B,K]\) | 模型 device | 是 |
| effective_k | Python int | host | 不适用 |
| selection_pool_size | Python int | host | 不适用 |

Train/eval 分叉：

- train 且 supervisor 非空：生成 payload、计算 SGQS loss 和 diagnostics；
- train 且 supervisor 为空：保持基础 criterion；
- eval：不调用 supervisor、不输出 payload，SGQS 本身零额外推理计算；
- shared invalid-anchor mask 仍作用于 eval，因此必须单独做旧/新 selector 的推理回归。

## 5. Shared invalid-anchor correctness fix

原 anchor 生成代码会把 invalid anchor 的 box logit 设为 \(+\infty\)，但旧 selector 仍可能按其有限分类 logit 将其选入 Top-K。SGQS 实现同时增加：

\[
z_{i,k}^{\mathrm{select}}=
\begin{cases}
z_{i,k},&i\in\mathcal V,\\
-\infty,&i\notin\mathcal V.
\end{cases}
\]

真实 selector 与 SGQS cutoff 使用同一份 selection_logits。

这是所有配置共享的正确性修复，不是 SGQS 贡献。它会改变历史 baseline 的选择结果，因此：

- clean control 与 SGQS 必须在同一当前提交重跑；
- 历史 0.864248 只能作为旧 selector 参考；
- invalid-mask delta 必须单独报告；
- 不允许把 shared fix 的收益记到 SGQS。

## 6. 尺度感知 owner 分配

### 6.1 尺度定义

对增强完成后的 GT：

\[
d_g=\min(w_gW_b,h_gH_b).
\]

使用短边而不是面积，是因为表格单元格可能面积不小，但高度只有十几个像素。固定 640 的先验为：

\[
(a_3,a_4,a_5)=(32,64,128).
\]

层级顺序按下式从近到远排序：

\[
\Delta_{g,l}=
\left|\log\frac{d_g+\varepsilon}{a_l+\varepsilon}\right|.
\]

相同距离时优先 stride 更小的层。

### 6.2 GT 处理顺序

所有 GT 按以下键稳定排序：

    (pixel short side, original GT index)

小短边 GT 先分配，避免较大目标提前占用更稀缺的 P3 中心位置。

### 6.3 唯一 owner

每个 GT 最多分配一个 owner，每个候选最多服务一个 GT。精确过程：

    used = empty set
    for GT in ascending short-side order:
        compute preferred level and continuous center

        if preferred center is valid, unused, and no other GT shares it:
            choose it directly
        else:
            for level in scale-distance order:
                for radius from initial_radius to max_radius:
                    enumerate valid and unused cells in the square neighborhood
                    choose minimum squared center distance
                    break when found

        if found:
            add owner to used
        else:
            record assignment_miss

默认参数：

| 参数 | 值 |
|---|---:|
| initial_radius | 1 |
| max_radius | 4 |
| ignore_radius | 1 |

距离相同时用展平后的 global index 作为确定性 tie-break。

### 6.4 Invalid GT

中心或宽高不是有限数、宽度不大于 0、或高度不大于 0 的 GT 不参与 owner loss，并记为 assignment miss。该行为用于暴露数据问题，不静默制造伪 owner。

### 6.5 Ignore ring

owner 周围同层 Chebyshev 半径 1 的有效候选不作为 dense negative：

\[
\mathcal I_b=
\bigcup_g
\left\{
i:
l_i=l_{o_g},
\|(r_i,c_i)-(r_{o_g},c_{o_g})\|_\infty\le1
\right\}
\setminus\mathcal O_b.
\]

如果某个位置同时是其他 GT 的 owner，owner 优先，不能被 ignore。

## 7. Owner box 分支

不对全部 8,400 个位置额外运行 box MLP。只 gather owner memory，并复用原 enc_bbox_head：

\[
\hat b_g=
\sigma\left(
h_{\mathrm{box}}(m_{o_g})+
a_{o_g}^{\mathrm{unact}}
\right).
\]

因此：

- 不新增参数；
- enc_bbox_head 同时接受原 Top-K encoder loss 和新增 owner regression；
- 框梯度可以进入 owner memory、HybridEncoder 与 backbone；
- 额外 box-head 计算量与 owner 数量成正比，而不是与全部候选数成正比。

## 8. Top-K cutoff 与真实入选集合

排序池严格复现 query_select_method：

- default：每个位置取类别最大 logit；
- agnostic：使用唯一通道；
- one2many：展平位置和类别，入选后再映射回位置。

Supervisor 保存：

- 第 \(K\) 大分数 topk_cutoff；
- 实际 selected_indices；
- effective_k；
- selection_pool_size。

cutoff 和 selected_indices 均 detach，不允许通过第 \(K\) 名候选反向传播。owner 自身的 dense logit 保持梯度。

## 9. 损失组合

完整推导见 [losses.md](losses.md)。代码输出四项：

| 日志键 | 原始数学项 | 配置权重 |
|---|---|---:|
| loss_sgqs_dense | \(L_{\mathrm{dense}}\) | 1.0 |
| loss_sgqs_bbox | \(L_{\mathrm{point-L1}}\) | 2.5 |
| loss_sgqs_giou | \(L_{\mathrm{point-GIoU}}\) | 1.0 |
| loss_sgqs_survival | \(r(t)L_{\mathrm{survive}}\) | 0.25 |

总损失为：

\[
L_{\mathrm{total}}
=L_{\mathrm{base}}
+L_{\mathrm{dense}}
+2.5L_{\mathrm{point-L1}}
+L_{\mathrm{point-GIoU}}
+0.25r(t)L_{\mathrm{survive}}.
\]

### 9.1 质量目标

\[
u_g=\operatorname{stopgrad}[\operatorname{IoU}(\hat b_g,g)],
\qquad
q_g=0.25+0.75u_g.
\]

q_min=0.25 有两个目的：

1. 训练早期细长框 IoU 为 0 时，dense positive 仍有梯度；
2. survival 使用 \(q_g^\eta\) 相对加权时，最弱 owner 也不会被完全关闭。

这不是绝对质量门控。由于 survival 最后除以 \(\sum q_g^\eta\)，它主要改变 owner 之间的相对权重。

### 9.2 生存预算门控

当 \(G_b>K_b\) 时，让所有 GT 都进入 Top-K 不可能，因此该图关闭 survival：

\[
\chi_b=\mathbf1[0<G_b\le K_b<M_b].
\]

dense classification 与 owner regression 仍正常计算。

### 9.3 Ramp

正式训练从第 0 epoch 开始时：

    survival_start_epoch = 0
    survival_ramp_epochs = 1

从 checkpoint0027 继续时：

    survival_start_epoch = 28
    survival_ramp_epochs = 1

progress 使用：

\[
t=e+\frac{\mathrm{step}}{\mathrm{epoch\_step}}.
\]

## 10. 梯度边界

| 量 | 是否 detach | 原因 |
|---|---|---|
| owner 离散索引 | 是 | 离散 GT 分配不可微 |
| IoU 质量 \(u_g\) | 是 | 防止分类/生存权重反向操纵 box 质量 |
| 质量目标 \(q_g\) | 是 | 只作为 soft target/相对权重 |
| Top-K cutoff \(\theta_b\) | 是 | 不让边界候选成为移动的反向目标 |
| selected_indices | 是 | Top-K 索引不可微 |
| owner dense logit | 否 | 接受 dense positive 与 survival 梯度 |
| 有效 negative logits | 否 | 接受 focal-style negative 梯度 |
| owner box | 否 | 接受 L1/GIoU 梯度 |
| owner memory | 否 | 同时接受分类、生存和框梯度 |

梯度路径：

    L_dense / L_survive
      -> enc_score_head
      -> output_memory
      -> HybridEncoder
      -> backbone

    L_point-L1 / L_point-GIoU
      -> enc_bbox_head
      -> owner output_memory
      -> HybridEncoder
      -> backbone

## 11. DDP 归一化

每项计数或连续权重先跨 rank 求和，再除以 world size：

\[
\bar n=
\max\left(
\varepsilon,
\frac1R\sum_{r=1}^R n^{(r)}
\right).
\]

每个 rank 的本地 loss sum 除以同一个 \(\bar n\)，随后 DDP 平均梯度。这样在全局计数非零时等价于全局总 loss 除以全局计数。

不能把平均 rank 分母强制截到 1。例如 8 卡全局只有 1 个 P5 owner 时，正确分母是 \(1/8\)；截到 1 会让该样本梯度额外缩小 8 倍。

## 12. 非损失机制指标

以下指标通过 SGQSCriterion.get_logging_metrics 返回，det_engine 单独写入控制台和 TensorBoard Metric/ 命名空间。它们绝不进入 loss_dict 的求和：

| 键 | 定义/目的 |
|---|---|
| sgqs_assignment_miss_rate | 未找到 owner 的 GT 比例 |
| sgqs_owner_coverage | 成功分配 owner 的 GT 比例 |
| sgqs_owner_count | 每图平均 owner 数 |
| sgqs_budget_overflow_rate | 非空图中 \(G>K\) 的比例 |
| sgqs_owner_iou | owner 初始框平均 IoU |
| sgqs_pre_topk_survival_rate | eligible owner 实际进入 Top-K 的比例 |
| sgqs_owner_cutoff_gap | owner GT-class logit 减 cutoff 的均值 |
| sgqs_owner_p3_count | 每图平均 P3 owner 数 |
| sgqs_owner_p4_count | 每图平均 P4 owner 数 |
| sgqs_owner_p5_count | 每图平均 P5 owner 数 |
| sgqs_survival_ramp | 当前 survival ramp |

机制指标使用全局 SUM 后再计算比例，避免简单平均各 rank 比例造成偏差。

## 13. 配置

正式配置：

    configs/sgqs/sgqs_hgnetv2_m_table_cell_fixed640.yml

它继承 clean fixed-640 VFL control，只覆盖：

    criterion: SGQSCriterion

    DFINETransformer:
      sgqs_supervisor:
        type: SGQSQuerySupervisor
        scale_priors: [32.0, 64.0, 128.0]
        initial_radius: 1
        max_radius: 4
        ignore_radius: 1

    SGQSCriterion:
      base_criterion: DEIMCriterion
      weight_dict:
        loss_sgqs_dense: 1.0
        loss_sgqs_bbox: 2.5
        loss_sgqs_giou: 1.0
        loss_sgqs_survival: 0.25

首轮明确关闭：

    small_object_weight: 1.0
    small_object_box_weight: 1.0
    small_object_local_weight: 1.0

Probe 配置只把 survival ramp 起点改为 epoch 28。

## 14. 首轮实验设计

训练当前仍由诊断 Gate 阻止。允许启动后，唯一合法对照是同一提交上的 clean fixed-640 VFL control。

必须固定：

- 初始化 checkpoint；
- train/val 数据与 annotation SHA-256；
- 数据顺序和 seed；
- global batch；
- optimizer、LR、scheduler 状态；
- optimizer steps 或 epoch；
- 输入 640；
- query 数 \(K=640\)；
- 增强；
- maxDets=300 主评估。

建议消融顺序：

1. 当前提交的 clean control；
2. 仅 dense；
3. dense + owner regression；
4. 完整 SGQS；
5. 短边层级分配替换为面积分配。

每组至少三个 seed。先检查机制指标，再检查终局指标。

成功条件：

- owner coverage 与 pre-Top-K survival 提升；
- P3/低短边 GT 的 selection 指标提升；
- 相同步数 mAP、AP75、AR 或短边分桶指标提升；
- 中大目标不能出现不可接受退化。

否定条件：

- 机制指标不改善：否定当前 SGQS loss；
- 机制改善但 AP 不改善：瓶颈转向 Decoder/后处理，不宣称 SGQS 有效；
- 只在 maxDets=640 改善而 maxDets=300 不改善：只能声称机制改善；
- 增益来自 invalid-mask、更多 steps、更大 batch 或不同增强：实验无效。

## 15. 评估口径

论文主表：

    COCO AP@[0.50:0.95], maxDets=300

同时报告：

- AP50、AP75；
- AR@300；
- COCO area small/medium/large；
- 按 GT 像素短边分桶的 AP/AR；
- evaluator 截断前的逐 GT generation/selection/decoder coverage；
- 辅助 maxDets=640，用于观察超过 300 个 GT 的密图，但不能替代主表。

## 16. 单元测试覆盖

tests/test_sgqs.py 当前覆盖 11 项：

1. 短边尺度分层和展平 offset；
2. owner 冲突唯一性；
3. invalid point 跳过；
4. 空 GT；
5. \(G>K\) 只关闭 survival；
6. dense/box/memory 梯度；
7. IoU=0 owner 仍有 survival 梯度；
8. YAML 注册与构建；
9. 开关 supervisor 的 state_dict 键一致；
10. 同代码版本 eval 输出逐元素一致；
11. 极高 invalid-anchor logit 不能进入真实 Top-K。

论文仓库另有 4 项 repository contract 测试，共 15 项通过。

## 17. 运行模板

以下只记录复现入口，不表示现在应启动训练。

SGQS probe：

    cd /path/to/DEIM
    RUN_DIR="$PWD/output/probe12800_sgqs_fixed640"
    mkdir -p "$RUN_DIR"
    DATA_ROOT=/path/to/paddlex_l_cell_merged \
    CONFIG_FILE="$PWD/configs/sgqs/sgqs_hgnetv2_m_table_cell_fixed640_probe.yml" \
    RESUME="$PWD/output/dfine_hgnetv2_m_table_cell/checkpoint0027.pth" \
    QUICK_STEPS=12800 QUICK_OUTPUT_DIR="$RUN_DIR" \
    GPUS=0,1,2,3 BATCH_SIZE_PER_GPU=4 NUM_WORKERS=4 \
    SEED=0 MASTER_PORT=29531 \
    bash tools/table_cells/train_dfine_m_table_cell_multigpu.sh \
    2>&1 | tee "$RUN_DIR/console.log"

同提交 control 只把 CONFIG_FILE 改为：

    configs/table_cells/dfine_hgnetv2_m_table_cell_fixed640_probe.yml

两个 run 必须使用不同输出目录。

## 18. 已知风险与待验证项

1. owner 分配当前在 CPU 上执行离散搜索，并把 GT/valid mask 暂时复制到 CPU；大 batch 时可能增加训练 wall-clock。
2. \((32,64,128)\) 是首轮手工先验，不代表最优值。
3. 一个 GT 一个 owner 可能不足以覆盖非常细长目标的全部局部结构。
4. 一个候选只能属于一个 GT；极端密集且中心重合时可能 assignment miss。
5. survival 使用相对质量加权，不是绝对质量门控。
6. 固定 \(K=640\) 时，若单图 \(G>K\)，完整 survival 目标不可满足。
7. 当前重点是单类别；多类别 default/one2many 虽有实现路径，但未形成任务级验证。
8. 多 rank 数值等价性尚缺真实 1/2/4/8 卡梯度对照测试。
9. invalid-mask 的 ONNX/TensorRT 导出回归尚未测试。
10. 训练速度、峰值显存和固定 CPU 推理延迟仍需正式测量。
11. 正式 GT JSON、数据 split 哈希和 checkpoint artifact 尚未全部进入公开证据链。
12. 方法原创性仍需系统文献检索，implemented 不等于论文贡献已成立。

## 19. 版本与变更纪律

- fd0c110：SGQS 代码、配置、训练日志指标和 11 项测试。
- 43590db：完整损失公式、研究协议、源码快照和实验登记。
- 后续任何公式、默认权重、owner 规则或 selector 行为变化，都必须同时修改代码、配置、losses.md、本文、测试和 source snapshot。
- 未经消融验证的新增组件不得直接并入 SGQS v1 主配置。

## 20. Loss 执行伪代码

    outputs = model(samples, targets)
    base_losses = DEIMCriterion(outputs, targets)

    if sgqs_aux_outputs exists:
        qualities, owner_l1, owner_giou = owner_quality_and_regression(...)
        dense = dense_loss(...)
        survival = survival_loss(...)

        sgqs_losses = {
            dense * 1.0,
            owner_l1 * 2.5,
            owner_giou * 1.0,
            survival * 0.25,
        }
        losses = base_losses + sgqs_losses
        diagnostics = detached_global_metrics(...)
    else:
        losses = base_losses
        diagnostics = empty

    total = sum(losses.values())
    backward(total)
    log(losses under Loss/)
    log(diagnostics under Metric/)

SGQS 四项只计算一次，不进入 Decoder auxiliary layer 循环。原 enc_aux_outputs 的匹配和 loss 保留，因此 owner 监督与原 Top-K encoder auxiliary 监督是叠加关系，而不是替换关系。

## 21. Dense 与 survival 的关系

两个分类目标可能产生局部张力：

- dense positive 希望 \(\sigma(z_{o_g})\) 接近软目标 \(q_g\)；
- survival 希望 \(z_{o_g}\ge\theta_b+m\)。

如果 cutoff 很高而 \(q_g\) 较低，survival 会继续推高 logit，而 dense soft target 可能产生反向约束。v1 采用以下方式控制：

1. survival 权重仅为 0.25；
2. 使用一整个 epoch 的 ramp；
3. 使用 \(q_g^\eta\) 做相对质量加权；
4. cutoff detach，避免形成互相追逐的移动边界；
5. 同时记录 cutoff gap、owner IoU 和 survival rate。

这仍是待实验验证的设计，不假定两项目标天然兼容。若出现 dense loss 下降但 cutoff gap 持续恶化，应消融 survival margin、权重或改成 rank-only 目标。

## 22. 完整超参数契约

| 参数 | 默认值 | 合法范围 | 单位/作用 | 首轮建议消融 |
|---|---:|---|---|---|
| scale_priors | [32,64,128] | 长度等于特征层数、全正数 | 输入像素短边 | [16,32,64]；[32,64,128] |
| initial_radius | 1 | 整数且 \(\ge0\) | feature cell | 0、1 |
| max_radius | 4 | 整数且不小于 initial | feature cell | 2、4 |
| ignore_radius | 1 | 整数且 \(\ge0\) | feature cell | 0、1 |
| q_min | 0.25 | \(0<q_{\min}\le1\) | soft target 下限 | 0.10、0.25、0.40 |
| dense_alpha | 0.75 | \(\ge0\) | negative focal 系数 | 固定；必要时 0.25 |
| dense_gamma | 2.0 | \(\ge0\) | negative focal 指数 | 固定；必要时 1、2 |
| negative_weight | 1.0 | \(\ge0\) | dense negative 相对权重 | 0.5、1.0 |
| survival_margin | 0.2 | 实数，建议 \(\ge0\) | logit margin | 0、0.1、0.2 |
| survival_temperature | 0.1 | \(>0\) | softplus 温度 | 0.05、0.1、0.2 |
| survival_quality_power | 1.0 | \(\ge0\) | 相对质量幂 | 0、1、2 |
| survival_start_epoch | 0/28 | 非负进度 | ramp 起点 | 随训练身份固定 |
| survival_ramp_epochs | 1.0 | \(\ge0\) | epoch | 0、1、2 |
| lambda_dense | 1.0 | \(\ge0\) | 总损失权重 | 0、1 |
| lambda_reg | 0.5 | \(\ge0\) | \(5:2\) 框组合外权重 | 0、0.5 |
| lambda_survival | 0.25 | \(\ge0\) | 总损失权重 | 0、0.1、0.25 |

禁止一次同时扫多个维度。首轮先做组件消融，再对唯一有正信号的组件调参。

## 23. Resume 身份与输出规范

checkpoint0027 continuation 是 resume 探针，不是从头训练的新方法基线。必须记录：

- checkpoint 文件 SHA-256；
- checkpoint 中 last_epoch/optimizer/scheduler/EMA 状态；
- 是完整 resume 还是只加载 model 权重的 tuning；
- source commit；
- resolved config；
- 启动命令和环境；
- 每卡 batch、world size 与 global batch；
- QUICK_STEPS 的定义是每 rank optimizer step，不是样本数；
- 独立 RUN_DIR，禁止覆盖 control 或其他 seed。

如果 launcher 的 resume 会恢复旧 optimizer/LR 状态，则 control 和 SGQS 必须以完全相同方式恢复。不得把从头训练与 checkpoint continuation 放在同一主表列中。

建议目录：

    output/probe12800_fixed640_control_maskfix_s0/
    output/probe12800_fixed640_sgqs_s0/

每个目录至少保留：

    console.log
    log.txt
    resolved_config.yml
    command.txt
    environment.txt
    checkpoint/probe SHA-256
    eval metrics
    TensorBoard events

## 24. 候选预算边界

固定 640 当前共有 8,400 个位置，其中 8,084 个 anchor 有效，\(K=640\)，因此：

\[
K=640<8084.
\]

当前代码对 num_queries 大于原始排序池会取 effective_k，但真实 selector 仍直接调用 topk(K)；对 valid candidate 数不足 \(K\) 的情况也没有正式 runtime assertion。当前配置不触发，但推广到更低分辨率或更大 \(K\) 前必须增加：

    assert K <= finite valid selection count for every image

否则 cutoff 可能为 \(-\infty\)，真实 selector 也可能重新选入 invalid 位置。

如果 GT assignment miss，未分配 GT 周围没有 owner 保护；除已有 owner ignore ring 外，该区域仍可能作为 dense negative。这是已知风险，因此 assignment_miss_rate 必须持续记录，非零时不能直接解释该 GT 的训练结果。

## 25. 训练与部署开销

训练期新增开销来自：

1. GT boxes 与 valid mask 的 GPU 到 CPU 复制；
2. Python owner 邻域搜索；
3. owner memory gather 与额外 enc_bbox_head；
4. 为 cutoff/selected_indices 再执行一次 Top-K；
5. loss 分母和 diagnostics 的多次 DDP all-reduce；
6. 全候选 dense negative BCE。

理论上 eval 不执行 SGQS supervisor，因此 SGQS 本身不增加参数或推理 FLOPs；但 shared invalid mask 仍存在。所谓“推理零额外开销”必须通过固定 CPU 协议重新实测 latency、峰值内存和输出一致性后才能写入论文。

## 26. 验证矩阵

提交训练结果前至少完成：

| 类别 | 必测项 |
|---|---|
| 数值 | FP32、AMP、loss finite、梯度 finite |
| 数据 | 空 GT、无效 GT、极密 GT、\(G>K\)、多 batch |
| 尺度 | 真实 640/8,400 候选；未来动态尺寸单独验证 |
| 分布式 | 1/2/4/8 卡同 global synthetic batch 的 loss/梯度等价 |
| checkpoint | model、EMA、optimizer、scheduler strict resume |
| selector | default；agnostic/one2many 只算代码路径测试，尚非任务结论 |
| 导出 | ONNX 与 TensorRT 对 \(-\infty\) mask 的兼容性 |
| 推理 | PyTorch eval 输出、固定 CPU latency、峰值内存 |
| 评估 | maxDets=300 主表、640 辅助、逐 GT stage coverage |

当前正式适用域只声明为 fixed-640、单类别、default selector。其他路径存在实现不代表已经验证。

## 27. 实验身份与公平性清单

每个 run 必须冻结并登记：

- source commit 与 working tree diff；
- method/config 文件 SHA-256；
- dataset root alias、annotation JSON SHA-256、split；
- checkpoint artifact ID 与 SHA-256；
- seed、sampler epoch、数据顺序；
- GPU 型号/数量、每卡 batch、global batch；
- optimizer、基础 LR、参数组 LR、scheduler、warmup；
- epoch、optimizer steps、实际样本数、wall-clock、GPU-hours；
- 全部增强及其概率；
- 输入尺度、P3/P4/P5 stride、query 数 \(K\)；
- AMP、EMA、gradient clip；
- evaluator commit、maxDets、IoU 阈值；
- CPU latency 的线程数、warmup、repeat、runtime 和输入集合。

计算公平性：

- 主要比较固定 optimizer steps 与 global batch；
- 同时报告 wall-clock/GPU-hours，防止靠更多训练计算换指标；
- SGQS 训练开销必须单列；
- 推理 latency 使用同一固定 CPU 基准；
- 任何资源变化都不是单因素 SGQS 对照。

## 28. 机制指标的统计域

- pre_topk_survival_rate 只统计 \(0<G_b\le K_b<M_b\) 且成功分配 owner 的 eligible owner；
- owner_cutoff_gap 使用 GT 类别 owner logit 减当图 cutoff；
- owner P3/P4/P5 count 是跨 rank 总 owner 数除全局图片数；
- owner_coverage 是成功 owner 数除全部 GT 数；
- assignment_miss_rate 是 miss 数除全部 GT 数；
- 当前训练日志没有直接记录完整 owner rank、短边分桶或重复 query 占位率，这些必须由逐 GT 诊断脚本补充，不能伪称已经具备。

正式机制报告还应增加：

- 按 GT 像素短边分桶的 best dense IoU、best rank、Top-K coverage；
- unique-GT coverage，避免多个 query 重复覆盖同一 GT；
- Top-K 内重复占位率；
- P3/P4/P5 对各短边桶的贡献。

## 29. 预注册推进阈值

在正式多 seed 方差未知前使用以下 provisional gate：

1. 12,800-step 单 seed 探针中，低短边桶 pre-Top-K coverage 至少提升 5 个百分点；
2. 同时总 mAP 相对同提交 control 至少提升 0.002；
3. 低短边桶 AR 至少提升 0.01；
4. medium/large AP 任一不得下降超过 0.002；
5. loss、梯度、assignment miss 和训练开销无异常。

满足全部条件才进入三 seed 复现。正式论文成功标准由三 seed 均值、标准差和置信区间决定，不以单 seed 过线代替统计结论。

invalid-anchor mask 必须先作为独立 control 对旧 selector 做一次 checkpoint0027 冻结评估；它与 SGQS 的训练结果分开登记。
