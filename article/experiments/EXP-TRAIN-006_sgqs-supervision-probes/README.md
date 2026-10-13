# EXP-TRAIN-006：SGQS 28+7 监督来源探针

状态：**诊断代码已实现，真实训练/回放尚未运行**。2026-10-14。

依据为已归档的 [B0 日志](../../../outputs/resume_control_e28_to_e34_gb64/console.log)、[B1 日志](../../../outputs/e35_r7_sgqse/resume_b1_sgqs_seed0/console.log)、[B3 日志](../../../outputs/e35_r7_sgqse/resume_b3_sgqs_e_seed0/console.log)。下述 loss 数值取控制台整轮均值，AP 仅有三位小数精度。

## 首要嫌疑与可证伪条件

| 优先级 | 假设及现有依据 | 探针与否定条件 |
|---|---|---|
| 1 | SGQS关联的浅层DN蒸馏停滞：B1第一层DN-DDF e30→34为2.4786→2.5379，B0为2.3448→1.4498；B3同样没有沿基线路径下降 | 分离positive、真实negative、padding的裸KL/权重/加权KL/teacher熵，记录实际GO来源rho；若主要是padding或权重，不能称小框正样本定位失败 |
| 2 | GO/guard选择性遗漏困难实例：总owner覆盖增加不能排除小框子群持续被排除 | 分尺度GO query数、GO零覆盖、owner过滤前三种冲突与原始rank；小框组没有恶化则降低此假设优先级 |
| 3 | 候选评分/输出名额重排，而非几何退化 | 同一GT在dense encoder、实际Top-640、decoder全部query、Top-300的最佳IoU与首次合格rank；AR@640稳定但AR@300退化支持排名路径 |
| 4 | 小框泛化退化/过拟合，或EMA瞬态 | 同一固定train/val子集逐epoch同时评估raw和EMA；只有训练小框改善而验证退化才支持过拟合；随机训练loss下降不够 |

B1相对B0在e30的small/large AP约+2.7/-1.1点，e34约-0.4/0点；large主要恢复基线，不能直接推出大框抢梯度。B1/B3复现同seed走势不等于独立seed验证。P3不等于COCO-small，loss标量份额不等于梯度份额。

e30→34，B1 总 owner 过滤率约 73.79%→55.08%，保留率 26.21%→44.92%，保留集合内生存率 93.27%→97.78%。因此“后期过滤更多”与现有数据相反；真正未排除的是剩余困难小框的选择偏差，以及只优化保留集合的代理目标与最终检出脱节。e28 的前100步已经发生大幅候选重排，当时 survival 升温不足2%；初期收益也不能全部归因于 survival。

## 设计与实现

- [SGQS配置](../../../configs/probes/sgqs_e28_r7.yml) / [B0配置](../../../configs/probes/b0_e28_r7.yml) 严格继承现有e35/r7；总epoch=35，恢复last_epoch=27，训练28..34。
- [显式入口](../../../tools/table_cells/train_with_probes.py) 注册[观测wrapper](../../../tools/table_cells/supervision_probe.py)，再执行原始train.py。原loss、Matcher、GO、DN、模型、推理与现有启动器均不修改。
- 原epoch27断点、两份标注SHA-256由[预检](../../../tools/table_cells/prepare_probe_run.py)核对；生成resolved.yml、代码哈希和运行manifest。8 GPU×8，与已归档B0/B1相同。对照采用同一提交和同一硬件。
- 每100 step取样；e28前200 step每10步；每轮最后一步必取样。各rank写独立JSONL，数据无四位小数截断。**不是每一步完整记录，也不是无偏整轮均值**；早期加密阶段必须单独分析。
- checkpoint_freq=1，保存e28..34每轮的在线模型、EMA和原优化器状态；e27由输入checkpoint保留。离线回放不会改变训练RNG或插入额外验证数据加载。

## 字段和数学口径

`probes/train.rankN.jsonl` 保存实际epoch/step、rank/world_size、input_size、加权loss、全batch分组的count/sum/sum_sq和最多64条GT示例。GT示例按顺序截取，**不能用于频率估计**；分组充分统计覆盖该rank全部GT。使用[汇总器](../../../tools/table_cells/summarize_probe.py)合并所有rank，它拒绝缺失或重复rank记录。

训练尺度为**增强后输入面积**的small/medium/large，不是原图COCO面积。示例另记input_short_px、每图GT数、中心L1相对距离≤2的邻居数。GT索引仅是当前增强batch索引，不能跨随机batch强行当稳定annotation_id。

- `go_query_count`/`go_missing`：每GT普通回归/FGL的GO归属数量与零覆盖；`*_go_iou_sum`配合query_count恢复真实质量权重份额。GO按query多数票并不保证每GT至少一个query。
- `owner_assigned/kept/selected_raw/iou_raw/rank_raw`：先于过滤的owner质量与排名；未分配GT相应量补0，质量均值须用assigned总数作分母。
- `conflict_background/encoder_foreign/go_only`：三类互斥归因；`go_any`额外保留GO重叠数。它们删除的是SGQS新增项，不删除基础分类和DN。
- `encoder_selected_*`：**已经选择出的K个Encoder预测**的GT覆盖；不是8400个dense候选。`decoder_*`是普通最终query预测。独立GT的best-IoU覆盖可重复使用同一query，**不能称为COCO recall**。
- 离线回放记录全部有效dense候选和实际Top-K框，保留选择器真实入选集合；rank使用稳定分数排序，同分排序本身可能不唯一。缺少合格框时first_rank=候选数+1。

DDF保存每个浅层的

\[
L_l=1.5\frac{\rho_+\overline{wT^2KL}_+ +\rho_-\overline{wT^2KL}_-}{\rho_++\rho_-},\quad T=5.
\]

teacher为同次末层预测，正样本w为当前IoU，负样本w为teacher分数；rho从实际criterion缓存读取，即普通query GO数量的平方根系数。正、真负、padding分别记录，真负按生成它的GT尺度分组。记录E[wKL]而不是E[w]E[KL]，复算值与原加权loss一并保存；AMP下存在计算精度差异，比较时使用合理容差。

第一层DN还记录实际FGL/DDF对**该层返回的分布logits**的梯度平方范数与内积。使用autograd.grad，保留主反向图，不写parameter.grad。它只检测局部目标冲突，**不是共享参数梯度，更不是AdamW更新归因**。后续若要断言大框抢梯度，需要在相同checkpoint/batch上补测共享参数的分尺度梯度及真实Adam/clip更新；本轮不作该结论。

梯度日志另存 `autograd_dtype`；探针读取未经过 GradScaler 放大的梯度，若实际返回 FP16，极小分量可能下溢。不能把低精度下的零范数当作没有监督；需要时在保存断点上用 FP32 复核。

## 固定样本回放与过拟合判据

[子集工具](../../../tools/table_cells/prepare_probe_subset.py)用文件名哈希固定选择train/val各128图，流式读取6GB标注；排除两组重名图，并拒绝选中子集中相同图像字节。不是模型驱动挑选，也不是全数据泄漏审计。归档源标注、子集和图像SHA-256，pilot若small GT太少需先扩大预注册样本再统一重放全部checkpoint，不能只补某一轮。

[回放工具](../../../tools/table_cells/replay_probe.py)关闭增强，使用与原验证一致的640缩放，strict加载raw/EMA各自权重；对固定train和val记录真正的COCO AP/AR@300、@640及逐annotation_id的候选数据。对比e27、e28、e29、e30、e31、e34，建议七轮都回放。

| 观察 | 支持的解释 | 仍不能直接推出 |
|---|---|---|
| 训练小框AP升、验证小框AP降 | 泛化差距扩大 | 具体由哪项loss造成 |
| 全query最佳IoU稳定，Top300覆盖/AR降，AR640稳定 | 排序/预算路径 | 大框梯度主导 |
| dense有合格框，TopK没有 | Encoder选择问题 | owner应强制保留 |
| TopK有合格框，decoder全部query也退化 | 后续表示/回归路径 | DN-DDF一定有害 |
| 小框GO零覆盖增加、质量权重份额下降 | 选择性监督不足 | 全部小框没有监督，DN仍在 |
| 仅EMA出现峰值、raw没有 | 参数混合/时间滞后值得检查 | 训练过拟合已证实 |

获得对应checkpoint后，第二阶段才做短干预：原样继续、只暂停SGQS新增项、只改变第一层DN-DDF，保持数据顺序、优化器和步数相同；不同时改owner、Matcher和小框权重。

## 服务器命令

在仓库根目录执行。先更新仓库，使用现有训练Python环境；替换下列路径。下面不会由Codex自动启动GPU训练。

```bash
git pull --ff-only
PY=/path/to/existing/venv/bin/python
DATA=/aicamera-mlp/chenkai/data/table_det_data/paddlex_l_cell_merged
CKPT="$PWD/output/dfine_hgnetv2_m_table_cell/checkpoint0027.pth"
RUN="$PWD/outputs/probes/sgqs_e28_r7_seed0"

# 先核验全部冻结输入，生成独立输出目录；现有目录会拒绝覆盖。
"$PY" tools/table_cells/prepare_probe_run.py --arm sgqs --resume "$CKPT" \
  --data-root "$DATA" --output "$RUN"

set -o pipefail
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "$PY" -m torch.distributed.run \
  --nproc_per_node=8 --master_port=29716 tools/table_cells/train_with_probes.py \
  -c "$RUN/resolved.yml" -r "$CKPT" --seed 0 --use-amp 2>&1 | tee "$RUN/console.log"

"$PY" tools/table_cells/summarize_probe.py --run "$RUN" --output "$RUN/probe_steps.csv"
```

B0配对：把arm改为`b0`、RUN改为新目录`outputs/probes/b0_e28_r7_seed0`，另选端口；保留其余数据、batch、seed和GPU类型。建议先完成B1以获得关键轨迹，再运行B0同探针对照；历史B0缺这些字段，不能代替。

```bash
# 子集只创建一次，两臂和所有epoch复用同一manifest。
"$PY" -m pip install ijson
"$PY" tools/table_cells/prepare_probe_subset.py \
  --train-ann "$DATA/annotations/instance_train.json" \
  --val-ann "$DATA/annotations/instance_val.json" --images "$DATA/images" \
  --output "$PWD/_local/probe_fixed128" --images-per-split 128

MANIFEST="$PWD/_local/probe_fixed128/manifest.json"
"$PY" tools/table_cells/replay_probe.py --config "$RUN/resolved.yml" \
  --checkpoint "$CKPT" --manifest "$MANIFEST" --device cuda \
  --output "$RUN/replay_e27"
for E in 0028 0029 0030 0031 0032 0033 0034; do
  "$PY" tools/table_cells/replay_probe.py --config "$RUN/resolved.yml" \
    --checkpoint "$RUN/checkpoint${E}.pth" --manifest "$MANIFEST" --device cuda \
    --output "$RUN/replay_e${E}"
done
```

归档run_manifest、resolved配置、console/log.txt、所有rank探针、固定子集manifest、回放summary/GT行和各epoch权重哈希；权重留外部存储或LFS，不把不存在的RUN/AP结果写进注册表。采样诊断有额外运行时间，预算按更新步数对齐，不能宣称训练耗时不变。

## 本地验证范围

[测试](../../../tests/test_supervision_probe.py)覆盖实际含DN训练的loss/参数梯度/RNG逐项等价、DDF重建、padding分离、GO遗漏反例、空GT、候选捕获不改变输出，以及真实COCO evaluator的Top300/640排序反例。CPU检查不是CUDA吞吐、真实数据过拟合或AP提升结论。

本次通过根目录39项测试、6项仓库契约测试，以及[双进程两步 CPU/Gloo 检查](../../../tests/probe_ddp_smoke.py)：loss、参数梯度、AdamW 更新后权重、RNG 逐项一致。流式子集 CLI 已用合成 COCO 检查确定性选择、图像哈希和跨组排除。复核命令：

```bash
"$PY" -m unittest discover -s tests -v
"$PY" -m unittest discover -s article/tests -p test_repository_contract.py -v
"$PY" tests/probe_ddp_smoke.py --local-spawn --output /tmp/sgqs_probe_ddp_fresh
```
