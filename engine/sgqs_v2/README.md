# SGQS v2 附加训练框架与留档契约

本目录提供训练时的几何支持、候选分配，以及独立训练生命周期。它复用 DEIM 模型、数据、原始损失和 optimizer，拥有自己的启动校验、训练循环、验证、统计与恢复逻辑；不修改推理结构，不接入旧监督探针包装器。B0、support、完整 v2、可选旧 SGQS 使用同一记录和验证策略。

方法公式见 [M-003](../../article/methods/M-003_sgqs_v2/README.md)，命令见 [SERVER_COMMANDS](../../configs/sgqs_v2/SERVER_COMMANDS.md)。本次完善运行和证据保存能力，不改变已确定的 IoS、接收者、溢出门控和损失公式，没有新的精度结论。

## 通用入口、方法配置与实验配方

通用入口不导入 EXP-TRAIN-008、冻结输入清单或 arm 表，不规定 epoch27、八卡、seed0、640 query、AMP/EMA 或调度器。支持单进程 CPU/CUDA、torchrun 多进程；shell 包装器支持单机任意显式进程数，也可直接使用 torchrun 启动 Python 入口。单类型 DFINE 和可追溯 Encoder query 是当前方法/留档的结构要求。

- `configs/sgqs_v2/method.yml` 只定义训练附加方法；`train.yml` 组合模型/数据变换与方法。
- `train_sgqs_v2.py` 通过命令行指定运行参数；`run_sgqs_v2.sh` 只转发参数、设置设备并启动进程，不分派实验臂、不填数据路径。
- 固定 28+7 配方放在 [实验示例](../../configs/sgqs_v2/recipes/EXP-TRAIN-008.md)，它调用同一个通用入口。

新运行必须显式提供 config、数据路径、总 epochs、global train/val batch、workers、seed、AMP/EMA 开关、输入尺寸、query 数、评价 budgets、保存/验证间隔和初始验证开关。v2 还要求显式给出绝对升温起点、升温长度。优化器与方法细节来自命令行选择的 `-c`，使用 `-u optimizer.lr=...` 等覆盖。最终值和原始 CLI 写入 resolved.yml/run_manifest.json，不静默钳回某个实验设置。完整可复制命令见 [SERVER_COMMANDS](../../configs/sgqs_v2/SERVER_COMMANDS.md)。

四种模式：scratch 创建新训练状态；tuning 只导入匹配权重、重新计时；resume 从任意完整 checkpoint 创建新续训运行；recover 继续同一个中断运行。epochs 是绝对终点，例如完成 epoch4 后传 epochs=9，将训练 epoch5–8。scratch 的骨干预训练选项仍来自明确选择的模型配置。

recover 只接受输出目录、设备、拓扑和新 attempt，沿用已记录配置；如果要改参数、换拓扑或延长已完成训练，使用 resume 和新输出目录。新运行只检查 batch 能被实际 world size 整除，不强制某个卡数。文件锁阻止同目录并发，旧 attempt 不覆盖。

旧固定协议运行不能直接用新 schema recover，但可用其完整 checkpoint 作为新 resume 输入。旧 prepare_sgqs_v2_run.py 仅保留为历史协议工具，不是通用入口的依赖。运行时记录实际源码与标注哈希；图像目录应维持同一数据版本，目前不逐图冻结图像内容。

## 每步与每轮统计

`attempts/NNNN/epochEEEE.steps.jsonl` 每次 optimizer 尝试记录：绝对 epoch/step/global_step、所有加权 loss 原名、诊断指标、各参数组更新前后 LR、AMP scale/跳步、裁剪前总梯度范数和耗时。DN、DDF、Encoder、Decoder 各辅助层损失不折叠。梯度范数是 rank0 同步参数的总梯度范数；溢出记 null，保留跳步事实。

v2 另外记录未加权 dense/bbox/giou/allocation 与统一升温系数；GT 覆盖缺口、可靠代表、保护/筛选各门、接收者、实际 pair、无 pair 图、overflow 数，以及 score/gap 原始和。保存 owner 过滤前后数量和本地 small 的留存数量。训练内 small 指增强后归一化框映射到该次运行声明的输入尺寸，不能混称未增强验证 GT。

每个 rank 的 `epochEEEE.rankRR.samples.jsonl` 记录主样本 image_id。mosaic 可能额外取图，主 image_id 不是完整增强配方。

`epochEEEE.summary.json` 区分：

- `global_counts`：已跨 rank 汇总的数量/原始和，沿时间求和，不再乘 world size。
- `population_rates`：全轮分子和 / 全轮分母和，空分母为 null。例如不能以各步 coverage 均值替代全轮 covered_GT / GT。
- `step_means`：loss 及其余逐步量的时间均值。保留的逐步比率均值只描述时间走势，群体结论使用 population_rates。
- `warnings`：无有效分配、AMP 跳步等事实。零 allocation loss 不等于训练成功。

逐步 seconds 不含等待下一批数据和当行写盘；全轮 epoch_wall_seconds 包含数据加载与记录，另报实际 optimizer_updates，用于比较训练开销。

JSONL 每行 flush，主逐步记录每 50 步及轮末 fsync；突然断电不保证最后 fsync 之后的行。轮末统计也嵌入 checkpoint。异常文件记录 rank、epoch、step、主 image_id 和错误，NaN 不写入正常 checkpoint。

附加 loss 的有限性跨 rank 一致检查后才做指标汇总，总 loss 在 backward/optimizer 前检查。AMP 跳步时 EMA 不更新；LR 和升温继续沿绝对 attempted-step 时钟前进。各 rank 跳步不同则终止，拒绝保存不同步状态。

## checkpoint 与恢复

新运行先保存初始化边界：scratch/tuning 为 checkpoint_init.pth，resume 为 checkpointEEEE.pth（实际加载的 epoch）。之后按 --checkpoint-every 保存，验证轮与最后一轮总会保存；不自动删除旧 checkpoint。每份包含 raw、optimizer、scaler（关闭 AMP 时为空状态）、启用时的 EMA、原生 scheduler 状态、last_epoch、各 rank RNG、下一绝对 global_step、run 身份、轮末统计及 attempt 起点。

支持配置的原生 scheduler 或 flatcosine，也可不配置 scheduler。原生状态从 checkpoint 还原，flatcosine 按配置/绝对步数重建。没有“指标下降就重载 best”或重设 EMA decay；数据增强阶段切换继续遵循所选配置。

写入：临时文件 → flush/fsync → 原子 rename → SHA sidecar → latest.json。失败不覆盖上一完整轮；恢复选择最高完整命名的 checkpoint，校验失败不静默退回旧轮。如果 rename 成功而 sidecar 尚未写入，可通过内嵌身份接管完整文件。best_raw.json / best_ema.json 是带 epoch、分数、checkpoint SHA 的指针，记录已配置预算下整体、本地 small 及实际启用子群的 AP；EMA 关闭时不生成 EMA 文件。

**恢复单位是 epoch 边界。** 每轮先存训练状态、后验证。验证中断只补未完成验证，不重训该轮；训练中断重做最近 checkpoint 之后的轮，重做范围由保存间隔决定。旧 attempt 的未提交/重做日志不能重复计入新运行统计，checkpoint 中的 attempt 与 attempt_start_epoch 用于追溯实际提交区间。worker 缓存、增强和 GPU 算子影响数值复现，不承诺任意环境的逐 bit mid-step 恢复。

## 完整验证及预测留档

--eval-initial 决定是否验证起点；--eval-every N 决定周期，非零时最后一轮也验证，0 表示不做后续周期验证。每次保存 raw；--ema 开启时再验证 EMA。--eval-budgets 决定全部评价预算，不预过滤分数。COCO 各尺度、本地 small 均为严格一对一 AP/AR，并给出 AP75/AP95。

本地 small 的面积/短边阈值仍为 32² / 16，映射坐标系来自该次显式输入尺寸；COCO-small 独立报告。只有显式 --dense-min-gt N 才生成对应高密度子群，空子群为 null，不再默认绑定 301。

`validation/epochEEEE/{raw,ema}/` 中，每 128 图保存压缩 NPZ。临时 hook 从同次前向捕获真实 Encoder TopK，校验 source index、anchor 和初始回归框一致，不额外执行 bbox head，不改变推理输出。rank0 checkpoint 的 BN/EMA buffers 临时广播到各 rank，验证后恢复各 rank 原缓冲，避免一个指标混用不同状态；训练 RNG 同样隔离并恢复。

| 字段 | 内容 |
|---|---|
| image_ids / orig_sizes | 图像 ID、原尺寸，遵循仓库 width,height 顺序 |
| dense_scores | 全部 dense 单类 sigmoid 分数，无效 anchor 为 0 |
| source_indices | 实际进入 Decoder 的源索引，保持 query 对应 |
| encoder_boxes / encoder_scores | 入选框归一化 cxcywh、Encoder 分数 |
| decoder_boxes / decoder_scores | 对应 query 最终框和分数，未过滤 |
| predictions | PostProcessor 像素 xywh，列为 image_id,x,y,w,h,score,category |

正式验证按 rank 对图像不补齐分片，避免重复图和近似同分排序差异；合并器仍防御性检查重复预测一致和完整 val 覆盖。分片附 SHA；所有分片及 AP 成功后才写 complete.json，绑定 checkpoint SHA、run 身份、指标和分片。恢复仅复用校验成功的 complete；部分分片留在旧 attempt，新 attempt 重做该状态验证。completed.json 表示训练和配套验证完成。

现有留档支持离线定位/排名、同 GT 堆叠、Encoder→Decoder 去向、阈值过滤、所选预算间的差异和图像子群 AP 分析。**不包含未入选候选的回归框、Decoder 中间激活、逐 loss 参数梯度**，不能用 loss 数值证明梯度争夺。实际减少 Decoder K、反事实训练机制仍需专门实验，不能以截断原有输出代替。

## 布局与成本

```text
run/
  resolved.yml, run_manifest.json
  checkpoint_init.pth, checkpointEEEE.pth
  checkpointEEEE.json, latest.json
  best_raw.json, best_ema.json, completed.json
  attempts/0000/start.json
  attempts/0000/epochEEEE.{steps.jsonl,summary.json}
  attempts/0000/epochEEEE.rankRR.samples.jsonl
  attempts/0001/…                           # 恢复记录独立保存
  validation/epochEEEE/raw/complete.json
  validation/epochEEEE/raw/attemptNNNN/rankRR.chunkCCCCC.npz
  validation/epochEEEE/ema/…
```

磁盘需求由保存轮数、验证间隔、EMA 开关、图像数和 query 数决定。例如 640 输入/640 query 时每图每状态数组约 100 KB 未压缩。配置更多验证状态会增加时间和空间；压缩比例依赖数据。输出目录需支持文件锁、原子 rename。

## 验证范围

测试覆盖原子写失败、RNG 还原、群体分母、预测覆盖/去重/校验、hook 不改变真实 D-FINE 输出、保存后验证前中断恢复，以及真实小型 D-FINE/SGQS v2 后续训练的 raw/EMA 一致性；另运行双进程 Gloo 分配测试，并测试真正通用入口从 epoch0 启动、关闭 AMP/EMA、原生调度器、不同预算/保存间隔和中断恢复。当前机器未执行正式八卡 NCCL 和真实数据完整训练，测试不代表精度、吞吐或云端验收。
