# SGQS v2 附加训练框架与留档契约

本目录提供训练时的几何支持、候选分配，以及独立训练生命周期。它复用 DEIM 模型、数据、原始损失和 optimizer，拥有自己的启动校验、训练循环、验证、统计与恢复逻辑；不修改推理结构，不接入旧监督探针包装器。B0、support、完整 v2、可选旧 SGQS 使用同一记录和验证策略。

方法公式见 [M-003](../../article/methods/M-003_sgqs_v2/README.md)，命令见 [SERVER_COMMANDS](../../configs/sgqs_v2/SERVER_COMMANDS.md)。本次完善运行和证据保存能力，不改变已确定的 IoS、接收者、溢出门控和损失公式，没有新的精度结论。

## 范围与启动

注册协议：冻结 epoch27 完整状态，续训 epoch28–34；Linux 单机 8 GPU、每卡 8 图、seed 0、AMP/EMA、640 query。它不是任意拓扑的通用训练产品。入口核对初始化、标注、配置、运行源码的 SHA256；恢复要求同一准备结果、代码和拓扑。图像目录须保持同一版本：当前冻结标注哈希，不逐张锁定训练图像内容。

```bash
bash tools/table_cells/run_sgqs_v2.sh                 # 默认 v2
bash tools/table_cells/run_sgqs_v2.sh b0 support v2   # 最小公平比较
# 恢复同一运行，不重新 prepare；再次中断时递增 ATTEMPT。
RUN="$PWD/outputs/sgqs_v2/v2_e28_r7_seed0" ATTEMPT=1 \
  bash tools/table_cells/recover_sgqs_v2.sh
```

Linux 文件锁防止同一目录同时训练，进程退出后自动释放。已有 attempt、checkpoint、分片不覆盖。旧 schema 的运行不直接套用新恢复机制，应保留原结果并新建运行。

## 每步与每轮统计

`attempts/NNNN/epochEEEE.steps.jsonl` 每次 optimizer 尝试记录：绝对 epoch/step/global_step、所有加权 loss 原名、诊断指标、各参数组更新前后 LR、AMP scale/跳步、裁剪前总梯度范数和耗时。DN、DDF、Encoder、Decoder 各辅助层损失不折叠。梯度范数是 rank0 同步参数的总梯度范数；溢出记 null，保留跳步事实。

v2 另外记录未加权 dense/bbox/giou/allocation 与统一升温系数；GT 覆盖缺口、可靠代表、保护/筛选各门、接收者、实际 pair、无 pair 图、overflow 数，以及 score/gap 原始和。保存 owner 过滤前后数量和本地 small 的留存数量。训练内 small 指增强后归一化框映射到 640，不能混称未增强验证 GT。

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

先保存本地 `checkpoint0027.pth`，此后保存 0028…0034，共八份，全部保留。每份含 raw model、EMA module/updates/before_start、optimizer、AMP scaler、可序列化 scheduler、last_epoch、各 rank 的 Python/NumPy/PyTorch/CUDA RNG、每轮步数、下一绝对 global_step、run 身份和轮末统计。flatcosine 按冻结配置与绝对步数重建，optimizer 保存实际 LR；恢复校验 loader 长度。没有指标下降自动重载 best、EMA decay 重设或时钟重置。

写入：临时文件 → flush/fsync → 原子 rename → SHA sidecar → latest.json。失败不覆盖上一完整轮；恢复选择最高完整命名的 checkpoint，校验失败不静默退回旧轮。如果 rename 成功而 sidecar 尚未写入，可通过内嵌身份接管完整文件。best_raw.json / best_ema.json 是带 epoch、分数、checkpoint SHA 的指针，分别记录整体、本地 small、高密度本地 small 的 AP@640。

**恢复单位是 epoch 边界。** 每轮先存训练状态、后验证。验证中断只补未完成验证，不重训该轮；训练中断重做未完成轮，部分日志留在旧 attempt，不能并入新完整轮。worker 缓存、增强和 GPU 算子影响数值复现，不承诺任意环境的逐 bit mid-step 恢复。

## 完整验证及预测留档

epoch27 初始状态和 epoch28–34，均验证完整 val 的 raw/EMA，共 16 状态/臂。每次给出 @640 主结果、@300 附加结果，COCO 各尺度及本地 hybrid-small 的严格一对一 AP/AR、AP75/AP95；另报告固定标注中非 crowd GT≥301 的高密度子群，空子群为 null。本地 small：640 下 area<32² OR short side<16。

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

现有留档支持离线定位/排名、同 GT 堆叠、Encoder→Decoder 去向、阈值过滤、300/640 差异和图像子群 AP 分析。**不包含未入选候选的回归框、Decoder 中间激活、逐 loss 参数梯度**，不能用 loss 数值证明梯度争夺。实际减少 Decoder K、反事实训练机制仍需专门实验，不能以截断 640 输出代替。

## 布局与成本

```text
run/
  resolved.yml, run_manifest.json
  checkpoint0027.pth … checkpoint0034.pth
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

磁盘需八份完整 checkpoint 加十六份全量验证留档。640 输入常有 8400 dense source；每图每状态数组约 100 KB 未压缩，1000 图×16 状态约 1.6 GB，再加 checkpoint。压缩比例依赖数据。每轮 raw/EMA 全量验证增加时间，这是用当轮留档减少后续重跑，不宣称零记录开销。输出目录需支持文件锁、原子 rename。

## 验证范围

测试覆盖原子写失败、RNG 还原、群体分母、预测覆盖/去重/校验、hook 不改变真实 D-FINE 输出、保存后验证前中断恢复，以及真实小型 D-FINE/SGQS v2 后续训练的 raw/EMA 一致性；另运行双进程 Gloo 分配测试。当前机器未执行正式八卡 NCCL 和真实数据完整训练，测试不代表精度、吞吐或云端验收。
