# SGQS-E 实现与运行

状态：implemented / unvalidated。以下测试验证代码和数学约束，不代表真实数据集收敛或 AP 结论。

## 实现边界

| 位置（相对仓库根） | 改动 |
|---|---|
| `engine/sgqs/edge_localization.py` | 一次构建增强后像素近邻图；固定 kappa=1/2；逐边有界补偿；支持显式续训课程起点/持续时间 |
| `engine/deim/deim_criterion.py` | 在原 loss_fgl 中替换权重，沿用真实 GO/DN 对应；暴露实际 Encoder 监督索引 |
| `engine/sgqs/criterion.py` | 可选非 owner 预算门槛、原 Encoder 监督保护、合并 EFGL 日志 |
| `engine/sgqs/candidate_supervision.py` | 提供真实 mask 后的选择分数，并校验有效候选数量 |
| `engine/solver/det_engine.py` | 把 samples 的实际 H/W 与已有 epoch/step 一并传给损失 |
| `engine/core/yaml_utils.py` | 消除多次配置加载共用默认字典的问题 |
| `tests/test_sgqs_e.py` | 数学边界、梯度、配置隔离、原生训练入口与推理 trace 验证 |

以上源文件、测试和新配置同步维护到 `article/` 的兼容运行镜像。根目录是 canonical 训练入口，历史 portable 配置差异不在这次变更中重写。

模型 forward、DN 生成函数、DDF/GO-LSD、Matcher、query 数、后处理没有新增方法分支。EFGL 是 criterion 内的训练功能，没有任何模型参数或 state_dict 键。使用 `efgl.enabled: False` 回到基线；旧 SGQS 配置保留旧门槛语义。

## 四个正式配置

| 配置（相对根） | SGQS | EFGL |
|---|---|---|
| `configs/sgqs_e/b0_baseline.yml` | 关闭 | 关闭 |
| `configs/sgqs_e/b1_sgqs.yml` | 非 owner 门槛 + 监督保护 | 关闭 |
| `configs/sgqs_e/b2_efgl.yml` | 关闭 | 开启 |
| `configs/sgqs_e/b3_sgqs_e.yml` | 同 B1 | 同 B2 |

B2、B3 共用 `configs/sgqs_e/efgl.yml`。全部继承现有 clean fixed-640 / K=640 控制配方，保持既有学习率与增强以隔离方法效果；本轮没有声称该控制配方已经是最佳训练日程。真实数据上的完整 DEIM/MAL + Dense O2O 复验是后续实验，不能用 clean VFL 的结果替代。

Linux 训练服务器上，设置实际数据与初始化权重后使用既有入口：

```bash
CONFIG_FILE=configs/sgqs_e/b3_sgqs_e.yml \
OUTPUT_DIR=./output/sgqs_e_b3_joint \
DATA_ROOT="$DATASET_ROOT" PRETRAINED="$INIT_CHECKPOINT" \
GPUS=0,1,2,3 SEED=0 \
bash tools/table_cells/train_dfine_m_table_cell_multigpu.sh
```

四组必须使用不同输出目录、相同初始化/seed/global batch/步数。`-r` 恢复时课程跟随恢复后的真实 epoch；`-t` 微调从新训练的 epoch 0 开始。EFGL 结束时间为 `epoches * compensation_fraction`，默认 120×.30=36 epoch，不能悄悄把恢复训练当作新的课程起点。自定义训练循环必须向 criterion 传 `epoch`、`step`、`epoch_step` 和真实 `input_size=(H,W)`，缺必需信息会报错。

## Epoch-27 continuation screen

`configs/e35_r7/` 另外预注册了一组明确标记为 screen 的短续训：
四臂都从 `last_epoch=27` 的同一 checkpoint 恢复，把绝对结束
epoch 设为 35，因此只执行 epoch 28--34。B0 已在外部运行，新入口
默认只启动 B1/B2/B3。

该 screen 不会隐式重置课程，而是在叶子配置显式声明：

```yaml
SGQSCriterion:
  survival_start_epoch: 28.0
  survival_ramp_epochs: 1.0
efgl:
  compensation_start_epoch: 28.0
  compensation_duration_epochs: 2.1
```

`2.1 = 7 × 0.30`，使 EFGL 在短续训的前 30% 线性退火。新字段的
默认值仍为从 epoch 0 开始、持续 `epoches×compensation_fraction`，
所以不改变正式从头训练配置。如果不显式加这个 offset，epoch 28 时
EFGL 已经归零，B2 会数值退化成 B0，B3 会退化成 B1。

这组结果只能作为 checkpoint continuation 的低成本打靶，不能替代
EXP-TRAIN-004 的同初始化、从头训练、多 seed 正式消融。

本屏选的规范数学与实现基准固定为
`6fd2c301bd8c09ff3d49fa077c34eaf75ef35b9a`。除
`engine/sgqs/edge_localization.py` 增加的课程时间原点/持续时间参数外，
方法关键文件与该提交按 canonical-LF SHA-256 完全一致。该唯一差异只实现

\[
c_{r7}(e)=\max\left(0,1-\frac{e-28}{7\times0.30}\right),\quad e\ge 28,
\]

而逐边权重、owner、SGQS 四项损失及系数、Matcher、GO/DDF、DN 均不变。
默认 `start=0,duration=null` 时仍严格回到提交中的
`c(e)=max(0,1-e/(T*0.30))`。机器可核验记录见
`configs/e35_r7/method_provenance.yml`；启动器在训练前校验方法快照和最终
解析配置，不能仅凭叶子 YAML 文件名宣称实验身份。

正式四组计划登记为 [EXP-TRAIN-004](../../experiments/EXP-TRAIN-004_sgqs-e-b0-b3/README.md)，当前 runs 为空。EXP-TRAIN-003 是历史 M-001 计划，不作为本版 B1 的已完成对照。

## 日志

- `efgl_weight_bonus_mean`：主普通分支相对原 IoU 权重的平均补偿，按边计数跨卡归一化；不包含 DN 分支统计。
- `efgl_compensation_ramp`：当前 c(t)。
- `sgqs_conflicting_owner_fraction`：被一致性保护跳过的 owner 比例。
- 原 SGQS 日志继续报告实际 Top-K 的存活率和 cutoff gap。`owner_coverage` 是保护后的有效 owner 覆盖，`assignment_miss_rate` 仍是原在线分配失败率；两者差异包含冲突保护。

日志通过 `get_logging_metrics()` 返回，不添加到 loss 字典，不参加反向传播。教师 KL 保持原权重；EFGL 不会因为学生低 IoU 而盲目放大教师监督。

## 验证

在具有项目依赖的环境中，从仓库根执行：

```bash
python -m unittest discover -s tests -v
python -m unittest discover -s article/tests -p test_repository_contract.py -v
python article/scripts/maintenance/check_markdown_links.py
```

本轮 CPU 测试覆盖：

- 原 11 项 SGQS 测试继续通过。
- K=A 的自追逐反例、新门槛的排名充分条件、无效候选边界。
- 非 owner 原 Encoder 正点保护、冲突 owner 屏蔽、保留 owner 框头梯度。
- 小框零 IoU 时的直接 FGL 梯度、邻域方向性、共享边界与重复框、跨类别隔离、320 GT 分块构图。
- 补偿退出后所有原损失逐项相等，分支开关和非法参数检查。
- 两层真实 D-FINE Decoder + DN + VFL/MAL 两步优化，包括部分空图与全空 batch；原生 `train_one_epoch` 入口。
- 同权重下 DN/普通输出与 eval 输出逐元素一致；微型 Decoder 的 TorchScript trace 算子种类序列和输出一致。
- 完整 HGNetv2-M、640×640、K=640 的 B0/B3 同权重 CPU eval：19,468,325 个参数、1,053 个 state_dict 键相同；分类与框输出最大绝对差均为 0。
- 四组 YAML 实例化与连续配置读取隔离。

共 29 项方法测试和 4 项仓库契约测试通过；`article/` 镜像运行同一套方法测试也通过，镜像结果不重复计为独立测试。验证环境为 PyTorch 2.13.0+cpu / torchvision 0.28.0+cpu。尚未进行 CUDA/AMP、多卡真实训练、全模型 ONNX/TensorRT 导出、部署延迟或数据集 AP 测量。TorchScript trace 只验证该微型 Decoder 的实际 eval 计算，不替代目标部署后端验证。

## 源码快照范围

旧 `docs/reproducibility/source_snapshot.sha256` 和 `code_provenance.yaml` 保留为历史记录。旧清单混用了 LF/CRLF 字节哈希；即使统一换行，审阅基准 `927b2c6` 中 `engine/data/dataloader.py`、`engine/data/transforms/_transforms.py` 和 `engine/deim/deim_criterion.py` 也已偏离该历史清单。本轮没有把旧清单重写成“全部通过”。

本版使用 [sgqs_e_source_snapshot.sha256](../../docs/reproducibility/sgqs_e_source_snapshot.sha256)，记录根目录及 article 镜像的源码、配置、测试、运行工具和依赖文件，路径相对仓库根。它统一按 UTF-8 文本的 LF 换行计算 SHA-256，保留其他字节内容；不包含数据、checkpoint 或实验结果。从仓库根校验：

```powershell
./article/scripts/reproduce/verify_source_snapshot.ps1 -RepositoryRoot . -ManifestPath article/docs/reproducibility/sgqs_e_source_snapshot.sha256 -CanonicalLF
```
