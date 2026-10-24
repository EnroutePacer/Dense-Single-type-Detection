# SGQS v2 服务器启动命令

在仓库根目录、现有训练 Python 环境执行。入口不安装依赖。固定 8 张可见 GPU、每卡 8 图、global batch 64；预检不允许静默更改拓扑、数据、共同初始化或追加已有运行。

## 1. 启动完整新方案

```bash
export PY=python
export DATA=/aicamera-mlp/chenkai/data/table_det_data/paddlex_l_cell_merged
export CKPT="$PWD/output/dfine_hgnetv2_m_table_cell/checkpoint0027.pth"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 默认只执行 v2；从冻结 epoch27 恢复，训练 epoch28..34。
bash tools/table_cells/run_sgqs_v2.sh
```

输出为 `outputs/sgqs_v2/v2_e28_r7_seed0/`。准备工具验证 checkpoint 含 model、EMA module、optimizer、scaler，核对数据标注和 checkpoint 的冻结哈希；每个臂独立从同一个 checkpoint 恢复。

最小主比较明确选择三臂：

```bash
bash tools/table_cells/run_sgqs_v2.sh b0 support v2
```

若默认命令已跑过 `v2`，只补 `b0 support`，不要重复运行 `v2`。可选历史原法参照：

```bash
bash tools/table_cells/run_sgqs_v2.sh sgqs
```

`support` 与 `v2` 仅 allocation 权重不同；`sgqs` 保留旧 v1 的 dense/owner 无升温语义，不能当作仅去掉 allocation 的控制。

运行目录已存在会拒绝覆盖。补跑失败运行应设置新目录，例如：

```bash
RUN_ROOT="$PWD/outputs/sgqs_v2_retry1" bash tools/table_cells/run_sgqs_v2.sh v2
```

不要用失败臂产生的中间 checkpoint 作为其他臂初始化。如果部署目录没有 `.git`，设置 `REV` 为同步方提供的完整 40 位提交号；有 Git 时记录实际 HEAD、工作区状态与运行文件哈希。`runtime_files.sha256` 不匹配时应同步完整源码，不能绕过检查。

单臂的底层命令等价于：

```bash
RUN="$PWD/outputs/sgqs_v2/v2_e28_r7_seed0"
"$PY" tools/table_cells/prepare_sgqs_v2_run.py \
  --arm v2 --resume "$CKPT" --data-root "$DATA" --output "$RUN" --workers 4
"$PY" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 \
  tools/table_cells/train_sgqs_v2.py -c "$RUN/resolved.yml" \
  2>&1 | tee "$RUN/console.log"
```

专用训练入口显式传入 `--seed 0 --use-amp`，拒绝任意额外 `-u`、`--tuning` 等覆盖。新增分支从绝对 epoch 28 开始，一轮内统一升温；不会因工具重启把时钟从零重算。此入口用于从共同 epoch27 开始的独立七轮比较，已有运行不允许原地恢复追加。

保留 `console.log`、`log.txt`、`resolved.yml`、`run_manifest.json` 和每轮 checkpoint。新诊断由 criterion 记录到训练日志，包含覆盖缺口、门控/保护、rescue/prefix 配对、分数间隔和升温状态。它没有旧 `probes/train.rank*.jsonl` 或 `probe_steps.csv`，无需执行旧 `summarize_probe.py`。

## 2. 完整验证集主评价，固定 K=640

每臂 × epoch28..34 × raw/EMA。三臂共有 42 个模型状态；每个状态从完整 640 输出计算 @640 主结果与 @300 附加结果，不预过滤 small、score 或图像子集。

```bash
"$PY" tools/table_cells/evaluate_sgqs_v2_matrix.py \
  --run-root "$PWD/outputs/sgqs_v2" --arms b0 support v2 \
  --epochs 28 29 30 31 32 33 34 --states raw ema --query-budgets 640 \
  --annotation "$DATA/annotations/instance_val.json" --images "$DATA/images" \
  --output "$PWD/outputs/sgqs_v2_full_val" --device cuda
```

只跑了新方法时，改为 `--arms v2`；原 `sgqs` 参照可显式加入。所有请求的 checkpoint 必须存在，不会静默跳过缺失状态。

主结果包含整体和 COCO-small/medium/large，以及本地 hybrid-small 的严格一对一 AP/AR、AP75/AP95，并报告按固定标注定义的 GT 数 ≥301 高密度场景。它们不能用独立 best-IoU coverage 替代。完整矩阵、预测身份与环境写入评估目录，只有完整成功才写完成标记。

中断后原命令追加 `--skip-completed`；只复用权重、图像内容、配置、源码、K 与后端环境一致的结果。更改任何请求坐标请使用新的评估目录。

## 3. 实际 Decoder 输入预算比较

先用 e34 比较实际 K；必要时再显式增加其他 epoch，不默认扩大训练矩阵。

```bash
"$PY" tools/table_cells/evaluate_sgqs_v2_matrix.py \
  --run-root "$PWD/outputs/sgqs_v2" --arms b0 support v2 \
  --epochs 34 --states raw ema --query-budgets 640 512 384 256 \
  --annotation "$DATA/annotations/instance_val.json" --images "$DATA/images" \
  --output "$PWD/outputs/sgqs_v2_query_budgets" --device cuda
```

这是每个 K 改变 Decoder 输入后重新前向，主输出评价上限固定为 640；K=256 时只有 256 个实际输出，不能宣称与原 640 前向后截断等价。脚本拒绝不兼容的 learned-query-content 模式，检查实际输出 query 数，并分别标识各 K 的缓存。

同步的模型 forward 耗时排除图像 I/O，不等于部署端完整延迟。必须同时检查整体与固定困难场景的 AP/AR，不能仅以候选数或均分报告收益。本轮尚无真实训练、GPU 性能或多 seed 稳定性结论。