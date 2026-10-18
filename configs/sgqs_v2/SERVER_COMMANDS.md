# SGQS v2 服务器启动命令

在仓库根目录、现有训练 Python 环境执行。脚本不安装依赖。
默认使用 8 张可见 GPU；不足 8 张会拒绝启动。

```bash
export PY=python
export DATA=/aicamera-mlp/chenkai/data/table_det_data/paddlex_l_cell_merged
export CKPT="$PWD/output/dfine_hgnetv2_m_table_cell/checkpoint0027.pth"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 顺序执行四臂，每臂都从同一个 checkpoint0027.pth 独立恢复。
bash tools/table_cells/run_sgqs_v2.sh
```

只跑一个分支：

```bash
bash tools/table_cells/run_sgqs_v2.sh r2
```

可显式选择顺序，例如 `bash tools/table_cells/run_sgqs_v2.sh r0 r2 r3 r1`。
输出分别位于 `outputs/sgqs_v2/r{0,1,2,3}_e28_r7_seed0/`。
目录已存在会停止，不覆盖、不追加；补跑失败臂需设置新的 `RUN_ROOT`，
例如 `RUN_ROOT="$PWD/outputs/sgqs_v2_retry1" bash tools/table_cells/run_sgqs_v2.sh r2`。
不要把失败臂的后续 checkpoint 当作这四臂的共同初始化。

如果部署目录没有 `.git`，设置 `REV` 为同步方提供的**完整 40 位部署提交号**。
Git 目录中可省略；预检会记录实际 HEAD、工作区状态和运行文件哈希。
冻结输入不匹配会停止，请核对数据/权重来源，不要绕过校验。

每臂内部等价于：

```bash
RUN="$PWD/outputs/sgqs_v2/r2_e28_r7_seed0"
"$PY" tools/table_cells/prepare_sgqs_v2_run.py \
  --arm r2 --resume "$CKPT" --data-root "$DATA" --output "$RUN" --workers 4
"$PY" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 \
  tools/table_cells/train_sgqs_v2.py -c "$RUN/resolved.yml" \
  2>&1 | tee "$RUN/console.log"
"$PY" tools/table_cells/summarize_probe.py \
  --run "$RUN" --output "$RUN/probe_steps.csv"
```

专用训练入口从已校验配置显式传入 `--seed 0 --use-amp`，防止上游 CLI 默认值
覆盖 YAML；不接受额外 `-u`、`--tuning` 等配置覆盖。

保留 `console.log`、`log.txt`、`resolved.yml`、`run_manifest.json`、
`probes/train.rank*.jsonl`（含抽样 GT 行）、`probe_steps.csv` 和逐轮 checkpoint。
探针聚合必须有全部 8 rank，不能用 rank 0 均值代替。

## 同预算回放

训练内评估仍是原 maxDets=300。用同一个固定 train/val 子集回放每臂的
epoch 28--34、raw/EMA；现有工具在同一组预测上同时计算 @300 和 @640：

```bash
# 只创建一次。若已有 EXP-TRAIN-006 的固定子集 manifest，应复用原文件。
SUBSET="$PWD/outputs/sgqs_v2_fixed_subset"
"$PY" tools/table_cells/prepare_probe_subset.py \
  --train-ann "$DATA/annotations/instance_train.json" \
  --val-ann "$DATA/annotations/instance_val.json" --images "$DATA/images" \
  --output "$SUBSET" --images-per-split 128 --seed 20261014

for arm in r0 r1 r2 r3; do
  RUN="$PWD/outputs/sgqs_v2/${arm}_e28_r7_seed0"
  for epoch in 28 29 30 31 32 33 34; do
    printf -v CKNAME 'checkpoint%04d.pth' "$epoch"
    "$PY" tools/table_cells/replay_probe.py --config "$RUN/resolved.yml" \
      --checkpoint "$RUN/$CKNAME" --manifest "$SUBSET/manifest.json" \
      --output "$RUN/replay_e${epoch}" --device cuda --states raw ema
  done
done
```

128 图仅作固定子集诊断，不替代全验证集结果；small GT 过少时统一扩充子集后
重放全部臂。正式结论还需同标注哈希/同后端的全验证集 @300/@640 评估。
独立 best-IoU 覆盖不等于 COCO 一对一召回。
