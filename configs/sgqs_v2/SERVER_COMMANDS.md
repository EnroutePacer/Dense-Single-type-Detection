# 通用 SGQS 训练命令

2026-10-28 兼容修复：同步完整源码后可保持现有启动参数，不需要安装、升级或编译 COCO 包。启动优先使用现有 `faster_coco_eval`，其次使用现有 `pycocotools`；两者均不可用时，自动使用仓库内置的 COCO bbox 参考实现。后者保持匹配/指标口径，但 CPU 评估较慢；当前 bbox 训练无需环境修改。后端名称/版本写入评价结果，不能用更换后端的耗时判断模型速度。分割/RLE 不在内置回退支持范围内。

CUDA AMP 同时兼容旧版 `torch.cuda.amp.GradScaler` API。日志中 TensorFlow 的重复 factory 注册提示不是此次启动退出的根因；已确认的根因是缺失 `faster_coco_eval`。本次更新同时改变 source coverage 训练语义，已有成功运行若要切换新代码，应按下文 `resume + 新目录` 建新运行；不绕过原运行的源码一致性检查。

新运行的 `--output-dir` 可以不存在，也可以是调度器预建的空目录。为了支持 `mkdir -p "$RUN_DIR"` 加 `tee "$RUN_DIR/console.log"`，目录中仅有 `console.log` 也视为全新。任何其他文件（尤其是 `resolved.yml`、manifest 或 checkpoint）都会继续 fail-closed，必须使用 `recover` 或换新目录，不会覆盖旧实验。

入口不再自动选择实验臂、GPU 数、数据路径、checkpoint、训练轮数或 seed。运行参数必须通过命令行显式给出；模型、增强和损失细节由 `-c` 指定，`-u key=value` 覆盖。启动后会保存原始命令参数、最终配置、输入/代码哈希。下面的数值是示例，可直接修改，不是入口的限制。

## 从零创建训练状态

```bash
bash tools/table_cells/run_sgqs_v2.sh \
  --python python --nproc-per-node 2 --gpus 0,1 --master-port 29501 \
  --device cuda --mode scratch \
  -c configs/sgqs_v2/train.yml --output-dir outputs/my_sgqs_run \
  --train-images /data/images --train-annotations /data/annotations/instance_train.json \
  --val-images /data/images --val-annotations /data/annotations/instance_val.json \
  --epochs 100 --batch-size 16 --val-batch-size 16 --workers 4 --seed 42 \
  --amp --ema --input-size 640 640 --num-queries 512 \
  --eval-budgets 300 512 --checkpoint-every 1 --eval-every 1 --eval-initial \
  --sgqs-start-epoch 0 --sgqs-ramp-epochs 1 \
  -u optimizer.lr=0.0001 lrsheduler=flatcosine lr_gamma=0.1 \
     flat_epoch=50 no_aug_epoch=0 warmup_iter=500
```

`--batch-size` 和 `--val-batch-size` 是全局 batch，必须可被进程数整除。`--epochs` 是绝对总轮数，例如 100 表示 epoch0–99。进程数、GPU 列表和 master port 也显式设置。原生 AdamW 的参数组可能有单独 LR；`optimizer.lr` 只覆盖默认组，组级配置通过 `-u optimizer.params=...` 明确调整。

`--input-size H W` 设置数据管线的 Resize、名义 collate base size 和验证尺寸；若所选配置启用多尺度训练，实际训练尺寸仍遵循显式配置的 scales/final_size，不会静默删除增强。v2 的 small 统计映射到此处声明的输入尺寸。

--no-amp、--no-ema 可关闭相应机制。--eval-every 0 关闭周期验证；--no-eval-initial 关闭起点验证。每个验证边界及最终轮都会保存 checkpoint，其余轮按 --checkpoint-every 保存。启用 EMA 时每次验证 raw/EMA，否则只有 raw。想报告高密度子群时显式加 `--dense-min-gt 301` 或其他阈值；省略时不预设该子群。

CPU 单进程可直接运行 `python tools/table_cells/train_sgqs_v2.py`，传入上述训练参数，改为 `--device cpu --no-amp`，无需 shell 的 GPU/进程/端口参数。CPU 多进程可使用 shell 的 `--device cpu --nproc-per-node N --master-port PORT`，不传 --gpus，并显式配置 `-u sync_bn=False`（PyTorch 的 SyncBatchNorm 不支持 CPU DDP）。外部多机 torchrun 也可以直接调用 Python 入口，但各 rank 必须共享相同输出目录和源码。

## 权重初始化与从任意 checkpoint 续训

仍使用完整的新运行命令，按目的替换模式和输出目录：

```text
--mode tuning --checkpoint /weights/model.pth --weights-state raw --output-dir outputs/new_tuning_run
--mode resume --checkpoint /runs/old/checkpoint0049.pth --output-dir outputs/new_resume_run
```

tuning 必须显式选择 --weights-state raw 或 ema，仅加载匹配权重，optimizer/时钟从头开始；resume 加载完整训练状态。若从 epoch49 续训到 epoch79，显式传 --epochs 80。v2 升温起点同样是绝对 epoch，必须显式指定，不会自动改成 28。启用 EMA/AMP 的完整 resume 需要 checkpoint 中有对应状态；只有模型权重时选择 tuning。

改参数、改变 world size、延长已完成训练，用 resume + 新目录；旧结果不覆盖。若要纯随机初始化，还应通过配置或 `-u HGNetv2.pretrained=False` 关闭骨干预训练。

## 恢复同一个中断运行

```bash
bash tools/table_cells/run_sgqs_v2.sh \
  --python python --nproc-per-node 2 --gpus 0,1 --master-port 29501 \
  --device cuda --mode recover --output-dir outputs/my_sgqs_run --attempt 1
```

恢复时沿用运行内 resolved.yml，不再指定 config、epochs、数据、AMP 等新参数；拓扑和设备类型必须匹配。再次中断使用新 attempt（2、3…）。这是保证恢复语义，不限制新实验参数。恢复脚本 `recover_sgqs_v2.sh` 只是上述命令添加 `--mode recover` 的薄包装，也要求显式运行目录、拓扑和 attempt。

训练中断重做最近完整 checkpoint 之后的轮；验证中断只补未完成验证。结构化记录保留在不同 attempts 中。控制台如需另存，可自行使用 `2>&1 | tee /path/to/console.log`，避免覆盖已有日志。

## 对照与固定实验

- B0：`-c configs/table_cells/dfine_hgnetv2_m_table_cell_fixed640.yml`，不传 SGQS 升温参数，其余运行参数显式给出。
- support：选择通用 v2 配置，额外 `-u SGQSV2Criterion.allocation_weight=0`。
- v2：选择通用 v2 配置，使用完整附加方法。

固定的八卡、epoch27→34 配方移到 [EXP-TRAIN-008 示例](recipes/EXP-TRAIN-008.md)。历史配置和准备工具保留供追溯，不是通用入口前置条件。

完整统计、checkpoint、预测留档及恢复边界见 [框架 README](../../engine/sgqs_v2/README.md)。历史 `evaluate_sgqs_v2_matrix.py` 仍是 EXP-TRAIN-008 专用的实际 K 回放工具，不是任意新运行的通用命令；本轮通用训练内的验证已随 CLI 预算/尺寸运行。
