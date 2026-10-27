# EXP-TRAIN-008：固定配方示例

本文件是实验协议的调用示例，不是训练框架的默认值。按原协议核对共同 checkpoint 确为 epoch27、数据版本一致；新运行 manifest 记录真实输入哈希。历史冻结身份仍可在 `configs/e35_r7/frozen_inputs.yml` 追溯。

```bash
bash tools/table_cells/run_sgqs_v2.sh \
  --python python --nproc-per-node 8 --gpus 0,1,2,3,4,5,6,7 --master-port 29500 \
  --device cuda --mode resume \
  --checkpoint /weights/checkpoint0027.pth \
  -c configs/sgqs_v2/v2_e28_r7.yml --output-dir outputs/sgqs_v2/v2_e28_r7_seed0 \
  --train-images /data/images --train-annotations /data/annotations/instance_train.json \
  --val-images /data/images --val-annotations /data/annotations/instance_val.json \
  --epochs 35 --batch-size 64 --val-batch-size 64 --workers 4 --seed 0 \
  --amp --ema --input-size 640 640 --num-queries 640 --eval-budgets 300 640 \
  --dense-min-gt 301 --checkpoint-every 1 --eval-every 1 --eval-initial \
  --sgqs-start-epoch 28 --sgqs-ramp-epochs 1
```

support 将 config 改为 support_e28_r7.yml、使用独立输出目录；B0 改为 b0_e28_r7.yml 并去掉两个 SGQS 升温参数；可选原 sgqs 使用 sgqs_e28_r7.yml，也去掉 v2 升温参数。每个臂均显式从共同 checkpoint 启动，没有 shell 内置实验分派。

保留最小三臂、同预算和原预注册终点，不将框架通用化当作新的研究结果。方法参数仍见 [M-003](../../../article/methods/M-003_sgqs_v2/README.md)；通用参数说明见 [SERVER_COMMANDS](../SERVER_COMMANDS.md)。
