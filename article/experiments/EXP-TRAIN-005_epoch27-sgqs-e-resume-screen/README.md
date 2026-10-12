# EXP-TRAIN-005：Epoch-27 SGQS-E 七 epoch 续训屏选

状态：planned / partial。B0 已完成并归档原始 console/JSON/job log；B1/B2/B3
仍未启动。B0 运行早于新 launcher，没有 at-start manifest，因此保留这一证据
边界，不把后续 preflight 伪装成它的历史 manifest。

## 设计

四臂都从同一 `checkpoint0027.pth` 严格 resume，设 `epoches=35`，实际执行
epoch 28--34，共 7 epoch。固定 640x640、query 640、8 卡×每卡 8、
global batch 64、seed 0 和 maxDets=300。数据与初始权重 SHA-256 见
`configs/e35_r7/frozen_inputs.yml`，启动脚本默认强制校验。

| 臂 | 新增机制 | 课程 |
|---|---|---|
| B0 | 无 | 已在跑，禁止重复启动 |
| B1 | SGQS | survival 在 epoch 28 开始，1 epoch ramp |
| B2 | EFGL | epoch 28 开始，2.1 epoch 线性退火 |
| B3 | SGQS + EFGL | 与 B1/B2 分别相同 |

EFGL 的 2.1 epoch 是续训长度 7 的 30%。如果继续使用默认“从
epoch 0 起算”的课程，到 epoch 28 补偿已经是 0，B2/B3 消融会退化。

方法规范基准为完整提交
`6fd2c301bd8c09ff3d49fa077c34eaf75ef35b9a`。实验允许的唯一方法代码差异
是把同一 EFGL 退火公式的时间原点平移到 epoch 28，并把持续时间设为
`7×0.30=2.1`；损失与权重公式本身不变。精确文件哈希和不变量由
`configs/e35_r7/method_provenance.yml` 冻结，launcher 的 contract preflight
必须通过后才能训练。

## 执行

使用 `tools/table_cells/train_e35_r7_sgqse_8gpu.sh`。默认只新启动 B1/B2/B3：

```bash
bash tools/table_cells/train_e35_r7_sgqse_8gpu.sh resume_b1_sgqs
bash tools/table_cells/train_e35_r7_sgqse_8gpu.sh resume_b2_efgl
bash tools/table_cells/train_e35_r7_sgqse_8gpu.sh resume_b3_sgqs_e
```

正式启动前分别加 `PREFLIGHT_ONLY=1` 运行同一命令。预检成功不会创建
输出目录或启动 `torchrun`。

每个输出目录必须是新目录。运行结束后把 `run_manifest.txt`、
`config_source.yml`、`resolved_config.yml`、`console.log`、`log.txt`、指标与必需
checkpoint 纳入 `output/` 的 Git/LFS 记录。

## 已完成的 B0 负对照

B0 的 console 明确记录：8 卡、每卡 batch 8、global batch 64、seed 0、
fixed-640、query 640、VFL、无 Mosaic/MixUp/多尺度、从
`checkpoint0027.pth` resume 到 epoch 34，并按 `maxDets=300` 评估。

| 时点 | mAP@50:95 | AP50 | AP-small | AR@300 |
|---|---:|---:|---:|---:|
| 恢复前 epoch 27 | 0.864310288258 | 0.880（console 三位数） | 0.678（三位数） | 0.870（三位数） |
| epoch 34 | 0.866667088526 | 0.880899235273 | 0.685633491224 | 0.871900060019 |

最终 mAP 相对恢复点增加 `+0.002356800268`。这只是 B1/B2/B3 的负对照
终点，不构成 SGQS/EFGL 的收益证据。原始材料见
`outputs/resume_control_e28_to_e34_gb64/`，ART-000024--ART-000027。

## 证据边界

这个 screen 使用已经收敛 28 epoch 的共同基座，可用来快速判断新梯度在
当前状态附近的方向，但无法回答方法是否改变早期表征和候选生成。
它只用于决定是否值得执行 EXP-TRAIN-004，不能代替正式从头、多 seed 消融。
