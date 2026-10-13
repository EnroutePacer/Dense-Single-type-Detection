# EXP-TRAIN-005：Epoch-27 SGQS-E 七 epoch 续训屏选

状态：partial / partial。B0、B1、B2 已完成；B1/B2 的完整 console 附件于
2026-10-13 归档，B3 尚无日志。三组均缺可用的历史 at-start manifest，
不把后续 preflight 或本轮 ingestion provenance 伪装成运行前证据。

最新入口：[B1/B2 配对诊断](diagnostics/20261013_b1_b2/README.md)、
[原设计、损失与问题的数学说明](diagnostics/20261013_b1_b2/math_diagnosis.md)。

## 设计

四臂都从同一 `checkpoint0027.pth` 严格 resume，设 `epoches=35`，实际执行
epoch 28--34，共 7 epoch。固定 640x640、query 640、8 卡×每卡 8、
global batch 64、seed 0 和 maxDets=300。计划的数据与初始权重 SHA-256 见
`configs/e35_r7/frozen_inputs.yml`，实际执行身份应与运行前 manifest 核对。

| 臂 | 新增机制 | 课程 |
|---|---|---|
| B0 | 无 | 已完成，禁止重复启动 |
| B1 | SGQS | survival 在 epoch 28 开始，1 epoch ramp |
| B2 | EFGL | epoch 28 开始，2.1 epoch 线性退火 |
| B3 | SGQS + EFGL | 与 B1/B2 分别相同 |

EFGL 的 2.1 epoch 是续训长度 7 的 30%。如果继续使用默认“从
epoch 0 起算”的课程，到 epoch 28 补偿已经是 0，B2/B3 消融会退化。

方法规范基准为完整提交
`6fd2c301bd8c09ff3d49fa077c34eaf75ef35b9a`。实验允许的唯一方法代码差异
是把同一 EFGL 退火公式的时间原点平移到 epoch 28，并把持续时间设为
`7×0.30=2.1`；损失与权重公式本身不变。精确文件哈希和不变量由
`configs/e35_r7/method_provenance.yml` 冻结，实验身份须独立核对。
远程 `4844254` 已将 launcher 改为通用运行入口，不再内嵌 e35/r7 专用契约；
不能把当前入口或事后的冻结清单当作这些历史运行已通过契约的证据。

## 执行

当前统一使用 `tools/table_cells/run_table_cell_experiment.sh`。launcher 不按实验名
选择配置，B1/B2/B3 的命令必须分别显式声明 `CONFIG_FILE`、`RESUME`、
`GPUS`、batch、绝对终点 epoch、输出目录和其余运行参数。冻结命令见
`configs/e35_r7/SERVER_COMMANDS.md`。把同一命令中的
`PREFLIGHT_ONLY=0` 改为 `1` 可做只读预检；成功时不会创建输出目录或启动
`torchrun`。

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

## 已完成的 B1/B2 与诊断

| epoch 34 | B0 | B1 SGQS | B2 EFGL |
|---|---:|---:|---:|
| EMA mAP | 0.866667089 | 0.867608455 | 0.866626083 |
| AP-small | 0.685633491 | 约 0.682 | 约 0.685 |
| AR-small | 0.690892250 | 约 0.686 | 约 0.690 |

B1/B2 小目标指标仅有 console 三位小数精度；共同恢复点评价也是 EMA，
不是 raw 模型。B1 的 AP-small 在 epoch 30 达到约 0.706，终点收益未保持；
B2 终点基本等同 B0。SGQS 冲突保护大面积撤销新增 owner 监督，97.78%
生存率只以过滤后的集合为分母；EFGL 的平均补偿很弱且在 epoch 30.1 结束。
这些是优先验证的机制问题，不是已证明的唯一因果解释。

原始日志位于 [B1](../../../outputs/e35_r7_sgqse/resume_b1_sgqs_seed0/README.md)
与 [B2](../../../outputs/e35_r7_sgqse/resume_b2_efgl_seed0/README.md)。
缺失的服务器 `log.txt`、checkpoint 和历史 manifest 不补造；不依据目录名
推断 B3 已运行。新增证据登记为 ART-000028--ART-000037，三组历史执行仍缺
独立 RUN 元数据，因此注册表保持 `run_migration_status=unmapped`。

## 证据边界

这个 screen 使用已经收敛 28 epoch 的共同基座，可用来快速判断新梯度在
当前状态附近的方向，但无法回答方法是否改变早期表征和候选生成。
它只用于决定是否值得执行 EXP-TRAIN-004，不能代替正式从头、多 seed 消融。

旧 `targeted_e28_r7_training_bundle.tar.gz` 的 small-reg、MAL-only、
Mosaic-only 和 MAL+Dense-O2O 四组已重写为 C1--C4 辅助诊断对照。它们与
B0--B3 共享 checkpoint、预算和评估协议，但不进入 SGQS-E 的主效应/交互
计算，也不能替代 B1/B2/B3。
