# EXP-DIAG-007 图片索引

- `hypothesis_trajectory.png`：SGQS/B0 AP-small、尺度 AP 差、整轮 loss 与 P3/P4/P5 owner
  失衡。
- `early_resume_shock.png`：epoch 28 前 500 sampled steps；step 0 的基础项相同，差值完全
  来自新增 SGQS。
- `image5367_forensics.png`：image 5367 的 source level、近似唯一框、阈值框数、rank
  分数和尾部重复轨迹。
- `image5367/query_probe_image_5367_ema_e28_e34.png`：EXP-DIAG-006 已登记的七轮
  Encoder Top-640 / Decoder all-640 实例图。
- `image5367/representative_ema_e30_e34_loss_iou75.png`：正式一对一匹配的 e30→34
  loss 代表图，image 5367 是其中 encoder-selection strata 的样本之一。
- `baseline_test/`：Baseline_test 全部 13 张 contact sheet、aggregate trajectory、源 CSV
  与读图说明。Encoder Top-300 只是诊断前缀；架构实际选择 Top-640。Decoder 640、正式
  Top-300、尾部 301--640、后处理 all-640 与 `score>0.5` 部署视图已分开。

Baseline_test 没有 GT，不能从 contact sheet 宣称 recall 或 AP；image 5367 与 formal
transition 图才带注册 annotation 对应关系。
