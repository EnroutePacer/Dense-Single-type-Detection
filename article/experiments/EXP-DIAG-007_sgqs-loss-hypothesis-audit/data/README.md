# EXP-DIAG-007 数据说明

- `hypothesis_epoch_trajectory.csv`：SGQS/B0 e28--e34 的精确结构化日志，含 COCO 指标、
  SGQS 四项、基础 Encoder/Decoder/pre 聚合及 owner 代理指标。
- `early_resume_steps.csv`：epoch 28 前 500 step 的两臂同采样点 loss；step 0 是最干净的
  新增目标差值，后续参数已分叉。
- `probe_sampled_epoch_summary.csv`：两臂训练探针的分组充分统计；这是采样 batch 汇总，
  e28 前 200 step 被加密采样，不能称无偏整轮均值。
- `owner_level_group_summary.csv`：采样 SGQS kept owner 的 P3/P4/P5 与增强后 input-area
  group 交叉表。input-area group 不是 COCO evaluator size。
- `full_val_small_budget_trajectory.csv`：从 EXP-DIAG-006 提取的 EMA evaluator-small
  `maxDets=300/640` 轨迹。
- `full_val_transition_summary.csv`：三组正式一对一转移的十 IoU cause 合计，单位为
  `GT × IoU transition events`，不是互异 GT，也不是 AP 的可加分解。
- `query_probe_image_forensics.csv`：29 张机制 probe × 7 epoch 的 Decoder 尾部、整数像素
  近似唯一框、分数边界及 Encoder source-level 统计。29 图是定向选样，不能估总体比例。
- `image5367_forensics.csv`：上表的 image 5367 子集。
- `query_probe_change_summary.csv`：29 张定向 probe 相对 e28 的方向计数，保留选择偏差警告。
- `baseline_topk_summary.csv`：Baseline_test 13 图 × 7 epoch 的原始精确导出，含 UTF-8
  BOM，字节保持不变。
- `hypothesis_verdicts.csv`：三个用户假说的结构化判定、反证和最小因果实验。
- `input_manifest.csv`：输入文件大小和 SHA-256；不保存私有原图或 annotation 内容。

所有 CSV 均由 `scripts/build_hypothesis_audit.py` 生成或按原字节复制。
