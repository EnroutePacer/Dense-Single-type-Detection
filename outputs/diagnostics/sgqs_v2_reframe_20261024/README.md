# SGQS v2 方向重评的可复算证据

2026-10-24 发布；原始输入为提交119bfe2中 `baseline_test/output/data_threshold_0.30_compare/` 的 b0/sgqs epoch34 导出。两个名字含raw的目录实际均为 `ema.module`。固定13张图没有GT；此处不报告AP、FP或真实剪枝收益。

运行 `python outputs/diagnostics/sgqs_v2_reframe_20261024/recompute.py`，使用已有numpy/pandas，生成同目录 `verified_candidate_audit.json`。JSON包含输入CSV的SHA256；运行commit字段随复算提交改变，输入哈希与统计才是固定证据。

| 统计 | B0 | SGQS |
|---|---:|---:|
| 入选Encoder平均分 | .192057 | .328136 |
| 入选Encoder中位分 | .020383 | .206202 |
| Encoder Top256对原Decoder score>=.5候选的静态保留率 | .995159 | .963432 |
| Encoder>=.3且Decoder<.1，最终与另一Decoder>=.5框IoU>=.7 | 39 | 768 |

最终框使用归一化cxcywh转未裁边xyxy，query按image_key/source_dense_index连接并验证分数一致。所有统计来自完整640前向，不是删除后重跑。

全部query的IoU>=.7重叠框对比例为.6261%/1.1880%，但仅Decoder>=.5时重叠对为13/12。最终高分重复接近与内部低分冗余增加可以同时成立；框对比例不是重复框比例。未复现用户所述20→11px/68%→89%的联合口径，因此未将其作为新设计成立条件。

完整设计与证据解释见 [M-003](../../../article/methods/M-003_sgqs_v2/README.md)。
