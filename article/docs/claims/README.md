# 论文主张—证据系统

证据链固定为：

```text
ART 原始证据 -> RUN 实际运行 -> EXP 受控实验 -> F 事实 -> C 论文主张 -> 论文图表/章节
```

研究链固定为：

```text
F 观察 -> H 可证伪假设 -> EXP 验证 -> 新 F -> C 是否成立
```

当前所有方法性主张都是 `target`，没有一项已满足论文准入条件。机器可读条目在 [claim_evidence_matrix.yaml](claim_evidence_matrix.yaml)。

## 当前目标主张

| ID | 目标主张 | 最低准入条件 |
|---|---|---|
| C-001 | 固定 `K=640` 时提高 unique-GT coverage、降低 SSR。 | 带 GT、冻结模型，同 K 对比 global Top-K、简单 quota/calibration、本文方法和 GT oracle。 |
| C-002 | coverage 改善转化为相同 epoch/steps 下更高 AP/AR。 | 相同初始化与训练协议、`maxDets=300`、至少三个 seed。 |
| C-003 | 方法没有把错误从 selection 转移至 decoder 或输出端。 | 完整逐阶段 GT survival、重复框、FP 与校准。 |
| C-004 | 收益集中于密集、短边小或细长目标，且不明显损害其他分桶。 | 预注册分桶、样本数、均值和方差。 |
| C-005 | 固定 query 数下部署成本可控。 | 实测参数、FLOPs、selector/model/pipeline latency。 |
| C-006 | 方法不是表格专用技巧。 | 第二公开数据集或公开可复算的合成密度基准。 |

## 当前禁止写成结论

- “绿色 P3 点少，证明 encoder 漏掉大量小框。”
- “小框主要死在 encoder Top-K。”
- “小框主要死在 decoder 重评分。”
- “P3 就是小目标层。”
- “增加 P3 quota、K 或 coverage-aware selection 必然提高 mAP。”
- “方法没有推理开销。”
