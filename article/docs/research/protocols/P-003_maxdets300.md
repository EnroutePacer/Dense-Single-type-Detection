# maxDets=300 评估协议

## 为什么必须显式设置

COCO 默认 `maxDets` 的最大值通常是 100。自然图像上单图目标数很少，这通常足够；表格单元格验证集中大量图片超过 100 个目标，检测器即使正确预测，也会在 evaluator 入口被截掉，形成结构性的假漏检。

当前代码在 evaluator 创建与 cleanup 两处都设置：

```python
self.coco_eval[iou_type].params.maxDets = [1, 10, 300]
```

这正是旧 mAP≈0.56 恢复到约 0.864 的关键原因。

## 固定主协议

- 验证集和 category mapping 固定。
- `num_queries=640`，PostProcessor 输出上限 640。
- COCO evaluator `maxDets=[1,10,300]`。
- COCO 评估时不使用人为 score threshold；按分数排序交给 evaluator。
- 主表统一报告 AP、AP50、AP75、APS/M/L、AR@300 和 ARS/M/L。
- 表头明确写 `maxDets=300`，不得把 100 与 300 的结果放在同一增益列。

## 为什么不直接把主指标改成 640

300 与既有当前结果、PaddleX 输出预算和已建立的实验口径一致。为了研究极密集场景，可以额外报告 `maxDets=640`、GT-count 分桶 recall 和每图 recall，但应作为补充指标，不能悄悄替换主表口径。

建议分桶：

```text
GT <= 100
100 < GT <= 300
300 < GT <= 640
```

这样可区分模型错误与 evaluator 容量上限。

## 暂定理论上限

此前基于验证标注做过“完美预测但最多保留 300 个框”的计算：

```text
AP 上限约 0.881188
AR 上限约 0.883315
```

这说明 0.864 已接近该特定截断协议的天花板，但在数据集 JSON 与复算脚本归档前，该数字只作为规划参考，不能直接写成论文最终事实。

## 必须保留的审计项

- evaluator 源码哈希；
- 解析后的 `num_queries` 和 `num_top_queries`；
- 标注 JSON 哈希；
- 每张图 GT 数分布；
- COCOeval params 的运行时打印；
- 评估 checkpoint SHA-256。

