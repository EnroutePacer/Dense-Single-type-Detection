# T001/T002：epoch-27 checkpoint 的 maxDets 复评

> **Historical snapshot**：旧路径指向本地私有 legacy 证据，公开追溯请使用 [artifact registry](../../docs/evidence/artifact_registry.csv)。canonical 结论见本实验的 [README](README.md)。

## 目的

确认 M 模型训练日志中的 mAP 五十多是否来自检测能力不足，还是 COCO evaluator 的输出上限不适用于密集单元格。

## 模型与数据

- 模型：D-FINE-M 表格单元格模型，checkpoint 文件名为 `checkpoint0027.pth`。
- 类别：1。
- queries / 后处理上限：640。
- 验证数据：`paddlex_l_cell_merged/annotations/instance_val.json`。
- 模型 checkpoint 未包含在当前服务器归档，需补取。

## 两种口径

旧训练日志在 epoch 27 记录：

```text
mAP@50:95 0.566899
AP50       0.574078
AR         0.567101
```

当前 evaluator 明确设置：

```python
self.coco_eval[iou_type].params.maxDets = [1, 10, 300]
```

同一 checkpoint 在多份快速探针开始前复评得到约：

```text
mAP@50:95 0.864247
AP50       0.880287
AP75       0.871
APS        0.680
APM        0.859
APL        0.920
AR@300     0.869
```

## 结论

0.56 不是与 0.86 可直接比较的训练指标；它主要受 `maxDets=100` 截断。所有后续主结果固定使用 `maxDets=300`，并在表头显式写出口径。

由于验证集中部分图像可能超过 300 个 GT，`maxDets=300` 仍不是“无限检测”评估。此前计算的完美预测 AP/AR 上限约 0.881188/0.883315 目前为暂定值；待标注 JSON 归档后应提供可复算脚本与哈希。

## 证据

- `../01_sources/server_DEIM_20261010_182837/DEIM/output/dfine_hgnetv2_m_table_cell/log.txt`
- `../01_sources/server_DEIM_20261010_182837/DEIM/engine/data/dataset/coco_eval.py`
- `../01_sources/server_DEIM_20261010_182837/DEIM/output/probe3200_fixed640_small_vfl_w1p5_correct/console.log`
- `../01_sources/server_DEIM_20261010_182837/DEIM/output/probe12800_fixed640_small_reg_4gpu/console.log`

