# 图片状态

`pending`。`plot_full_val.py` 将从真实派生表生成 metrics、逐 IoU budget gap、阶段
coverage、formal transition attribution、domain/density/short-side 分桶和 global PR
score separation 图。在真实表生成前不放置占位结果图。

formal analysis 与 canonical replay 均通过门禁后，
`render_representative_transitions.py` 还将生成：

- `representative_ema_e28_e29_gain_iou50.png`
- `representative_ema_e28_e29_gain_iou75.png`
- `representative_ema_e28_e29_gain_iou95.png`
- `representative_ema_e30_e34_loss_iou50.png`
- `representative_ema_e30_e34_loss_iou75.png`
- `representative_ema_e30_e34_loss_iou95.png`
- `representative_transition_figure_manifest.csv`

每页固定为 EMA、formal evaluator-small、`maxDets=300`，行是五类直接缺失 cause ×
wired/wireless；没有代表样本的 strata 显式写 `absent`。洋红框是选中 GT，绿色框是正式
TP query，橙色框是 FN 端可复核的 Decoder query。图片只呈现最后一层 Decoder 几何；
dense/Encoder cause 来自表中阶段证据，不表示图里画出了 Encoder 输入或输出。
