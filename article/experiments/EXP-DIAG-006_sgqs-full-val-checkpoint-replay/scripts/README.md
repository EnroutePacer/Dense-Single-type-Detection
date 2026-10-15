# 复算入口

`analyze_full_val.py` 读取每个 checkpoint 的 compact replay 和 formal matching 目录。
正式 matching 必须显式使用十个 COCO IoU 阈值：

```text
0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95
```

以下 PowerShell 片段展示参数组织方式。路径仅为占位符：

```powershell
$analysisArgs = @()
foreach ($epoch in 28..34) {
    $checkpointDir = "checkpoint{0:D4}" -f $epoch
    $analysisArgs += @('--replay', "$epoch=<replay-root>\$checkpointDir")
    foreach ($state in @('raw', 'ema')) {
        $analysisArgs += @('--match', "${epoch}:${state}=<match-root>\$checkpointDir\$state")
    }
}

python .\analyze_full_val.py `
    @analysisArgs `
    --output <derived-data-output>
```

默认执行 e28→e29 到 e33→e34 的全部相邻配对，并额外执行 e28→e30、e30→e34；
状态为 raw/EMA，使用 `maxDets=300/640` 和 IoU `.50:.05:.95`。`--skip-global-pr`
只允许合成单元测试使用；正式结果不得启用，
因为它会跳过跨图 score 排序、PR 曲线和逐 IoU evaluator 对账。

新版 matcher 会同时生成 `detection_status_manifest.json` 与
`detection_status.partNNNN.npz`。分析器优先使用其中按
`maxDets × scope × IoU × image × query` 保存的正式 TP/FP/ignored/truncated 状态，
一次落盘后重建全局 PR；这避免对 14 个状态再次运行 evaluator。每个 COCO-small
切片仍须与 replay summary 在 `1e-8` 内一致；含 TP/FP 同分的合成回归会把全部切片与
matcher 独立重建函数及 evaluator 逐值复核。正式运行每个状态只对完整 score 流做一次
stable sort，再沿相同顺序筛各 status slice。旧 matcher 目录没有 detection-status 时才
退回 evaluator 重建路径。

默认是 fail-closed 正式模式：固定要求 e28--e34、raw+EMA、12,841 图 / 1,458,060 GT、
十个 IoU、`maxDets=300/640`，并校验同一 config、annotation 和 inference backend。
`<match-root>/matrix.complete.json` 还必须逐状态绑定 replay prediction、summary、state
marker、checkpoint contract 与 matcher complete；每个 state complete 再经
`gt_shard_manifest.json` 根到 summary、GT CSV 和 detection-status NPZ。audit 只记录逻辑
名称、hash 和脱敏 runtime identity，不写机器绝对路径。
`--testing` 才允许子集或合成矩阵，生成的 audit 固定标为 `testing_noncomplete`；
`--skip-global-pr` 只能与 `--testing` 同用。逐 annotation 转移明细写为确定性 gzip
`transition_details.csv.gz`，不再生成无压缩百万行 CSV。

绘图：

```powershell
python .\plot_full_val.py `
    --data-dir <derived-data-output> `
    --output <figure-output>
```

绘图入口同样 fail closed，并且在读取任何结果表之前先验证 `audit.json` 与
`artifact_manifest.csv`。只接受 `status=complete` 且 `formal_contract=true` 的正式
e28--e34 × raw/EMA 分析；12,841 图 / 1,458,060 non-crowd GT、十个 IoU、
`maxDets=300/640`、全局 PR、上游 replay/matcher SHA-256 链和一致 runtime/backend
身份必须齐全。manifest 中登记的每个分析产物都会重新核对 bytes 与 SHA-256；
`testing_noncomplete`、缺文件、人口不符、hash 篡改或 backend gate 缺失时均拒绝绘图。
CPU 正式回放可通过该结构门禁，但其 runtime identity 会被保留，不能据此冒充 H20
逐候选数值复现。

正式分析完成后，可从已保存的 final-decoder prediction 生成代表转移图；该步骤不重新
运行模型：

```powershell
python .\render_representative_transitions.py `
    --data-dir <derived-data-output> `
    --replay-root <replay-root> `
    --images-root <merged-images-root> `
    --output <figure-output>
```

该入口首先复用 `plot_full_val.py` 的完整 formal audit/manifest 门禁，再逐一核对 e28
metadata 和 e28/e29/e30/e34 EMA prediction 的 bytes、SHA-256、schema、state、image/query
轴。压缩 prediction 只按最终入选的少量 image row 顺序流式解压，不把四个全量数组同时
载入内存。默认固定生成 e28→e29 gain 与 e30→e34 loss 在 IoU 0.50/0.75/0.95 的六张
contact sheet，并写 `representative_transition_figure_manifest.csv`；cause × domain 中没有
合格样本时保留明确的 `absent` panel。

图中只画原始 GT 与已持久化的最后一层 640-query Decoder 几何。dense-all 与 Encoder
Top-640 的 box 没有被 full-val replay 持久化，因此 encoder 类 cause 只展示正式阶段
IoU/rank 归因，不能把这些图片称为 Encoder 几何可视化。`representative_annotations.csv`
保存渲染所需的 before/after TP/FN、formal query/rank/score/IoU 以及四阶段 IoU/rank，
renderer 会用 prediction 重新计算 Decoder IoU/rank并与表中证据对账。

合成 schema、确定性和旧 val128 归因规则回归：

```powershell
python .\test_analyze_full_val.py
python .\test_plot_full_val_gate.py
python .\test_render_representative_transitions.py
```

测试夹具只在临时目录生成，不能当作实验数据或结果引用。

## 归因规则

对于 formal TP→FN 的 loss，读取 end checkpoint；对于 FN→TP 的 gain，读取 start
checkpoint。按“仍然存活的最深阶段”分类：

1. dense-all 不覆盖：`candidate_absence`；
2. dense-all 覆盖、Encoder Top-640 不覆盖：`encoder_selection`；
3. Encoder Top-640 覆盖、Decoder all-640 不覆盖：`decoder_geometry`；
4. Decoder all-640 覆盖、Decoder Top-300 不覆盖：`rank_gt300`；
5. Decoder Top-300 独立覆盖但 formal FN：`one_to_one_competition`。

这只定位 formal FN 的直接缺失端，不宣称训练级根因。global PR 另行重建每图匹配后
的数据集级稳定 score 排序，并与 replay summary 的逐 IoU small AP/AR 做 `1e-8`
误差门禁。
