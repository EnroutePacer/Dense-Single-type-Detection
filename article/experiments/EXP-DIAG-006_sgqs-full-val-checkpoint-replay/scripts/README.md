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

`stage_coverage.png` 固定并列展示两个不同 population：formal evaluator-small
（原始 annotation `area <= 32^2`）与本地 hybrid-small（直接缩放到 640 后
`area < 32^2 OR short side < 16`）。这里是每个 GT 对该阶段所有候选的独立 best-IoU
coverage，不是一对一 COCO recall。`bucket_recall.png` 则固定读取
`scope=coco_small_eval`，展示 evaluator-small 在 domain、密度和输入短边分桶下的正式
一对一 TP/FN recall；不得用 `scope=all` 的全面积结果冒充 small 分析。

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

### 选择性 Encoder/Decoder checkpoint probe

`render_checkpoint_query_probes.py` 补充 baseline_test 同类的 Encoder query 来源与
Decoder 全 640 / local-hybrid-small 可视化，但不会重新回放完整验证集。它只从已经通过
formal manifest 门禁的 `representative_annotations.csv` 选择 IoU=0.75 的两组端点：

- e28→e29 formal evaluator-small gain 与 e30→e34 formal evaluator-small loss；
- 相同端点中 `match_scope=all, small_partition=hybrid_only` 的本地标准样本。

每个实际存在的 cause×domain 固定保留一条，随后按 image ID 去重，因此最多选择 40 张
图。七个 checkpoint 只读取 `ema.module`；这是对 AP-small EMA 倒 U 的定性补图，raw/EMA
的完整定量比较仍来自 14-state full-val replay。运行命令为：

```powershell
$env:OMP_NUM_THREADS = '20'
$env:MKL_NUM_THREADS = '20'
python .\render_checkpoint_query_probes.py `
    --analysis-data <derived-data-output> `
    --replay-root D:\TabledetectCode\DEIM_2\_local\fullval_cpu `
    --annotation D:\TabledetectCode\DEIM_2\_local\fullval_dataset_view\annotations\instance_val.json `
    --images-root D:\TabledetectCode\DEIM_2\_local\fullval_dataset_view\images `
    --config D:\TabledetectCode\DEIM_2\_local\server_sgqs_probe_20261014\resolved.yml `
    --checkpoint-root D:\TabledetectCode\DEIM_2\_local\server_sgqs_probe_20261014 `
    --output <figure-output>\query_probes
```

入口要求 config、annotation、七个 checkpoint 容器、metadata 与 canonical EMA prediction
的 SHA-256 全部匹配 formal audit，并进一步要求 `<replay-root>/matrix.complete.json` 的
SHA-256 等于 formal matcher matrix 当时绑定的 replay matrix。当前
`full_val_replay.py`、`supervision_probe.py`、完整 `engine/**/*.py` tree，以及
device、Python、Torch/CUDA/cuDNN、GPU 身份、Torch intra/inter-op threads、OMP/MKL 环境
必须与该 replay matrix 逐字段一致，且 canonical batch size 必须为 1。上例中的线程值 20
来自当前 `_local/fullval_cpu/matrix.complete.json`；换 replay 根时必须使用其记录值，不能把
不同后端产生的 Encoder 捕获冒充正式重放证据。每次选择性 forward 都用
`full_val_replay.CaptureCandidates` 捕获真实 Encoder Top-640；若其逐 query Decoder box、
score、稳定 rank 顺序或 local-hybrid-small mask 与已保存 NPZ 不是位级一致，脚本立即失败，
不出图。

每张代表图生成一张 e28--e34 × 三列 contact sheet：

1. Encoder Top-640 的真实 dense source level（P3/P4/P5，单图互斥且和为 640）；
2. Decoder 最后一层全部 640 query；
3. Decoder 最后一层中按 640 输入尺度满足 `area < 32² OR short side < 16` 的 query。

这里的 P3/P4/P5 是被选中 dense candidate 的实际特征层来源，不是根据 Decoder 最终框
短边事后计算的 SGQS preferred scale。输出还包括 `query_probe_selection.csv`、
`query_probe_data.npz` 与 `query_probe_manifest.csv`。NPZ 按
`[image,e28..e34,query]` 持久化 Encoder dense index/source level/normalized box 以及 Decoder
原图 xywh/score/hybrid mask，并保存 image size、stride、spatial shape 和 640 输入尺度；
因此无需 checkpoint 也能离线复核或重画。writer 采用临时文件校验后原子替换，固定字段
顺序和 dtype，同一输入必须生成逐字节相同的 archive。manifest 记录该 NPZ 的 bytes/hash、
正式 replay matrix/source/engine/runtime 身份、每图 P3/P4/P5 数量、hybrid-small 数量、
捕获数组 hash 及 Decoder 对账最大误差，并拒绝任何绝对路径。长时间 forward 前后还会
复核 probe/analyzer/plotter/renderer 源码、formal audit/manifest 与代表表未发生变化；发布前
逐行反查 NPZ，要求完整 `image × e28..e34` 网格、P3/P4/P5 和为 640、Decoder 位级一致标志
全真、slice hash/count 与 NPZ 相同。图片是从全量正式转移中确定性抽出的
**定性索引**，不能用其出现频率估计 AP、recall 或机制占比。

合成 schema、确定性和旧 val128 归因规则回归：

```powershell
python .\test_analyze_full_val.py
python .\test_plot_full_val_gate.py
python .\test_render_representative_transitions.py
python .\test_render_checkpoint_query_probes.py
```

测试夹具只在临时目录生成，不能当作实验数据或结果引用。

整包分析、图和文档定稿后，从本目录运行根清单构建器：

```powershell
python .\build_manifest.py
python .\build_manifest.py --check
python .\test_build_manifest.py
```

输出固定为实验根目录的 `artifact_manifest.csv`，按 POSIX 相对路径排序并记录实际读取
的 bytes 与 SHA-256。输入集合由 Git 定义：实验目录内已跟踪文件，加上尚未 `git add`
但未被 ignore 的生成文件；根清单自身除外，嵌套的同名 CSV 仍会纳入。因此 `.gitignore`
排除的本地日志、cache 和私有原始输入不会因磁盘遍历而误收。构建器拒绝 symlink、越出
实验根的路径和读取期间发生变化的文件，并通过同目录临时文件原子替换清单。

普通 Git 文件大于 10 MiB 时构建直接失败；只有 `git check-attr filter` 对该精确路径返回
小写 `lfs` 才允许纳入。不得用大小写近似值、扩展名推断或手工跳过绕过该门禁。
`--check` 不写文件，只把当前 Git 发布集合重新序列化并逐字节核对已有清单；清单缺失、
文件内容变化或成员增删都会返回失败。任何待发布产物变化后先重建，提交前再运行
`--check`。

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
