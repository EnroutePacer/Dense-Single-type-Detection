# 复算说明

`analyze_val128.py` 和 `analyze_val128_ema.py` 读取七轮 replay 的
`val.ema.gt.jsonl`/`summary.json`，生成本实验包中的阶段 coverage、转移和 rank
表。原始 JSONL 与固定子集标注未公开入库；取得受控原始材料后，在其父目录运行：

```powershell
$env:SGQS_REPLAY_ROOT = 'D:\path\to\server_sgqs_probe'
$env:SGQS_REPLAY_ANNOTATION = 'D:\path\to\probe_fixed128_valonly\val.json'
$env:SGQS_REPLAY_OUTPUT = 'D:\path\to\output\val128_ema_analysis'
python .\analyze_val128_ema.py
```

`plot_diagnostics.py` 只读取已入库的小型 CSV，可在没有 checkpoint 和私有标注时
重新生成三张统计图：

```powershell
python .\plot_diagnostics.py
```

定性 contact sheet 还需要本地 13 图可视化目录；它们没有 GT，不参与阶段归因。
需要重画时显式传入 `--baseline-root D:\path\to\baseline_output`。

`checkpoint_state_hash.py` 对 checkpoint 文件本身及 raw/EMA 参数分别计算 SHA-256。
参数哈希按参数名排序，并纳入 dtype、shape 和连续 tensor bytes，用来识别“容器字节
不同但模型状态完全相同”的文件。

`build_manifest.py` 为实验目录中除 `artifact_manifest.csv` 自身以外的每个文件记录
字节数和 SHA-256；文档、数据、脚本或图片更新后必须重新运行。
