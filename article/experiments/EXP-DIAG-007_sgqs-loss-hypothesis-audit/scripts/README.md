# 复算脚本

在仓库根目录执行：

```powershell
python article/experiments/EXP-DIAG-007_sgqs-loss-hypothesis-audit/scripts/build_hypothesis_audit.py
python article/experiments/EXP-DIAG-007_sgqs-loss-hypothesis-audit/scripts/build_manifest.py
python article/experiments/EXP-DIAG-007_sgqs-loss-hypothesis-audit/scripts/build_manifest.py --check
```

第一条命令只读取已登记的 SGQS/B0 logs、probe summaries、EXP-DIAG-006 query-probe/full-val
表与 `D:\Baseline_test\output\SGQS_e28_e34_baseline_topk_postprocess`，不会训练或推理。
若 Baseline_test 位于其他目录，传入 `--baseline-dir`。脚本将重建派生 CSV/PNG，并按原
字节复制 13 张 contact sheet。

需要 Python、NumPy 和 Matplotlib。发布 manifest 排除自身，覆盖实验目录内其余文件。
