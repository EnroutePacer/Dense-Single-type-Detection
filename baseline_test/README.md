# baseline_test：链路还原与可视化诊断

这个目录是训练仓库外层的测试包，用来还原表格单元格检测链路，并把 Encoder 候选生成、Top-K 截断和 Decoder 查询结果可视化。它服务于当前的密集单类别检测研究，不是第二套训练源码，也不承担正式训练配置的维护。

## 当前保留内容

- `DATA/`：固定的 13 张 wired / wireless 诊断图片。
- `output/`：DEIM 基线、SGQS 续训和 Top-K 后处理的诊断结果；每个结果目录自带运行说明。
- `tools/diagnose_deim_query_selection.py`：导出 Top-K 前的密集 Encoder 候选和 640 个 Decoder 查询。
- `tools/benchmark_deim_cpu.py`：提供诊断脚本复用的固定 CPU 预处理、模型加载和推理基准。
- `tools/render_sgqs_checkpoint_baseline_topk.py`：从诊断结果重建 SGQS 的 Top-K 可视化。
- `models/which_weight_to_use/`：只保存权重定位规则；模型源码和权重使用仓库根目录或外部存储中的正式副本。
- `DEIM_DFINE_pipeline_and_multiscale_nms070_audit.md`：早期推理链路审计记录。

## 复现约定

诊断脚本固定输入尺寸、CPU 线程和 affinity，输出目录中的 CSV、PNG、JSON 是可追溯证据。运行时显式指定仓库中的配置和 checkpoint，例如：

```powershell
python baseline_test/tools/diagnose_deim_query_selection.py `
  --data-dir baseline_test/DATA `
  --config configs/table_cells/dfine_hgnetv2_m_table_cell.yml `
  --checkpoint outputs/e35_r7_sgqse/deim_core_e35_seed0/checkpoint0034.pth `
  --output-dir baseline_test/output/DEIM_query_diagnostic_no_post_topk_640
```

这里的 `checkpoint` 仅为示例，实际运行应以 `models/which_weight_to_use/` 和对应实验记录登记的权重为准。正式训练、损失和配置代码位于仓库根目录的 `engine/`、`configs/` 和 `tools/`。

## 证据边界

这些图和 CSV 用于定位链路中的候选缺失、Top-K 排名以及查询分布问题。它们没有标注集 GT，不能单独推出 recall、AP 或训练收益；正式指标必须以 COCO 评估输出为准。
