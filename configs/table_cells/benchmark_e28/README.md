# Epoch-index-27 table-cell benchmark

These configurations train for 28 complete epochs (indices 0 through 27) from
the same detector pretraining checkpoint. They keep single-scale 640 input,
640 queries, one class, global batch 16, and COCO evaluation with maxDets=300.

Core causal matrix:

| config | loss | augmentation |
|---|---|---|
| `vfl_clean.yml` | VFL | clean |
| `mal_clean.yml` | MAL | clean |
| `vfl_dense_o2o.yml` | VFL | Dense O2O |
| `mal_dense_o2o.yml` | MAL | Dense O2O |

Diagnostics:

- `vfl_mosaic_only.yml`: isolates Mosaic.
- `vfl_mixup_only.yml`: isolates MixUp.
- `deim_official.yml`: MAL + Dense O2O + official DEIM SiLU/optimizer/LR
  settings, while retaining fixed 640 and 640 queries.

Use `tools/table_cells/train_e28_benchmark_8gpu.sh`; it deliberately disables
resume and quick-probe mode and fixes 8 GPUs x 2 images/GPU = global batch 16.

