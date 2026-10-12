# Epoch-index-27 table-cell benchmark

These configurations train for 28 complete epochs (indices 0 through 27) from
the same detector pretraining checkpoint. They keep single-scale 640 input,
640 queries, one class, global batch 16, and COCO evaluation with maxDets=300.
The accompanying bundle includes `engine/data/dataset/coco_eval.py`, which
sets `[1, 10, 300]` both when the evaluator is created and when it is reset.

The existing single-scale-640 D-FINE `checkpoint0027.pth` result (mAP
0.864248) remains a valid historical reference. `vfl_clean.yml` deliberately
repeats that recipe so the benchmark also has three current-code seeds and a
mean/std computed under the same execution protocol as every new baseline.

Core causal matrix:

| config | loss | augmentation |
|---|---|---|
| `vfl_clean.yml` | VFL | clean |
| `vfl_small_reg.yml` | VFL + small-object localization reweighting | clean |
| `mal_clean.yml` | MAL | clean |
| `vfl_dense_o2o.yml` | VFL | Dense O2O |
| `mal_dense_o2o.yml` | MAL | Dense O2O |

Diagnostics:

- `vfl_small_reg.yml`: formal epoch-length follow-up to the 12,800-step
  small-regression probe (box 1.30, local 1.15).
- `vfl_mosaic_only.yml`: isolates Mosaic.
- `vfl_mixup_only.yml`: isolates MixUp.
- `deim_official.yml`: MAL + Dense O2O + official DEIM SiLU/optimizer/LR
  settings, while retaining fixed 640 and 640 queries.

Use `tools/table_cells/train_e28_benchmark_8gpu.sh`; it deliberately disables
resume and quick-probe mode and fixes 8 GPUs x 2 images/GPU = global batch 16.
Run seeds 0, 1, and 2 for all eight configurations (24 independent jobs).
