# B0 vs SGQS e34: thresholded COCO AP at maxDets=640

## Metric contract

The dense table-cell validation set can contain more than 300 valid objects in
one image, so the legacy COCO `maxDets=300` result is not used here.  The
replacement comparison is:

1. Use all 640 Decoder queries from the e34 EMA checkpoint.
2. Remove predictions whose sigmoid score is below the fixed cutoff.
3. Run COCO bbox evaluation at IoU `0.50:0.05:0.95` with `maxDets=640`.
4. Use the same cutoff for both models.  Report cutoffs 0.3, 0.4, 0.5 and 0.6;
   use **score >= 0.3** as the primary result.

This is denoted `AP@[.50:.95], score>=t, maxDets=640`.  There is no Top-300
selection before evaluation.

## Full-validation results

Dataset: 12,841 images.  Model state: `ema.module` from checkpoint e34.

| score cutoff | model | AP | AP50 | AP75 | APs | APm | APl | mean detections/image |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0.3 | B0 | 0.975310 | 0.989798 | 0.979691 | 0.943444 | 0.976461 | 0.966078 | 115.53 |
| 0.3 | SGQS | **0.975372** | **0.989834** | **0.979740** | **0.948592** | **0.977489** | 0.963223 | 115.56 |
| 0.4 | B0 | 0.973391 | 0.989798 | 0.979691 | 0.942490 | 0.976461 | 0.965117 | 114.49 |
| 0.4 | SGQS | **0.975372** | **0.989834** | **0.979740** | **0.947627** | **0.977489** | 0.962282 | 114.46 |
| 0.5 | B0 | 0.971451 | 0.989798 | 0.979691 | 0.941520 | 0.975484 | 0.964160 | 113.86 |
| 0.5 | SGQS | **0.973428** | **0.989834** | **0.979740** | **0.945687** | **0.977489** | 0.961323 | 113.76 |
| 0.6 | B0 | 0.970471 | 0.989798 | 0.979691 | 0.939578 | 0.974503 | 0.963191 | 113.34 |
| 0.6 | SGQS | **0.971484** | **0.989834** | **0.979740** | **0.945687** | **0.976509** | 0.958409 | 113.24 |

At the primary cutoff 0.3, overall AP is effectively tied (`+0.000063` for
SGQS).  SGQS improves APs by `+0.005148`, while B0 improves APl by `+0.002855`.

## Reproduction

```powershell
python outputs/diagnostics/b0_sgqs_e34_threshold_map640/evaluate.py `
  --annotation _local/fullval_dataset_view/annotations/instance_val.json `
  --b0-predictions _local/fullval_cpu_b0_b4/checkpoint0034/val.ema.predictions.npz `
  --sgqs-predictions _local/fullval_cpu/checkpoint0034/val.ema.predictions.npz `
  --output outputs/diagnostics/b0_sgqs_e34_threshold_map640/metrics.csv
```

`metrics.csv` and `metrics.json` contain the full AP/AR output and detection
counts.  The large prediction and annotation caches remain local and are not
committed.
