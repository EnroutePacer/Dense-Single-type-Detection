# B1: SGQS, epoch-27 continuation

Completed absolute epochs 28--34, 5,631 steps/epoch, seed 0, 8 GPUs with
batch 8/GPU, fixed 640 input, 640 queries and COCO `maxDets=300`.

| Metric | Exact epoch-34 value |
|---|---:|
| mAP@50:95 | 0.867608455076 |
| AP50 | 0.880952489102 |
| AP75 | 0.870974272985 |
| AP-small | 0.682071410226 |
| AR@300 | 0.872612260996 |
| AR-small | 0.686096938776 |

- `console.log` is the byte-preserving user attachment registered before the
  direct server sync; `provenance.json` records that ingestion.
- `server_console.log` is the separately downloaded server file. It differs
  from the attachment only by trailing blank lines and is retained as source
  evidence rather than replacing it.
- `run_manifest.txt`, `resolved_config.yml`, `runtime_sha256.txt` and
  `method_provenance.yml` are the at-start contract evidence.
- `log.txt`, `summary/` and the four `.pth` files are the structured metrics,
  TensorBoard data and model artifacts downloaded from the completed run.

The earlier B1/B2-only analysis remains an immutable time-point snapshot:
[paired diagnosis](../../../article/experiments/EXP-TRAIN-005_epoch27-sgqs-e-resume-screen/diagnostics/20261013_b1_b2/README.md).
