# B2: EFGL, epoch-27 continuation

Completed absolute epochs 28--34, 5,631 steps/epoch, seed 0, 8 GPUs with
batch 8/GPU, fixed 640 input, 640 queries and COCO `maxDets=300`.

| Metric | Exact epoch-34 value |
|---|---:|
| mAP@50:95 | 0.866626082543 |
| AP50 | 0.880877245319 |
| AP75 | 0.870928135001 |
| AP-small | 0.685311726499 |
| AR@300 | 0.871863225585 |
| AR-small | 0.690269679300 |

- `console.log` is the byte-preserving user attachment registered before the
  direct server sync; `provenance.json` records that ingestion.
- `server_console.log` is the separately downloaded server file and is kept
  distinct even though the substantive non-empty lines match.
- `run_manifest.txt`, `resolved_config.yml`, `runtime_sha256.txt` and
  `method_provenance.yml` are the at-start contract evidence.
- `log.txt`, `summary/` and the four `.pth` files are the structured metrics,
  TensorBoard data and model artifacts downloaded from the completed run.

The earlier B1/B2-only analysis remains an immutable time-point snapshot:
[paired diagnosis](../../../article/experiments/EXP-TRAIN-005_epoch27-sgqs-e-resume-screen/diagnostics/20261013_b1_b2/README.md).
