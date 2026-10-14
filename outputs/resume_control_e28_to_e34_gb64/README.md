# B0 fixed-640 VFL continuation control

This is the completed negative-control arm for the epoch-27 continuation
screen. It resumed `checkpoint0027.pth`, executed absolute epochs 28--34 and
ended at `epoches=35`.

The console evidence resolves to: 8 GPUs, per-GPU batch 8, global batch 64,
seed 0, one 640x640 dataset resize, no multiscale/Mosaic/MixUp, 640 decoder
queries, VFL + box + GIoU + FGL + DDF, and COCO `maxDets=300`. The initial EMA
epoch-27 evaluation was mAP 0.864310288258. The final epoch-34 result was:

| Metric | Value |
|---|---:|
| mAP@50:95 | 0.866667088526 |
| AP50 | 0.880899235273 |
| AP75 | 0.870946341535 |
| AP-small | 0.685633491224 |
| AP-medium | 0.861307225133 |
| AP-large | 0.928902582379 |
| AR@300 | 0.871900060019 |
| AR-small | 0.690892249757 |
| AR-medium | 0.864603760749 |
| AR-large | 0.933702910420 |

Thus the seven-epoch continuation improved mAP by +0.002356800268 over the
EMA resumed checkpoint. This B0 result is the comparison endpoint for B1/B2/B3;
it is not evidence for SGQS or EFGL.

Evidence files:

- `console.log`: server-side tee output.
- `resume_control_e28_to_e34_gb64.log`: external job log, including the same
  completed run plus scheduler capture.
- `log.txt`: one exact JSON metric row for each epoch 28--34.
- `metrics.csv`: lossless 12-decimal extraction of the COCO arrays in
  `log.txt`.
- `best_stg1.pth`, `checkpoint0031.pth`, `last.pth` and `eval/latest.pth`:
  server model/evaluator artifacts stored through Git LFS.
- `summary/`: the eight-worker TensorBoard event set from the completed run.

The run predates the new fail-closed launcher, so it has no generated
`run_manifest.txt`. Its complete resolved configuration is nevertheless printed
at the start of both console logs. The subsequently deployed preflight verifies
the current checkpoint/data hashes, but must not be misrepresented as a
historical at-start manifest for this run.
