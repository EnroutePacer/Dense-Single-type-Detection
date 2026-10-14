# E35/R7 continuation screen

All completed arms use the same epoch-27 checkpoint, fixed 640 input, 640
queries, global batch 64, seed 0, epochs 28--34 and COCO `maxDets=300`.
Values below come from the final exact JSON row in each `log.txt`.

| Arm | State | mAP@50:95 | AP-small | AR-small |
|---|---|---:|---:|---:|
| B0 VFL control | complete | 0.866667089 | 0.685633491 | 0.690892250 |
| B1 SGQS | complete | 0.867608455 | 0.682071410 | 0.686096939 |
| B2 EFGL | complete | 0.866626083 | 0.685311726 | 0.690269679 |
| B3 SGQS-E | complete | 0.867687360 | 0.682243934 | 0.686048348 |
| C1 small-reg | complete auxiliary control | 0.866651559 | 0.685224968 | 0.690312196 |
| C2 MAL-only | complete auxiliary control | 0.868764798 | 0.677576517 | 0.682780612 |
| C3 Mosaic-only | complete auxiliary control | 0.865379299 | 0.682975777 | 0.687603256 |

B0 is stored in the sibling directory
`outputs/resume_control_e28_to_e34_gb64/`. B1--B3 and C2 include the
server run manifest, resolved configuration, structured log, checkpoints and
TensorBoard records. `source_checkpoint/` contains the shared immutable
resume checkpoint.

The C1 and C3 directories contain their complete console, structured log,
resolved configuration, run manifest and TensorBoard records. Their final
checkpoints have not been copied into this repository. The server-side C4
DEIM-core retrofit remained incomplete and its files were still changing at
the final audit, so no C4 snapshot is committed as a final artifact.
`deim_core_e35_seed0` is an older interrupted partial run at epoch 0 step
2200/5631 and is not presented as a valid endpoint.

This is a one-seed, already-converged checkpoint continuation screen. C1 is
effectively identical to B0; C3 is worse; C2 gains `+0.002097709` total mAP
while losing `-0.008056974` AP-small. B1--B3 also fail to improve AP-small.
The compact phase summary is archived under
`article/experiments/EXP-TRAIN-005_epoch27-sgqs-e-resume-screen/diagnostics/20261014_phase1_conventional_methods/`.
