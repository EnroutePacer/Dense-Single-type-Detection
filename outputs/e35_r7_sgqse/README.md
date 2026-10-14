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
| C2 MAL-only | complete auxiliary control | 0.868764798 | 0.677576517 | 0.682780612 |

B0 is stored in the sibling directory
`outputs/resume_control_e28_to_e34_gb64/`. B1--B3 and C2 include the
server run manifest, resolved configuration, structured log, checkpoints and
TensorBoard records. `source_checkpoint/` contains the shared immutable
resume checkpoint.

`deim_core_e35_seed0`, `resume_c1_small_reg_seed0` and
`resume_c3_mosaic_only_seed0` are interrupted partial runs. They contain only
the small text evidence required to diagnose identity and stopping point; no
partial checkpoint is presented as a valid result.

This is a one-seed, already-converged checkpoint continuation screen. B3 is
`+0.001020272` mAP over B0 but its AP-small is `-0.003389557`; C2 has the
highest total mAP but its AP-small is `-0.008056974` below B0. These results
do not establish stable gain, small-object improvement or method synergy.
