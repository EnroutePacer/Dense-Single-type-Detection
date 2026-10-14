# C2: MAL-only auxiliary continuation

This completed auxiliary control uses the same checkpoint and epoch-28--34
budget as B0--B3, while replacing VFL with MAL without enabling SGQS/EFGL.

| Metric | Exact epoch-34 value |
|---|---:|
| mAP@50:95 | 0.868764797642 |
| AP50 | 0.881034334354 |
| AP75 | 0.871055851457 |
| AP-small | 0.677576516799 |
| AR@300 | 0.873083289034 |
| AR-small | 0.682780612245 |

The physical output directory ends in `_correct0`, while `run_manifest.txt`
records the logical `run_id` as `resume_c2_mal_only_seed0`. Both identities are
preserved; neither has been rewritten after the run.

C2 is `+0.002097709115` total mAP over B0, but AP-small is
`-0.008056974425`. It is an auxiliary loss control and does not establish an
SGQS-E contribution or a small-object improvement.
