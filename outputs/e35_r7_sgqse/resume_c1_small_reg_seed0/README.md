# C1: small-reg auxiliary continuation

This completed auxiliary control uses the same epoch-27 checkpoint and
epoch-28--34 budget as B0, adding only area-thresholded box/local regression
weights (`1.30` / `1.15`) for normalized box area below `0.0025`.

| Metric | Exact epoch-34 value |
|---|---:|
| mAP@50:95 | 0.866651558764 |
| AP50 | 0.880889029274 |
| AP75 | 0.870917992438 |
| AP-small | 0.685224967827 |
| AR@300 | 0.871948829632 |
| AR-small | 0.690312196307 |

Relative to B0, mAP changes by `-0.000015529762` and AP-small by
`-0.000408523397`. Within this single-seed continuation screen, the historical
small-reg weighting is effectively neutral and does not improve small objects.

The complete console, structured log, resolved configuration, run manifest and
TensorBoard records are archived here. The final checkpoint was not copied.
