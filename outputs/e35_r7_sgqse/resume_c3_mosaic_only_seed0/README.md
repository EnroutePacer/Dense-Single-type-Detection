# C3: Mosaic-only auxiliary continuation

This completed auxiliary control uses the same epoch-27 checkpoint and
epoch-28--34 budget as B0, adding only Mosaic with probability `0.5`; the
network input remains fixed at 640x640 and MixUp stays disabled.

| Metric | Exact epoch-34 value |
|---|---:|
| mAP@50:95 | 0.865379299191 |
| AP50 | 0.880778059642 |
| AP75 | 0.870868753820 |
| AP-small | 0.682975777018 |
| AR@300 | 0.870191340135 |
| AR-small | 0.687603255588 |

Relative to B0, mAP changes by `-0.001287789335` and AP-small by
`-0.002657714206`. This Mosaic setting is negative in the present continuation
screen and does not improve small objects.

The complete console, structured log, resolved configuration, run manifest and
TensorBoard records are archived here. The final checkpoint was not copied.
