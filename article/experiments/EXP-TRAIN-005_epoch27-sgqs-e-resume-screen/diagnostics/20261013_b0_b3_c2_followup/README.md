# B0--B3 core matrix and C2 follow-up

This follow-up supersedes the *current-status conclusions* of the earlier
B1/B2-only snapshot without changing that registered historical artifact.
All exact values below are read from the epoch-34 JSON row in each downloaded
server `log.txt`; all 95 downloaded files passed server-to-local SHA-256
comparison.

| Arm | mAP@50:95 | AP-small | AR-small | ΔmAP vs B0 | ΔAP-small vs B0 |
|---|---:|---:|---:|---:|---:|
| B0 VFL | 0.866667089 | 0.685633491 | 0.690892250 | — | — |
| B1 SGQS | 0.867608455 | 0.682071410 | 0.686096939 | +0.000941367 | -0.003562081 |
| B2 EFGL | 0.866626083 | 0.685311726 | 0.690269679 | -0.000041006 | -0.000321765 |
| B3 SGQS-E | 0.867687360 | 0.682243934 | 0.686048348 | +0.001020272 | -0.003389557 |
| C2 MAL-only | 0.868764798 | 0.677576517 | 0.682780612 | +0.002097709 | -0.008056974 |

For the factorial B0--B3 screen, B3 exceeds B1 by only
`+0.000078905207` mAP and the single-seed interaction
`B3 - B1 - B2 + B0` is `+0.000119911191`. Those differences are much too
small to establish stable complementarity. More importantly, B1 and B3 both
finish below B0 on AP-small and AR-small. C2 reaches the highest total mAP but
has the lowest small-object endpoint among the completed arms.

The evidence therefore supports only a narrow screening conclusion: SGQS
produced a small total-mAP signal near an already-converged checkpoint, EFGL
did not add a meaningful endpoint signal, and none of these continuations
demonstrated better small-object detection. Formal claims require from-start
training and paired multi-seed repeats with the candidate-chain diagnostics.

Auxiliary status at audit time:

- C1 small-reg stopped at epoch 30 step 3800/5631;
- C3 Mosaic-only stopped at epoch 28 step 500/5631;
- DEIM-core from-start stopped at epoch 0 step 2200/5631;
- no active `torchrun`/`train.py` process was found, and the captured consoles
  contain no explicit exception explaining the termination.

The source logs, configs, manifests, weights and TensorBoard files are under
`outputs/`; [metrics.csv](metrics.csv) is the compact exact endpoint table.
