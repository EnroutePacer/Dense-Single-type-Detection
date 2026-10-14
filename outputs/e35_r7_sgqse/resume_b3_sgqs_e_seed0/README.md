# B3: SGQS-E, epoch-27 continuation

Completed absolute epochs 28--34 under the same fixed-640, query-640,
global-batch-64, seed-0 and `maxDets=300` protocol as B0--B2.

| Metric | Exact epoch-34 value |
|---|---:|
| mAP@50:95 | 0.867687360283 |
| AP50 | 0.880947799774 |
| AP75 | 0.870967687072 |
| AP-small | 0.682243934177 |
| AR@300 | 0.872698893938 |
| AR-small | 0.686048347911 |

The registered `console.log` is the user-supplied capture and includes the
preflight prefix. `server_console.log` is the independently downloaded server
tee log and must not replace it. `provenance.json` identifies the attachment;
`run_manifest.txt` and the YAML/hash files identify the actual server run.

B3 is `+0.001020271757` mAP over B0 and `+0.000078905207` over B1, while
AP-small is `-0.003389557047` below B0. With one seed and a seven-epoch
continuation, this is a screening signal only, not evidence of stable synergy
or improved small-object detection.
