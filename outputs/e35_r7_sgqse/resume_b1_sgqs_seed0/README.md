# B1: SGQS, epoch-27 continuation

Completed absolute epochs 28--34, 5,631 steps/epoch, seed 0, 8 GPUs with
batch 8/GPU, fixed 640 input, 640 queries and COCO maxDets=300.
Final EMA mAP is 0.8676084550759916; AP-small is approximately 0.682.
AP-small/AR-small are console-rounded to 0.001.

- [console.log](console.log): byte-for-byte user-provided attachment, message (2).txt.
- [provenance.json](provenance.json): ingestion provenance and SHA-256;
  this is not an at-start training manifest.
- [Paired diagnosis](../../../article/experiments/EXP-TRAIN-005_epoch27-sgqs-e-resume-screen/diagnostics/20261013_b1_b2/README.md).

The resolved configuration and initial mAP agree with B0. No independent
runtime source/data hash manifest, server log.txt or checkpoint was supplied.
Existing server paths in the immutable console record are retained for evidence
continuity; no new local machine paths are added by the analysis script.
