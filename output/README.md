# Experiment outputs

This directory is a durable research record, not disposable build output.

- Commit text logs, metrics, summaries, resolved configs, commands and manifests with ordinary Git.
- Commit checkpoints, evaluator tensors, TensorBoard event files and transfer archives through Git LFS.
- Never reuse a run directory. Failed, interrupted and superseded runs remain recorded and should be labelled instead of deleted.
- Every committed artifact is listed in `ARTIFACT_MANIFEST.csv` with its byte size, SHA-256 and Git storage class.
- Before pushing, inspect logs and configs for credentials, tokens and private dataset paths.

The files restored on 2026-10-12 came from the unchanged local safety copy at `_local/cleanup_20261012/output/`. That safety copy is intentionally not overwritten or removed.

`targeted_e28_r7_training_bundle.tar.gz` is retained as historical evidence and
is superseded; it must not be extracted over the current source tree because it
contains an older criterion implementation. The active server overlay for the
35e/r7 matrix is `e35_r7_sgqse_runtime_bundle.tar.gz`, verified by
`tools/table_cells/e35_r7_runtime_sha256.txt` after extraction.
It pins SGQS-E to commit `6fd2c301bd8c09ff3d49fa077c34eaf75ef35b9a`
and includes the provenance/contract preflight used before any new run.
