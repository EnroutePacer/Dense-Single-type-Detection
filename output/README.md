# Experiment outputs

This directory is a durable research record, not disposable build output.

- Commit text logs, metrics, summaries, resolved configs, commands and manifests with ordinary Git.
- Commit checkpoints, evaluator tensors, TensorBoard event files and transfer archives through Git LFS.
- Never reuse a run directory. Failed, interrupted and superseded runs remain recorded and should be labelled instead of deleted.
- Every committed artifact is listed in `ARTIFACT_MANIFEST.csv` with its byte size, SHA-256 and Git storage class.
- Before pushing, inspect logs and configs for credentials, tokens and private dataset paths.

The files restored on 2026-10-12 came from the unchanged local safety copy at `_local/cleanup_20261012/output/`. That safety copy is intentionally not overwritten or removed.
