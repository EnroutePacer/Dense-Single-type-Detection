# Table-cell training configurations

## Active configuration tree

```text
generic dataset/runtime/optimizer/model bases
├── dfine_hgnetv2_m_table_cell_fixed640.yml
└── dfine_hgnetv2_s_table_cell.yml
```

`dfine_hgnetv2_m_table_cell_fixed640.yml` is the only active D-FINE-M control
base. It fixes train and validation input to 640x640, keeps 640 queries, uses
VFL, and disables scale sampling and MixUp. SGQS uses this same control through
`configs/sgqs/`; historical probes are preserved in the article evidence and
Git history rather than the active configuration tree.

This directory intentionally contains task bases only. The controlled 35-epoch
DEIM-core run and the epoch-27 continuation screen live together under
`configs/e35_r7/`; method definitions remain under `configs/sgqs/` and
`configs/sgqs_e/`. This keeps one-off experiment schedules out of the canonical
task-base directory.

## Historical material

Invalid, superseded and completed probe configurations are not active runtime
entry points. Their conclusions and provenance remain under
`article/experiments/`, `article/docs/` and the repository history.
