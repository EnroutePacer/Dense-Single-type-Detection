# Table-cell training configurations

## Canonical versus evidence configurations

The repository-root `configs/table_cells/` directory is the canonical runtime
entry and intentionally contains only the fixed-640 M base and the S base.
This `article/` mirror also retains historical probe configurations as research
evidence; their presence here does not make them active training defaults.

`dfine_hgnetv2_m_table_cell_fixed640.yml` is the only active D-FINE-M control
base. It fixes the model input and validation input to 640x640, keeps 640
queries, uses VFL, and disables scale sampling and MixUp.

The DEIM probe changes only the classification loss from VFL to MAL and the
training augmentation from the plain control pipeline to Dense O2O. It copies
those method fields explicitly instead of including `configs/base/deim.yml`,
so activation and learning-rate scheduling remain identical to the control.

The current controlled 35-epoch DEIM-core and epoch-27 continuation screen are
mirrored under `article/configs/e35_r7/` and launched from the repository-root
tooling. They are kept outside `table_cells/` so task bases and one-off schedules
cannot be confused.

## Compatibility and legacy files

- `dfine_hgnetv2_m_table_cell.yml` is only a compatibility alias to the active
  fixed-640 base. New configs must not inherit it.
- `dfine_hgnetv2_m_table_cell_fixed640_probe.yml` is also a compatibility alias.
- `dfine_hgnetv2_m_table_cell_multiscale_legacy.yml` archives the abandoned
  Resize-800 plus 640/704/800 scale-sampling recipe.
- `dfine_hgnetv2_m_table_cell_weighted_singlepass_probe.yml` is retained only
  to reproduce the abandoned high-scale experiment.
