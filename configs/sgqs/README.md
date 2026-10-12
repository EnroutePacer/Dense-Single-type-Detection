# SGQS configurations

These configurations add Scale-aware GT Query Survival supervision before the
encoder global Top-K operation.  They keep the DEIM/D-FINE model, matcher,
decoder query budget, data pipeline, and inference path unchanged.

- `sgqs_hgnetv2_m_table_cell_fixed640.yml`: canonical fixed-640 SGQS training.
- `sgqs_hgnetv2_m_table_cell_fixed640_probe.yml`: continuation from the
  historical epoch-27 checkpoint; its survival ramp starts at epoch 28.

The first controlled comparison must use the fixed-640 VFL baseline.  Do not
combine SGQS with MAL, Mosaic, MixUp, small-object reweighting, or a different
query budget until the standalone ablation is complete.

The same code revision also masks invalid boundary anchors before selection for
both the control and SGQS configurations.  This is a shared correctness fix,
not an SGQS contribution.  Re-run the clean fixed-640 control on this revision;
do not attribute a comparison against a historical selector to SGQS.
