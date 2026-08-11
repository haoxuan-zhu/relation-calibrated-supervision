# Reproducibility guide

## Levels of reproduction

1. **Evidence rebuild.** Install the analysis dependencies and run `analysis/build_tables.py`. This reads only the locked JSON summaries under `outputs/`.
2. **Protocol tests.** Run `pytest -q` to check the released calibration, projection, data-interface, and aggregation code.
3. **Training rerun.** Download the corresponding public dataset, place it under `data/`, and run the configuration named for the study. Paths in the released YAML files are repository-relative.

The training programs write to `outputs/` by default. Large feature archives and checkpoints are intentionally absent from the review package.
Copy-ready commands for each principal interface are listed in [COMMANDS.md](COMMANDS.md).

## Principal configurations

- Light Tunnel supervision budget: `configs/config_relation_budget_curve_v1.yaml`
- Calibrated Relation Tube: `configs/config_calibrated_relation_tube_k0_seed3407.yaml`
- Tube budget sweep: `configs/config_isotropic_relation_tube_budget_k11.yaml`
- Spring replication: `configs/config_causalverse_spring_correctness_k40_v4.yaml`
- Slope held-out evaluation: `configs/config_causalverse_slope_heldout_k26.yaml`
- MOVi-B validation and held-out evaluation: `outputs/movi_sparse_collision_formal_v2/configs_v2/formal_validation_v3.yaml` and `formal_heldout_v3.yaml`
- Welding validation and held-out evaluation: `configs/config_resistance_spot_welding_relation_k43_validation.yaml` and `configs/config_resistance_spot_welding_relation_k43_heldout.yaml`

Every released file is listed in `release-manifest.json` with its SHA-256. Administrative path strings are replaced by repository-relative paths during export; numerical arrays, metrics, and protocol fields are not changed.
