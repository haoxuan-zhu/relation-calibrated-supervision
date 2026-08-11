# Reproduction commands

Run these commands from the repository root after installing the package:

```bash
python -m pip install -e ".[analysis,test]"
```

All programs refuse to overwrite locked outputs. Use a new output directory for a fresh run. Dataset locations in the exported configurations are relative to `data/`; obtain the fixed upstream revisions listed in [DATASETS.md](DATASETS.md) before training.

## Rebuild the reported tables and figure

```bash
PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider
python analysis/build_tables.py
python analysis/plot_budget_curve.py \
  --aggregate outputs/relation_budget_curve_v1/aggregate/aggregate_results.json \
  --raw-aggregate outputs/raw_ridge_propagation_aggregate_v1/raw/raw_aggregate_results.json \
  --output-prefix analysis/generated/relation_budget_curve
python tools/audit_anonymous_repository.py .
```

## Light Tunnel supervision and Relation Tube

One budget-condition run:

```bash
python scripts/run_relation_budget_condition.py \
  --config configs/config_relation_budget_curve_v1.yaml \
  --budget 80 --condition-kind physical --seed 3407 \
  --output-dir outputs/reproduction/light_tunnel_k80_physical
```

The selected Tube can be checked before training and then run with the same frozen configuration:

```bash
python scripts/run_calibrated_relation_tube_k0.py \
  --config configs/config_calibrated_relation_tube_k0_seed3407.yaml \
  --mode preflight --output-root outputs/reproduction/tube_k80_seed3407
for condition in center_only_correct unbounded_correct tube_correct tube_permuted_matched; do
  python scripts/run_calibrated_relation_tube_k0.py \
    --config configs/config_calibrated_relation_tube_k0_seed3407.yaml \
    --mode train --condition "$condition" \
    --output-root outputs/reproduction/tube_k80_seed3407
done
python scripts/run_calibrated_relation_tube_k0.py \
  --config configs/config_calibrated_relation_tube_k0_seed3407.yaml \
  --mode readout \
  --output-root outputs/reproduction/tube_k80_seed3407
```

The four-budget isotropic protocol uses the same `preflight`, `train`, and `readout` sequence:

```bash
python scripts/run_isotropic_relation_tube_budget_k11.py \
  --config configs/config_isotropic_relation_tube_budget_k11.yaml \
  --mode preflight --budget 20 \
  --output-root outputs/reproduction/isotropic_k11
python scripts/run_isotropic_relation_tube_budget_k11.py \
  --config configs/config_isotropic_relation_tube_budget_k11.yaml \
  --mode train --budget 20 --seed 3407 \
  --output-root outputs/reproduction/isotropic_k11
python scripts/run_isotropic_relation_tube_budget_k11.py \
  --config configs/config_isotropic_relation_tube_budget_k11.yaml \
  --mode readout --budget 20 --seed 3407 \
  --output-root outputs/reproduction/isotropic_k11
```

The once-held-out K13 interface is released for verification, but it requires the K11 checkpoints named by its configuration; those large files are not stored in the review repository.

The target-fidelity control keeps each measured target unchanged and propagates the fitted relation only to the remaining rows:

```bash
python scripts/run_anchor_preserving_raw_ridge_condition.py \
  --config configs/config_anchor_preserving_raw_ridge_v1.yaml \
  --budget 80 --condition-kind empirical --seed 3407 \
  --output-dir outputs/reproduction/anchor_preserving_k80_seed3407
```

The compact outputs for all 12 anchor-preserving runs and the matched K=80 bounded/unbounded held-out audit are included for deterministic table regeneration. Re-evaluating the latter from checkpoints requires the archived Light Tunnel checkpoints and the exact source versions recorded by its configuration; those large historical assets are not copied into the review repository.

## Cross-system evaluations

Spring relation supervision:

```bash
python scripts/run_causalverse_spring_correctness_v3.py \
  --config configs/config_causalverse_spring_correctness_k40_v4.yaml \
  --mode train --condition correct_relation --seed 3407 \
  --output outputs/reproduction/spring_correct_seed3407 --device cuda
```

MOVi-B formal validation uses the released frozen config and its locked output path:

```bash
python scripts/evaluate_movi_collision_formal_relation_v1.py \
  --config outputs/movi_sparse_collision_formal_v2/configs_v2/formal_validation_v3.yaml \
  --output data/movi_b_128/formal_v4/results_v3/formal_validation_v3.json
```

Resistance spot welding validation:

```bash
python scripts/evaluate_resistance_spot_welding_relation_v1.py \
  --config configs/config_resistance_spot_welding_relation_k43_validation.yaml \
  --output data/resistance_spot_welding_v3/formal_k43/formal_validation_v1.json
```

The MOVi-B and welding commands require the feature archives named in their configurations. The compact result JSON files are included so table regeneration does not depend on those large archives.
