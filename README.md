# Relation-Calibrated Supervision

This repository contains the implementation, experiment configurations, and evaluation summaries for *Relation-calibrated supervision for causal representation learning from sparse measurements*. The code studies how sparse measured labels can define constraints for a larger image or video collection while retaining direct measurements and observation-specific residual information.

## Contents

- `src/relation_tube/`: model-independent calibration, projection, and anchor-preserving target utilities.
- `scripts/`: training, evaluation, and aggregation programs used for the reported systems.
- `configs/`: frozen experiment configurations with repository-relative data and output paths.
- `outputs/`: compact JSON evidence used to generate the manuscript tables; raw datasets and checkpoints are not redistributed.
- `analysis/`: deterministic table and figure generation.
- `tests/`: unit and protocol tests for the released implementation.

The protocol-specific Light Tunnel clipping and normalization sequence is implemented in `scripts/run_calibrated_relation_tube_k0.py`. The model-independent `fit_ridge_center_with_loo` helper returns unconstrained predictions; bounded RGB calls use `RidgeCenter.predict(..., clip=True)` explicitly.

## Quick check

Python 3.10 or newer is required.

```bash
python -m pip install -e ".[analysis,test]"
PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider
python analysis/build_tables.py
python analysis/plot_budget_curve.py \
  --aggregate outputs/relation_budget_curve_v1/aggregate/aggregate_results.json \
  --raw-aggregate outputs/raw_ridge_propagation_aggregate_v1/raw/raw_aggregate_results.json \
  --output-prefix analysis/generated/relation_budget_curve
```

The compact JSON files are sufficient to regenerate the tables and budget figure without downloading datasets. Re-running feature extraction or training requires the public data listed in [DATASETS.md](DATASETS.md).
Exact principal commands and the status of large checkpoint/feature dependencies are given in [COMMANDS.md](COMMANDS.md). Frozen stage identifiers are explained in [PROTOCOLS.md](PROTOCOLS.md).

## Main experiment entry points

| Study | Entry point |
|---|---|
| Sparse relation supervision on Light Tunnel | `scripts/run_relation_budget_condition.py` |
| Calibrated Relation Tube | `scripts/run_calibrated_relation_tube_k0.py` |
| Isotropic Tube budget and held-out evaluation | `scripts/run_isotropic_relation_tube_budget_k11.py`, `scripts/run_isotropic_relation_tube_heldout_k13.py` |
| Anchor-preserving Light Tunnel propagation | `scripts/run_anchor_preserving_raw_ridge_condition.py`, `scripts/aggregate_anchor_preserving_comparison.py` |
| Matched bounded/unbounded held-out audit | `scripts/audit_bounded_unbounded_heldout_k45_v2.py` |
| CausalVerse Spring and Slope | `scripts/run_causalverse_spring_correctness_v3.py`, `scripts/evaluate_causalverse_slope_heldout_k26.py` |
| MOVi-B collision protocol | `scripts/evaluate_movi_collision_formal_relation_v1.py` |
| Resistance spot welding | `scripts/evaluate_resistance_spot_welding_relation_v1.py` |

The numbered suffixes identify frozen experimental protocols. They are retained because the JSON results and configurations refer to those identifiers.

## Data and code availability

The datasets are hosted by their original providers and are not copied into this repository. Source URLs, fixed revisions, licenses, and the exact subsets used are listed in [DATASETS.md](DATASETS.md). Compact evaluation summaries are included under `outputs/`; large feature caches and model checkpoints can be regenerated from the listed data and configurations.
