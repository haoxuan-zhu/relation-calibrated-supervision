from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import aggregate_isotropic_relation_tube_heldout_k13 as k13agg
import extract_heldout_baseline_comparator_k13 as baseline
import run_isotropic_relation_tube_budget_k11 as k11


def test_aggregate_reports_all_pair_heldout_advantage() -> None:
    entries = {}
    validation = {"entries": {}}
    baselines = {"entries": {}}
    for budget in k11.BUDGETS:
        entries[budget] = {}
        validation["entries"][str(budget)] = {}
        baselines["entries"][str(budget)] = {
            condition: {} for condition in baseline.CONDITIONS
        }
        for seed in k11.SEEDS:
            metrics = {
                "raw_correlation": 0.85,
                "raw_r2": 0.7,
                "rgb_mcc": 0.86,
                "coordinatewise_r2": 0.75,
                "full_affine_r2": 0.8,
            }
            entries[budget][seed] = {"metrics": metrics}
            validation["entries"][str(budget)][str(seed)] = {
                "metrics": {key: value - 0.01 for key, value in metrics.items()}
            }
            for condition in baseline.CONDITIONS:
                baselines["entries"][str(budget)][condition][str(seed)] = {
                    "mean_direct_abs_rgb_correlation": 0.7
                }
    result = k13agg.aggregate(entries, validation, baselines)
    assert result["decision"]["raw_ridge_strict_positive_count"] == 12
    assert (
        result["decision"]["verdict"]
        == "heldout_tube_advantage_over_raw_propagation_all_pairs"
    )
    assert result["budgets"]["20"]["test_minus_validation"][
        "raw_correlation"
    ]["mean"] == pytest.approx(0.01)
