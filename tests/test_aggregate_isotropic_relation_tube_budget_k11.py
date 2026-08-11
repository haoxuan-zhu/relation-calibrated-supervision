from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import aggregate_isotropic_relation_tube_budget_k11 as aggregate
import run_isotropic_relation_tube_budget_k11 as k11


def test_aggregate_counts_seedwise_directions_without_hard_gate() -> None:
    entries = {}
    comparator = {"entries": {}}
    for budget_index, budget in enumerate(k11.BUDGETS):
        entries[budget] = {}
        comparator["entries"][str(budget)] = {}
        for condition in aggregate.COMPARATORS:
            comparator["entries"][str(budget)][condition] = {}
        for seed_index, seed in enumerate(k11.SEEDS):
            value = 0.5 + 0.01 * budget_index + 0.001 * seed_index
            entries[budget][seed] = {
                "metrics": {
                    "raw_correlation": value,
                    "raw_r2": value - 0.2,
                    "rgb_mcc": value,
                    "coordinatewise_r2": value - 0.1,
                    "full_affine_r2": value - 0.05,
                    "ccrl_total": 0.1,
                    "edge_auroc": 0.6,
                    "shd": 5.0,
                }
            }
            for condition in aggregate.COMPARATORS:
                comparator["entries"][str(budget)][condition][str(seed)] = {
                    "validation_rgb_direct_abs_correlation": [0.4, 0.4, 0.4],
                    "validation_rgb_mean_direct_abs_correlation": 0.4,
                    "validation_rgb_mean_r2": 0.1,
                    "final_ccrl_validation_total": 0.2,
                    "graph": {"edge_auroc": 0.5, "fixed_threshold_shd": 6},
                }
    result = aggregate.aggregate(entries, comparator)
    assert result["curve_audit"]["raw_correlation_monotone_nondecreasing"] is True
    assert all(
        count == 3
        for count in result["curve_audit"][
            "isotropic_minus_point_raw_correlation_positive_counts"
        ].values()
    )
    assert result["decision"]["geometry_selection_reopened"] is False
    assert result["decision"]["test_evaluated"] is False
