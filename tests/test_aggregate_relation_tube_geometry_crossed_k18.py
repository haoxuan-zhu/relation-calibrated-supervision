from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "aggregate_relation_tube_geometry_crossed_k18",
    ROOT / "scripts/aggregate_relation_tube_geometry_crossed_k18.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def cell(delta: float) -> dict:
    comparison = {
        "semantic_deltas": {metric: delta for metric in MODULE.SEMANTIC_METRICS},
        "edge_auroc_delta": delta,
        "full_minus_candidate_shd": 1,
        "candidate_minus_full_ccrl_total": -delta,
    }
    return {
        "protocol_version": MODULE.NEW_PROTOCOL,
        "isotropic_minus_diagonal": comparison,
    }


def test_crossed_aggregate_supports_consistent_isotropic_advantage() -> None:
    results = {
        (subset, seed): cell(0.01)
        for subset in MODULE.SUBSET_SEEDS
        for seed in MODULE.MODEL_SEEDS
    }
    result = MODULE.aggregate(results)
    assert result["decision"]["verdict"] == "isotropic_crossed_robustness_supported"
    assert result["decision"]["raw_correlation_positive_count"] == 9


def test_crossed_aggregate_rejects_subset_reversal() -> None:
    results = {
        (subset, seed): cell(-0.01 if subset == 20260821 else 0.01)
        for subset in MODULE.SUBSET_SEEDS
        for seed in MODULE.MODEL_SEEDS
    }
    result = MODULE.aggregate(results)
    assert result["decision"]["verdict"] == "axis_separable_tradeoff_no_isotropic_separation"
    assert result["decision"]["subset_primary_means_positive"] is False
