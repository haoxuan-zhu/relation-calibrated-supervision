from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


METRICS = load(
    "anchor_preserving_locked_metrics",
    "scripts/audit_anchor_preserving_locked_metrics.py",
)
AGGREGATE = load(
    "anchor_preserving_comparison",
    "scripts/aggregate_anchor_preserving_comparison.py",
)


def test_metric_summary_preserves_budget_and_split_structure() -> None:
    metrics = {
        str(budget): {
            str(seed): {
                split: {
                    "mean_direct_abs_correlation": 0.5,
                    "mean_direct_r2": 0.2,
                }
                for split in ("validation", "test")
            }
            for seed in METRICS.SEEDS
        }
        for budget in METRICS.BUDGETS
    }
    value = METRICS.summarize(metrics)
    assert np.isclose(value["80"]["test"]["mean_direct_r2"]["mean"], 0.2)
    assert value["20"]["validation"]["mean_direct_abs_correlation"]["population_sd"] == 0.0


def test_scalar_summary_reports_direction_counts() -> None:
    value = AGGREGATE.summary([0.1, -0.2, 0.3])
    assert value["strict_positive_count"] == 2
    assert value["strict_negative_count"] == 1
    assert value["minimum"] == -0.2
