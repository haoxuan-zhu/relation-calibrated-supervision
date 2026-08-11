from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate_causalverse_slope_heldout_k26 as heldout


def metric(correlation: float, r2: float, free: float = 0.8) -> dict[str, float]:
    return {
        "mean_direct_abs_correlation_relation4": correlation,
        "mean_direct_r2_relation4": r2,
        "mean_direct_abs_correlation_free3": free,
    }


def test_decision_requires_point_permuted_projection_and_free_safety() -> None:
    metrics = {
        "point": metric(0.7, 0.5),
        "coefficient_permuted_relation": metric(0.6, 0.4),
        "all_coordinate_manifold_projection": metric(0.85, 0.75, 0.9),
        "correct_relation": metric(0.9, 0.8, 0.79),
    }
    flags = heldout.seed_flags(metrics, 0.05)
    assert all(flags.values())
    all_seeds = {"3407": flags, "42": flags, "0": flags}
    assert heldout.aggregate_decision(True, all_seeds) == "heldout_relation_content_and_projection_confirmed"
    mixed = {key: dict(value) for key, value in all_seeds.items()}
    mixed["42"]["correct_beats_projection"] = False
    assert heldout.aggregate_decision(True, mixed) == "heldout_relation_content_confirmed_projection_not_beaten"


def test_paired_bootstrap_is_deterministic_and_positive_for_better_predictions() -> None:
    rng = np.random.default_rng(5)
    true = rng.normal(size=(80, 7))
    correct = true + rng.normal(scale=0.05, size=true.shape)
    baseline = true + rng.normal(scale=0.5, size=true.shape)
    first = heldout.paired_bootstrap(true, correct, baseline, 100, 9, 0.95)
    second = heldout.paired_bootstrap(true, correct, baseline, 100, 9, 0.95)
    assert first == second
    assert first["relation4_correlation"]["lower"] > 0
    assert first["relation4_r2"]["lower"] > 0
