from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import audit_causalverse_slope_projection_parity_k32 as parity
import causalverse_slope_preflight as slope


def test_unbounded_projection_recovers_exact_manifold_and_can_leave_box() -> None:
    parameters = slope.RelationParameters(0.9, 0.1, 0.5, 0.1, 9.81, 19.62)
    exact = parity.bounded_projection.state_from_free(
        np.asarray([1.2, 75.0, 16.0]), parameters
    )
    predicted = exact + np.asarray([0.02, -0.1, 0.1, 0.03, -0.02, 0.1, -0.05])
    projected, diagnostics = parity.project_unbounded(
        predicted[None, :],
        parameters,
        np.asarray([0.3, 12.0, 2.0, 0.25, 0.15, 2.5, 1.0]),
    )
    assert diagnostics["success_count"] == 1
    np.testing.assert_allclose(
        slope.relation_residuals_numpy(projected, parameters), 0.0, atol=1e-8
    )
    assert projected[0, 0] > 0.95


def test_aggregate_requires_all_seeds_all_variants_and_ci() -> None:
    passing = {
        "directional": {name: True for name in parity.UNBOUNDED_VARIANTS},
        "ci_positive": {name: True for name in parity.UNBOUNDED_VARIANTS},
        "projection_matches_or_exceeds": {name: False for name in parity.UNBOUNDED_VARIANTS},
    }
    outcomes = {str(seed): dict(passing) for seed in (3407, 42, 0)}
    assert parity.aggregate_decision(True, outcomes) == "published_projection_variants_beaten_all_seeds_ci"
    outcomes["42"] = {
        **passing,
        "ci_positive": {"unbounded_k40_std": False, "unbounded_k40_minmax": True},
    }
    assert parity.aggregate_decision(True, outcomes) == "published_projection_variants_beaten_directionally"


def test_projection_match_has_distinct_decision() -> None:
    outcome = {
        "directional": {"unbounded_k40_std": False, "unbounded_k40_minmax": True},
        "ci_positive": {"unbounded_k40_std": False, "unbounded_k40_minmax": True},
        "projection_matches_or_exceeds": {"unbounded_k40_std": True, "unbounded_k40_minmax": False},
    }
    assert (
        parity.aggregate_decision(True, {"3407": outcome})
        == "published_projection_matches_or_exceeds_relation"
    )
