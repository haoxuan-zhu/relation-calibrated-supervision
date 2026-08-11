from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_relation_tube_half_radius_instance_information_audit_k16 as k16


def master_config() -> dict:
    return yaml.safe_load(
        (ROOT / "configs" / "config_relation_tube_half_radius_instance_information_audit_k16.yaml").read_text(encoding="utf-8")
    )


def test_k16_contract_is_frozen_and_validation_only() -> None:
    config = master_config()
    k16.validate_master(config)
    assert config["upstream"]["radius_mode"] == "half_empirical"
    assert config["scope"]["test_evaluated"] is False
    assert config["evaluation"]["bootstrap_replicates"] == 2000


def test_k16_decision_requires_three_seed_directionality() -> None:
    config = master_config()
    controls = ("fixed_anchor_visual_derangement", "projected_residual_reassignment")

    def seed_result() -> dict:
        return {
            "deltas": {
                control: {
                    "correct_minus_control_output_full_affine_r2": 0.1,
                    "correct_minus_control_residual_full_affine_r2": 0.01,
                }
                for control in controls
            },
            "bootstrap": {
                "fixed_anchor_visual_derangement": {
                    "output_full_affine": {"ci_strictly_positive": True},
                    "residual_full_affine": {"ci_strictly_positive": True},
                }
            },
            "reproduction": {"metric": True},
            "permutation": {"fixed_points": 0},
            "test_evaluated": False,
        }

    results = {seed: seed_result() for seed in k16.SEEDS}
    decision = k16.decide(results, config)
    assert decision["verdict"] == "image_specific_signal_retained_under_half_radius"
    results[42]["deltas"][controls[0]]["correct_minus_control_output_full_affine_r2"] = -0.1
    assert k16.decide(results, config)["verdict"] == "mixed_signal_retention"


def test_k16_uses_the_same_deterministic_no_self_permutation_as_k14() -> None:
    first = k16.k14.make_derangement(1000, 20260803)
    second = k16.k14.make_derangement(1000, 20260803)
    assert np.array_equal(first, second)
    assert not np.any(first == np.arange(1000))
