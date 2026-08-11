from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "aggregate_calibrated_relation_tube_multiseed_k3",
    ROOT / "scripts/aggregate_calibrated_relation_tube_multiseed_k3.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def config() -> dict:
    return yaml.safe_load(
        (
            ROOT
            / "configs/config_calibrated_relation_tube_multiseed_aggregate_k3.yaml"
        ).read_text(encoding="utf-8")
    )


def metrics(r2: float, corr: float) -> dict:
    return {"mean_r2": r2, "mean_direct_abs_correlation": corr}


def seed_result(tube_r2: float, projected_r2: float) -> dict:
    return {
        "tube_machine_decision": {
            "mechanism_checks": {"a": True, "b": True},
        },
        "runs": {
            "tube_correct": {
                "raw": metrics(tube_r2 - 0.2, 0.84),
                "full_affine": metrics(tube_r2, 0.89),
            },
            "projected_physical_correct": {
                "raw": metrics(projected_r2 - 0.1, 0.82),
                "full_affine": metrics(projected_r2, 0.86),
            },
            "unbounded_correct": {
                "raw": metrics(-1.0, 0.3),
                "full_affine": metrics(0.1, 0.4),
            },
            "tube_permuted_matched": {
                "raw": metrics(-0.2, 0.2),
                "full_affine": metrics(0.2, 0.5),
            },
        },
        "test_evaluated": False,
    }


def test_config_freezes_three_seeds_and_readout() -> None:
    value = config()
    MODULE.validate_config(value)
    assert value["evaluation"]["seeds"] == [0, 42, 3407]
    assert value["calibration"]["budget"] == 80
    assert value["calibration"]["ridge"] == 0.001


def test_decision_requires_all_three_seeds() -> None:
    results = {
        "0": seed_result(0.76, 0.70),
        "42": seed_result(0.72, 0.68),
        "3407": seed_result(0.79, 0.72),
    }
    decision = MODULE.decide(results, config())
    assert decision["verdict"] == "relation_tube_multiseed_structure_supported"
    assert decision["positive_tube_minus_projected_seed_count"] == 3


def test_decision_reports_two_of_three_without_mean_rescue() -> None:
    results = {
        "0": seed_result(0.76, 0.70),
        "42": seed_result(0.64, 0.68),
        "3407": seed_result(0.79, 0.72),
    }
    decision = MODULE.decide(results, config())
    assert decision["verdict"] == "relation_tube_multiseed_mixed_two_of_three"
    assert decision["per_seed"]["42"]["all_checks_pass"] is False


def test_main_does_not_read_test_split() -> None:
    source = inspect.getsource(MODULE.main)
    assert "test_end" not in source
    assert '"test_evaluated": False' in source
