from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_relation_tube_instance_information_audit_k14 as k14


def master_config() -> dict:
    return yaml.safe_load(
        (
            ROOT
            / "configs"
            / "config_relation_tube_instance_information_audit_k14.yaml"
        ).read_text(encoding="utf-8")
    )


def test_k14_contract_and_derangement_are_frozen() -> None:
    config = master_config()
    k14.validate_master(config)
    first = k14.make_derangement(1000, 20260803)
    second = k14.make_derangement(1000, 20260803)
    assert np.array_equal(first, second)
    assert sorted(first.tolist()) == list(range(1000))
    assert not np.any(first == np.arange(1000))


def test_project_residual_matches_radial_tube_rule() -> None:
    embedding = SimpleNamespace(
        tube_precision=torch.eye(3), tube_radius_squared=torch.tensor(4.0)
    )
    raw = torch.tensor([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    projected, scale, score = k14.project_residual(embedding, raw)
    assert torch.allclose(score, torch.tensor([1.0, 9.0]))
    assert torch.allclose(scale, torch.tensor([1.0, 2.0 / 3.0]))
    assert torch.allclose(projected, torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]))


def test_frozen_readout_prefers_matched_predictions() -> None:
    rng = np.random.default_rng(17)
    calibration_truth = rng.normal(size=(80, 3))
    calibration_prediction = calibration_truth + rng.normal(scale=0.01, size=(80, 3))
    validation_truth = rng.normal(size=(200, 3))
    correct = validation_truth + rng.normal(scale=0.02, size=(200, 3))
    control = correct[k14.make_derangement(200, 19)]
    readout = k14.fit_frozen_readout(calibration_prediction, calibration_truth, 0.001)
    correct_metrics, correct_predictions = k14.apply_frozen_readout(
        correct, validation_truth, readout
    )
    control_metrics, control_predictions = k14.apply_frozen_readout(
        control, validation_truth, readout
    )
    assert (
        correct_metrics["full_affine"]["mean_r2"]
        > control_metrics["full_affine"]["mean_r2"]
    )
    bootstrap = k14.paired_bootstrap_mse_advantage(
        correct_predictions["full_affine"],
        control_predictions["full_affine"],
        validation_truth,
        200,
        23,
    )
    assert bootstrap["observed_control_minus_correct_mse"] > 0.0
    assert bootstrap["ci_strictly_positive"] is True


def test_reassignment_preserves_marginal_residual_distribution() -> None:
    rng = np.random.default_rng(31)
    residual = rng.normal(size=(1000, 3))
    permutation = k14.make_derangement(1000, 20260803)
    reassigned = residual[permutation]
    assert np.allclose(np.sort(np.linalg.norm(residual, axis=1)), np.sort(np.linalg.norm(reassigned, axis=1)))
    assert np.allclose(residual.mean(axis=0), reassigned.mean(axis=0))


def test_constant_residual_metrics_serialize_undefined_correlations_as_none() -> None:
    rng = np.random.default_rng(41)
    truth = rng.normal(size=(80, 3))
    prediction = np.zeros_like(truth)
    metrics = k14.safe_regression_metrics(prediction, truth)
    assert metrics["mean_direct_abs_correlation"] is None
    assert metrics["direct_abs_correlation"] == [None, None, None]
    cosine = k14.cosine_summary(prediction, truth)
    assert cosine["mean"] is None
    assert cosine["median"] is None


def test_nonconstant_metrics_follow_original_evaluator_exactly() -> None:
    rng = np.random.default_rng(43)
    truth = rng.normal(size=(100, 3)).astype(np.float32)
    prediction = (truth + rng.normal(scale=0.1, size=(100, 3))).astype(np.float32)
    expected = k14.v6.regression_metrics(prediction, truth)
    observed = k14.safe_regression_metrics(prediction, truth)
    assert observed.keys() == expected.keys()
    for key in ("mean_r2", "mean_direct_abs_correlation", "mse"):
        assert observed[key] == expected[key]
    assert np.array_equal(observed["r2"], expected["r2"])
    assert observed["direct_abs_correlation"] == expected["direct_abs_correlation"]
