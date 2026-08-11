from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "calibrated_relation_tube_k0",
    ROOT / "scripts/run_calibrated_relation_tube_k0.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def config() -> dict:
    return yaml.safe_load(
        (
            ROOT
            / "configs/config_calibrated_relation_tube_k0_seed3407.yaml"
        ).read_text(encoding="utf-8")
    )


def dummy_teacher() -> dict[str, np.ndarray | float]:
    return {
        "alpha": 0.1,
        "feature_mean": np.zeros(4, dtype=np.float32),
        "feature_scale": np.ones(4, dtype=np.float32),
        "coefficients": np.zeros((5, 3), dtype=np.float32),
    }


def dummy_geometry(radius_squared: float = 1.0) -> dict[str, np.ndarray | float]:
    return {
        "covariance": np.eye(3, dtype=np.float32),
        "precision": np.eye(3, dtype=np.float32),
        "radius_squared": radius_squared,
    }


def test_frozen_config_and_condition_registry_are_exact() -> None:
    value = config()
    MODULE.validate_config(value)
    assert tuple(value["training"]["conditions"]) == MODULE.CONDITIONS
    assert value["training"]["shuffle_seed"] == 20260730
    assert MODULE.condition_spec("center_only_correct").train_encoder is False
    assert MODULE.condition_spec("tube_permuted_matched").center == "permuted"
    assert {
        "audit_physics_functional_anchor.py",
        "run_physics_functional_anchor_training.py",
        "run_supervised_continuation_diagnostic.py",
    }.issubset(MODULE.SOURCE_FILES)


def test_confirmation_protocol_accepts_only_registered_seed_hashes() -> None:
    value = config()
    value["protocol_version"] = MODULE.CONFIRMATION_PROTOCOL
    for seed, expected_hash in MODULE.REGISTERED_INITIAL_STATE_HASHES.items():
        candidate = yaml.safe_load(yaml.safe_dump(value))
        candidate["training"]["seed"] = seed
        candidate["initialization"]["seed"] = seed
        candidate["initialization"]["expected_state_dict_sha256"] = expected_hash
        MODULE.validate_config(candidate)
    value["training"]["seed"] = 42
    value["initialization"]["seed"] = 42
    value["initialization"]["expected_state_dict_sha256"] = "wrong"
    try:
        MODULE.validate_config(value)
    except ValueError as error:
        assert "initial-state hash" in str(error)
    else:
        raise AssertionError("unregistered initial state was accepted")


def test_seed0_seed42_confirmation_configs_are_frozen_and_valid() -> None:
    expected = {
        0: (0.8455364975177991, 0.5983039140701294),
        42: (0.8236767195182968, 0.4961322546005249),
    }
    for seed, projected in expected.items():
        value = yaml.safe_load(
            (
                ROOT / f"configs/config_calibrated_relation_tube_k3_seed{seed}.yaml"
            ).read_text(encoding="utf-8")
        )
        MODULE.validate_config(value)
        assert value["protocol_version"] == MODULE.CONFIRMATION_PROTOCOL
        assert value["training"]["seed"] == seed
        assert np.isclose(
            value["evaluation"]["projected_physical_validation_correlation"],
            projected[0],
        )
        assert np.isclose(
            value["evaluation"]["projected_physical_validation_r2"], projected[1]
        )


def test_tube_geometry_uses_registered_finite_sample_order_statistic() -> None:
    generator = np.random.default_rng(20260802)
    residuals = generator.normal(size=(80, 3))
    geometry, audit = MODULE.build_tube_geometry(
        residuals, coverage=0.95, diagonal_ridge=1e-6
    )
    assert audit["score_order_index_one_based"] == 77
    assert audit["empirical_coverage"] >= 0.95
    assert audit["finite"] is True
    assert np.linalg.eigvalsh(geometry["covariance"]).min() > 0
    assert isinstance(bool(audit["covariance_eigenvalues"].min() > 0), bool)


def test_tube_projection_is_inside_registered_ellipsoid() -> None:
    encoder = MODULE.CalibratedRelationTubeEncoder(
        learned_dim=3,
        anchor_dim=2,
        hidden_channels=8,
        conv_layers=2,
        teacher=dummy_teacher(),
        geometry=dummy_geometry(radius_squared=0.25),
        latent_mean=np.zeros(5, dtype=np.float32),
        latent_std=np.ones(5, dtype=np.float32),
        residual_mode="tube",
        clip_center=True,
    )
    x = torch.rand(7, 3, 64, 64)
    anchors = torch.zeros(7, 2)
    _, parts = encoder.components(x, anchors)
    projected_score = torch.einsum(
        "ni,ij,nj->n",
        parts["projected_residual"],
        encoder.tube_precision,
        parts["projected_residual"],
    )
    assert float(projected_score.detach().max()) <= 0.25 + 1e-5


def test_model_preserves_registered_trainable_state_contract() -> None:
    value = config()
    MODULE.base.set_seed(int(value["training"]["seed"]))
    model = MODULE.CalibratedRelationTubeModel(
        value,
        dummy_teacher(),
        dummy_geometry(),
        np.zeros(5, dtype=np.float32),
        np.ones(5, dtype=np.float32),
        "tube",
    )
    state = model.state_dict()
    baseline_state, baseline_hash = MODULE.v16.make_initial_state(
        value, torch.device("cpu")
    )
    assert not any(key.startswith("embedding.tube_") for key in state)
    assert state.keys() == baseline_state.keys()
    assert MODULE.v6.sha256_state_dict(state) == baseline_hash
    assert sum(parameter.numel() for parameter in model.parameters()) == value[
        "implementation"
    ]["expected_parameter_count"]


def test_decision_separates_mechanism_from_paper_competitiveness() -> None:
    semantic = {
        "center_only_correct": {
            "mean_direct_abs_correlation": 0.80,
            "mean_r2": 0.50,
        },
        "unbounded_correct": {
            "mean_direct_abs_correlation": 0.79,
            "mean_r2": 0.48,
        },
        "tube_correct": {
            "mean_direct_abs_correlation": 0.82,
            "mean_r2": 0.56,
        },
        "tube_permuted_matched": {
            "mean_direct_abs_correlation": 0.60,
            "mean_r2": 0.10,
        },
    }
    ccrl = {
        condition: {"total": 0.2} for condition in MODULE.CONDITIONS
    }
    ccrl["center_only_correct"]["total"] = 0.21
    audits = {
        condition: {"boundary_active_fraction": 0.2}
        for condition in MODULE.CONDITIONS
    }
    decision = MODULE.decide(semantic, ccrl, audits, config())
    assert decision["verdict"] == "tube_structure_supported_not_paper_competitive"
    assert all(decision["mechanism_checks"].values())
    assert not all(decision["paper_competitive_checks"].values())


def test_direct_center_checks_are_native_json_booleans() -> None:
    value = config()
    semantic = {
        "center_only_correct": {
            "mean_direct_abs_correlation": value["evaluation"][
                "direct_center_validation_correlation"
            ],
            "mean_r2": value["evaluation"]["direct_center_validation_r2"],
        }
    }
    checks = MODULE.direct_center_checks(semantic, value)
    assert all(checks.values())
    assert all(type(flag) is bool for flag in checks.values())


def test_joint_readout_has_no_test_evaluator_path() -> None:
    source = inspect.getsource(MODULE.readout_main)
    assert "evaluate_model" not in source
    assert "test_end" not in source
    assert '"test_evaluated": False' in source
