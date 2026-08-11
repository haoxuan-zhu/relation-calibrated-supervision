from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_relation_tube_radius_scale_ablation_k15 as k15


def master_config() -> dict:
    return yaml.safe_load(
        (ROOT / "configs" / "config_relation_tube_radius_scale_ablation_k15.yaml").read_text(
            encoding="utf-8"
        )
    )


def empirical_geometry() -> dict:
    return {
        "precision": np.eye(3) * 4.0,
        "radius_squared": 4.0 * 0.8**2,
        "covariance": np.eye(3) / 4.0,
    }


def test_k15_contract_is_frozen() -> None:
    config = master_config()
    k15.validate_master(config)
    assert k15.TRAIN_MODES == ("half_empirical", "fixed_unit", "double_empirical")
    assert k15.SEEDS == (3407, 0, 42)


def test_scaled_geometries_have_exact_effective_radii() -> None:
    config = master_config()
    expected = {"half_empirical": 0.4, "fixed_unit": 1.0, "double_empirical": 1.6}
    for mode, target in expected.items():
        geometry, audit = k15.scaled_geometry(empirical_geometry(), mode, config)
        assert np.isclose(audit["effective_euclidean_radius"], target)
        observed = np.sqrt(geometry["radius_squared"] / geometry["precision"][0][0])
        assert np.isclose(observed, target)
        assert audit["finite"] is True


def test_run_config_changes_protocol_but_not_training_seed_or_budget() -> None:
    config = master_config()
    parent = yaml.safe_load(
        (ROOT / "configs" / "config_relation_tube_geometry_ablation_k4_seed3407.yaml").read_text(
            encoding="utf-8"
        )
    )
    run = k15.run_config(parent, config, 3407, "half_empirical")
    assert run["protocol_version"] == k15.PROTOCOL
    assert run["calibration"]["budget"] == 80
    assert run["training"]["seed"] == 3407
    assert run["training"]["epochs"] == 100
    assert run["selected_radius_run"] == {"seed": 3407, "mode": "half_empirical"}
