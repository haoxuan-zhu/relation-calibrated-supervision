from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "relation_tube_geometry_crossed_k18",
    ROOT / "scripts/run_relation_tube_geometry_crossed_k18.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_k18_registers_the_six_missing_crossed_cells() -> None:
    seen = set()
    for subset_seed in MODULE.k8.SUBSET_SEEDS:
        for model_seed in MODULE.MODEL_SEEDS:
            path = (
                ROOT
                / "configs"
                / f"config_relation_tube_geometry_crossed_k18_s{subset_seed}_seed{model_seed}.yaml"
            )
            config = yaml.safe_load(path.read_text(encoding="utf-8"))
            MODULE.validate_config(config)
            assert config["audit"]["test_evaluated"] is False
            seen.add((config["calibration"]["subset_seed"], config["training"]["seed"]))
    assert len(seen) == 6


def test_k18_keeps_equal_coverage_geometries() -> None:
    residuals = np.random.default_rng(18).normal(size=(80, 3))
    registry, audits = MODULE.k8.build_geometry_registry(residuals, 0.95, 1e-6)
    assert set(MODULE.GEOMETRIES).issubset(registry)
    assert {
        audits[name]["score_order_index_one_based"] for name in MODULE.GEOMETRIES
    } == {77}
    assert len({audits[name]["empirical_coverage"] for name in MODULE.GEOMETRIES}) == 1


def test_k18_comparison_is_isotropic_minus_diagonal() -> None:
    def run(value: float) -> dict:
        return {
            "semantic": {
                "raw": {"mean_direct_abs_correlation": value, "mean_r2": value},
                "rgb_hungarian_mcc": value,
                "coordinatewise_affine": {"mean_r2": value},
                "full_affine": {"mean_r2": value},
            },
            "graph": {"edge_auroc": value, "fixed_threshold_shd": 8},
            "ccrl_validation": {"total": value},
        }

    comparison = MODULE.k8.comparison(run(0.4), run(0.6))
    assert comparison["semantic_deltas"]["raw_correlation"] > 0
    assert comparison["semantic_deltas"]["coordinatewise_r2"] > 0
