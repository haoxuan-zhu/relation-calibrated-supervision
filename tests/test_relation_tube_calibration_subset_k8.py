from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "relation_tube_calibration_subset_k8",
    ROOT / "scripts/run_relation_tube_calibration_subset_k8.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_k8_configs_use_the_three_pre_registered_subsets() -> None:
    seen = set()
    for subset_seed in MODULE.SUBSET_SEEDS:
        path = ROOT / f"configs/config_relation_tube_calibration_subset_k8_{subset_seed}.yaml"
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        MODULE.validate_config(config)
        seen.add(config["calibration"]["subset_seed"])
        assert config["audit"]["test_evaluated"] is False
    assert seen == set(MODULE.SUBSET_SEEDS)


def test_k8_geometries_match_coverage_without_changing_centers() -> None:
    residuals = np.random.default_rng(9).normal(size=(80, 3))
    registry, audits = MODULE.build_geometry_registry(residuals, 0.95, 1e-6)
    assert set(registry) == set(MODULE.GEOMETRIES)
    assert {audits[name]["score_order_index_one_based"] for name in MODULE.GEOMETRIES} == {77}
    assert len({audits[name]["empirical_coverage"] for name in MODULE.GEOMETRIES}) == 1


def test_k8_candidate_comparison_keeps_semantic_and_graph_separate() -> None:
    def run(raw_corr, raw_r2, mcc, coordinatewise, full_r2, edge, shd, ccrl):
        return {
            "semantic": {
                "raw": {"mean_direct_abs_correlation": raw_corr, "mean_r2": raw_r2},
                "rgb_hungarian_mcc": mcc,
                "coordinatewise_affine": {"mean_r2": coordinatewise},
                "full_affine": {"mean_r2": full_r2},
            },
            "graph": {"edge_auroc": edge, "fixed_threshold_shd": shd},
            "ccrl_validation": {"total": ccrl},
        }

    full = run(0.8, 0.5, 0.8, 0.7, 0.75, 0.6, 8, 0.1)
    candidate = run(0.85, 0.6, 0.85, 0.75, 0.8, 0.65, 7, 0.2)
    comparison = MODULE.comparison(full, candidate)
    assert comparison["complete_semantic_advantage"] is True
    assert comparison["graph_advantage"] is True
    assert comparison["candidate_minus_full_ccrl_total"] > 0
