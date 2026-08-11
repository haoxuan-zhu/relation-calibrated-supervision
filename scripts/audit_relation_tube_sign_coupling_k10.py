"""Audit sign compensation and local cross-axis coupling of relation tubes."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

import run_calibrated_relation_tube_k0 as tube
import run_conflict_projected_physics_k80_closed as v17
import run_diagnostic as base
import run_dynamic_residual_propagation_k0 as dynamic
import run_physics_functional_anchor_training as v11
import run_relation_tube_calibration_subset_k8 as k8
import run_relation_tube_oas_shrinkage_k9 as k9


PROTOCOL = "relation_tube_sign_coupling_k10_v1"
GEOMETRIES = ("full_anisotropic", "oas", "diagonal", "isotropic")


def validate_config(config: dict[str, Any]) -> None:
    if config["protocol_version"] != PROTOCOL:
        raise ValueError("unexpected K10 protocol")
    if tuple(int(seed) for seed in config["subsets"].keys()) != k8.SUBSET_SEEDS:
        raise ValueError("K10 subset registry changed")
    if not np.isclose(float(config["audit"]["axis_probe_score_multiple"]), 2.25):
        raise ValueError("K10 axis-probe scale changed")
    if config["audit"]["test_evaluated"] is not False:
        raise ValueError("K10 test must remain closed")


def load_json(path_text: str, expected_sha: str, label: str) -> tuple[dict, Path]:
    path = Path(path_text).resolve()
    if base.sha256_file(path) != expected_sha:
        raise ValueError(f"{label} hash mismatch")
    return json.loads(path.read_text(encoding="utf-8")), path


def summary(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "maximum": float(array.max()),
    }


def sign_orbit_audit(
    residuals: np.ndarray, geometry: dict[str, Any], tolerance: float
) -> dict[str, Any]:
    magnitudes = np.abs(np.asarray(residuals, dtype=np.float64))
    signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=3)))
    orbit = magnitudes[:, None, :] * signs[None, :, :]
    precision = np.asarray(geometry["precision"], dtype=np.float64)
    radius = float(geometry["radius_squared"])
    scores = np.einsum("nsi,ij,nsj->ns", orbit, precision, orbit)
    score_min = np.maximum(scores.min(axis=1), 1e-15)
    score_ratio = scores.max(axis=1) / score_min
    scales = np.minimum(1.0, np.sqrt(radius / np.maximum(scores, 1e-15)))
    scale_range = scales.max(axis=1) - scales.min(axis=1)
    norms = np.linalg.norm(orbit, axis=2)
    correction = (1.0 - scales) * norms
    correction_range = correction.max(axis=1) - correction.min(axis=1)
    boundary = scores > radius
    boundary_flip = np.any(boundary, axis=1) & ~np.all(boundary, axis=1)
    return {
        "score_sign_ratio": summary(score_ratio),
        "projection_scale_range": summary(scale_range),
        "radial_correction_l2_range": summary(correction_range),
        "boundary_decision_flip_fraction": float(boundary_flip.mean()),
        "sign_invariant": bool(
            np.max(np.abs(scores - scores[:, :1])) <= tolerance
        ),
        "sample_count": int(len(residuals)),
        "sign_count": int(len(signs)),
    }


def axis_probe_audit(
    geometry: dict[str, Any], score_multiple: float, tolerance: float
) -> dict[str, Any]:
    precision = np.asarray(geometry["precision"], dtype=np.float64)
    radius = float(geometry["radius_squared"])
    dimension = precision.shape[0]
    rows = []
    for axis in range(dimension):
        residual = np.zeros(dimension, dtype=np.float64)
        residual[axis] = np.sqrt(score_multiple * radius / precision[axis, axis])
        score = float(residual @ precision @ residual)
        scale = float(np.sqrt(radius / score))
        jacobian = scale * np.eye(dimension) - scale * np.outer(
            residual, precision @ residual
        ) / score
        off_diagonal = jacobian - np.diag(np.diag(jacobian))
        active_row = jacobian[axis]
        cross = np.delete(active_row, axis)
        rows.append(
            {
                "axis": axis,
                "score_multiple": score / radius,
                "projection_scale": scale,
                "off_diagonal_frobenius_ratio": float(
                    np.linalg.norm(off_diagonal) / max(np.linalg.norm(jacobian), 1e-15)
                ),
                "active_row_cross_ratio": float(
                    np.linalg.norm(cross) / max(np.linalg.norm(active_row), 1e-15)
                ),
                "jacobian": jacobian,
            }
        )
    maximum = max(row["off_diagonal_frobenius_ratio"] for row in rows)
    return {
        "axes": rows,
        "maximum_off_diagonal_frobenius_ratio": maximum,
        "cross_axis_local_coupling": bool(maximum > tolerance),
    }


def recompute_residuals(config: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    images = np.load(Path(config["dataset"]["cache_path"]), mmap_mode="r")
    raw_latents = base.load_latents(config).astype(np.float64)
    subset, subset_audit = dynamic.build_subset(config)
    train_end = int(config["split"]["train_end"])
    features = tube.calibration_features(images, raw_latents, subset, train_end)
    targets = raw_latents[0, subset, :3].astype(np.float64) / 255.0
    _, loo_predictions, teacher_audit = tube.fit_teacher(
        features,
        targets,
        float(config["calibration"]["raw_ridge_alpha"]),
        bool(config["calibration"]["clip_predictions_to_unit_interval"]),
    )
    _, _, latent_std = v17.normalize_with_k80_rgb_statistics(
        raw_latents, train_end, subset
    )
    residuals = (targets - loo_predictions) * 255.0 / latent_std[:3]
    return residuals, {"subset": subset_audit, "teacher": teacher_audit}


def run(config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    tolerance = float(config["audit"]["sign_tolerance"])
    score_multiple = float(config["audit"]["axis_probe_score_multiple"])
    subsets: dict[str, Any] = {}
    for seed_text, item in config["subsets"].items():
        seed = int(seed_text)
        k8_config_path = Path(item["k8_config_path"]).resolve()
        if base.sha256_file(k8_config_path) != item["k8_config_sha256"]:
            raise ValueError(f"K10 K8 config hash mismatch: {seed}")
        k8_config = yaml.safe_load(k8_config_path.read_text(encoding="utf-8"))
        k8.validate_config(k8_config)
        k8_preflight, k8_path = load_json(
            item["k8_preflight_path"], item["k8_preflight_sha256"], f"K8 {seed}"
        )
        k9_preflight, k9_path = load_json(
            item["k9_preflight_path"], item["k9_preflight_sha256"], f"K9 {seed}"
        )
        residuals, recomputed = recompute_residuals(k8_config)
        checks = {
            "subset_matches_k8": recomputed["subset"] == k8_preflight["subset"],
            "teacher_coefficients_match_k8": recomputed["teacher"][
                "coefficients_sha256"
            ]
            == k8_preflight["teachers"]["correct"]["audit"][
                "coefficients_sha256"
            ],
            "k9_subset_matches_k8": k9_preflight["subset"] == k8_preflight["subset"],
            "test_not_read": True,
        }
        if not all(checks.values()):
            raise RuntimeError(f"K10 input reproduction failed: {seed}/{checks}")
        geometries = {
            "full_anisotropic": k8_preflight["geometries"]["full_anisotropic"],
            "oas": k9_preflight["geometry"],
            "diagonal": k8_preflight["geometries"]["diagonal"],
            "isotropic": k8_preflight["geometries"]["isotropic"],
        }
        audits = {
            name: {
                "sign_orbit": sign_orbit_audit(residuals, geometry, tolerance),
                "axis_probe": axis_probe_audit(
                    geometry, score_multiple, tolerance
                ),
            }
            for name, geometry in geometries.items()
        }
        if not audits["diagonal"]["sign_orbit"]["sign_invariant"]:
            raise RuntimeError("diagonal geometry lost sign invariance")
        if not audits["isotropic"]["sign_orbit"]["sign_invariant"]:
            raise RuntimeError("isotropic geometry lost sign invariance")
        subsets[str(seed)] = {
            "inputs": {
                "k8_config": {
                    "path": str(k8_config_path),
                    "sha256": base.sha256_file(k8_config_path),
                },
                "k8_preflight": {
                    "path": str(k8_path),
                    "sha256": base.sha256_file(k8_path),
                },
                "k9_preflight": {
                    "path": str(k9_path),
                    "sha256": base.sha256_file(k9_path),
                },
            },
            "checks": checks,
            "audits": audits,
        }
    full_nonzero = all(
        subsets[str(seed)]["audits"]["full_anisotropic"]["axis_probe"][
            "cross_axis_local_coupling"
        ]
        for seed in k8.SUBSET_SEEDS
    )
    oas_nonzero = all(
        subsets[str(seed)]["audits"]["oas"]["axis_probe"][
            "cross_axis_local_coupling"
        ]
        for seed in k8.SUBSET_SEEDS
    )
    axis_separable_zero = all(
        not subsets[str(seed)]["audits"][name]["axis_probe"][
            "cross_axis_local_coupling"
        ]
        for seed in k8.SUBSET_SEEDS
        for name in ("diagonal", "isotropic")
    )
    return {
        "protocol_version": PROTOCOL,
        "mode": "calibration_residual_sign_orbit_and_axis_probe_audit",
        "config_path": str(config_path),
        "config_sha256": base.sha256_file(config_path),
        "subsets": subsets,
        "decision": {
            "full_cross_axis_coupling_all_subsets": full_nonzero,
            "oas_cross_axis_coupling_all_subsets": oas_nonzero,
            "diagonal_and_isotropic_zero_axis_probe_coupling_all_subsets": (
                axis_separable_zero
            ),
            "structural_mechanism_supported": bool(
                full_nonzero and oas_nonzero and axis_separable_zero
            ),
            "downstream_causality_claimed": False,
            "test_evaluated": False,
        },
        "test_evaluated": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    output = run(config_path, config)
    output_path = Path(config["output_path"]).resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite K10 result: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    v11.atomic_json(output_path, output)


if __name__ == "__main__":
    main()
