"""Evaluate the preregistered formal MOVi relation-supervision table."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

import audit_movi_collision_conic_relation_tube_v1 as conic_v1
import audit_movi_collision_cross_fitted_residual_gate_v1 as gate_v1
import audit_movi_collision_physics_budget_v1 as budget_v1
import audit_movi_collision_physics_budget_v2 as budget_v2
import audit_movi_collision_relation_tube_pilot_v1 as tube_v1
import audit_movi_collision_state_headroom_v1 as headroom
import audit_movi_collision_visual_residual_v1 as visual_v1


PROTOCOL = "movi_collision_formal_relation_validation_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_config(config: dict[str, Any]) -> None:
    if config.get("protocol_version") != PROTOCOL:
        raise ValueError("unexpected formal MOVi relation protocol")
    phase = config.get("phase")
    if phase not in {"validation", "heldout"}:
        raise ValueError("phase must be validation or heldout")
    data = config["data"]
    if data.get("evaluation_split") != phase:
        raise ValueError("evaluation archive does not match the declared phase")
    expected_read = phase == "heldout"
    if data.get("heldout_target_evaluated") is not expected_read:
        raise ValueError("heldout read flag does not match the phase")
    architecture = config["architecture"]
    if architecture.get("visual_representation") != "equal_weight_multiview_v1":
        raise ValueError("registered visual representation changed")
    if architecture.get("normalize_full_state_block") is not True:
        raise ValueError("registered full-state block normalization changed")
    if float(architecture["ridge_alpha"]) != 0.1 or architecture.get("fit_intercept") is not True:
        raise ValueError("registered ridge changed")
    probe = config["probe"]
    if [int(value) for value in probe["budgets"]] != [20, 40, 80, 160]:
        raise ValueError("registered budgets changed")
    if [float(value) for value in probe["gain_bounds"]] != [1.0, 2.0]:
        raise ValueError("registered gain interval changed")
    if float(probe["coverage"]) != 0.95:
        raise ValueError("registered coverage changed")
    expected_integers = {
        "propagation_derangement_seed_base": 20264110,
        "residual_derangement_seed_base": 20264120,
        "relation_derangement_seed_base": 20264130,
        "bootstrap_seed_base": 20264200,
        "bootstrap_replicates": 5000,
        "subgroup_minimum_events": 20,
    }
    for key, expected in expected_integers.items():
        if int(probe[key]) != expected:
            raise ValueError(f"registered {key} changed")
    implementation = config.get("implementation", {}).get("files", [])
    if not implementation:
        raise ValueError("formal implementation hashes are missing")
    if not str(config.get("output", {}).get("audit_json_path", "")).endswith(".json"):
        raise ValueError("formal result path is not locked")


def verify_inputs(config: dict[str, Any]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for key in (
        "train_npz",
        "evaluation_npz",
        "train_feature_npz",
        "evaluation_feature_npz",
    ):
        path = Path(config["data"][f"{key}_path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = sha256_file(path)
        if observed != str(config["data"][f"{key}_sha256"]):
            raise ValueError(f"{key} hash changed")
        paths[key] = path
    if config["phase"] == "heldout":
        path = Path(config["data"]["validation_result_path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = sha256_file(path)
        if observed != str(config["data"]["validation_result_sha256"]):
            raise ValueError("validation result changed before held-out evaluation")
        paths["validation_result"] = path
    return paths


def verify_implementation(config: dict[str, Any]) -> dict[str, str]:
    observed = {}
    for row in config["implementation"]["files"]:
        path = Path(row["path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = sha256_file(path)
        if digest != str(row["sha256"]):
            raise ValueError(f"formal implementation changed: {path}")
        observed[str(path)] = digest
    return observed


def reorder_features(names: np.ndarray, archive: Any, key: str) -> np.ndarray:
    return tube_v1.reorder(names, np.asarray(archive["video_name"]), np.asarray(archive[key]))


def normalized_visual_blocks(names: np.ndarray, feature_archive: Any) -> tuple[np.ndarray, np.ndarray]:
    pair = reorder_features(names, feature_archive, "pair_cls").astype(np.float64)
    objects = reorder_features(names, feature_archive, "object_cls").astype(np.float64)
    motion = reorder_features(names, feature_archive, "mask_motion").astype(np.float64).reshape(
        len(names), -1
    )
    pair /= np.maximum(np.linalg.norm(pair, axis=2, keepdims=True), 1.0e-12)
    objects /= np.maximum(np.linalg.norm(objects, axis=3, keepdims=True), 1.0e-12)
    views = np.concatenate([pair.reshape(len(names), -1), objects.reshape(len(names), -1)], axis=1)
    return views, motion


def visual_features(
    train_names: np.ndarray,
    evaluation_names: np.ndarray,
    train_archive: Any,
    evaluation_archive: Any,
    architecture: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if architecture.get("visual_representation") != "equal_weight_multiview_v1":
        raise ValueError("unexpected visual representation")
    train_views, train_motion = normalized_visual_blocks(train_names, train_archive)
    eval_views, eval_motion = normalized_visual_blocks(evaluation_names, evaluation_archive)
    motion_mean = train_motion.mean(axis=0)
    motion_scale = train_motion.std(axis=0)
    train_motion = (train_motion - motion_mean) / np.where(motion_scale > 1.0e-8, motion_scale, 1.0)
    eval_motion = (eval_motion - motion_mean) / np.where(motion_scale > 1.0e-8, motion_scale, 1.0)
    train_motion /= np.maximum(np.linalg.norm(train_motion, axis=1, keepdims=True), 1.0e-12)
    eval_motion /= np.maximum(np.linalg.norm(eval_motion, axis=1, keepdims=True), 1.0e-12)
    scale = np.sqrt(10.0)
    train_visual = np.concatenate([train_views, train_motion], axis=1) / scale
    eval_visual = np.concatenate([eval_views, eval_motion], axis=1) / scale
    return train_visual, eval_visual, {
        "name": "equal_weight_multiview_v1",
        "dino_view_blocks": 9,
        "motion_block_dimension": train_motion.shape[1],
        "feature_dimension": train_visual.shape[1],
        "block_scale": float(scale),
    }


def rotate_pair(values: np.ndarray, transforms: np.ndarray) -> np.ndarray:
    return np.einsum("nij,nkj->nki", transforms, np.asarray(values, dtype=np.float64))


def camera_arrays(archive: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    transforms = np.asarray(archive["view_from_world"], dtype=np.float64)
    identity = np.eye(3)
    error = np.max(np.abs(transforms @ np.transpose(transforms, (0, 2, 1)) - identity))
    determinant = np.linalg.det(transforms)
    if error > 1.0e-5 or np.max(np.abs(determinant - 1.0)) > 1.0e-5:
        raise ValueError("camera transforms are not proper rotations")
    base = np.einsum("nij,nj->ni", transforms, np.asarray(archive["base_impulse"], dtype=np.float64))
    target = np.einsum("nij,nj->ni", transforms, np.asarray(archive["target"], dtype=np.float64))
    return transforms, base, target


def state_features(archive: Any, transforms: np.ndarray) -> np.ndarray:
    masses = np.asarray(archive["mass_pair"], dtype=np.float64)
    positions = rotate_pair(archive["position_pair"], transforms)
    velocity = rotate_pair(archive["linear_velocity_pair"], transforms)
    angular = rotate_pair(archive["angular_velocity_pair"], transforms)
    base = np.einsum("nij,nj->ni", transforms, np.asarray(archive["base_impulse"], dtype=np.float64))
    return np.concatenate(
        [masses, positions.reshape(len(masses), -1), velocity.reshape(len(masses), -1), angular.reshape(len(masses), -1), base],
        axis=1,
    )


def standardize(train: np.ndarray, evaluation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale = np.where(scale > 1.0e-8, scale, 1.0)
    return (train - mean) / scale, (evaluation - mean) / scale


def ridge(train_x: np.ndarray, train_y: np.ndarray, evaluation_x: np.ndarray, alpha: float) -> np.ndarray:
    return tube_v1.dual_ridge(train_x, train_y, evaluation_x, alpha, fit_intercept=True)


def isotropic_radius(base: np.ndarray, target: np.ndarray, bounds: tuple[float, float], coverage: float) -> float:
    residual_norms = []
    for index in range(len(base)):
        keep = np.arange(len(base)) != index
        gain = budget_v2.fit_bounded_gain(base[keep], target[keep], bounds)
        residual_norms.append(float(np.linalg.norm(target[index] - gain * base[index])))
    return conic_v1.rank_value(np.asarray(residual_norms), coverage)


def project_isotropic(point: np.ndarray, center: np.ndarray, radius: float) -> tuple[np.ndarray, float]:
    residual = point - center
    norms = np.linalg.norm(residual, axis=1)
    factors = np.minimum(1.0, radius / np.maximum(norms, 1.0e-12))
    return center + factors[:, None] * residual, float(np.mean(factors < 1.0 - 1.0e-12))


def cross_fitted_conic_branch(
    train_features: np.ndarray,
    evaluation_features: np.ndarray,
    train_base: np.ndarray,
    train_target: np.ndarray,
    evaluation_center: np.ndarray,
    calibration: np.ndarray,
    *,
    alpha: float,
    bounds: tuple[float, float],
    coverage: float,
) -> dict[str, Any]:
    residual_target = train_target[calibration] - (
        budget_v2.fit_bounded_gain(
            train_base[calibration], train_target[calibration], bounds
        )
        * train_base[calibration]
    )
    residual = ridge(
        train_features[calibration],
        residual_target,
        evaluation_features,
        alpha,
    )
    beta, _, _ = gate_v1.cross_fitted_beta(
        train_features[calibration],
        train_features[calibration],
        train_base[calibration],
        train_target[calibration],
        alpha,
        bounds,
    )
    unbounded = evaluation_center + residual
    gated = evaluation_center + beta * residual
    angle_radius, log_radius, informative = conic_v1.calibrate_product_geometry(
        train_base[calibration], train_target[calibration], bounds, coverage
    )
    conic, direction_active, radial_active = conic_v1.project_product(
        gated, evaluation_center, angle_radius, log_radius
    )
    return {
        "unbounded": unbounded,
        "gated": gated,
        "conic": conic,
        "beta": beta,
        "angle_radius": angle_radius,
        "log_radius": log_radius,
        "informative": informative,
        "direction_boundary_rate": float(np.mean(direction_active)),
        "radial_boundary_rate": float(np.mean(radial_active)),
    }


def subgroup_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    archive: Any,
    minimum_size: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in ("material_label_pair", "shape_label_pair"):
        groups = np.asarray(archive[key])
        records: dict[str, Any] = {}
        for value in sorted({tuple(sorted(row.tolist())) for row in groups}):
            keep = np.asarray([tuple(sorted(row.tolist())) == value for row in groups])
            if int(np.sum(keep)) >= minimum_size:
                records["-".join(str(item) for item in value)] = {
                    "events": int(np.sum(keep)),
                    "metrics": headroom.metrics(truth[keep], prediction[keep]),
                }
        result[key] = records
    return result


def evaluate(
    config: dict[str, Any],
    paths: dict[str, Path],
    implementation_sha256: dict[str, str] | None = None,
) -> dict[str, Any]:
    train = np.load(paths["train_npz"], allow_pickle=False)
    evaluation = np.load(paths["evaluation_npz"], allow_pickle=False)
    train_feature = np.load(paths["train_feature_npz"], allow_pickle=False)
    evaluation_feature = np.load(paths["evaluation_feature_npz"], allow_pickle=False)
    train_names = np.asarray(train["video_name"])
    evaluation_names = np.asarray(evaluation["video_name"])
    train_transform, train_base, train_target = camera_arrays(train)
    evaluation_transform, evaluation_base, evaluation_target = camera_arrays(evaluation)
    architecture = config["architecture"]
    train_visual, evaluation_visual, visual_record = visual_features(
        train_names, evaluation_names, train_feature, evaluation_feature, architecture
    )
    train_state, evaluation_state = standardize(
        state_features(train, train_transform), state_features(evaluation, evaluation_transform)
    )
    train_state_unit = train_state / np.maximum(
        np.linalg.norm(train_state, axis=1, keepdims=True), 1.0e-12
    )
    evaluation_state_unit = evaluation_state / np.maximum(
        np.linalg.norm(evaluation_state, axis=1, keepdims=True), 1.0e-12
    )
    train_full = np.concatenate([train_visual, train_state_unit], axis=1) / np.sqrt(2.0)
    evaluation_full = np.concatenate(
        [evaluation_visual, evaluation_state_unit], axis=1
    ) / np.sqrt(2.0)
    label_order = np.argsort(
        np.asarray(train["label_order_hash"]).astype(str), kind="stable"
    )
    probe = config["probe"]
    alpha = float(architecture["ridge_alpha"])
    bounds = tuple(float(value) for value in probe["gain_bounds"])
    coverage = float(probe["coverage"])
    budgets: dict[str, Any] = {}
    for label_budget in (int(value) for value in probe["budgets"]):
        calibration = label_order[:label_budget]
        gain = budget_v2.fit_bounded_gain(
            train_base[calibration], train_target[calibration], bounds
        )
        center_train = gain * train_base
        center = gain * evaluation_base
        point_visual = ridge(
            train_visual[calibration], train_target[calibration], evaluation_visual, alpha
        )
        point_full = ridge(train_full[calibration], train_target[calibration], evaluation_full, alpha)
        state_only = ridge(
            train_state[calibration], train_target[calibration], evaluation_state, alpha
        )
        propagation = ridge(train_visual, center_train, evaluation_visual, alpha)
        propagation_permutation = budget_v1.derangement(
            len(train_names), int(probe["propagation_derangement_seed_base"]) + label_budget
        )
        deranged_propagation = ridge(
            train_visual, center_train[propagation_permutation], evaluation_visual, alpha
        )
        residual_target = train_target[calibration] - center_train[calibration]
        residual = ridge(train_full[calibration], residual_target, evaluation_full, alpha)
        unbounded = center + residual
        radius = isotropic_radius(
            train_base[calibration], train_target[calibration], bounds, coverage
        )
        relation_tube, isotropic_rate = project_isotropic(unbounded, center, radius)
        residual_branches = {
            "visual": cross_fitted_conic_branch(
                train_visual,
                evaluation_visual,
                train_base,
                train_target,
                center,
                calibration,
                alpha=alpha,
                bounds=bounds,
                coverage=coverage,
            ),
            "state": cross_fitted_conic_branch(
                train_state_unit,
                evaluation_state_unit,
                train_base,
                train_target,
                center,
                calibration,
                alpha=alpha,
                bounds=bounds,
                coverage=coverage,
            ),
            "full": cross_fitted_conic_branch(
                train_full,
                evaluation_full,
                train_base,
                train_target,
                center,
                calibration,
                alpha=alpha,
                bounds=bounds,
                coverage=coverage,
            ),
        }
        gated = residual_branches["full"]["gated"]
        conic_gated = residual_branches["full"]["conic"]
        beta = residual_branches["full"]["beta"]
        angle_radius = residual_branches["full"]["angle_radius"]
        log_radius = residual_branches["full"]["log_radius"]
        informative = residual_branches["full"]["informative"]
        residual_derangement = budget_v1.derangement(
            label_budget, int(probe["residual_derangement_seed_base"]) + label_budget
        )
        deranged_beta, _, _ = gate_v1.cross_fitted_beta(
            train_full[calibration],
            train_full[calibration][residual_derangement],
            train_base[calibration],
            train_target[calibration],
            alpha,
            bounds,
        )
        deranged_residual = ridge(
            train_full[calibration][residual_derangement],
            residual_target,
            evaluation_full,
            alpha,
        )
        deranged_gated = center + deranged_beta * deranged_residual
        deranged_conic, _, _ = conic_v1.project_product(
            deranged_gated, center, angle_radius, log_radius
        )
        relation_permutation = budget_v1.derangement(
            label_budget, int(probe["relation_derangement_seed_base"]) + label_budget
        )
        permuted_gain = budget_v2.fit_bounded_gain(
            train_base[calibration][relation_permutation],
            train_target[calibration],
            bounds,
        )
        permuted_relation = permuted_gain * evaluation_base
        predictions = {
            "point_visual": point_visual,
            "point_full": point_full,
            "state_only": state_only,
            "center_only": center,
            "center_propagation": propagation,
            "deranged_propagation": deranged_propagation,
            "unbounded_residual": unbounded,
            "relation_tube": relation_tube,
            "permuted_relation": permuted_relation,
            "cross_fitted_residual": gated,
            "cross_fitted_conic_tube": conic_gated,
            "deranged_cross_fitted_conic_tube": deranged_conic,
            "visual_cross_fitted_residual": residual_branches["visual"]["gated"],
            "visual_cross_fitted_conic_tube": residual_branches["visual"]["conic"],
            "state_cross_fitted_residual": residual_branches["state"]["gated"],
            "state_cross_fitted_conic_tube": residual_branches["state"]["conic"],
        }
        conditions = {
            name: headroom.metrics(evaluation_target, prediction)
            for name, prediction in predictions.items()
        }
        comparisons_spec = {
            "propagation_minus_point": (propagation, point_visual),
            "propagation_minus_deranged": (propagation, deranged_propagation),
            "tube_minus_center": (relation_tube, center),
            "tube_minus_unbounded": (relation_tube, unbounded),
            "relation_minus_permuted": (center, permuted_relation),
            "gated_minus_unbounded": (gated, unbounded),
            "conic_minus_gated": (conic_gated, gated),
            "conic_minus_center": (conic_gated, center),
            "conic_minus_deranged": (conic_gated, deranged_conic),
            "visual_conic_minus_center": (
                residual_branches["visual"]["conic"],
                center,
            ),
            "state_conic_minus_center": (
                residual_branches["state"]["conic"],
                center,
            ),
            "full_conic_minus_state_conic": (
                conic_gated,
                residual_branches["state"]["conic"],
            ),
            "full_conic_minus_visual_conic": (
                conic_gated,
                residual_branches["visual"]["conic"],
            ),
        }
        comparisons = {}
        base_seed = int(probe["bootstrap_seed_base"]) + label_budget * 100
        for offset, (name, (first, second)) in enumerate(comparisons_spec.items()):
            comparisons[name] = visual_v1.bootstrap_delta(
                evaluation_target,
                first,
                second,
                replicates=int(probe["bootstrap_replicates"]),
                seed=base_seed + offset,
            )
        budgets[str(label_budget)] = {
            "labels": label_budget,
            "gain": gain,
            "permuted_gain": permuted_gain,
            "isotropic_radius": radius,
            "isotropic_boundary_rate": isotropic_rate,
            "cross_fitted_beta": beta,
            "deranged_cross_fitted_beta": deranged_beta,
            "cross_fitted_beta_by_input": {
                name: branch["beta"] for name, branch in residual_branches.items()
            },
            "conic_angle_radius_degrees": float(np.degrees(angle_radius)),
            "conic_log_radius": log_radius,
            "conic_informative_labels": informative,
            "conic_boundary_rate_by_input": {
                name: {
                    "direction": branch["direction_boundary_rate"],
                    "radial": branch["radial_boundary_rate"],
                }
                for name, branch in residual_branches.items()
            },
            "conditions": conditions,
            "comparisons": comparisons,
            "subgroups": subgroup_metrics(
                evaluation_target,
                conic_gated,
                evaluation,
                int(probe["subgroup_minimum_events"]),
            ),
        }
    phase = str(config["phase"])
    return {
        "protocol_version": PROTOCOL,
        "status": f"completed_formal_{phase}_relation_evaluation",
        "benchmark_metrics": True,
        "evaluation_split": phase,
        "heldout_target_evaluated": phase == "heldout",
        "decision": "formal_validation_completed" if phase == "validation" else "formal_heldout_completed",
        "input_sha256": {key: sha256_file(path) for key, path in paths.items()},
        "implementation_sha256": implementation_sha256 or {},
        "counts": {"train": len(train_names), phase: len(evaluation_names)},
        "architecture": {
            "visual_representation": visual_record,
            "state_dimension": train_state.shape[1],
            "full_dimension": train_full.shape[1],
            "ridge_alpha": alpha,
        },
        "budgets": budgets,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    validate_config(config)
    if args.output != Path(config["output"]["audit_json_path"]):
        raise ValueError("CLI output differs from the hash-locked formal result path")
    paths = verify_inputs(config)
    implementation_sha256 = verify_implementation(config)
    result = evaluate(config, paths, implementation_sha256)
    result["config"] = {"path": str(args.config), "sha256": sha256_file(args.config)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"decision": result["decision"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
