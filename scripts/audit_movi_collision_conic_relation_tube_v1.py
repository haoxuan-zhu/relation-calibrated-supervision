"""Evaluate a cone-times-radial relation geometry for MOVi impulse vectors."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml

import audit_movi_collision_physics_budget_v1 as budget_v1
import audit_movi_collision_physics_budget_v2 as budget_v2
import audit_movi_collision_projected_point_tube_v1 as point_v1
import audit_movi_collision_relation_tube_pilot_v1 as tube_v1
import audit_movi_collision_state_headroom_v1 as headroom
import audit_movi_collision_visual_residual_v1 as visual_v1


PROTOCOL = "movi_collision_conic_relation_tube_v1"


def validate_config(config: dict[str, Any]) -> None:
    if config.get("protocol_version") != PROTOCOL:
        raise ValueError("unexpected conic Relation Tube protocol")
    if config["data"].get("development_only") is not True:
        raise ValueError("conic pilot must remain development-only")
    if config["data"].get("test_evaluated") is not False:
        raise ValueError("test must remain closed")
    probe = config["probe"]
    if [int(value) for value in probe["budgets"]] != [20, 40]:
        raise ValueError("registered budgets changed")
    if int(probe["repeats"]) != 5 or float(probe["ridge_alpha"]) != 0.1:
        raise ValueError("registered repeat count or ridge changed")
    if float(probe["coverage"]) != 0.95:
        raise ValueError("registered coverage changed")


def rank_value(values: np.ndarray, coverage: float) -> float:
    finite = np.asarray(values, dtype=np.float64)
    if finite.ndim != 1 or len(finite) < 3 or not np.all(np.isfinite(finite)):
        raise ValueError("invalid calibration residuals")
    rank = min(len(finite), int(math.ceil((len(finite) + 1) * coverage)))
    return float(np.sort(finite)[rank - 1])


def calibrate_product_geometry(
    base: np.ndarray,
    target: np.ndarray,
    bounds: tuple[float, float],
    coverage: float,
) -> tuple[float, float, int]:
    base_norm = np.linalg.norm(base, axis=1)
    target_norm = np.linalg.norm(target, axis=1)
    valid = (base_norm > 1.0e-9) & (target_norm > 1.0e-9)
    if int(np.sum(valid)) < 3:
        raise ValueError("too few nonzero relation centers")
    cosine = np.sum(base[valid] * target[valid], axis=1) / (
        base_norm[valid] * target_norm[valid]
    )
    angles = np.arccos(np.clip(cosine, -1.0, 1.0))
    radial = []
    indices = np.flatnonzero(valid)
    for index in indices:
        keep = np.arange(len(base)) != index
        gain = budget_v2.fit_bounded_gain(base[keep], target[keep], bounds)
        center_norm = gain * base_norm[index]
        radial.append(abs(math.log(target_norm[index] / max(center_norm, 1.0e-12))))
    return (
        rank_value(angles, coverage),
        rank_value(np.asarray(radial), coverage),
        int(np.sum(valid)),
    )


def project_product(
    point: np.ndarray,
    center: np.ndarray,
    angle_radius: float,
    log_radius: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    output = np.asarray(point, dtype=np.float64).copy()
    point_norm = np.linalg.norm(point, axis=1)
    center_norm = np.linalg.norm(center, axis=1)
    valid = (point_norm > 1.0e-12) & (center_norm > 1.0e-12)
    direction_active = np.zeros(len(point), dtype=bool)
    radial_active = np.zeros(len(point), dtype=bool)
    for index in np.flatnonzero(valid):
        source = point[index] / point_norm[index]
        axis = center[index] / center_norm[index]
        cosine = float(np.clip(np.dot(axis, source), -1.0, 1.0))
        angle = math.acos(cosine)
        direction = source
        if angle > angle_radius:
            orthogonal = source - cosine * axis
            orthogonal_norm = float(np.linalg.norm(orthogonal))
            if orthogonal_norm <= 1.0e-12:
                basis = np.eye(3)[int(np.argmin(np.abs(axis)))]
                orthogonal = basis - float(np.dot(basis, axis)) * axis
                orthogonal_norm = float(np.linalg.norm(orthogonal))
            tangent = orthogonal / orthogonal_norm
            direction = math.cos(angle_radius) * axis + math.sin(angle_radius) * tangent
            direction_active[index] = True
        magnitude = point_norm[index]
        if log_radius is not None:
            lower = center_norm[index] * math.exp(-log_radius)
            upper = center_norm[index] * math.exp(log_radius)
            clipped = min(max(magnitude, lower), upper)
            radial_active[index] = abs(clipped - magnitude) > 1.0e-12
            magnitude = clipped
        output[index] = magnitude * direction
    return output, direction_active, radial_active


def run(config: dict[str, Any], paths: dict[str, Path]) -> dict[str, Any]:
    validate_config(config)
    hashes = tube_v1.verify_inputs(config, paths)
    clip = np.load(paths["clip_npz"], allow_pickle=False)
    pair = np.load(paths["pair_feature_npz"], allow_pickle=False)
    objects = np.load(paths["object_feature_npz"], allow_pickle=False)
    camera = np.load(paths["camera_npz"], allow_pickle=False)
    names = np.asarray(clip["video_name"])
    transforms = tube_v1.reorder(
        names, np.asarray(camera["video_name"]), np.asarray(camera["view_from_world"])
    )
    base = np.einsum("nij,nj->ni", transforms, np.asarray(clip["base_impulse"], dtype=np.float64))
    target = np.einsum("nij,nj->ni", transforms, np.asarray(clip["target"], dtype=np.float64))
    rows = [{"video_name": str(name), "index": i} for i, name in enumerate(names.tolist())]
    split = config["split"]
    pool_rows, evaluation_rows = budget_v1.split_events(
        rows,
        int(split["evaluation_modulus"]),
        {int(value) for value in split["evaluation_remainders"]},
        str(split["split_salt"]),
    )
    pool_index = np.asarray([row["index"] for row in pool_rows], dtype=np.int64)
    evaluation_index = np.asarray([row["index"] for row in evaluation_rows], dtype=np.int64)
    visual = tube_v1.visual_descriptor(clip, pair, objects, pool_index)
    features = {
        "visual": visual,
        "full": point_v1.state_augmented_features(visual, base, pool_index),
    }
    truth = target[evaluation_index]
    probe = config["probe"]
    bounds = tuple(float(value) for value in probe["gain_bounds"])
    repeats = []
    for repeat in range(int(probe["repeats"])):
        seed = int(probe["order_seed_base"]) + repeat
        ordered = pool_index[np.random.default_rng(seed).permutation(len(pool_index))]
        budgets: dict[str, Any] = {}
        for label_budget in (int(value) for value in probe["budgets"]):
            calibration = ordered[:label_budget]
            gain = budget_v2.fit_bounded_gain(base[calibration], target[calibration], bounds)
            center_eval = gain * base[evaluation_index]
            angle_radius, log_radius, informative = calibrate_product_geometry(
                base[calibration],
                target[calibration],
                bounds,
                float(probe["coverage"]),
            )
            target_scale = target[calibration].std(axis=0)
            target_scale = np.where(target_scale > 1.0e-6, target_scale, 1.0)
            conditions = {"center_only": headroom.metrics(truth, center_eval)}
            comparisons: dict[str, Any] = {}
            activity: dict[str, Any] = {}
            for branch, values in features.items():
                point_normalized = tube_v1.dual_ridge(
                    values[calibration],
                    target[calibration] / target_scale,
                    values[evaluation_index],
                    float(probe["ridge_alpha"]),
                    fit_intercept=True,
                )
                point = point_normalized * target_scale
                direction_only, direction_active, _ = project_product(
                    point, center_eval, angle_radius, None
                )
                product, product_direction_active, radial_active = project_product(
                    point, center_eval, angle_radius, log_radius
                )
                conditions[f"point_{branch}"] = headroom.metrics(truth, point)
                conditions[f"direction_cone_{branch}"] = headroom.metrics(truth, direction_only)
                conditions[f"conic_tube_{branch}"] = headroom.metrics(truth, product)
                activity[branch] = {
                    "direction_rate": float(np.mean(product_direction_active)),
                    "radial_rate": float(np.mean(radial_active)),
                    "applicability_rate": float(
                        np.mean(np.linalg.norm(center_eval, axis=1) > 1.0e-12)
                    ),
                }
                bootstrap_seed = (
                    int(probe["bootstrap_seed_base"])
                    + repeat * 100
                    + label_budget
                    + (0 if branch == "visual" else 10_000)
                )
                comparisons[f"conic_{branch}_minus_point"] = visual_v1.bootstrap_delta(
                    truth,
                    product,
                    point,
                    replicates=int(probe["bootstrap_replicates"]),
                    seed=bootstrap_seed,
                )
                comparisons[f"conic_{branch}_minus_center"] = visual_v1.bootstrap_delta(
                    truth,
                    product,
                    center_eval,
                    replicates=int(probe["bootstrap_replicates"]),
                    seed=bootstrap_seed + 20_000,
                )
            budgets[str(label_budget)] = {
                "gain": gain,
                "angle_radius_degrees": math.degrees(angle_radius),
                "log_magnitude_radius": log_radius,
                "informative_calibration_events": informative,
                "activity": activity,
                "conditions": conditions,
                "comparisons": comparisons,
            }
        repeats.append({"repeat": repeat, "order_seed": seed, "budgets": budgets})
    summary: dict[str, Any] = {}
    for label_budget in (str(value) for value in probe["budgets"]):
        summary[label_budget] = {
            "angle_radius_degrees_mean": float(
                np.mean(
                    [row["budgets"][label_budget]["angle_radius_degrees"] for row in repeats]
                )
            ),
            "log_magnitude_radius_mean": float(
                np.mean(
                    [row["budgets"][label_budget]["log_magnitude_radius"] for row in repeats]
                )
            ),
        }
        for branch in ("visual", "full"):
            row_summary: dict[str, Any] = {
                "direction_rate_mean": float(
                    np.mean(
                        [row["budgets"][label_budget]["activity"][branch]["direction_rate"] for row in repeats]
                    )
                ),
                "radial_rate_mean": float(
                    np.mean(
                        [row["budgets"][label_budget]["activity"][branch]["radial_rate"] for row in repeats]
                    )
                ),
            }
            for comparator in ("point", "center"):
                name = f"conic_{branch}_minus_{comparator}"
                values = np.asarray(
                    [row["budgets"][label_budget]["comparisons"][name]["observed"] for row in repeats]
                )
                row_summary[f"minus_{comparator}"] = {
                    "positive_repeats": int(np.sum(values > 0.0)),
                    "mean": float(np.mean(values)),
                    "range": [float(np.min(values)), float(np.max(values))],
                    "ci95_positive_repeats": int(
                        sum(
                            row["budgets"][label_budget]["comparisons"][name]["ci95_low"] > 0.0
                            for row in repeats
                        )
                    ),
                }
            summary[label_budget][branch] = row_summary
    primary = summary["40"]["full"]
    passed = all(
        primary[name]["positive_repeats"] >= 3 and primary[name]["mean"] > 0.0
        for name in ("minus_point", "minus_center")
    )
    return {
        "protocol_version": PROTOCOL,
        "status": "completed_development_conic_relation_tube",
        "benchmark_metrics": False,
        "test_evaluated": False,
        "decision": (
            "conic_relation_tube_signal_observed"
            if passed
            else "conic_relation_tube_not_confirmed"
        ),
        "input_sha256": hashes,
        "split": {"pool_events": len(pool_index), "evaluation_events": len(evaluation_index)},
        "summary": summary,
        "repeats": repeats,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    data = config["data"]
    paths = {
        "clip_npz": Path(data["clip_npz_path"]),
        "pair_feature_npz": Path(data["pair_feature_npz_path"]),
        "object_feature_npz": Path(data["object_feature_npz_path"]),
        "camera_npz": Path(data["camera_npz_path"]),
    }
    result = run(config, paths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"decision": result["decision"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
