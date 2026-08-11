"""Test cross-sample physical-center propagation into an RGB-only MOVi predictor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

import audit_movi_collision_physics_budget_v1 as budget_v1
import audit_movi_collision_physics_budget_v2 as budget_v2
import audit_movi_collision_state_headroom_v1 as headroom
import audit_movi_collision_visual_residual_v1 as visual_v1


PROTOCOL = "movi_collision_center_propagation_v1"


def validate_config(config: dict[str, Any]) -> None:
    if config.get("protocol_version") != PROTOCOL:
        raise ValueError("unexpected center-propagation protocol")
    if config["data"].get("development_only") is not True:
        raise ValueError("center propagation must remain development-only")
    if config["data"].get("test_evaluated") is not False:
        raise ValueError("test must remain closed")
    representation = config["representation"]
    if int(representation["pool_only_pca_components"]) != 16:
        raise ValueError("registered PCA changed")
    if float(representation["ridge_alpha"]) != 1.0:
        raise ValueError("registered ridge changed")
    probe = config["probe"]
    if [int(value) for value in probe["budgets"]] != [20, 40]:
        raise ValueError("registered budgets changed")
    if int(probe["repeats"]) != 5:
        raise ValueError("registered repeats changed")
    if [float(value) for value in probe["gain_bounds"]] != [1.0, 2.0]:
        raise ValueError("gain interval changed")


def reorder_features(names: np.ndarray, feature_names: np.ndarray, features: np.ndarray) -> np.ndarray:
    source = [str(value) for value in feature_names.tolist()]
    target = [str(value) for value in names.tolist()]
    if len(source) != len(set(source)) or set(source) != set(target):
        raise ValueError("clip and feature video sets differ")
    lookup = {name: index for index, name in enumerate(source)}
    return features[np.asarray([lookup[name] for name in target], dtype=np.int64)]


def fit_pca(features: np.ndarray, pool_index: np.ndarray, components: int) -> np.ndarray:
    pool = features[pool_index]
    mean = pool.mean(axis=0)
    _, _, vectors = np.linalg.svd(pool - mean, full_matrices=False)
    return (features - mean) @ vectors[:components].T


def fit_ridge(
    features: np.ndarray, target: np.ndarray, evaluation: np.ndarray, alpha: float
) -> np.ndarray:
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale = np.where(scale > 1.0e-8, scale, 1.0)
    target_mean = target.mean(axis=0)
    design = (features - mean) / scale
    weights = np.linalg.solve(
        design.T @ design + alpha * np.eye(design.shape[1]),
        design.T @ (target - target_mean),
    )
    return target_mean + ((evaluation - mean) / scale) @ weights


def run(config: dict[str, Any], clip_path: Path, feature_path: Path) -> dict[str, Any]:
    validate_config(config)
    clip_hash = visual_v1.sha256_file(clip_path)
    feature_hash = visual_v1.sha256_file(feature_path)
    if clip_hash != str(config["data"]["clip_npz_sha256"]):
        raise ValueError("clip archive hash does not match")
    if feature_hash != str(config["data"]["feature_npz_sha256"]):
        raise ValueError("feature archive hash does not match")
    clips = np.load(clip_path, allow_pickle=False)
    frozen = np.load(feature_path, allow_pickle=False)
    names = np.asarray(clips["video_name"])
    frame_features = reorder_features(
        names,
        np.asarray(frozen["video_name"]),
        np.asarray(frozen["cls"], dtype=np.float64),
    )
    features = frame_features.reshape(len(names), -1)
    base = np.asarray(clips["base_impulse"], dtype=np.float64)
    target = np.asarray(clips["target"], dtype=np.float64)
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
    representation = config["representation"]
    projected = fit_pca(
        features, pool_index, int(representation["pool_only_pca_components"])
    )
    alpha = float(representation["ridge_alpha"])
    evaluation_target = target[evaluation_index]
    probe = config["probe"]
    bounds = tuple(float(value) for value in probe["gain_bounds"])
    repeats = []
    for repeat in range(int(probe["repeats"])):
        order_seed = int(probe["order_seed_base"]) + repeat
        ordered = pool_index[np.random.default_rng(order_seed).permutation(len(pool_index))]
        budgets: dict[str, Any] = {}
        for budget in (int(value) for value in probe["budgets"]):
            calibration = ordered[:budget]
            gain = budget_v2.fit_bounded_gain(base[calibration], target[calibration], bounds)
            propagated_target = gain * base[pool_index]
            point_prediction = fit_ridge(
                projected[calibration], target[calibration], projected[evaluation_index], alpha
            )
            propagation_prediction = fit_ridge(
                projected[pool_index], propagated_target, projected[evaluation_index], alpha
            )
            permutation = budget_v1.derangement(
                len(pool_index),
                int(probe["propagation_derangement_seed_base"]) + repeat * 100 + budget,
            )
            deranged_prediction = fit_ridge(
                projected[pool_index],
                propagated_target[permutation],
                projected[evaluation_index],
                alpha,
            )
            predictions = {
                "point_replay": point_prediction,
                "center_propagation": propagation_prediction,
                "deranged_center_propagation": deranged_prediction,
            }
            conditions = {
                name: headroom.metrics(evaluation_target, prediction)
                for name, prediction in predictions.items()
            }
            bootstrap_seed = int(probe["bootstrap_seed_base"]) + repeat * 100 + budget
            comparisons = {
                "propagation_minus_point": visual_v1.bootstrap_delta(
                    evaluation_target,
                    propagation_prediction,
                    point_prediction,
                    replicates=int(probe["bootstrap_replicates"]),
                    seed=bootstrap_seed,
                ),
                "propagation_minus_deranged": visual_v1.bootstrap_delta(
                    evaluation_target,
                    propagation_prediction,
                    deranged_prediction,
                    replicates=int(probe["bootstrap_replicates"]),
                    seed=bootstrap_seed + 10_000,
                ),
            }
            budgets[str(budget)] = {
                "bounded_gain": gain,
                "propagated_training_events": len(pool_index),
                "point_training_events": budget,
                "conditions": conditions,
                "comparisons": comparisons,
            }
        repeats.append({"repeat": repeat, "order_seed": order_seed, "budgets": budgets})
    summary: dict[str, Any] = {}
    for budget in (str(value) for value in probe["budgets"]):
        summary[budget] = {}
        for comparison in ("propagation_minus_point", "propagation_minus_deranged"):
            values = np.asarray(
                [row["budgets"][budget]["comparisons"][comparison]["observed"] for row in repeats]
            )
            summary[budget][comparison] = {
                "positive_repeats": int(np.sum(values > 0.0)),
                "mean": float(np.mean(values)),
                "range": [float(np.min(values)), float(np.max(values))],
                "ci95_positive_repeats": int(
                    sum(
                        row["budgets"][budget]["comparisons"][comparison]["ci95_low"] > 0.0
                        for row in repeats
                    )
                ),
            }
    passed = all(
        summary["40"][name]["positive_repeats"] >= 3
        and summary["40"][name]["mean"] > 0.0
        for name in ("propagation_minus_point", "propagation_minus_deranged")
    )
    return {
        "protocol_version": PROTOCOL,
        "status": "completed_development_center_propagation_audit",
        "benchmark_metrics": False,
        "test_evaluated": False,
        "decision": (
            "cross_sample_center_propagation_observed_tube_unlocked"
            if passed
            else "center_propagation_not_confirmed_stop_before_tube"
        ),
        "relation_tube_unlocked": passed,
        "test_time_instrument_state_used": False,
        "clip_npz_sha256": clip_hash,
        "feature_npz_sha256": feature_hash,
        "split": {"calibration_pool_events": len(pool_index), "evaluation_events": len(evaluation_index)},
        "summary": summary,
        "repeats": repeats,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--clip", required=True, type=Path)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite result: {args.output}")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    result = run(config, args.clip, args.features)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "decision": result["decision"]}, sort_keys=True))


if __name__ == "__main__":
    main()
