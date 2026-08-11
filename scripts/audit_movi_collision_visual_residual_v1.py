"""Evaluate frozen visual residual signal beyond the bounded MOVi collision center."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

import audit_movi_collision_physics_budget_v1 as v1
import audit_movi_collision_physics_budget_v2 as v2
import audit_movi_collision_state_headroom_v1 as headroom


PROTOCOL = "movi_collision_visual_residual_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_config(config: dict[str, Any]) -> None:
    if config.get("protocol_version") != PROTOCOL:
        raise ValueError("unexpected MOVi visual-residual protocol")
    if config["data"].get("development_only") is not True:
        raise ValueError("visual audit must remain development-only")
    if config["data"].get("test_evaluated") is not False:
        raise ValueError("test must remain closed")
    probe = config["probe"]
    if [int(value) for value in probe["budgets"]] != [20, 40]:
        raise ValueError("registered visual budgets changed")
    if int(probe["repeats"]) != 5 or int(probe["pca_components"]) != 16:
        raise ValueError("registered repeat or PCA contract changed")
    if float(probe["ridge_alpha"]) != 1.0:
        raise ValueError("registered ridge alpha changed")
    if [float(value) for value in probe["gain_bounds"]] != [1.0, 2.0]:
        raise ValueError("bounded-center interval changed")


def fit_pca(features: np.ndarray, components: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = features.mean(axis=0)
    scale = np.ones(features.shape[1], dtype=np.float64)
    _, _, right = np.linalg.svd(features - mean, full_matrices=False)
    return mean, scale, right[:components]


def transform_pca(
    features: np.ndarray, mean: np.ndarray, scale: np.ndarray, components: np.ndarray
) -> np.ndarray:
    return ((features - mean) / scale) @ components.T


def fit_ridge(
    features: np.ndarray, target: np.ndarray, evaluation_features: np.ndarray, alpha: float
) -> np.ndarray:
    mean_x = features.mean(axis=0)
    scale_x = features.std(axis=0)
    scale_x = np.where(scale_x > 1.0e-8, scale_x, 1.0)
    mean_y = target.mean(axis=0)
    x = (features - mean_x) / scale_x
    y = target - mean_y
    gram = x.T @ x + float(alpha) * np.eye(x.shape[1])
    weights = np.linalg.solve(gram, x.T @ y)
    return mean_y + ((evaluation_features - mean_x) / scale_x) @ weights


def mean_coordinate_r2(target: np.ndarray, prediction: np.ndarray) -> float:
    denominator = np.sum((target - target.mean(axis=0)) ** 2, axis=0)
    numerator = np.sum((target - prediction) ** 2, axis=0)
    valid = denominator > 1.0e-12
    return float(np.mean(1.0 - numerator[valid] / denominator[valid]))


def bootstrap_delta(
    target: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    observed = mean_coordinate_r2(target, first) - mean_coordinate_r2(target, second)
    rng = np.random.default_rng(seed)
    values = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sample = rng.integers(0, len(target), size=len(target))
        values[index] = mean_coordinate_r2(target[sample], first[sample]) - mean_coordinate_r2(
            target[sample], second[sample]
        )
    return {
        "observed": observed,
        "ci95_low": float(np.quantile(values, 0.025)),
        "ci95_high": float(np.quantile(values, 0.975)),
    }


def run(
    config: dict[str, Any], clip_path: Path, feature_path: Path, feature_manifest_path: Path
) -> dict[str, Any]:
    validate_config(config)
    clip_hash = sha256_file(clip_path)
    if clip_hash != str(config["data"]["clip_npz_sha256"]):
        raise ValueError("clip archive hash does not match the frozen config")
    feature_manifest = json.loads(feature_manifest_path.read_text(encoding="utf-8"))
    feature_hash = sha256_file(feature_path)
    if feature_hash != feature_manifest["feature_npz"]["sha256"]:
        raise ValueError("feature archive hash does not match its manifest")
    clips = np.load(clip_path, allow_pickle=False)
    feature_archive = np.load(feature_path, allow_pickle=False)
    names = np.asarray(clips["video_name"])
    if names.tolist() != np.asarray(feature_archive["video_name"]).tolist():
        raise ValueError("clip and feature video order differ")
    base = np.asarray(clips["base_impulse"], dtype=np.float64)
    target = np.asarray(clips["target"], dtype=np.float64)
    features = np.asarray(feature_archive["cls"], dtype=np.float64).reshape(len(names), -1)
    rows = [
        {"video_name": str(name), "index": index}
        for index, name in enumerate(names.tolist())
    ]
    split = config["split"]
    pool_rows, evaluation_rows = v1.split_events(
        rows,
        int(split["evaluation_modulus"]),
        {int(value) for value in split["evaluation_remainders"]},
        str(split["split_salt"]),
    )
    pool_index = np.asarray([row["index"] for row in pool_rows], dtype=np.int64)
    evaluation_index = np.asarray(
        [row["index"] for row in evaluation_rows], dtype=np.int64
    )
    probe = config["probe"]
    pca_mean, pca_scale, pca_components = fit_pca(
        features[pool_index], int(probe["pca_components"])
    )
    projected = transform_pca(features, pca_mean, pca_scale, pca_components)
    evaluation_base = base[evaluation_index]
    evaluation_target = target[evaluation_index]
    bounds = tuple(float(value) for value in probe["gain_bounds"])
    repeats: list[dict[str, Any]] = []
    for repeat in range(int(probe["repeats"])):
        order_seed = int(probe["order_seed_base"]) + repeat
        order = np.random.default_rng(order_seed).permutation(len(pool_index))
        ordered_index = pool_index[order]
        budgets: dict[str, Any] = {}
        for budget in (int(value) for value in probe["budgets"]):
            calibration_index = ordered_index[:budget]
            gain = v2.fit_bounded_gain(
                base[calibration_index], target[calibration_index], bounds
            )
            calibration_residual = target[calibration_index] - gain * base[calibration_index]
            center_prediction = gain * evaluation_base
            residual_mean_prediction = center_prediction + calibration_residual.mean(axis=0)
            correct_residual = fit_ridge(
                projected[calibration_index],
                calibration_residual,
                projected[evaluation_index],
                float(probe["ridge_alpha"]),
            )
            permutation = v1.derangement(
                budget,
                int(probe["derangement_seed_base"]) + repeat * 100 + budget,
            )
            deranged_residual = fit_ridge(
                projected[calibration_index][permutation],
                calibration_residual,
                projected[evaluation_index],
                float(probe["ridge_alpha"]),
            )
            correct_prediction = center_prediction + correct_residual
            deranged_prediction = center_prediction + deranged_residual
            conditions = {
                "bounded_center": headroom.metrics(evaluation_target, center_prediction),
                "bounded_center_plus_mean_residual": headroom.metrics(
                    evaluation_target, residual_mean_prediction
                ),
                "bounded_center_plus_correct_visual": headroom.metrics(
                    evaluation_target, correct_prediction
                ),
                "bounded_center_plus_deranged_visual": headroom.metrics(
                    evaluation_target, deranged_prediction
                ),
            }
            bootstrap_seed = int(probe["bootstrap_seed_base"]) + repeat * 100 + budget
            comparisons = {
                "correct_minus_center": bootstrap_delta(
                    evaluation_target,
                    correct_prediction,
                    center_prediction,
                    replicates=int(probe["bootstrap_replicates"]),
                    seed=bootstrap_seed,
                ),
                "correct_minus_mean_residual": bootstrap_delta(
                    evaluation_target,
                    correct_prediction,
                    residual_mean_prediction,
                    replicates=int(probe["bootstrap_replicates"]),
                    seed=bootstrap_seed + 10_000,
                ),
                "correct_minus_deranged": bootstrap_delta(
                    evaluation_target,
                    correct_prediction,
                    deranged_prediction,
                    replicates=int(probe["bootstrap_replicates"]),
                    seed=bootstrap_seed + 20_000,
                ),
            }
            budgets[str(budget)] = {
                "gain": gain,
                "conditions": conditions,
                "comparisons": comparisons,
            }
        repeats.append({"repeat": repeat, "order_seed": order_seed, "budgets": budgets})

    summary: dict[str, Any] = {}
    for budget in (str(value) for value in probe["budgets"]):
        rows_at_budget = [row["budgets"][budget] for row in repeats]
        summary[budget] = {}
        for comparison in (
            "correct_minus_center",
            "correct_minus_mean_residual",
            "correct_minus_deranged",
        ):
            deltas = np.asarray(
                [row["comparisons"][comparison]["observed"] for row in rows_at_budget]
            )
            summary[budget][comparison] = {
                "positive_repeats": int(np.sum(deltas > 0.0)),
                "mean": float(np.mean(deltas)),
                "range": [float(np.min(deltas)), float(np.max(deltas))],
                "ci95_positive_repeats": int(
                    sum(
                        row["comparisons"][comparison]["ci95_low"] > 0.0
                        for row in rows_at_budget
                    )
                ),
            }
    k40 = summary["40"]
    comparisons_pass = all(
        k40[name]["positive_repeats"] >= 3 and k40[name]["mean"] > 0.0
        for name in (
            "correct_minus_center",
            "correct_minus_mean_residual",
            "correct_minus_deranged",
        )
    )
    return {
        "protocol_version": PROTOCOL,
        "status": "completed_development_visual_residual_audit",
        "benchmark_metrics": False,
        "test_evaluated": False,
        "decision": (
            "visual_signal_observed_compact_model_unlocked"
            if comparisons_pass
            else "frozen_visual_residual_signal_not_confirmed_stop_branch"
        ),
        "compact_model_unlocked": comparisons_pass,
        "clip_npz_sha256": clip_hash,
        "feature_npz_sha256": feature_hash,
        "feature_manifest_sha256": sha256_file(feature_manifest_path),
        "backbone": feature_manifest["backbone"],
        "split": {
            "calibration_pool_events": len(pool_index),
            "evaluation_events": len(evaluation_index),
        },
        "pca": {
            "fit_scope": "unlabeled_calibration_pool_only",
            "components": int(probe["pca_components"]),
        },
        "summary": summary,
        "repeats": repeats,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--clip", required=True, type=Path)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--feature-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite result: {args.output}")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    result = run(config, args.clip, args.features, args.feature_manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "decision": result["decision"]}, sort_keys=True))


if __name__ == "__main__":
    main()
