"""Development-only bounded residual pilot for the MOVi collision system."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml

import audit_movi_collision_center_propagation_v1 as propagation
import audit_movi_collision_physics_budget_v1 as budget_v1
import audit_movi_collision_physics_budget_v2 as budget_v2
import audit_movi_collision_state_headroom_v1 as headroom
import audit_movi_collision_visual_residual_v1 as visual_v1
import extract_movi_collision_multiview_dinov2_v1 as multiview


PROTOCOL_V1 = "movi_collision_tube_pilot_v1"
PROTOCOL_V2 = "movi_collision_tube_pilot_v2"
PROTOCOL = PROTOCOL_V1


def validate_config(config: dict[str, Any]) -> None:
    if config.get("protocol_version") not in (PROTOCOL_V1, PROTOCOL_V2):
        raise ValueError("unexpected MOVi tube pilot protocol")
    if config["data"].get("development_only") is not True:
        raise ValueError("tube pilot must remain development-only")
    if config["data"].get("test_evaluated") is not False:
        raise ValueError("test must remain closed")
    probe = config["probe"]
    if [int(value) for value in probe["budgets"]] != [20, 40]:
        raise ValueError("registered budgets changed")
    if int(probe["repeats"]) != 5 or float(probe["ridge_alpha"]) != 0.1:
        raise ValueError("registered repeat count or ridge changed")
    if float(probe["coverage"]) != 0.95:
        raise ValueError("registered coverage changed")
    expected_intercept = config["protocol_version"] == PROTOCOL_V2
    if bool(probe.get("fit_intercept", False)) != expected_intercept:
        raise ValueError("ridge intercept does not match the protocol version")


def verify_inputs(config: dict[str, Any], paths: dict[str, Path]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name, path in paths.items():
        observed = visual_v1.sha256_file(path)
        expected = str(config["data"][f"{name}_sha256"])
        if observed != expected:
            raise ValueError(f"{name} hash does not match")
        hashes[name] = observed
    return hashes


def reorder(names: np.ndarray, source_names: np.ndarray, values: np.ndarray) -> np.ndarray:
    return propagation.reorder_features(names, source_names, values)


def l2_blocks(values: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.maximum(norm, 1.0e-12)


def visual_descriptor(
    clip: Any,
    pair_archive: Any,
    object_archive: Any,
    pool_index: np.ndarray,
) -> np.ndarray:
    names = np.asarray(clip["video_name"])
    pair = reorder(names, np.asarray(pair_archive["video_name"]), pair_archive["cls"])
    objects = reorder(
        names, np.asarray(object_archive["video_name"]), object_archive["cls"]
    )
    pair_blocks = l2_blocks(np.asarray(pair, dtype=np.float64)).reshape(len(names), -1)
    object_blocks = l2_blocks(np.asarray(objects, dtype=np.float64)).reshape(len(names), -1)
    geometry = np.stack(
        [
            multiview.mask_motion_features(rgb, masks).reshape(-1)
            for rgb, masks in zip(clip["rgb"], clip["pair_mask"], strict=True)
        ]
    ).astype(np.float64)
    mean = geometry[pool_index].mean(axis=0)
    scale = geometry[pool_index].std(axis=0)
    scale = np.where(scale > 1.0e-8, scale, 1.0)
    geometry = (geometry - mean) / scale
    geometry /= np.maximum(np.linalg.norm(geometry, axis=1, keepdims=True), 1.0e-12)
    return np.concatenate([pair_blocks, object_blocks, geometry], axis=1) / math.sqrt(10.0)


def dual_ridge(
    train_x: np.ndarray,
    train_y: np.ndarray,
    evaluation_x: np.ndarray,
    alpha: float,
    fit_intercept: bool = False,
) -> np.ndarray:
    if fit_intercept:
        feature_mean = train_x.mean(axis=0)
        target_mean = train_y.mean(axis=0)
    else:
        feature_mean = np.zeros(train_x.shape[1], dtype=np.float64)
        target_mean = np.zeros(train_y.shape[1], dtype=np.float64)
    centered_train = train_x - feature_mean
    centered_evaluation = evaluation_x - feature_mean
    centered_target = train_y - target_mean
    kernel = centered_train @ centered_train.T
    coefficients = np.linalg.solve(
        kernel + alpha * np.eye(len(train_x)), centered_target
    )
    return target_mean + centered_evaluation @ centered_train.T @ coefficients


def loo_radius(
    base: np.ndarray,
    target: np.ndarray,
    target_scale: np.ndarray,
    bounds: tuple[float, float],
    coverage: float,
) -> float:
    norms = []
    for index in range(len(base)):
        keep = np.arange(len(base)) != index
        gain = budget_v2.fit_bounded_gain(base[keep], target[keep], bounds)
        residual = (target[index] - gain * base[index]) / target_scale
        norms.append(float(np.linalg.norm(residual)))
    rank = min(len(norms), int(math.ceil((len(norms) + 1) * coverage)))
    return float(np.sort(np.asarray(norms))[rank - 1])


def run(config: dict[str, Any], paths: dict[str, Path]) -> dict[str, Any]:
    validate_config(config)
    hashes = verify_inputs(config, paths)
    clip = np.load(paths["clip_npz"], allow_pickle=False)
    pair = np.load(paths["pair_feature_npz"], allow_pickle=False)
    objects = np.load(paths["object_feature_npz"], allow_pickle=False)
    camera = np.load(paths["camera_npz"], allow_pickle=False)
    names = np.asarray(clip["video_name"])
    transforms = reorder(
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
    features = visual_descriptor(clip, pair, objects, pool_index)
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
            center = gain * base
            target_scale = target[calibration].std(axis=0)
            target_scale = np.where(target_scale > 1.0e-6, target_scale, 1.0)
            residual_target = (target[calibration] - center[calibration]) / target_scale
            residual_prediction = dual_ridge(
                features[calibration],
                residual_target,
                features[evaluation_index],
                float(probe["ridge_alpha"]),
                fit_intercept=bool(probe.get("fit_intercept", False)),
            )
            radius = loo_radius(
                base[calibration],
                target[calibration],
                target_scale,
                bounds,
                float(probe["coverage"]),
            )
            norms = np.linalg.norm(residual_prediction, axis=1)
            factors = np.minimum(1.0, radius / np.maximum(norms, 1.0e-12))
            center_eval = center[evaluation_index]
            unbounded = center_eval + residual_prediction * target_scale
            tube = center_eval + residual_prediction * factors[:, None] * target_scale
            truth = target[evaluation_index]
            conditions = {
                "center_only": headroom.metrics(truth, center_eval),
                "unbounded_residual": headroom.metrics(truth, unbounded),
                "relation_tube": headroom.metrics(truth, tube),
            }
            bootstrap_seed = int(probe["bootstrap_seed_base"]) + repeat * 100 + label_budget
            comparisons = {
                "tube_minus_center": visual_v1.bootstrap_delta(
                    truth,
                    tube,
                    center_eval,
                    replicates=int(probe["bootstrap_replicates"]),
                    seed=bootstrap_seed,
                ),
                "tube_minus_unbounded": visual_v1.bootstrap_delta(
                    truth,
                    tube,
                    unbounded,
                    replicates=int(probe["bootstrap_replicates"]),
                    seed=bootstrap_seed + 10_000,
                ),
            }
            budgets[str(label_budget)] = {
                "bounded_gain": gain,
                "target_scale": target_scale.tolist(),
                "radius": radius,
                "boundary_rate": float(np.mean(factors < 1.0 - 1.0e-12)),
                "conditions": conditions,
                "comparisons": comparisons,
            }
        repeats.append({"repeat": repeat, "order_seed": seed, "budgets": budgets})
    summary: dict[str, Any] = {}
    for label_budget in (str(value) for value in probe["budgets"]):
        summary[label_budget] = {}
        for comparison in ("tube_minus_center", "tube_minus_unbounded"):
            values = np.asarray(
                [row["budgets"][label_budget]["comparisons"][comparison]["observed"] for row in repeats]
            )
            summary[label_budget][comparison] = {
                "positive_repeats": int(np.sum(values > 0.0)),
                "mean": float(np.mean(values)),
                "range": [float(np.min(values)), float(np.max(values))],
                "ci95_positive_repeats": int(
                    sum(
                        row["budgets"][label_budget]["comparisons"][comparison]["ci95_low"] > 0.0
                        for row in repeats
                    )
                ),
            }
        summary[label_budget]["boundary_rate_mean"] = float(
            np.mean([row["budgets"][label_budget]["boundary_rate"] for row in repeats])
        )
    passed = all(
        summary["40"][name]["positive_repeats"] >= 3
        and summary["40"][name]["mean"] > 0.0
        for name in ("tube_minus_center", "tube_minus_unbounded")
    )
    return {
        "protocol_version": str(config["protocol_version"]),
        "status": "completed_development_relation_tube_pilot",
        "benchmark_metrics": False,
        "test_evaluated": False,
        "decision": (
            "bounded_residual_signal_observed_formal_tube_retained"
            if passed
            else "bounded_residual_not_confirmed_propagation_only_retained"
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
