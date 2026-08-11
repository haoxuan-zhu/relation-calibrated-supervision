"""Project ordinary sparse point predictors into a calibrated MOVi relation tube."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

import audit_movi_collision_physics_budget_v1 as budget_v1
import audit_movi_collision_physics_budget_v2 as budget_v2
import audit_movi_collision_relation_tube_pilot_v1 as tube_v1
import audit_movi_collision_state_headroom_v1 as headroom
import audit_movi_collision_visual_residual_v1 as visual_v1


PROTOCOL = "movi_collision_projected_point_tube_v1"


def validate_config(config: dict[str, Any]) -> None:
    if config.get("protocol_version") != PROTOCOL:
        raise ValueError("unexpected projected-point Tube protocol")
    if config["data"].get("development_only") is not True:
        raise ValueError("projected-point pilot must remain development-only")
    if config["data"].get("test_evaluated") is not False:
        raise ValueError("test must remain closed")
    probe = config["probe"]
    if [int(value) for value in probe["budgets"]] != [20, 40]:
        raise ValueError("registered budgets changed")
    if int(probe["repeats"]) != 5 or float(probe["ridge_alpha"]) != 0.1:
        raise ValueError("registered repeat count or ridge changed")
    if float(probe["coverage"]) != 0.95:
        raise ValueError("registered coverage changed")


def state_augmented_features(
    visual: np.ndarray, base: np.ndarray, pool_index: np.ndarray
) -> np.ndarray:
    mean = base[pool_index].mean(axis=0)
    scale = base[pool_index].std(axis=0)
    scale = np.where(scale > 1.0e-8, scale, 1.0)
    state = (base - mean) / scale
    state /= np.maximum(np.linalg.norm(state, axis=1, keepdims=True), 1.0e-12)
    return np.concatenate([visual, state], axis=1) / np.sqrt(2.0)


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
        "full": state_augmented_features(visual, base, pool_index),
    }
    probe = config["probe"]
    bounds = tuple(float(value) for value in probe["gain_bounds"])
    truth = target[evaluation_index]
    repeats = []
    for repeat in range(int(probe["repeats"])):
        seed = int(probe["order_seed_base"]) + repeat
        ordered = pool_index[np.random.default_rng(seed).permutation(len(pool_index))]
        budgets: dict[str, Any] = {}
        for label_budget in (int(value) for value in probe["budgets"]):
            calibration = ordered[:label_budget]
            gain = budget_v2.fit_bounded_gain(base[calibration], target[calibration], bounds)
            center = gain * base
            center_eval = center[evaluation_index]
            target_scale = target[calibration].std(axis=0)
            target_scale = np.where(target_scale > 1.0e-6, target_scale, 1.0)
            radius = tube_v1.loo_radius(
                base[calibration],
                target[calibration],
                target_scale,
                bounds,
                float(probe["coverage"]),
            )
            conditions = {"center_only": headroom.metrics(truth, center_eval)}
            comparisons: dict[str, Any] = {}
            boundary_rates: dict[str, float] = {}
            for branch, values in features.items():
                point_normalized = tube_v1.dual_ridge(
                    values[calibration],
                    target[calibration] / target_scale,
                    values[evaluation_index],
                    float(probe["ridge_alpha"]),
                    fit_intercept=True,
                )
                point = point_normalized * target_scale
                raw_residual = (point - center_eval) / target_scale
                norms = np.linalg.norm(raw_residual, axis=1)
                factors = np.minimum(1.0, radius / np.maximum(norms, 1.0e-12))
                projected = center_eval + raw_residual * factors[:, None] * target_scale
                conditions[f"point_{branch}"] = headroom.metrics(truth, point)
                conditions[f"projected_{branch}"] = headroom.metrics(truth, projected)
                boundary_rates[branch] = float(np.mean(factors < 1.0 - 1.0e-12))
                bootstrap_seed = (
                    int(probe["bootstrap_seed_base"])
                    + repeat * 100
                    + label_budget
                    + (0 if branch == "visual" else 10_000)
                )
                comparisons[f"projected_{branch}_minus_point"] = visual_v1.bootstrap_delta(
                    truth,
                    projected,
                    point,
                    replicates=int(probe["bootstrap_replicates"]),
                    seed=bootstrap_seed,
                )
                comparisons[f"projected_{branch}_minus_center"] = visual_v1.bootstrap_delta(
                    truth,
                    projected,
                    center_eval,
                    replicates=int(probe["bootstrap_replicates"]),
                    seed=bootstrap_seed + 20_000,
                )
            budgets[str(label_budget)] = {
                "bounded_gain": gain,
                "target_scale": target_scale.tolist(),
                "radius": radius,
                "boundary_rates": boundary_rates,
                "conditions": conditions,
                "comparisons": comparisons,
            }
        repeats.append({"repeat": repeat, "order_seed": seed, "budgets": budgets})

    summary: dict[str, Any] = {}
    for label_budget in (str(value) for value in probe["budgets"]):
        summary[label_budget] = {}
        for branch in ("visual", "full"):
            branch_summary: dict[str, Any] = {
                "boundary_rate_mean": float(
                    np.mean(
                        [row["budgets"][label_budget]["boundary_rates"][branch] for row in repeats]
                    )
                )
            }
            for comparison in ("point", "center"):
                name = f"projected_{branch}_minus_{comparison}"
                values = np.asarray(
                    [row["budgets"][label_budget]["comparisons"][name]["observed"] for row in repeats]
                )
                branch_summary[f"minus_{comparison}"] = {
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
            summary[label_budget][branch] = branch_summary
    primary = summary["40"]["full"]
    passed = all(
        primary[name]["positive_repeats"] >= 3 and primary[name]["mean"] > 0.0
        for name in ("minus_point", "minus_center")
    )
    return {
        "protocol_version": PROTOCOL,
        "status": "completed_development_projected_point_tube",
        "benchmark_metrics": False,
        "test_evaluated": False,
        "decision": (
            "projected_point_tube_signal_observed"
            if passed
            else "projected_point_tube_not_confirmed"
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
