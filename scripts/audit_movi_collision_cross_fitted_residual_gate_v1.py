"""Cross-fit a scalar reliability gate for the MOVi visual residual."""

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


PROTOCOL = "movi_collision_cross_fitted_residual_gate_v1"


def validate_config(config: dict[str, Any]) -> None:
    if config.get("protocol_version") != PROTOCOL:
        raise ValueError("unexpected cross-fitted residual-gate protocol")
    if config["data"].get("development_only") is not True:
        raise ValueError("residual gate must remain development-only")
    if config["data"].get("test_evaluated") is not False:
        raise ValueError("test must remain closed")
    probe = config["probe"]
    if [int(value) for value in probe["budgets"]] != [20, 40]:
        raise ValueError("registered budgets changed")
    if int(probe["repeats"]) != 5 or float(probe["ridge_alpha"]) != 0.1:
        raise ValueError("registered repeats or ridge changed")
    if probe.get("fit_intercept") is not True:
        raise ValueError("registered residual intercept changed")
    if [float(value) for value in probe["reliability_bounds"]] != [0.0, 1.0]:
        raise ValueError("registered reliability bounds changed")


def reliability(true_residual: np.ndarray, predicted_residual: np.ndarray) -> float:
    if true_residual.shape != predicted_residual.shape or true_residual.ndim != 2:
        raise ValueError("residual arrays must have the same matrix shape")
    denominator = float(np.sum(predicted_residual**2))
    if denominator <= 1.0e-15:
        return 0.0
    return float(np.clip(np.sum(true_residual * predicted_residual) / denominator, 0.0, 1.0))


def cross_fitted_beta(
    features: np.ndarray,
    source_features: np.ndarray,
    base: np.ndarray,
    target: np.ndarray,
    alpha: float,
    bounds: tuple[float, float],
) -> tuple[float, np.ndarray, np.ndarray]:
    if len(features) < 3 or features.shape != source_features.shape:
        raise ValueError("invalid cross-fitting rows")
    true_rows = np.empty_like(target, dtype=np.float64)
    predicted_rows = np.empty_like(target, dtype=np.float64)
    for index in range(len(features)):
        keep = np.arange(len(features)) != index
        gain = budget_v2.fit_bounded_gain(base[keep], target[keep], bounds)
        center = gain * base
        scale = target[keep].std(axis=0)
        scale = np.where(scale > 1.0e-6, scale, 1.0)
        train_residual = (target[keep] - center[keep]) / scale
        predicted = tube_v1.dual_ridge(
            source_features[keep],
            train_residual,
            features[index : index + 1],
            alpha,
            fit_intercept=True,
        )[0]
        true_rows[index] = target[index] - center[index]
        predicted_rows[index] = predicted * scale
    return reliability(true_rows, predicted_rows), true_rows, predicted_rows


def residual_prediction(
    source_features: np.ndarray,
    evaluation_features: np.ndarray,
    base: np.ndarray,
    target: np.ndarray,
    alpha: float,
    bounds: tuple[float, float],
) -> tuple[np.ndarray, float, np.ndarray]:
    gain = budget_v2.fit_bounded_gain(base, target, bounds)
    center = gain * base
    scale = target.std(axis=0)
    scale = np.where(scale > 1.0e-6, scale, 1.0)
    residual = (target - center) / scale
    predicted = tube_v1.dual_ridge(
        source_features,
        residual,
        evaluation_features,
        alpha,
        fit_intercept=True,
    )
    return predicted * scale, gain, scale


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
    features = tube_v1.visual_descriptor(clip, pair, objects, pool_index)
    probe = config["probe"]
    bounds = tuple(float(value) for value in probe["gain_bounds"])
    alpha = float(probe["ridge_alpha"])
    repeats: list[dict[str, Any]] = []
    for repeat in range(int(probe["repeats"])):
        order_seed = int(probe["order_seed_base"]) + repeat
        ordered = pool_index[np.random.default_rng(order_seed).permutation(len(pool_index))]
        budget_rows: dict[str, Any] = {}
        for label_budget in (int(value) for value in probe["budgets"]):
            calibration = ordered[:label_budget]
            x = features[calibration]
            derangement = budget_v1.derangement(
                label_budget,
                int(probe["derangement_seed_base"]) + repeat * 100 + label_budget,
            )
            beta, _, _ = cross_fitted_beta(
                x, x, base[calibration], target[calibration], alpha, bounds
            )
            deranged_beta, _, _ = cross_fitted_beta(
                x, x[derangement], base[calibration], target[calibration], alpha, bounds
            )
            residual, gain, scale = residual_prediction(
                x,
                features[evaluation_index],
                base[calibration],
                target[calibration],
                alpha,
                bounds,
            )
            deranged_residual, _, _ = residual_prediction(
                x[derangement],
                features[evaluation_index],
                base[calibration],
                target[calibration],
                alpha,
                bounds,
            )
            center = gain * base[evaluation_index]
            ungated = center + residual
            gated = center + beta * residual
            deranged = center + deranged_beta * deranged_residual
            truth = target[evaluation_index]
            conditions = {
                "center_only": headroom.metrics(truth, center),
                "ungated_residual": headroom.metrics(truth, ungated),
                "cross_fitted_gate": headroom.metrics(truth, gated),
                "deranged_cross_fitted_gate": headroom.metrics(truth, deranged),
            }
            seed = int(probe["bootstrap_seed_base"]) + repeat * 100 + label_budget
            comparisons = {
                "gated_minus_center": visual_v1.bootstrap_delta(
                    truth,
                    gated,
                    center,
                    replicates=int(probe["bootstrap_replicates"]),
                    seed=seed,
                ),
                "gated_minus_ungated": visual_v1.bootstrap_delta(
                    truth,
                    gated,
                    ungated,
                    replicates=int(probe["bootstrap_replicates"]),
                    seed=seed + 10_000,
                ),
                "gated_minus_deranged": visual_v1.bootstrap_delta(
                    truth,
                    gated,
                    deranged,
                    replicates=int(probe["bootstrap_replicates"]),
                    seed=seed + 20_000,
                ),
            }
            budget_rows[str(label_budget)] = {
                "gain": gain,
                "target_scale": scale.tolist(),
                "beta": beta,
                "deranged_beta": deranged_beta,
                "conditions": conditions,
                "comparisons": comparisons,
            }
        repeats.append({"repeat": repeat, "order_seed": order_seed, "budgets": budget_rows})
    summary: dict[str, Any] = {}
    for label_budget in (str(value) for value in probe["budgets"]):
        summary[label_budget] = {}
        for comparison in (
            "gated_minus_center",
            "gated_minus_ungated",
            "gated_minus_deranged",
        ):
            records = [row["budgets"][label_budget]["comparisons"][comparison] for row in repeats]
            values = np.asarray([record["observed"] for record in records])
            summary[label_budget][comparison] = {
                "mean": float(np.mean(values)),
                "range": [float(np.min(values)), float(np.max(values))],
                "positive_repeats": int(np.sum(values > 0.0)),
                "ci95_positive_repeats": int(sum(record["ci95_low"] > 0.0 for record in records)),
            }
        summary[label_budget]["beta"] = [
            float(row["budgets"][label_budget]["beta"]) for row in repeats
        ]
        summary[label_budget]["deranged_beta"] = [
            float(row["budgets"][label_budget]["deranged_beta"]) for row in repeats
        ]
    passed = all(
        summary["40"][name]["positive_repeats"] >= 3
        and summary["40"][name]["mean"] > 0.0
        for name in (
            "gated_minus_center",
            "gated_minus_ungated",
            "gated_minus_deranged",
        )
    )
    return {
        "protocol_version": PROTOCOL,
        "status": "completed_development_cross_fitted_residual_gate",
        "benchmark_metrics": False,
        "test_evaluated": False,
        "decision": (
            "cross_fitted_residual_gate_signal_observed"
            if passed
            else "cross_fitted_residual_gate_not_confirmed"
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
