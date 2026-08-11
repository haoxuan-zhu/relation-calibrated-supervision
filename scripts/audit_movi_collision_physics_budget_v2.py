"""Audit the physically bounded MOVi collision gain on the frozen v1 subsets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

import audit_movi_collision_frame_alignment_v1 as alignment
import audit_movi_collision_metadata_v1 as metadata_v1
import audit_movi_collision_physics_budget_v1 as v1
import audit_movi_collision_physics_proxy_v1 as proxy
import audit_movi_collision_state_headroom_v1 as headroom
from relation_tube.calibration import calibrate_tube


PROTOCOL = "movi_collision_physics_budget_v2"


def validate_config(config: dict[str, Any]) -> None:
    if config.get("protocol_version") != PROTOCOL:
        raise ValueError("unexpected bounded physics-budget protocol")
    if config["data"].get("development_only") is not True:
        raise ValueError("budget audit must remain development-only")
    if config["data"].get("test_evaluated") is not False:
        raise ValueError("test must remain closed")
    calibration = config["calibration"]
    if [int(value) for value in calibration["budgets"]] != [4, 8, 20, 40]:
        raise ValueError("registered budgets changed")
    if int(calibration["repeats"]) != 5:
        raise ValueError("registered repeats changed")
    if [float(value) for value in calibration["gain_bounds"]] != [1.0, 2.0]:
        raise ValueError("physical gain bounds changed")


def fit_bounded_gain(base: np.ndarray, target: np.ndarray, bounds: tuple[float, float]) -> float:
    return float(np.clip(proxy.fit_nonnegative_gain(base, target), bounds[0], bounds[1]))


def bounded_loo_residuals(
    base: np.ndarray, target: np.ndarray, bounds: tuple[float, float]
) -> np.ndarray:
    residuals = np.empty_like(target)
    for index in range(base.shape[0]):
        keep = np.arange(base.shape[0]) != index
        gain = fit_bounded_gain(base[keep], target[keep], bounds)
        residuals[index] = target[index] - gain * base[index]
    return residuals


def run(config: dict[str, Any], shards: list[Path]) -> dict[str, Any]:
    validate_config(config)
    raw_examples = list(proxy.load_shards(shards))
    rows = [alignment.event_at_offset(example, -1) for example in raw_examples]
    selected = [row for row in rows if row is not None]
    split = config["split"]
    pool, evaluation = v1.split_events(
        selected,
        int(split["evaluation_modulus"]),
        {int(value) for value in split["evaluation_remainders"]},
        str(split["split_salt"]),
    )
    evaluation_base = np.stack([row["base_impulse"] for row in evaluation])
    evaluation_target = np.stack([row["target"] for row in evaluation])
    calibration = config["calibration"]
    bounds = tuple(float(value) for value in calibration["gain_bounds"])
    repeats: list[dict[str, Any]] = []
    for repeat in range(int(calibration["repeats"])):
        order = np.random.default_rng(int(calibration["order_seed_base"]) + repeat).permutation(
            len(pool)
        )
        ordered = [pool[index] for index in order]
        budget_rows: dict[str, Any] = {}
        for budget in (int(value) for value in calibration["budgets"]):
            subset = ordered[:budget]
            base = np.stack([row["base_impulse"] for row in subset])
            target = np.stack([row["target"] for row in subset])
            unbounded_gain = proxy.fit_nonnegative_gain(base, target)
            bounded_gain = fit_bounded_gain(base, target, bounds)
            unbounded_prediction = unbounded_gain * evaluation_base
            bounded_prediction = bounded_gain * evaluation_base
            geometry = calibrate_tube(
                bounded_loo_residuals(base, target, bounds),
                coverage=float(config["tube"]["coverage"]),
                ridge=float(config["tube"]["covariance_ridge"]),
                geometry=str(config["tube"]["geometry"]),
            )
            budget_rows[str(budget)] = {
                "unbounded_gain": unbounded_gain,
                "bounded_gain": bounded_gain,
                "gain_was_clipped": bounded_gain != unbounded_gain,
                "unbounded_center": headroom.metrics(evaluation_target, unbounded_prediction),
                "bounded_center": headroom.metrics(evaluation_target, bounded_prediction),
                "bounded_minus_unbounded_r2": headroom.metrics(
                    evaluation_target, bounded_prediction
                )["mean_coordinate_r2"]
                - headroom.metrics(evaluation_target, unbounded_prediction)["mean_coordinate_r2"],
                "tube": {
                    "calibration_radius_squared": geometry.radius_squared,
                    "calibration_empirical_coverage": geometry.empirical_coverage,
                    "evaluation_coverage": v1.tube_coverage(
                        evaluation_target - bounded_prediction, geometry
                    ),
                },
            }
        repeats.append(
            {
                "repeat": repeat,
                "order_seed": int(calibration["order_seed_base"]) + repeat,
                "budgets": budget_rows,
            }
        )

    summary: dict[str, Any] = {}
    for budget in (str(value) for value in calibration["budgets"]):
        values = [row["budgets"][budget] for row in repeats]
        gains = np.asarray([row["bounded_gain"] for row in values])
        r2 = np.asarray([row["bounded_center"]["mean_coordinate_r2"] for row in values])
        coverage = np.asarray([row["tube"]["evaluation_coverage"] for row in values])
        summary[budget] = {
            "bounded_gain_range": [float(np.min(gains)), float(np.max(gains))],
            "clipped_repeats": int(sum(bool(row["gain_was_clipped"]) for row in values)),
            "bounded_center_r2_mean": float(np.mean(r2)),
            "bounded_center_r2_range": [float(np.min(r2)), float(np.max(r2))],
            "tube_coverage_mean": float(np.mean(coverage)),
            "tube_coverage_range": [float(np.min(coverage)), float(np.max(coverage))],
        }
    return {
        "protocol_version": PROTOCOL,
        "status": "completed_protocol_development_bounded_physics_budget_audit",
        "benchmark_metrics": False,
        "test_evaluated": False,
        "physical_basis": "gain_equals_one_plus_restitution_in_frictionless_center_line_limit",
        "gain_bounds": list(bounds),
        "shards": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": metadata_v1.sha256_file(path)}
            for path in shards
        ],
        "split": {"calibration_pool_events": len(pool), "evaluation_events": len(evaluation)},
        "summary": summary,
        "repeats": repeats,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--shards", nargs="+", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    validate_config(config)
    output_path = args.output or Path(config["output"]["json_path"])
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite result: {output_path}")
    result = run(config, args.shards)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path), "benchmark_metrics": False}, sort_keys=True))


if __name__ == "__main__":
    main()
