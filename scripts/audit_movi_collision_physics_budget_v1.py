"""Nested K-label audit for the source-aligned MOVi physical relation center."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

import audit_movi_collision_frame_alignment_v1 as alignment
import audit_movi_collision_metadata_v1 as metadata_v1
import audit_movi_collision_physics_proxy_v1 as proxy
import audit_movi_collision_state_headroom_v1 as headroom
from relation_tube.calibration import calibrate_tube


PROTOCOL = "movi_collision_physics_budget_v1"


def validate_config(config: dict[str, Any]) -> None:
    if config.get("protocol_version") != PROTOCOL:
        raise ValueError("unexpected MOVi physics-budget protocol")
    if config["data"].get("development_only") is not True:
        raise ValueError("budget audit must remain development-only")
    if config["data"].get("test_evaluated") is not False:
        raise ValueError("test must remain closed")
    if [int(value) for value in config["calibration"]["budgets"]] != [4, 8, 20, 40]:
        raise ValueError("registered budgets changed")
    if int(config["calibration"]["repeats"]) != 5:
        raise ValueError("registered repeats changed")
    tube = config["tube"]
    if tube.get("geometry") != "isotropic" or float(tube["coverage"]) != 0.95:
        raise ValueError("registered tube changed")


def split_events(
    rows: list[dict[str, Any]], modulus: int, remainders: set[int], salt: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pool, evaluation = [], []
    for row in rows:
        bucket = headroom.fold_id(str(row["video_name"]), salt, modulus)
        (evaluation if bucket in remainders else pool).append(row)
    if len(pool) < 40 or len(evaluation) < 20:
        raise ValueError("development split is too small")
    return pool, evaluation


def loo_residuals(base: np.ndarray, target: np.ndarray) -> np.ndarray:
    if base.shape != target.shape or base.shape[0] < 3:
        raise ValueError("invalid gain calibration rows")
    residuals = np.empty_like(target)
    for index in range(base.shape[0]):
        keep = np.arange(base.shape[0]) != index
        gain = proxy.fit_nonnegative_gain(base[keep], target[keep])
        residuals[index] = target[index] - gain * base[index]
    return residuals


def derangement(size: int, seed: int) -> np.ndarray:
    identity = np.arange(size)
    rng = np.random.default_rng(seed)
    for _ in range(10_000):
        candidate = rng.permutation(size)
        if np.all(candidate != identity):
            return candidate
    raise RuntimeError("failed to construct derangement")


def tube_coverage(residuals: np.ndarray, geometry: Any) -> float:
    scores = np.einsum("ni,ij,nj->n", residuals, geometry.precision, residuals)
    return float(np.mean(scores <= geometry.radius_squared + 1.0e-12))


def run(config: dict[str, Any], shards: list[Path]) -> dict[str, Any]:
    validate_config(config)
    raw_examples = list(proxy.load_shards(shards))
    rows = [alignment.event_at_offset(example, -1) for example in raw_examples]
    selected = [row for row in rows if row is not None]
    split = config["split"]
    pool, evaluation = split_events(
        selected,
        int(split["evaluation_modulus"]),
        {int(value) for value in split["evaluation_remainders"]},
        str(split["split_salt"]),
    )
    evaluation_base = np.stack([row["base_impulse"] for row in evaluation])
    evaluation_target = np.stack([row["target"] for row in evaluation])
    evaluation_mean = np.broadcast_to(evaluation_target.mean(axis=0), evaluation_target.shape)

    calibration = config["calibration"]
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
            gain = proxy.fit_nonnegative_gain(base, target)
            prediction = gain * evaluation_base
            permutation = derangement(
                budget, int(calibration["derangement_seed_base"]) + repeat * 100 + budget
            )
            permuted_gain = proxy.fit_nonnegative_gain(base[permutation], target)
            permuted_prediction = permuted_gain * evaluation_base
            target_mean = target.mean(axis=0)
            mean_prediction = np.broadcast_to(target_mean, evaluation_target.shape)
            residuals = loo_residuals(base, target)
            geometry = calibrate_tube(
                residuals,
                coverage=float(config["tube"]["coverage"]),
                ridge=float(config["tube"]["covariance_ridge"]),
                geometry=str(config["tube"]["geometry"]),
            )
            evaluation_residuals = evaluation_target - prediction
            budget_rows[str(budget)] = {
                "gain": gain,
                "permuted_gain": permuted_gain,
                "correct_center": headroom.metrics(evaluation_target, prediction),
                "permuted_center": headroom.metrics(evaluation_target, permuted_prediction),
                "k_label_mean": headroom.metrics(evaluation_target, mean_prediction),
                "evaluation_target_mean_reference": headroom.metrics(
                    evaluation_target, evaluation_mean
                ),
                "tube": {
                    "calibration_radius_squared": geometry.radius_squared,
                    "calibration_empirical_coverage": geometry.empirical_coverage,
                    "evaluation_coverage": tube_coverage(evaluation_residuals, geometry),
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
        gains = np.asarray([row["budgets"][budget]["gain"] for row in repeats])
        r2_values = np.asarray(
            [row["budgets"][budget]["correct_center"]["mean_coordinate_r2"] for row in repeats]
        )
        coverages = np.asarray(
            [row["budgets"][budget]["tube"]["evaluation_coverage"] for row in repeats]
        )
        summary[budget] = {
            "gain_mean": float(np.mean(gains)),
            "gain_range": [float(np.min(gains)), float(np.max(gains))],
            "center_r2_mean": float(np.mean(r2_values)),
            "center_r2_range": [float(np.min(r2_values)), float(np.max(r2_values))],
            "tube_coverage_mean": float(np.mean(coverages)),
            "tube_coverage_range": [float(np.min(coverages)), float(np.max(coverages))],
        }
    return {
        "protocol_version": PROTOCOL,
        "status": "completed_protocol_development_physics_budget_audit",
        "benchmark_metrics": False,
        "test_evaluated": False,
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
