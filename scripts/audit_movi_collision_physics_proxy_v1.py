"""Evaluate a center-line rigid-body proxy on selected MOVi collision episodes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

import audit_movi_collision_metadata_v1 as metadata_v1
import audit_movi_collision_state_headroom_v1 as headroom


PROTOCOL = "movi_collision_physics_proxy_v1"
FEATURE_DESCRIPTION = metadata_v1.FEATURE_DESCRIPTION | {
    "instances/positions": "float",
    "instances/friction": "float",
    "instances/restitution": "float",
}


def validate_config(config: dict[str, Any]) -> None:
    if config.get("protocol_version") != PROTOCOL:
        raise ValueError("unexpected MOVi physics-proxy protocol")
    if config["data"].get("development_only") is not True:
        raise ValueError("physics proxy audit must remain development-only")
    if config["data"].get("test_evaluated") is not False:
        raise ValueError("test must remain closed")
    if int(config["cross_validation"]["outer_folds"]) != 5:
        raise ValueError("registered outer folds changed")


def physics_example(example: dict[str, Any]) -> dict[str, Any]:
    result = metadata_v1.audit_example(example)
    if not result["selected"]:
        return result
    objects = int(result["objects"])
    frames = int(result["frames"])
    event = result["event"]
    first, second = (int(value) for value in event["objects"])
    start = int(event["start_frame"])
    positions = np.asarray(example["instances/positions"], dtype=np.float64).reshape(
        objects, frames, 3
    )
    velocities = np.asarray(example["instances/velocities"], dtype=np.float64).reshape(
        objects, frames, 3
    )
    masses = np.asarray(example["instances/mass"], dtype=np.float64).reshape(objects)
    friction = np.asarray(example["instances/friction"], dtype=np.float64).reshape(objects)
    restitution = np.asarray(example["instances/restitution"], dtype=np.float64).reshape(objects)

    displacement = positions[second, start] - positions[first, start]
    distance = float(np.linalg.norm(displacement))
    if distance <= 1.0e-12:
        raise ValueError("collision object centers coincide")
    center_direction = displacement / distance
    relative_velocity = velocities[second, start] - velocities[first, start]
    signed_separation_speed = float(np.dot(relative_velocity, center_direction))
    closing_speed = max(0.0, -signed_separation_speed)
    reduced_mass = float(masses[first] * masses[second] / (masses[first] + masses[second]))
    base_impulse = reduced_mass * closing_speed * center_direction
    target = np.asarray(event["impulse"], dtype=np.float64)
    target_norm = float(np.linalg.norm(target))
    target_direction = target / target_norm if target_norm > 1.0e-12 else np.zeros(3)
    event["physics_proxy"] = {
        "positions_at_start": positions[[first, second], start].reshape(-1).tolist(),
        "center_distance": distance,
        "center_direction": center_direction.tolist(),
        "relative_velocity": relative_velocity.tolist(),
        "signed_separation_speed": signed_separation_speed,
        "closing_speed": closing_speed,
        "reduced_mass": reduced_mass,
        "base_impulse": base_impulse.tolist(),
        "target_center_direction_cosine": float(np.dot(target_direction, center_direction)),
        "friction_pair": friction[[first, second]].tolist(),
        "restitution_pair": restitution[[first, second]].tolist(),
    }
    return result


def load_shards(paths: list[Path]) -> Iterable[dict[str, Any]]:
    try:
        from tfrecord.reader import tfrecord_loader
    except ImportError as error:
        raise RuntimeError("install the 'movi' optional dependency") from error
    for path in paths:
        yield from tfrecord_loader(str(path), None, FEATURE_DESCRIPTION)


def fit_nonnegative_gain(base: np.ndarray, target: np.ndarray) -> float:
    denominator = float(np.sum(base**2))
    if denominator <= 1.0e-15:
        return 0.0
    return max(0.0, float(np.sum(base * target) / denominator))


def oof_gain(
    base: np.ndarray, target: np.ndarray, names: list[str], folds: int, salt: str
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    fold_ids = np.asarray([headroom.fold_id(name, salt, folds) for name in names])
    prediction = np.empty_like(target)
    records: list[dict[str, Any]] = []
    for fold in range(folds):
        validation = fold_ids == fold
        training = ~validation
        gain = fit_nonnegative_gain(base[training], target[training])
        prediction[validation] = gain * base[validation]
        records.append(
            {
                "fold": fold,
                "training_events": int(np.sum(training)),
                "validation_events": int(np.sum(validation)),
                "gain": gain,
            }
        )
    return prediction, records


def run(config: dict[str, Any], shards: list[Path]) -> dict[str, Any]:
    validate_config(config)
    examples = [physics_example(example) for example in load_shards(shards)]
    selected = [row for row in examples if row["selected"]]
    if len(selected) < 50:
        raise ValueError("too few selected events for physics proxy audit")
    names = [str(row["video_name"]) for row in selected]
    target = np.asarray([row["event"]["impulse"] for row in selected], dtype=np.float64)
    base = np.asarray(
        [row["event"]["physics_proxy"]["base_impulse"] for row in selected], dtype=np.float64
    )
    cosines = np.asarray(
        [
            row["event"]["physics_proxy"]["target_center_direction_cosine"]
            for row in selected
        ],
        dtype=np.float64,
    )
    closing = np.asarray(
        [row["event"]["physics_proxy"]["closing_speed"] for row in selected], dtype=np.float64
    )
    cross_validation = config["cross_validation"]
    fitted, fold_records = oof_gain(
        base,
        target,
        names,
        int(cross_validation["outer_folds"]),
        str(cross_validation["split_salt"]),
    )
    mean_prediction = np.broadcast_to(target.mean(axis=0), target.shape).copy()
    evaluation = config["evaluation"]
    return {
        "protocol_version": PROTOCOL,
        "status": "completed_protocol_development_physics_proxy_audit",
        "benchmark_metrics": False,
        "test_evaluated": False,
        "shards": [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": metadata_v1.sha256_file(path),
            }
            for path in shards
        ],
        "selected_events": len(selected),
        "conditions": {
            "global_mean": headroom.metrics(target, mean_prediction),
            "unit_gain_center_line_proxy": headroom.metrics(target, base),
            "oof_fitted_gain_center_line_proxy": headroom.metrics(target, fitted),
        },
        "folds": fold_records,
        "geometry": {
            "center_direction_cosine": {
                "minimum": float(np.min(cosines)),
                "q05": float(np.quantile(cosines, 0.05)),
                "median": float(np.median(cosines)),
                "q95": float(np.quantile(cosines, 0.95)),
                "maximum": float(np.max(cosines)),
                "mean": float(np.mean(cosines)),
            },
            "positive_closing_speed_events": int(np.sum(closing > 0.0)),
            "closing_speed_fraction": float(np.mean(closing > 0.0)),
        },
        "paired_bootstrap": {
            "fitted_proxy_minus_mean": headroom.bootstrap_r2_delta(
                target,
                fitted,
                mean_prediction,
                int(evaluation["bootstrap_replicates"]),
                int(evaluation["bootstrap_seed"]),
                float(evaluation["confidence_level"]),
            )
        },
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--shards", nargs="+", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    validate_config(config)
    for shard in args.shards:
        if not shard.is_file():
            raise FileNotFoundError(shard)
    output_path = args.output or Path(config["output"]["json_path"])
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite result: {output_path}")
    result = run(config, args.shards)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path), "benchmark_metrics": False}, sort_keys=True))


if __name__ == "__main__":
    main()
