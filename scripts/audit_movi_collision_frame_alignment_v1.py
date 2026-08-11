"""Check render-state alignment for MOVi step-level collision contacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

import audit_movi_collision_metadata_v1 as metadata_v1
import audit_movi_collision_physics_proxy_v1 as proxy
import audit_movi_collision_state_headroom_v1 as headroom


PROTOCOL = "movi_collision_frame_alignment_v1"


def validate_config(config: dict[str, Any]) -> None:
    if config.get("protocol_version") != PROTOCOL:
        raise ValueError("unexpected MOVi frame-alignment protocol")
    if config["data"].get("development_only") is not True:
        raise ValueError("alignment audit must remain development-only")
    if config["data"].get("test_evaluated") is not False:
        raise ValueError("test must remain closed")
    alignment = config["alignment"]
    if [int(value) for value in alignment["state_frame_offsets"]] != [-2, -1, 0]:
        raise ValueError("registered frame offsets changed")
    if int(alignment["primary_offset"]) != -1 or int(alignment["outer_folds"]) != 5:
        raise ValueError("registered primary alignment changed")


def event_at_offset(
    example: dict[str, Any],
    offset: int,
    *,
    input_offsets: tuple[int, ...] | None = None,
) -> dict[str, Any] | None:
    # State is read below at the requested offset. Event selection must also
    # work for a contact indexed immediately after the final rendered frame.
    selected = metadata_v1.audit_example(
        example,
        input_offsets=input_offsets,
        include_instrument_state=False,
    )
    if not selected["selected"]:
        return None
    objects = int(selected["objects"])
    frames = int(selected["frames"])
    event = selected["event"]
    first, second = (int(value) for value in event["objects"])
    state_frame = int(event["start_frame"]) + int(offset)
    if state_frame < 0 or state_frame >= frames:
        return None
    positions = np.asarray(example["instances/positions"], dtype=np.float64).reshape(
        objects, frames, 3
    )
    velocities = np.asarray(example["instances/velocities"], dtype=np.float64).reshape(
        objects, frames, 3
    )
    masses = np.asarray(example["instances/mass"], dtype=np.float64).reshape(objects)
    displacement = positions[second, state_frame] - positions[first, state_frame]
    distance = float(np.linalg.norm(displacement))
    if distance <= 1.0e-12:
        return None
    direction = displacement / distance
    relative_velocity = velocities[second, state_frame] - velocities[first, state_frame]
    separation_speed = float(np.dot(relative_velocity, direction))
    closing_speed = max(0.0, -separation_speed)
    reduced_mass = float(masses[first] * masses[second] / (masses[first] + masses[second]))
    base = reduced_mass * closing_speed * direction
    target = np.asarray(event["impulse"], dtype=np.float64)
    target_norm = float(np.linalg.norm(target))
    target_direction = target / target_norm if target_norm > 1.0e-12 else np.zeros(3)
    return {
        "video_name": str(selected["video_name"]),
        "objects": [first, second],
        "start_frame": int(event["start_frame"]),
        "base_impulse": base,
        "target": target,
        "closing": closing_speed > 0.0,
        "direction_cosine": float(np.dot(target_direction, direction)),
    }


def evaluate_offset(
    examples: list[dict[str, Any]], offset: int, folds: int, salt: str
) -> dict[str, Any]:
    rows = [event_at_offset(example, offset) for example in examples]
    selected = [row for row in rows if row is not None]
    names = [str(row["video_name"]) for row in selected]
    base = np.stack([np.asarray(row["base_impulse"]) for row in selected])
    target = np.stack([np.asarray(row["target"]) for row in selected])
    fitted, fold_records = proxy.oof_gain(base, target, names, folds, f"{salt}:offset{offset}")
    closing = np.asarray([bool(row["closing"]) for row in selected])
    cosine = np.asarray([float(row["direction_cosine"]) for row in selected])
    return {
        "events": len(selected),
        "closing_speed_fraction": float(np.mean(closing)),
        "direction_cosine": {
            "mean": float(np.mean(cosine)),
            "median": float(np.median(cosine)),
            "q05": float(np.quantile(cosine, 0.05)),
            "q95": float(np.quantile(cosine, 0.95)),
        },
        "unit_gain_metrics": headroom.metrics(target, base),
        "oof_fitted_gain_metrics": headroom.metrics(target, fitted),
        "folds": fold_records,
    }


def run(config: dict[str, Any], shards: list[Path]) -> dict[str, Any]:
    validate_config(config)
    examples = list(proxy.load_shards(shards))
    alignment = config["alignment"]
    offsets = [int(value) for value in alignment["state_frame_offsets"]]
    return {
        "protocol_version": PROTOCOL,
        "status": "completed_protocol_development_frame_alignment_audit",
        "benchmark_metrics": False,
        "test_evaluated": False,
        "source_order": "getContactPoints_then_read_state_then_stepSimulation",
        "primary_offset_registered_before_result": int(alignment["primary_offset"]),
        "shards": [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": metadata_v1.sha256_file(path),
            }
            for path in shards
        ],
        "offsets": {
            str(offset): evaluate_offset(
                examples,
                offset,
                int(alignment["outer_folds"]),
                str(alignment["split_salt"]),
            )
            for offset in offsets
        },
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
