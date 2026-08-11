"""Inspect MOVi collision metadata without decoding video frames."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from relation_tube.movi import (
    contact_episodes,
    select_isolated_episode,
    select_isolated_episode_at_offsets,
)


PROTOCOL = "movi_sparse_collision_metadata_v1"
FEATURE_DESCRIPTION = {
    "metadata/video_name": "byte",
    "metadata/num_frames": "int",
    "metadata/num_instances": "int",
    "events/collisions/instances": "int",
    "events/collisions/frame": "int",
    "events/collisions/force": "float",
    "events/collisions/contact_normal": "float",
    "instances/visibility": "int",
    "instances/mass": "float",
    "instances/velocities": "float",
    "instances/angular_velocities": "float",
    "instances/material_label": "int",
    "instances/shape_label": "int",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _scalar(value: np.ndarray) -> int:
    array = np.asarray(value).reshape(-1)
    if array.size != 1:
        raise ValueError("expected a scalar TFRecord feature")
    return int(array[0])


def audit_example(
    example: dict[str, Any],
    *,
    input_offsets: tuple[int, ...] | None = None,
    include_instrument_state: bool = True,
) -> dict[str, Any]:
    name_value = example["metadata/video_name"]
    video_name = name_value.decode("utf-8") if isinstance(name_value, bytes) else str(name_value)
    frames = _scalar(example["metadata/num_frames"])
    objects = _scalar(example["metadata/num_instances"])
    if frames <= 0 or objects <= 0:
        raise ValueError("invalid MOVi video dimensions")

    collision_frames = np.asarray(example["events/collisions/frame"], dtype=np.int64)
    collisions = {
        "instances": np.asarray(example["events/collisions/instances"], dtype=np.int64).reshape(-1, 2),
        "frame": collision_frames,
        "force": np.asarray(example["events/collisions/force"], dtype=np.float64),
        "contact_normal": np.asarray(
            example["events/collisions/contact_normal"], dtype=np.float64
        ).reshape(-1, 3),
    }
    episodes = contact_episodes(collisions)
    visibility = np.asarray(example["instances/visibility"], dtype=np.int64).reshape(objects, frames)
    selected = (
        select_isolated_episode(episodes, visibility)
        if input_offsets is None
        else select_isolated_episode_at_offsets(
            episodes, visibility, input_offsets=input_offsets
        )
    )
    foreground_rows = np.all(collisions["instances"] != 65535, axis=1)
    result: dict[str, Any] = {
        "video_name": video_name,
        "objects": objects,
        "frames": frames,
        "raw_contact_rows": int(collision_frames.size),
        "foreground_contact_rows": int(np.sum(foreground_rows)),
        "foreground_episodes": len(episodes),
        "selected": selected is not None,
    }
    if selected is None:
        result["rejection"] = (
            "no_foreground_contact_episode" if not episodes else "visibility_or_isolation"
        )
        return result

    materials = np.asarray(example["instances/material_label"], dtype=np.int64).reshape(objects)
    shapes = np.asarray(example["instances/shape_label"], dtype=np.int64).reshape(objects)
    a, b, start = selected.object_a, selected.object_b, selected.start_frame
    result["event"] = {
        "objects": [a, b],
        "start_frame": start,
        "end_frame": selected.end_frame,
        "contact_rows": selected.contact_rows,
        "impulse": selected.impulse.tolist(),
        "impulse_norm": float(np.linalg.norm(selected.impulse)),
        "material_pair": sorted([int(materials[a]), int(materials[b])]),
        "shape_pair": sorted([int(shapes[a]), int(shapes[b])]),
    }
    if include_instrument_state:
        masses = np.asarray(example["instances/mass"], dtype=np.float64).reshape(objects)
        velocities = np.asarray(example["instances/velocities"], dtype=np.float64).reshape(
            objects, frames, 3
        )
        angular = np.asarray(
            example["instances/angular_velocities"], dtype=np.float64
        ).reshape(objects, frames, 3)
        state = np.concatenate(
            [
                masses[[a, b]],
                velocities[[a, b], start].reshape(-1),
                angular[[a, b], start].reshape(-1),
            ]
        )
        if state.shape != (14,) or not np.all(np.isfinite(state)):
            raise ValueError("invalid selected-event instrument state")
        result["event"]["instrument_state"] = state.tolist()
    return result


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total <= 0 or successes < 0 or successes > total:
        raise ValueError("invalid binomial count")
    probability = successes / total
    denominator = 1.0 + z * z / total
    center = (probability + z * z / (2.0 * total)) / denominator
    half_width = z * np.sqrt(
        probability * (1.0 - probability) / total + z * z / (4.0 * total * total)
    ) / denominator
    return [float(center - half_width), float(center + half_width)]


def audit_records(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    examples = [audit_example(record) for record in records]
    if not examples:
        raise ValueError("no MOVi records found")
    selected = [item for item in examples if item["selected"]]
    rejection_counts: dict[str, int] = {}
    for item in examples:
        if not item["selected"]:
            reason = str(item["rejection"])
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
    impulse_norms = [float(item["event"]["impulse_norm"]) for item in selected]
    return {
        "videos": len(examples),
        "eligible_videos": len(selected),
        "eligible_fraction": len(selected) / len(examples),
        "eligible_fraction_wilson_95": wilson_interval(len(selected), len(examples)),
        "raw_contact_rows": int(sum(item["raw_contact_rows"] for item in examples)),
        "foreground_contact_rows": int(
            sum(item["foreground_contact_rows"] for item in examples)
        ),
        "foreground_episodes": int(sum(item["foreground_episodes"] for item in examples)),
        "rejections": rejection_counts,
        "selected_impulse_norm": {
            "minimum": min(impulse_norms) if impulse_norms else None,
            "median": float(np.median(impulse_norms)) if impulse_norms else None,
            "maximum": max(impulse_norms) if impulse_norms else None,
        },
        "examples": examples,
    }


def load_shards(paths: list[Path]) -> Iterable[dict[str, Any]]:
    try:
        from tfrecord.reader import tfrecord_loader
    except ImportError as error:
        raise RuntimeError("install the 'movi' optional dependency to read TFRecord shards") from error
    for path in paths:
        yield from tfrecord_loader(str(path), None, FEATURE_DESCRIPTION)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite result: {args.output}")
    for shard in args.shards:
        if not shard.is_file():
            raise FileNotFoundError(shard)
    result = {
        "protocol_version": PROTOCOL,
        "status": "upstream_validation_protocol_development_audit",
        "benchmark_metrics": False,
        "shards": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in args.shards
        ],
        "audit": audit_records(load_shards(args.shards)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "audit": result["audit"] | {"examples": "omitted"}}))


if __name__ == "__main__":
    main()
