"""Materialize deterministic MOVi train, validation and sealed held-out archives."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


PROTOCOL = "movi_collision_split_materialization_v1"
STATIC_KEYS = ("visual_input_offsets", "state_offset")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def materialize(
    aggregate_path: Path,
    aggregate_manifest_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    aggregate_manifest = json.loads(aggregate_manifest_path.read_text(encoding="utf-8"))
    expected_hash = str(aggregate_manifest["output"]["sha256"])
    observed_hash = sha256_file(aggregate_path)
    if observed_hash != expected_hash:
        raise ValueError("aggregate archive does not match its manifest")
    if aggregate_manifest.get("heldout_target_evaluated") is not False:
        raise ValueError("aggregate manifest already reports held-out evaluation")
    with np.load(aggregate_path, allow_pickle=False) as archive:
        if "split" not in archive.files or "video_name" not in archive.files:
            raise ValueError("aggregate archive has no deterministic split")
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    names = arrays["video_name"]
    splits = arrays["split"]
    event_keys = tuple(
        key
        for key, value in arrays.items()
        if key not in STATIC_KEYS and value.ndim > 0 and len(value) == len(names)
    )
    if "target" not in event_keys or set(np.unique(splits)) != {
        "train",
        "validation",
        "heldout",
    }:
        raise ValueError("aggregate lacks a target or one registered split")

    output_dir.mkdir(parents=True, exist_ok=True)
    records: dict[str, Any] = {}
    for split in ("train", "validation", "heldout"):
        output = output_dir / f"{split}.npz"
        if output.exists():
            raise FileExistsError(f"refusing to overwrite split archive: {output}")
        keep = splits == split
        if not np.any(keep):
            raise ValueError(f"empty split: {split}")
        payload = {key: arrays[key][keep] for key in event_keys}
        payload.update({key: arrays[key] for key in STATIC_KEYS})
        temporary = output.with_suffix(output.suffix + ".partial")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **payload)
        temporary.replace(output)
        records[split] = {
            "path": str(output),
            "events": int(np.sum(keep)),
            "bytes": output.stat().st_size,
            "sha256": sha256_file(output),
            "target_present": True,
            "evaluation_status": (
                "sealed_not_evaluated" if split == "heldout" else "available_for_development"
            ),
        }
    return {
        "protocol_version": PROTOCOL,
        "status": "completed_deterministic_split_materialization",
        "benchmark_metrics": False,
        "heldout_target_evaluated": False,
        "aggregate": {
            "path": str(aggregate_path),
            "sha256": observed_hash,
            "manifest_path": str(aggregate_manifest_path),
            "manifest_sha256": sha256_file(aggregate_manifest_path),
        },
        "event_keys": list(event_keys),
        "static_keys": list(STATIC_KEYS),
        "splits": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", required=True, type=Path)
    parser.add_argument("--aggregate-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    args = parser.parse_args()
    if args.split_manifest.exists():
        raise FileExistsError(f"refusing to overwrite split manifest: {args.split_manifest}")
    result = materialize(args.aggregate, args.aggregate_manifest, args.output_dir)
    args.split_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.split_manifest.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {name: row["events"] for name, row in result["splits"].items()},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
