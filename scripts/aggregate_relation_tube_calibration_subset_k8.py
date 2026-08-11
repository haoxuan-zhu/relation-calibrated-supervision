"""Aggregate the three frozen K8 calibration-subset audits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

import run_diagnostic as base
import run_physics_functional_anchor_training as v11


PROTOCOL = "relation_tube_calibration_subset_k8_aggregate_v1"
INPUT_PROTOCOL = "relation_tube_calibration_subset_k8_v1"
SUBSET_SEEDS = (20260811, 20260821, 20260831)
CANDIDATES = ("diagonal", "isotropic")
SEMANTIC_METRICS = (
    "raw_correlation",
    "raw_r2",
    "rgb_mcc",
    "coordinatewise_r2",
    "full_affine_r2",
)


def validate_config(config: dict[str, Any]) -> None:
    if config["protocol_version"] != PROTOCOL:
        raise ValueError("unexpected K8 aggregate protocol")
    if tuple(int(seed) for seed in config["inputs"].keys()) != SUBSET_SEEDS:
        raise ValueError("K8 aggregate subset registry changed")
    if config["test_evaluated"] is not False:
        raise ValueError("K8 aggregate cannot evaluate test")


def summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "sample_std": float(array.std(ddof=1)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def aggregate(results: dict[int, dict[str, Any]]) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    for candidate in CANDIDATES:
        rows = {
            str(seed): results[seed]["comparisons"][candidate]
            for seed in SUBSET_SEEDS
        }
        metric_summary = {}
        for metric in SEMANTIC_METRICS:
            values = [
                float(rows[str(seed)]["semantic_deltas"][metric])
                for seed in SUBSET_SEEDS
            ]
            metric_summary[metric] = {
                **summary(values),
                "strict_positive_count": sum(value > 0.0 for value in values),
                "strict_negative_count": sum(value < 0.0 for value in values),
            }
        comparisons[candidate] = {
            "subsets": rows,
            "semantic_deltas": metric_summary,
            "complete_semantic_advantage_count": sum(
                bool(row["complete_semantic_advantage"]) for row in rows.values()
            ),
            "graph_advantage_count": sum(
                bool(row["graph_advantage"]) for row in rows.values()
            ),
            "ccrl_total_delta": summary(
                [
                    float(rows[str(seed)]["candidate_minus_full_ccrl_total"])
                    for seed in SUBSET_SEEDS
                ]
            ),
        }

    preregistered_candidates = []
    for candidate in CANDIDATES:
        item = comparisons[candidate]
        if (
            item["complete_semantic_advantage_count"] == len(SUBSET_SEEDS)
            and item["semantic_deltas"]["raw_correlation"][
                "strict_negative_count"
            ]
            == 0
            and item["semantic_deltas"]["raw_r2"]["strict_negative_count"] == 0
        ):
            preregistered_candidates.append(candidate)

    isotropic = comparisons["isotropic"]
    direct_coordinate_metrics_positive_all_subsets = all(
        isotropic["semantic_deltas"][metric]["strict_positive_count"]
        == len(SUBSET_SEEDS)
        for metric in ("raw_correlation", "rgb_mcc", "coordinatewise_r2")
    )
    return {
        "comparisons": comparisons,
        "decision": {
            "preregistered_core_support": bool(preregistered_candidates),
            "preregistered_candidates": preregistered_candidates,
            "isotropic_direct_coordinate_metrics_positive_all_subsets": (
                direct_coordinate_metrics_positive_all_subsets
            ),
            "isotropic_raw_r2_positive_count": isotropic["semantic_deltas"][
                "raw_r2"
            ]["strict_positive_count"],
            "isotropic_full_affine_r2_positive_count": isotropic[
                "semantic_deltas"
            ]["full_affine_r2"]["strict_positive_count"],
            "test_evaluated": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_config(config)

    results: dict[int, dict[str, Any]] = {}
    inputs: dict[str, Any] = {}
    for seed_text, item in config["inputs"].items():
        seed = int(seed_text)
        path = Path(item["path"]).resolve()
        digest = base.sha256_file(path)
        if digest != item["sha256"]:
            raise ValueError(f"K8 subset {seed} input hash mismatch")
        result = json.loads(path.read_text(encoding="utf-8"))
        if (
            result["protocol_version"] != INPUT_PROTOCOL
            or int(result["subset_seed"]) != seed
            or result["test_evaluated"] is not False
            or result["semantic_validation_evaluated"] is not True
            or set(result["training_locks"]) != {
                "full_anisotropic",
                "diagonal",
                "isotropic",
            }
        ):
            raise ValueError(f"invalid K8 subset {seed} result")
        results[seed] = result
        inputs[str(seed)] = {"path": str(path), "sha256": digest}

    output = {
        "protocol_version": PROTOCOL,
        "mode": "three_calibration_subset_validation_only_aggregate",
        "config_path": str(config_path),
        "config_sha256": base.sha256_file(config_path),
        "inputs": inputs,
        **aggregate(results),
        "test_evaluated": False,
    }
    output_path = Path(config["output_path"]).resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite K8 aggregate: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    v11.atomic_json(output_path, output)


if __name__ == "__main__":
    main()
