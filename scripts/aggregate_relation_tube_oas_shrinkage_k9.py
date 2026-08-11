"""Aggregate the three exact-OAS K9 validation audits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

import run_diagnostic as base
import run_physics_functional_anchor_training as v11
import run_relation_tube_calibration_subset_k8 as k8
import run_relation_tube_oas_shrinkage_k9 as k9


PROTOCOL = "relation_tube_oas_shrinkage_k9_aggregate_v1"
COMPARISONS = ("oas_minus_full", "oas_minus_isotropic")
METRICS = (
    "raw_correlation",
    "raw_r2",
    "rgb_mcc",
    "coordinatewise_r2",
    "full_affine_r2",
)


def validate_config(config: dict[str, Any]) -> None:
    if config["protocol_version"] != PROTOCOL:
        raise ValueError("unexpected K9 aggregate protocol")
    if tuple(int(seed) for seed in config["inputs"].keys()) != k8.SUBSET_SEEDS:
        raise ValueError("K9 aggregate subset registry changed")
    if config["test_evaluated"] is not False:
        raise ValueError("K9 aggregate cannot evaluate test")


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
    for comparison in COMPARISONS:
        rows = {
            str(seed): results[seed]["comparisons"][comparison]
            for seed in k8.SUBSET_SEEDS
        }
        metric_summary = {}
        for metric in METRICS:
            values = [
                float(rows[str(seed)]["semantic_deltas"][metric])
                for seed in k8.SUBSET_SEEDS
            ]
            metric_summary[metric] = {
                **summary(values),
                "strict_positive_count": sum(value > 0.0 for value in values),
                "strict_negative_count": sum(value < 0.0 for value in values),
            }
        comparisons[comparison] = {
            "subsets": rows,
            "semantic_deltas": metric_summary,
            "complete_semantic_advantage_count": sum(
                bool(row["complete_semantic_advantage"]) for row in rows.values()
            ),
            "graph_advantage_count": sum(
                bool(row["graph_advantage"]) for row in rows.values()
            ),
        }

    shrinkages = [
        float(results[seed]["geometry_audit"]["shrinkage"])
        for seed in k8.SUBSET_SEEDS
    ]
    oas_full = comparisons["oas_minus_full"]
    oas_isotropic = comparisons["oas_minus_isotropic"]
    return {
        "shrinkage": summary(shrinkages),
        "comparisons": comparisons,
        "decision": {
            "oas_complete_semantic_advantage_over_full_all_subsets": (
                oas_full["complete_semantic_advantage_count"]
                == len(k8.SUBSET_SEEDS)
            ),
            "oas_complete_semantic_advantage_over_isotropic_any_subset": (
                oas_isotropic["complete_semantic_advantage_count"] > 0
            ),
            "isotropic_raw_correlation_above_oas_all_subsets": (
                oas_isotropic["semantic_deltas"]["raw_correlation"][
                    "strict_negative_count"
                ]
                == len(k8.SUBSET_SEEDS)
            ),
            "isotropic_raw_r2_above_oas_all_subsets": (
                oas_isotropic["semantic_deltas"]["raw_r2"][
                    "strict_negative_count"
                ]
                == len(k8.SUBSET_SEEDS)
            ),
            "interpretation": "automatic_covariance_shrinkage_does_not_replace_axis_prior",
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
            raise ValueError(f"K9 subset {seed} input hash mismatch")
        result = json.loads(path.read_text(encoding="utf-8"))
        if (
            result["protocol_version"] != k9.PROTOCOL
            or int(result["subset_seed"]) != seed
            or result["semantic_validation_evaluated"] is not True
            or result["test_evaluated"] is not False
            or result["geometry_audit"]["formula"] != "chen_2010_exact_eq23"
        ):
            raise ValueError(f"invalid K9 subset {seed} result")
        results[seed] = result
        inputs[str(seed)] = {"path": str(path), "sha256": digest}
    output = {
        "protocol_version": PROTOCOL,
        "mode": "three_calibration_subset_exact_oas_aggregate",
        "config_path": str(config_path),
        "config_sha256": base.sha256_file(config_path),
        "inputs": inputs,
        **aggregate(results),
        "test_evaluated": False,
    }
    output_path = Path(config["output_path"]).resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite K9 aggregate: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    v11.atomic_json(output_path, output)


if __name__ == "__main__":
    main()
