"""Aggregate the crossed calibration-subset by model-seed geometry audit."""

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


PROTOCOL = "relation_tube_geometry_crossed_k18_aggregate_v1"
NEW_PROTOCOL = "relation_tube_geometry_crossed_k18_v1"
OLD_PROTOCOL = "relation_tube_calibration_subset_k8_v1"
SUBSET_SEEDS = (20260811, 20260821, 20260831)
MODEL_SEEDS = (0, 42, 3407)
SEMANTIC_METRICS = (
    "raw_correlation",
    "raw_r2",
    "rgb_mcc",
    "coordinatewise_r2",
    "full_affine_r2",
)


def validate_config(config: dict[str, Any]) -> None:
    if config["protocol_version"] != PROTOCOL:
        raise ValueError("unexpected K18 aggregate protocol")
    if config["test_evaluated"] is not False:
        raise ValueError("K18 aggregate cannot evaluate test")
    expected = {
        f"{subset_seed}:{model_seed}"
        for subset_seed in SUBSET_SEEDS
        for model_seed in MODEL_SEEDS
    }
    if set(config["inputs"]) != expected:
        raise ValueError("K18 crossed input registry changed")


def summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "sample_std": float(array.std(ddof=1)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "strict_positive_count": int(np.sum(array > 0.0)),
        "strict_negative_count": int(np.sum(array < 0.0)),
    }


def cell_comparison(result: dict[str, Any]) -> dict[str, Any]:
    if result["protocol_version"] == NEW_PROTOCOL:
        return result["isotropic_minus_diagonal"]
    if result["protocol_version"] == OLD_PROTOCOL:
        return k8.comparison(result["runs"]["diagonal"], result["runs"]["isotropic"])
    raise ValueError("unexpected crossed-cell protocol")


def aggregate(results: dict[tuple[int, int], dict[str, Any]]) -> dict[str, Any]:
    cells = {
        f"{subset_seed}:{model_seed}": cell_comparison(
            results[(subset_seed, model_seed)]
        )
        for subset_seed in SUBSET_SEEDS
        for model_seed in MODEL_SEEDS
    }
    semantic = {
        metric: summary(
            [
                float(cell["semantic_deltas"][metric])
                for cell in cells.values()
            ]
        )
        for metric in SEMANTIC_METRICS
    }
    by_subset: dict[str, Any] = {}
    for subset_seed in SUBSET_SEEDS:
        selected = [cells[f"{subset_seed}:{seed}"] for seed in MODEL_SEEDS]
        by_subset[str(subset_seed)] = {
            metric: summary(
                [float(cell["semantic_deltas"][metric]) for cell in selected]
            )
            for metric in SEMANTIC_METRICS
        }
    by_model_seed: dict[str, Any] = {}
    for model_seed in MODEL_SEEDS:
        selected = [cells[f"{subset}:{model_seed}"] for subset in SUBSET_SEEDS]
        by_model_seed[str(model_seed)] = {
            metric: summary(
                [float(cell["semantic_deltas"][metric]) for cell in selected]
            )
            for metric in SEMANTIC_METRICS
        }

    primary = ("raw_correlation", "coordinatewise_r2")
    subset_means_positive = all(
        by_subset[str(subset)][metric]["mean"] > 0.0
        for subset in SUBSET_SEEDS
        for metric in primary
    )
    strict = (
        subset_means_positive
        and all(semantic[metric]["strict_positive_count"] >= 8 for metric in primary)
    )
    directional = subset_means_positive and all(
        semantic[metric]["mean"] > 0.0
        and semantic[metric]["strict_positive_count"] >= 6
        for metric in primary
    )
    if strict:
        verdict = "isotropic_crossed_robustness_supported"
    elif directional:
        verdict = "isotropic_crossed_directional_support_mixed"
    else:
        verdict = "axis_separable_tradeoff_no_isotropic_separation"
    return {
        "cells": cells,
        "semantic_deltas": semantic,
        "by_subset": by_subset,
        "by_model_seed": by_model_seed,
        "auxiliary": {
            "edge_auroc_delta": summary(
                [float(cell["edge_auroc_delta"]) for cell in cells.values()]
            ),
            "full_minus_isotropic_shd": summary(
                [float(cell["full_minus_candidate_shd"]) for cell in cells.values()]
            ),
            "isotropic_minus_diagonal_ccrl_total": summary(
                [
                    float(cell["candidate_minus_full_ccrl_total"])
                    for cell in cells.values()
                ]
            ),
        },
        "decision": {
            "verdict": verdict,
            "subset_primary_means_positive": subset_means_positive,
            "raw_correlation_positive_count": semantic["raw_correlation"][
                "strict_positive_count"
            ],
            "coordinatewise_r2_positive_count": semantic["coordinatewise_r2"][
                "strict_positive_count"
            ],
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
    results: dict[tuple[int, int], dict[str, Any]] = {}
    inputs: dict[str, Any] = {}
    for key, item in config["inputs"].items():
        subset_seed, model_seed = (int(value) for value in key.split(":"))
        path = Path(item["path"]).resolve()
        digest = base.sha256_file(path)
        if digest != item["sha256"]:
            raise ValueError(f"K18 input hash mismatch: {key}")
        result = json.loads(path.read_text(encoding="utf-8"))
        if (
            int(result["subset_seed"]) != subset_seed
            or result["test_evaluated"] is not False
            or result["semantic_validation_evaluated"] is not True
        ):
            raise ValueError(f"invalid K18 cell: {key}")
        if model_seed == 3407:
            if result["protocol_version"] != OLD_PROTOCOL:
                raise ValueError(f"invalid locked K8 cell: {key}")
        elif (
            result["protocol_version"] != NEW_PROTOCOL
            or int(result["model_seed"]) != model_seed
        ):
            raise ValueError(f"invalid new K18 cell: {key}")
        results[(subset_seed, model_seed)] = result
        inputs[key] = {"path": str(path), "sha256": digest}

    output = {
        "protocol_version": PROTOCOL,
        "mode": "three_subset_three_model_seed_diagonal_isotropic_aggregate",
        "config_path": str(config_path),
        "config_sha256": base.sha256_file(config_path),
        "inputs": inputs,
        **aggregate(results),
        "test_evaluated": False,
    }
    output_path = Path(config["output_path"]).resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite K18 aggregate: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    v11.atomic_json(output_path, output)


if __name__ == "__main__":
    main()
