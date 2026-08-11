"""Aggregate Tube, anchor-preserving, and ordinary propagation readouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

import run_diagnostic as base
import run_physics_functional_anchor_training as v11


PROTOCOL = "anchor_preserving_comparison_v1"
BUDGETS = (20, 40, 80, 160)
SEEDS = (0, 42, 3407)


def summary(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "population_sd": float(array.std(ddof=0)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "strict_positive_count": int(np.sum(array > 0.0)),
        "strict_negative_count": int(np.sum(array < 0.0)),
    }


def build_comparison(
    anchor: dict[str, Any], raw: dict[str, Any], tube: dict[str, Any]
) -> dict[str, Any]:
    entries: dict[str, Any] = {}
    pooled: dict[str, list[float]] = {
        "anchor_minus_raw_correlation": [],
        "anchor_minus_raw_r2": [],
        "tube_minus_anchor_correlation": [],
        "tube_minus_anchor_r2": [],
    }
    candidate_inputs = tube["inputs"]["candidate_readouts"]
    for budget in BUDGETS:
        key = str(budget)
        entries[key] = {}
        for seed in SEEDS:
            seed_key = str(seed)
            raw_item = raw["locked_metrics"][key][seed_key]["test"]
            anchor_item = anchor["locked_metrics"][key][seed_key]["test"]
            candidate_path = Path(candidate_inputs[key][seed_key]["path"])
            if not candidate_path.exists():
                raise FileNotFoundError(candidate_path)
            if base.sha256_file(candidate_path) != candidate_inputs[key][seed_key]["sha256"]:
                raise ValueError("Tube readout hash mismatch")
            candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
            tube_item = candidate["semantic"]["raw"]
            deltas = {
                "anchor_minus_raw_correlation": float(
                    anchor_item["mean_direct_abs_correlation"]
                    - raw_item["mean_direct_abs_correlation"]
                ),
                "anchor_minus_raw_r2": float(
                    anchor_item["mean_direct_r2"] - raw_item["mean_direct_r2"]
                ),
                "tube_minus_anchor_correlation": float(
                    tube_item["mean_direct_abs_correlation"]
                    - anchor_item["mean_direct_abs_correlation"]
                ),
                "tube_minus_anchor_r2": float(
                    tube_item["mean_r2"] - anchor_item["mean_direct_r2"]
                ),
            }
            entries[key][seed_key] = {
                "raw": {
                    "correlation": raw_item["mean_direct_abs_correlation"],
                    "r2": raw_item["mean_direct_r2"],
                },
                "anchor_preserving": {
                    "correlation": anchor_item["mean_direct_abs_correlation"],
                    "r2": anchor_item["mean_direct_r2"],
                },
                "tube": {
                    "correlation": tube_item["mean_direct_abs_correlation"],
                    "r2": tube_item["mean_r2"],
                },
                "deltas": deltas,
            }
            for name, value in deltas.items():
                pooled[name].append(value)
    return {
        "entries": entries,
        "overall": {name: summary(values) for name, values in pooled.items()},
        "performance_threshold_used": False,
        "tube_selection_reopened": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor-metrics", type=Path, required=True)
    parser.add_argument("--raw-metrics", type=Path, required=True)
    parser.add_argument("--tube-aggregate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = {
        name: path.resolve()
        for name, path in (
            ("anchor_metrics", args.anchor_metrics),
            ("raw_metrics", args.raw_metrics),
            ("tube_aggregate", args.tube_aggregate),
        )
    }
    values = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in inputs.items()
    }
    if values["anchor_metrics"]["protocol_version"] != "anchor_preserving_locked_metrics_v1":
        raise ValueError("unexpected anchor metric protocol")
    if values["anchor_metrics"]["all_runs_valid"] is not True:
        raise ValueError("anchor metric audit is invalid")
    if values["raw_metrics"]["protocol_version"] != "raw_ridge_raw_locked_metric_audit_v1":
        raise ValueError("unexpected raw metric protocol")
    if values["raw_metrics"]["all_runs_valid"] is not True:
        raise ValueError("raw metric audit is invalid")
    if values["tube_aggregate"]["protocol_version"] != "isotropic_relation_tube_heldout_k13_aggregate_v1":
        raise ValueError("unexpected Tube held-out protocol")

    result = {
        "protocol_version": PROTOCOL,
        "fact_type": "post_hoc_comparator_audit",
        "inputs": {
            name: {"path": str(path), "sha256": base.sha256_file(path)}
            for name, path in inputs.items()
        },
        "comparison": build_comparison(
            values["anchor_metrics"],
            values["raw_metrics"],
            values["tube_aggregate"],
        ),
        "training_performed": False,
        "test_evaluated": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    v11.atomic_json(args.output.resolve(), result)
    print(json.dumps({"output": str(args.output.resolve()), "sha256": base.sha256_file(args.output.resolve())}, sort_keys=True))


if __name__ == "__main__":
    main()
