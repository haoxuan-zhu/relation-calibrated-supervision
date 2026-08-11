"""Recompute anchor-preserving metrics from stored per-unit predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = target - prediction
    denominator = np.sum((target - target.mean()) ** 2)
    return {
        "r2": float(1.0 - np.sum(error**2) / denominator),
        "correlation": float(np.corrcoef(target, prediction)[0, 1]),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "normalized_rmse": float(np.sqrt(np.mean(error**2)) / target.std()),
        "mae": float(np.mean(np.abs(error))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = json.loads(args.result.read_text(encoding="utf-8"))
    max_metric_error = 0.0
    max_delta_error = 0.0
    checked_runs = 0
    for evaluation in result["evaluations"].values():
        for run in evaluation["runs"]:
            stored = run["evaluation_predictions"]
            target = np.asarray(stored["target"], dtype=np.float64)
            predictions = {
                name: np.asarray(values, dtype=np.float64)
                for name, values in stored.items()
                if name not in {"sample_ids", "target"}
            }
            for name, prediction in predictions.items():
                observed = metrics(target, prediction)
                for metric, value in observed.items():
                    max_metric_error = max(
                        max_metric_error, abs(value - run["conditions"][name][metric])
                    )
            pairs = {
                "anchor_preserving_minus_point": ("anchor_preserving", "point"),
                "anchor_preserving_minus_relation_propagation": (
                    "anchor_preserving",
                    "relation_propagation",
                ),
                "anchor_preserving_minus_anchor_preserving_permuted": (
                    "anchor_preserving",
                    "anchor_preserving_permuted",
                ),
            }
            for name, (candidate, baseline) in pairs.items():
                delta = metrics(target, predictions[candidate])["r2"] - metrics(
                    target, predictions[baseline]
                )["r2"]
                max_delta_error = max(
                    max_delta_error, abs(delta - run["comparisons"][name]["observed"])
                )
            checked_runs += 1
    audit = {
        "protocol": "resistance_spot_welding_anchor_preserving_independent_audit_v1",
        "source": str(args.result),
        "checked_runs": checked_runs,
        "max_metric_absolute_error": max_metric_error,
        "max_delta_absolute_error": max_delta_error,
        "passed": max_metric_error < 1.0e-12 and max_delta_error < 1.0e-12,
    }
    args.output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit))


if __name__ == "__main__":
    main()
