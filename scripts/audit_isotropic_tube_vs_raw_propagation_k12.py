"""Paired validation audit of K11 against G1 raw-ridge propagation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

import aggregate_isotropic_relation_tube_budget_k11 as k11agg
import extract_raw_ridge_validation_comparator_k12 as raw_extract
import run_diagnostic as base
import run_isotropic_relation_tube_budget_k11 as k11
import run_physics_functional_anchor_training as v11


PROTOCOL = "isotropic_tube_vs_raw_propagation_k12_v1"
RAW_COMPARATOR_SHA256 = "a004b234cac0bf0512be6db24eba48046b53fcb0f1f706a5a7e58d7e361e501c"
K80_PREFLIGHT_SHA256 = "19fedff09f232f5077dfbd8c1ee5b6eebe2852ff241848715dcdffa157a67221"
METRICS = ("raw_correlation", "raw_r2")
EXPECTED_PARAMETER_COUNT = 16902384


def summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "population_std": float(array.std(ddof=0)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def center_model(preflight: dict[str, Any]) -> dict[str, Any]:
    return preflight["teachers"]["correct"]["model"]


def center_difference(left: dict[str, Any], right: dict[str, Any]) -> float:
    values = []
    for key in ("feature_mean", "feature_scale", "coefficients"):
        values.append(
            float(
                np.max(
                    np.abs(
                        np.asarray(left[key], dtype=np.float64)
                        - np.asarray(right[key], dtype=np.float64)
                    )
                )
            )
        )
    return max(values)


def validate_raw_comparator(raw: dict[str, Any]) -> None:
    if tuple(int(value) for value in raw["budgets"]) != k11.BUDGETS:
        raise ValueError("K12 raw comparator budget registry mismatch")
    if tuple(int(value) for value in raw["seeds"]) != k11.SEEDS:
        raise ValueError("K12 raw comparator seed registry mismatch")
    for budget in k11.BUDGETS:
        teacher = raw["analytic_center"][str(budget)]
        if not np.isclose(float(teacher["selected_alpha"]), 0.1):
            raise ValueError(f"K{budget} raw center ridge mismatch")
        teacher_metrics = np.asarray(
            [
                teacher["mean_direct_correlation"],
                teacher["mean_direct_r2"],
                teacher["hungarian_mcc"],
                teacher["mae"],
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(teacher_metrics)):
            raise ValueError(f"K{budget} raw center has non-finite metrics")
        for seed in k11.SEEDS:
            item = raw["raw_representation"][str(budget)][str(seed)]
            if int(item["parameter_count"]) != EXPECTED_PARAMETER_COUNT:
                raise ValueError(
                    f"K{budget} seed{seed} raw parameter-count mismatch"
                )
            metrics = np.asarray(
                [
                    *item["direct_abs_correlation"],
                    *item["direct_r2"],
                    item["mean_direct_abs_correlation"],
                    item["mean_direct_r2"],
                ],
                dtype=np.float64,
            )
            if metrics.shape != (8,) or not np.all(np.isfinite(metrics)):
                raise ValueError(f"K{budget} seed{seed} raw metrics invalid")


def paired_audit(
    k11_result: dict[str, Any], raw: dict[str, Any]
) -> dict[str, Any]:
    budgets: dict[str, Any] = {}
    all_corr_deltas: list[float] = []
    all_r2_deltas: list[float] = []
    for budget in k11.BUDGETS:
        rows: dict[str, Any] = {}
        teacher = raw["analytic_center"][str(budget)]
        for seed in k11.SEEDS:
            tube_metrics = k11_result["entries"][str(budget)][str(seed)]["metrics"]
            raw_metrics = raw["raw_representation"][str(budget)][str(seed)]
            deltas = {
                "raw_correlation": float(
                    tube_metrics["raw_correlation"]
                    - raw_metrics["mean_direct_abs_correlation"]
                ),
                "raw_r2": float(
                    tube_metrics["raw_r2"] - raw_metrics["mean_direct_r2"]
                ),
            }
            gap_closed: dict[str, float | None] = {}
            for metric, teacher_key, raw_key in (
                (
                    "raw_correlation",
                    "mean_direct_correlation",
                    "mean_direct_abs_correlation",
                ),
                ("raw_r2", "mean_direct_r2", "mean_direct_r2"),
            ):
                denominator = float(teacher[teacher_key] - raw_metrics[raw_key])
                gap_closed[metric] = (
                    float(deltas[metric] / denominator)
                    if denominator > 0.0
                    else None
                )
            rows[str(seed)] = {
                "tube": {
                    "raw_correlation": float(tube_metrics["raw_correlation"]),
                    "raw_r2": float(tube_metrics["raw_r2"]),
                },
                "raw_representation": {
                    "raw_correlation": float(
                        raw_metrics["mean_direct_abs_correlation"]
                    ),
                    "raw_r2": float(raw_metrics["mean_direct_r2"]),
                },
                "analytic_center": {
                    "raw_correlation": float(teacher["mean_direct_correlation"]),
                    "raw_r2": float(teacher["mean_direct_r2"]),
                },
                "tube_minus_raw_representation": deltas,
                "analytic_gap_fraction_closed": gap_closed,
            }
            all_corr_deltas.append(deltas["raw_correlation"])
            all_r2_deltas.append(deltas["raw_r2"])
        metric_summaries: dict[str, Any] = {}
        for metric in METRICS:
            values = [
                rows[str(seed)]["tube_minus_raw_representation"][metric]
                for seed in k11.SEEDS
            ]
            fractions = [
                rows[str(seed)]["analytic_gap_fraction_closed"][metric]
                for seed in k11.SEEDS
                if rows[str(seed)]["analytic_gap_fraction_closed"][metric] is not None
            ]
            metric_summaries[metric] = {
                "delta": summary(values),
                "strict_positive_count": sum(value > 0.0 for value in values),
                "strict_negative_count": sum(value < 0.0 for value in values),
                "analytic_gap_fraction_closed": summary(fractions) if fractions else None,
            }
        budgets[str(budget)] = {
            "seeds": rows,
            "summary": metric_summaries,
        }
    corr_positive = sum(value > 0.0 for value in all_corr_deltas)
    r2_positive = sum(value > 0.0 for value in all_r2_deltas)
    if corr_positive == 12 and r2_positive == 12:
        verdict = "tube_outperforms_raw_propagation_all_budget_seed_both_metrics"
    elif corr_positive >= 9 and r2_positive >= 9:
        verdict = "tube_advantage_most_pairs_with_exceptions"
    else:
        verdict = "tube_vs_raw_propagation_mixed"
    return {
        "budgets": budgets,
        "overall": {
            "raw_correlation_delta": summary(all_corr_deltas),
            "raw_r2_delta": summary(all_r2_deltas),
            "raw_correlation_strict_positive_count": corr_positive,
            "raw_r2_strict_positive_count": r2_positive,
        },
        "decision": {
            "verdict": verdict,
            "threshold_beyond_direction_used": False,
            "geometry_selection_reopened": False,
            "test_evaluated": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k11-aggregate", type=Path, required=True)
    parser.add_argument("--raw-comparator", type=Path, required=True)
    parser.add_argument("--k11-output-root", type=Path, required=True)
    parser.add_argument("--k80-preflight", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite K12 audit: {output_path}")
    k11_path = args.k11_aggregate.resolve()
    raw_path = args.raw_comparator.resolve()
    k80_path = args.k80_preflight.resolve()
    if base.sha256_file(raw_path) != RAW_COMPARATOR_SHA256:
        raise ValueError("K12 raw comparator hash mismatch")
    if base.sha256_file(k80_path) != K80_PREFLIGHT_SHA256:
        raise ValueError("K12 K80 preflight hash mismatch")
    k11_result = json.loads(k11_path.read_text(encoding="utf-8"))
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    if (
        k11_result["protocol_version"] != k11agg.PROTOCOL
        or k11_result["test_evaluated"] is not False
        or k11_result["decision"]["geometry_selection_reopened"] is not False
    ):
        raise ValueError("invalid K11 aggregate")
    if (
        raw["protocol_version"] != raw_extract.PROTOCOL
        or raw["copied_test_fields"] is not False
        or raw["test_evaluated"] is not False
    ):
        raise ValueError("invalid K12 raw comparator")
    validate_raw_comparator(raw)
    preflight_inputs: dict[str, Any] = {}
    center_differences: dict[str, float] = {}
    for budget in k11.BUDGETS:
        path = (
            k80_path
            if budget == 80
            else args.k11_output_root.resolve()
            / f"k{budget}"
            / "preflight"
            / "isotropic_preflight.json"
        )
        digest = base.sha256_file(path)
        preflight = json.loads(path.read_text(encoding="utf-8"))
        if preflight["test_evaluated"] is not False:
            raise ValueError(f"K{budget} preflight read test")
        difference = center_difference(
            center_model(preflight),
            raw["analytic_center"][str(budget)]["center_model"],
        )
        if difference > 5e-8:
            raise ValueError(f"K{budget} relation center mismatch: {difference}")
        center_differences[str(budget)] = difference
        preflight_inputs[str(budget)] = {"path": str(path), "sha256": digest}
    result = {
        "protocol_version": PROTOCOL,
        "mode": "paired_four_budget_three_seed_validation_only_audit",
        "source_files_sha256": {
            "audit_isotropic_tube_vs_raw_propagation_k12.py": base.sha256_file(
                Path(__file__).resolve()
            ),
            "extract_raw_ridge_validation_comparator_k12.py": base.sha256_file(
                Path(raw_extract.__file__).resolve()
            ),
            "aggregate_isotropic_relation_tube_budget_k11.py": base.sha256_file(
                Path(k11agg.__file__).resolve()
            ),
            "run_isotropic_relation_tube_budget_k11.py": base.sha256_file(
                Path(k11.__file__).resolve()
            ),
        },
        "inputs": {
            "k11_aggregate": {
                "path": str(k11_path),
                "sha256": base.sha256_file(k11_path),
            },
            "raw_comparator": {
                "path": str(raw_path),
                "sha256": base.sha256_file(raw_path),
            },
            "preflights": preflight_inputs,
        },
        "center_identity": {
            "maximum_absolute_difference_by_budget": center_differences,
            "tolerance": 5e-8,
            "all_within_tolerance": True,
        },
        **paired_audit(k11_result, raw),
        "semantic_validation_evaluated": True,
        "test_evaluated": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    v11.atomic_json(output_path, result)


if __name__ == "__main__":
    main()
