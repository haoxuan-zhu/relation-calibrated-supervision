"""Extract test-free G1 representation and teacher metrics for K12."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import run_diagnostic as base
import run_physics_functional_anchor_training as v11


PROTOCOL = "raw_ridge_validation_comparator_k12_v1"
BUDGETS = (20, 40, 80, 160)
SEEDS = (0, 42, 3407)


def extract(
    aggregate: dict[str, Any], audit: dict[str, Any], teachers: dict[int, dict[str, Any]]
) -> dict[str, Any]:
    if aggregate["protocol_version"] != "raw_ridge_raw_aggregate_v1":
        raise ValueError("unexpected G1 aggregate protocol")
    if audit["protocol_version"] != "raw_ridge_raw_locked_metric_audit_v1":
        raise ValueError("unexpected G1 locked audit protocol")
    if not bool(audit["all_runs_valid"]) or not bool(audit["all_source_hashes_match"]):
        raise ValueError("G1 locked audit is invalid")
    entries: dict[str, Any] = {}
    teacher_metrics: dict[str, Any] = {}
    for budget in BUDGETS:
        entries[str(budget)] = {}
        for seed in SEEDS:
            aggregate_run = aggregate["raw_ridge"]["entries"][str(budget)][str(seed)]
            if not bool(aggregate_run["valid"]):
                raise ValueError(f"invalid G1 run K{budget} seed{seed}")
            validation = audit["locked_metrics"][str(budget)][str(seed)]["validation"]
            entries[str(budget)][str(seed)] = {
                "mean_direct_abs_correlation": float(
                    validation["mean_direct_abs_correlation"]
                ),
                "mean_direct_r2": float(validation["mean_direct_r2"]),
                "direct_abs_correlation": validation["direct_abs_correlation"],
                "direct_r2": validation["direct_r2"],
                "parameter_count": int(aggregate_run["parameter_count"]),
            }
        teacher = teachers[budget]
        if (
            teacher["protocol_version"] != "raw_ridge_propagation_v1"
            or int(teacher["budget"]) != budget
            or teacher["test_evaluated"] is not False
            or not bool(teacher["decision"]["all_valid"])
        ):
            raise ValueError(f"invalid G1 teacher audit K{budget}")
        selected = teacher["teachers"]["empirical_fit"]["selected_alpha"]
        metrics = teacher["teachers"]["post_fit_validation"]["raw_ridge"]
        center = teacher["teachers"]["empirical_coefficients"]
        teacher_metrics[str(budget)] = {
            "selected_alpha": float(selected),
            "center_model": {
                "feature_mean": center["feature_mean"],
                "feature_scale": center["feature_scale"],
                "coefficients": center["coefficients"],
            },
            "mean_direct_correlation": float(metrics["mean_direct_correlation"]),
            "mean_direct_r2": float(metrics["mean_direct_r2"]),
            "hungarian_mcc": float(metrics["hungarian_mcc"]),
            "mae": float(metrics["mae"]),
        }
    return {
        "protocol_version": PROTOCOL,
        "budgets": list(BUDGETS),
        "seeds": list(SEEDS),
        "raw_representation": entries,
        "analytic_center": teacher_metrics,
        "copied_test_fields": False,
        "test_evaluated": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--locked-audit", type=Path, required=True)
    parser.add_argument("--teacher-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite K12 comparator: {output}")
    aggregate_path = args.aggregate.resolve()
    audit_path = args.locked_audit.resolve()
    teachers: dict[int, dict[str, Any]] = {}
    teacher_sources: dict[str, Any] = {}
    for budget in BUDGETS:
        path = (args.teacher_dir / f"k{budget}.json").resolve()
        teachers[budget] = json.loads(path.read_text(encoding="utf-8"))
        teacher_sources[str(budget)] = {
            "path": str(path),
            "sha256": base.sha256_file(path),
        }
    result = extract(
        json.loads(aggregate_path.read_text(encoding="utf-8")),
        json.loads(audit_path.read_text(encoding="utf-8")),
        teachers,
    )
    result["sources"] = {
        "aggregate": {
            "path": str(aggregate_path),
            "sha256": base.sha256_file(aggregate_path),
        },
        "locked_audit": {
            "path": str(audit_path),
            "sha256": base.sha256_file(audit_path),
        },
        "teachers": teacher_sources,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    v11.atomic_json(output, result)


if __name__ == "__main__":
    main()
