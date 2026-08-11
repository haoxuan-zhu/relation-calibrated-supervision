"""Paired validation graph audit of K11 against G1 raw-ridge propagation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

import aggregate_isotropic_relation_tube_budget_k11 as k11agg
import extract_raw_ridge_downstream_comparator_k12b as raw_extract
import run_diagnostic as base
import run_isotropic_relation_tube_budget_k11 as k11
import run_physics_functional_anchor_training as v11


PROTOCOL = "isotropic_tube_vs_raw_downstream_k12b_v1"
RAW_COMPARATOR_SHA256 = "00e31267b1ba36536ba05049aee6bcb35f89013000ca6558a8f39253cfeb6d34"


def summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "population_std": float(array.std(ddof=0)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def paired_audit(k11_result: dict[str, Any], raw: dict[str, Any]) -> dict[str, Any]:
    budgets: dict[str, Any] = {}
    all_edge: list[float] = []
    all_shd: list[float] = []
    all_ccrl: list[float] = []
    joint_count = 0
    for budget in k11.BUDGETS:
        rows: dict[str, Any] = {}
        for seed in k11.SEEDS:
            tube = k11_result["entries"][str(budget)][str(seed)]["metrics"]
            baseline = raw["entries"][str(budget)][str(seed)]
            delta = {
                "edge_auroc": float(tube["edge_auroc"] - baseline["edge_auroc"]),
                "shd": float(tube["shd"] - baseline["fixed_threshold_shd"]),
                "ccrl_total": float(
                    tube["ccrl_total"] - baseline["ccrl_validation_total"]
                ),
            }
            joint = delta["edge_auroc"] > 0.0 and delta["shd"] <= 0.0
            joint_count += int(joint)
            all_edge.append(delta["edge_auroc"])
            all_shd.append(delta["shd"])
            all_ccrl.append(delta["ccrl_total"])
            rows[str(seed)] = {
                "tube": {
                    "edge_auroc": float(tube["edge_auroc"]),
                    "shd": int(tube["shd"]),
                    "ccrl_total": float(tube["ccrl_total"]),
                },
                "raw_representation": {
                    "edge_auroc": float(baseline["edge_auroc"]),
                    "shd": int(baseline["fixed_threshold_shd"]),
                    "ccrl_total": float(baseline["ccrl_validation_total"]),
                },
                "tube_minus_raw_representation": delta,
                "joint_graph_improvement": joint,
            }
        edges = [rows[str(seed)]["tube_minus_raw_representation"]["edge_auroc"] for seed in k11.SEEDS]
        shds = [rows[str(seed)]["tube_minus_raw_representation"]["shd"] for seed in k11.SEEDS]
        ccrls = [rows[str(seed)]["tube_minus_raw_representation"]["ccrl_total"] for seed in k11.SEEDS]
        budgets[str(budget)] = {
            "seeds": rows,
            "summary": {
                "edge_auroc_delta": {
                    **summary(edges),
                    "strict_positive_count": sum(value > 0.0 for value in edges),
                },
                "shd_delta": {
                    **summary(shds),
                    "strict_negative_count": sum(value < 0.0 for value in shds),
                    "nonpositive_count": sum(value <= 0.0 for value in shds),
                },
                "ccrl_total_delta": summary(ccrls),
                "joint_graph_improvement_count": sum(
                    rows[str(seed)]["joint_graph_improvement"] for seed in k11.SEEDS
                ),
            },
        }
    edge_positive = sum(value > 0.0 for value in all_edge)
    shd_nonpositive = sum(value <= 0.0 for value in all_shd)
    if joint_count == 12:
        verdict = "joint_semantic_and_graph_advantage_all_pairs"
    elif edge_positive >= 9 and shd_nonpositive >= 9:
        verdict = "semantic_advantage_with_most_graph_pairs_supportive"
    else:
        verdict = "semantic_advantage_with_mixed_graph_outcomes"
    return {
        "budgets": budgets,
        "overall": {
            "edge_auroc_delta": summary(all_edge),
            "edge_auroc_strict_positive_count": edge_positive,
            "shd_delta": summary(all_shd),
            "shd_strict_negative_count": sum(value < 0.0 for value in all_shd),
            "shd_nonpositive_count": shd_nonpositive,
            "joint_graph_improvement_count": joint_count,
            "ccrl_total_delta": summary(all_ccrl),
        },
        "decision": {
            "verdict": verdict,
            "semantic_k12_reopened": False,
            "ccrl_used_as_success_gate": False,
            "geometry_selection_reopened": False,
            "test_evaluated": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k11-aggregate", type=Path, required=True)
    parser.add_argument("--raw-comparator", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite K12b audit: {output}")
    k11_path = args.k11_aggregate.resolve()
    raw_path = args.raw_comparator.resolve()
    if base.sha256_file(raw_path) != RAW_COMPARATOR_SHA256:
        raise ValueError("K12b raw comparator hash mismatch")
    k11_result = json.loads(k11_path.read_text(encoding="utf-8"))
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    if (
        k11_result["protocol_version"] != k11agg.PROTOCOL
        or k11_result["test_evaluated"] is not False
    ):
        raise ValueError("invalid K11 aggregate")
    if (
        raw["protocol_version"] != raw_extract.PROTOCOL
        or raw["copied_test_fields"] is not False
        or raw["new_test_evaluation"] is not False
        or tuple(raw["budgets"]) != k11.BUDGETS
        or tuple(raw["seeds"]) != k11.SEEDS
    ):
        raise ValueError("invalid K12b raw comparator")
    result = {
        "protocol_version": PROTOCOL,
        "mode": "paired_four_budget_three_seed_validation_downstream_audit",
        "inputs": {
            "k11_aggregate": {
                "path": str(k11_path),
                "sha256": base.sha256_file(k11_path),
            },
            "raw_comparator": {
                "path": str(raw_path),
                "sha256": base.sha256_file(raw_path),
            },
        },
        "source_files_sha256": {
            "audit_isotropic_tube_vs_raw_downstream_k12b.py": base.sha256_file(
                Path(__file__).resolve()
            ),
            "extract_raw_ridge_downstream_comparator_k12b.py": base.sha256_file(
                Path(raw_extract.__file__).resolve()
            ),
        },
        **paired_audit(k11_result, raw),
        "semantic_validation_evaluated": True,
        "new_test_evaluation": False,
        "test_evaluated": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    v11.atomic_json(output, result)


if __name__ == "__main__":
    main()
