"""Aggregate the one-time K13 held-out readouts without model selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

import extract_heldout_baseline_comparator_k13 as baseline_extract
import run_diagnostic as base
import run_isotropic_relation_tube_budget_k11 as k11
import run_isotropic_relation_tube_heldout_k13 as k13
import run_physics_functional_anchor_training as v11


PROTOCOL = "isotropic_relation_tube_heldout_k13_aggregate_v1"
METRICS = (
    "raw_correlation",
    "raw_r2",
    "rgb_mcc",
    "coordinatewise_r2",
    "full_affine_r2",
)


def summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "population_std": float(array.std(ddof=0)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def normalize_semantic(semantic: dict[str, Any]) -> dict[str, float]:
    return {
        "raw_correlation": float(semantic["raw"]["mean_direct_abs_correlation"]),
        "raw_r2": float(semantic["raw"]["mean_r2"]),
        "rgb_mcc": float(semantic["rgb_hungarian_mcc"]),
        "coordinatewise_r2": float(
            semantic["coordinatewise_affine"]["mean_r2"]
        ),
        "full_affine_r2": float(semantic["full_affine"]["mean_r2"]),
    }


def aggregate(
    entries: dict[int, dict[int, dict[str, Any]]],
    validation: dict[str, Any],
    baselines: dict[str, Any],
) -> dict[str, Any]:
    budgets: dict[str, Any] = {}
    overall_deltas = {condition: [] for condition in baseline_extract.CONDITIONS}
    for budget in k11.BUDGETS:
        rows: dict[str, Any] = {}
        for seed in k11.SEEDS:
            test_metrics = entries[budget][seed]["metrics"]
            validation_metrics = validation["entries"][str(budget)][str(seed)][
                "metrics"
            ]
            gaps = {
                metric: float(test_metrics[metric] - validation_metrics[metric])
                for metric in METRICS
            }
            deltas = {}
            for condition in baseline_extract.CONDITIONS:
                reference = baselines["entries"][str(budget)][condition][str(seed)][
                    "mean_direct_abs_rgb_correlation"
                ]
                delta = float(test_metrics["raw_correlation"] - reference)
                deltas[condition] = delta
                overall_deltas[condition].append(delta)
            rows[str(seed)] = {
                "test": test_metrics,
                "validation": {
                    metric: float(validation_metrics[metric]) for metric in METRICS
                },
                "test_minus_validation": gaps,
                "test_correlation_minus_baseline": deltas,
            }
        budget_metrics = {
            metric: summary([rows[str(seed)]["test"][metric] for seed in k11.SEEDS])
            for metric in METRICS
        }
        gap_summaries = {
            metric: summary(
                [rows[str(seed)]["test_minus_validation"][metric] for seed in k11.SEEDS]
            )
            for metric in METRICS
        }
        comparisons = {}
        for condition in baseline_extract.CONDITIONS:
            values = [
                rows[str(seed)]["test_correlation_minus_baseline"][condition]
                for seed in k11.SEEDS
            ]
            comparisons[condition] = {
                **summary(values),
                "strict_positive_count": sum(value > 0.0 for value in values),
                "strict_negative_count": sum(value < 0.0 for value in values),
                "by_seed": {
                    str(seed): value for seed, value in zip(k11.SEEDS, values)
                },
            }
        budgets[str(budget)] = {
            "seeds": rows,
            "test_metrics": budget_metrics,
            "test_minus_validation": gap_summaries,
            "comparisons": comparisons,
        }
    raw_positive = sum(value > 0.0 for value in overall_deltas["raw_ridge"])
    if raw_positive == 12:
        verdict = "heldout_tube_advantage_over_raw_propagation_all_pairs"
    elif raw_positive >= 9:
        verdict = "heldout_tube_advantage_over_raw_propagation_most_pairs"
    else:
        verdict = "heldout_tube_vs_raw_propagation_mixed"
    return {
        "budgets": budgets,
        "overall_comparisons": {
            condition: {
                **summary(values),
                "strict_positive_count": sum(value > 0.0 for value in values),
                "strict_negative_count": sum(value < 0.0 for value in values),
            }
            for condition, values in overall_deltas.items()
        },
        "decision": {
            "verdict": verdict,
            "raw_ridge_strict_positive_count": raw_positive,
            "test_guided_selection_performed": False,
            "geometry_selection_reopened": False,
            "test_evaluated": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    config_path = args.config.resolve()
    master = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    k13.validate_master(master)
    root = Path(args.output_root or master["runtime"]["output_root"]).resolve()
    validation_path = Path(master["k11"]["validation_aggregate_path"]).resolve()
    baseline_path = Path(master["baseline_comparator"]["path"]).resolve()
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    baselines = json.loads(baseline_path.read_text(encoding="utf-8"))
    if (
        baselines["protocol_version"] != baseline_extract.PROTOCOL
        or baselines["candidate_test_evaluated"] is not False
        or base.sha256_file(baseline_path) != master["baseline_comparator"]["sha256"]
    ):
        raise ValueError("invalid K13 baseline comparator")
    entries: dict[int, dict[int, dict[str, Any]]] = {}
    for budget in k11.BUDGETS:
        entries[budget] = {}
        for seed in k11.SEEDS:
            path = root / f"k{budget}" / f"seed{seed}" / "test_readout.json"
            result = json.loads(path.read_text(encoding="utf-8"))
            checks = {
                "protocol": result["protocol_version"] == k13.PROTOCOL,
                "budget": int(result["budget"]) == budget,
                "seed": int(result["seed"]) == seed,
                "geometry": result["geometry"] == "isotropic",
                "no_training": result["training_performed"] is False,
                "rows": result["test_rows"] == [9000, 10000],
                "test": result["test_evaluated"] is True,
                "config": result["config_sha256"] == base.sha256_file(config_path),
                "source": result["source_files_sha256"] == k13.source_hashes(),
            }
            if not all(checks.values()):
                raise ValueError(f"invalid K13 readout K{budget} seed{seed}: {checks}")
            entries[budget][seed] = {
                "path": str(path),
                "sha256": base.sha256_file(path),
                "metrics": normalize_semantic(result["semantic"]),
            }
    output = root / "aggregate" / "heldout_aggregate.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite K13 aggregate: {output}")
    result = {
        "protocol_version": PROTOCOL,
        "mode": "four_budget_three_seed_one_time_heldout_aggregate",
        "config_path": str(config_path),
        "config_sha256": base.sha256_file(config_path),
        "inputs": {
            "validation_aggregate": {
                "path": str(validation_path),
                "sha256": base.sha256_file(validation_path),
            },
            "baseline_comparator": {
                "path": str(baseline_path),
                "sha256": base.sha256_file(baseline_path),
            },
            "candidate_readouts": {
                str(budget): {
                    str(seed): {
                        "path": entries[budget][seed]["path"],
                        "sha256": entries[budget][seed]["sha256"],
                    }
                    for seed in k11.SEEDS
                }
                for budget in k11.BUDGETS
            },
        },
        "source_files_sha256": {
            "aggregate_isotropic_relation_tube_heldout_k13.py": base.sha256_file(
                Path(__file__).resolve()
            ),
            **k13.source_hashes(),
        },
        **aggregate(entries, validation, baselines),
        "test_evaluated": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    v11.atomic_json(output, result)


if __name__ == "__main__":
    main()
