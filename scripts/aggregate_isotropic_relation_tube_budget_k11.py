"""Aggregate the validation-only isotropic relation-tube budget curve."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

import run_diagnostic as base
import run_isotropic_relation_tube_budget_k11 as k11
import run_physics_functional_anchor_training as v11


PROTOCOL = "isotropic_relation_tube_budget_k11_aggregate_v1"
K80_PROTOCOL = "relation_tube_geometry_downstream_audit_k6_v1"
COMPARATOR_PROTOCOL = "relation_budget_validation_comparator_v1"
COMPARATORS = ("point", "empirical", "physical")
METRICS = (
    "raw_correlation",
    "raw_r2",
    "rgb_mcc",
    "coordinatewise_r2",
    "full_affine_r2",
    "ccrl_total",
    "edge_auroc",
    "shd",
)


def summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "population_std": float(array.std(ddof=0)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def normalize_run(
    *,
    budget: int,
    seed: int,
    semantic: dict[str, Any],
    graph: dict[str, Any],
    ccrl: dict[str, Any],
    origin: str,
    path: Path,
    digest: str,
) -> dict[str, Any]:
    return {
        "budget": budget,
        "seed": seed,
        "origin": origin,
        "input_path": str(path),
        "input_sha256": digest,
        "metrics": {
            "raw_correlation": float(semantic["raw"]["mean_direct_abs_correlation"]),
            "raw_r2": float(semantic["raw"]["mean_r2"]),
            "rgb_mcc": float(semantic["rgb_hungarian_mcc"]),
            "coordinatewise_r2": float(
                semantic["coordinatewise_affine"]["mean_r2"]
            ),
            "full_affine_r2": float(semantic["full_affine"]["mean_r2"]),
            "ccrl_total": float(ccrl["total"]),
            "edge_auroc": float(graph["edge_auroc"]),
            "shd": float(graph["fixed_threshold_shd"]),
        },
    }


def load_new_run(root: Path, budget: int, seed: int) -> dict[str, Any]:
    path = root / f"k{budget}" / f"seed{seed}" / "formal" / "validation_readout.json"
    digest = base.sha256_file(path)
    result = json.loads(path.read_text(encoding="utf-8"))
    checks = {
        "protocol": result["protocol_version"] == k11.PROTOCOL,
        "budget": int(result["budget"]) == budget,
        "seed": int(result["seed"]) == seed,
        "semantic_validation": result["semantic_validation_evaluated"] is True,
        "test_unread": result["test_evaluated"] is False,
    }
    if not all(checks.values()):
        raise ValueError(f"invalid K11 run K{budget} seed{seed}: {checks}")
    return normalize_run(
        budget=budget,
        seed=seed,
        semantic=result["semantic"],
        graph=result["graph"],
        ccrl=result["ccrl_validation"],
        origin="new_k11_training",
        path=path,
        digest=digest,
    )


def load_k80_run(master: dict[str, Any], seed: int) -> dict[str, Any]:
    item = k11.keyed(master["k80_reuse"], seed)
    path = Path(item["path"]).resolve()
    digest = base.sha256_file(path)
    if digest != item["sha256"]:
        raise ValueError(f"K80 seed{seed} input hash mismatch")
    result = json.loads(path.read_text(encoding="utf-8"))
    checks = {
        "protocol": result["protocol_version"] == K80_PROTOCOL,
        "seed": int(result["seed"]) == seed,
        "semantic_validation": result["semantic_validation_evaluated"] is True,
        "test_unread": result["test_evaluated"] is False,
        "isotropic_present": "isotropic" in result["runs"],
    }
    if not all(checks.values()):
        raise ValueError(f"invalid K80 reuse seed{seed}: {checks}")
    run = result["runs"]["isotropic"]
    return normalize_run(
        budget=80,
        seed=seed,
        semantic=run["semantic"],
        graph=run["graph"],
        ccrl=run["ccrl_validation"],
        origin="locked_k6_isotropic_reuse",
        path=path,
        digest=digest,
    )


def load_comparator(master: dict[str, Any]) -> tuple[dict[str, Any], Path, str]:
    item = master["validation_comparator"]
    path = Path(item["path"]).resolve()
    digest = base.sha256_file(path)
    if digest != item["sha256"]:
        raise ValueError("validation comparator hash mismatch")
    result = json.loads(path.read_text(encoding="utf-8"))
    checks = {
        "protocol": result["protocol_version"] == COMPARATOR_PROTOCOL,
        "budgets": tuple(result["budgets"]) == k11.BUDGETS,
        "conditions": tuple(result["conditions"]) == COMPARATORS,
        "seeds": tuple(result["seeds"]) == k11.SEEDS,
        "no_test_copy": result["copied_test_fields"] is False,
        "test_unread": result["test_evaluated"] is False,
    }
    if not all(checks.values()):
        raise ValueError(f"invalid validation comparator: {checks}")
    return result, path, digest


def comparator_metrics(item: dict[str, Any]) -> dict[str, float]:
    return {
        "raw_correlation": float(item["validation_rgb_mean_direct_abs_correlation"]),
        "raw_r2": float(item["validation_rgb_mean_r2"]),
        "ccrl_total": float(item["final_ccrl_validation_total"]),
        "edge_auroc": float(item["graph"]["edge_auroc"]),
        "shd": float(item["graph"]["fixed_threshold_shd"]),
    }


def aggregate(
    entries: dict[int, dict[int, dict[str, Any]]], comparator: dict[str, Any]
) -> dict[str, Any]:
    budget_summaries: dict[str, Any] = {}
    for budget in k11.BUDGETS:
        rows = entries[budget]
        metrics = {
            metric: summary([rows[seed]["metrics"][metric] for seed in k11.SEEDS])
            for metric in METRICS
        }
        comparisons: dict[str, Any] = {}
        for condition in COMPARATORS:
            deltas: dict[str, Any] = {}
            for metric in ("raw_correlation", "raw_r2", "ccrl_total", "edge_auroc", "shd"):
                values = []
                by_seed = {}
                for seed in k11.SEEDS:
                    reference = comparator_metrics(
                        comparator["entries"][str(budget)][condition][str(seed)]
                    )[metric]
                    delta = float(rows[seed]["metrics"][metric] - reference)
                    values.append(delta)
                    by_seed[str(seed)] = delta
                deltas[metric] = {
                    **summary(values),
                    "by_seed": by_seed,
                    "strict_positive_count": sum(value > 0 for value in values),
                    "strict_negative_count": sum(value < 0 for value in values),
                }
            comparisons[condition] = deltas
        budget_summaries[str(budget)] = {
            "metrics": metrics,
            "comparisons": comparisons,
        }
    correlation_means = [
        budget_summaries[str(budget)]["metrics"]["raw_correlation"]["mean"]
        for budget in k11.BUDGETS
    ]
    return {
        "budget_summaries": budget_summaries,
        "curve_audit": {
            "raw_correlation_means_by_budget": {
                str(budget): value for budget, value in zip(k11.BUDGETS, correlation_means)
            },
            "raw_correlation_monotone_nondecreasing": all(
                right >= left for left, right in zip(correlation_means, correlation_means[1:])
            ),
            "isotropic_minus_point_raw_correlation_positive_counts": {
                str(budget): budget_summaries[str(budget)]["comparisons"]["point"][
                    "raw_correlation"
                ]["strict_positive_count"]
                for budget in k11.BUDGETS
            },
            "isotropic_minus_physical_raw_correlation_positive_counts": {
                str(budget): budget_summaries[str(budget)]["comparisons"]["physical"][
                    "raw_correlation"
                ]["strict_positive_count"]
                for budget in k11.BUDGETS
            },
            "isotropic_minus_empirical_raw_correlation_positive_counts": {
                str(budget): budget_summaries[str(budget)]["comparisons"]["empirical"][
                    "raw_correlation"
                ]["strict_positive_count"]
                for budget in k11.BUDGETS
            },
        },
        "decision": {
            "verdict": "budget_characterization_complete_no_test_read",
            "geometry_selection_reopened": False,
            "test_evaluated": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    config_path = args.config.resolve()
    master = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    k11.validate_master(master)
    root = Path(args.output_root or master["runtime"]["output_root"]).resolve()
    comparator, comparator_path, comparator_sha256 = load_comparator(master)
    entries: dict[int, dict[int, dict[str, Any]]] = {}
    for budget in k11.BUDGETS:
        entries[budget] = {}
        for seed in k11.SEEDS:
            entries[budget][seed] = (
                load_k80_run(master, seed)
                if budget == 80
                else load_new_run(root, budget, seed)
            )
    result = {
        "protocol_version": PROTOCOL,
        "mode": "four_budget_three_seed_validation_only_aggregate",
        "config_path": str(config_path),
        "config_sha256": base.sha256_file(config_path),
        "source_files_sha256": {
            "aggregate_isotropic_relation_tube_budget_k11.py": base.sha256_file(
                Path(__file__).resolve()
            ),
            **k11.source_hashes(),
        },
        "validation_comparator": {
            "path": str(comparator_path),
            "sha256": comparator_sha256,
            "copied_test_fields": False,
        },
        "entries": {
            str(budget): {str(seed): entries[budget][seed] for seed in k11.SEEDS}
            for budget in k11.BUDGETS
        },
        "expected_finite_sample_coverage": {
            str(budget): float(
                k11.keyed(
                    master["budget_curve"]["expected_empirical_coverage"], budget
                )
            )
            for budget in k11.BUDGETS
        },
        **aggregate(entries, comparator),
        "semantic_validation_evaluated": True,
        "test_evaluated": False,
    }
    output_path = root / "aggregate" / "budget_validation_aggregate.json"
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite K11 aggregate: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    v11.atomic_json(output_path, result)


if __name__ == "__main__":
    main()
