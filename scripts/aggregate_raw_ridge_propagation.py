"""Aggregate the two preregistered raw-ridge propagation studies from locked JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

import run_diagnostic as base
import run_physics_functional_anchor_training as v11


RAW_PROTOCOL = "raw_ridge_propagation_v1"
SUBSET_PROTOCOL = "raw_ridge_calibration_subset_robustness_v1"
REFERENCE_PROTOCOL = "relation_supervision_budget_curve_v1"
REFERENCE_AGGREGATE_SHA256 = (
    "18040bdecceeeb41d7b370726dc12b0b8153bb1dd110783d00c17e956e831d22"
)
BUDGETS = (20, 40, 80, 160)
MODEL_SEEDS = (0, 42, 3407)
SUBSET_SEEDS = (20260811, 20260821, 20260831)
KINDS = ("point", "empirical", "physical")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def mean_sd(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("summary values must be finite and nonempty")
    return {
        "mean": float(np.mean(array)),
        "population_sd": float(np.std(array, ddof=0)),
    }


def load_registry(path: Path, protocol: str) -> tuple[dict[str, Any], str]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if value["protocol_version"] != protocol:
        raise ValueError(f"unexpected protocol in {path}")
    return value, base.sha256_file(path)


def load_reference(path: Path) -> dict[str, Any]:
    digest = base.sha256_file(path)
    if digest != REFERENCE_AGGREGATE_SHA256:
        raise ValueError(f"locked reference aggregate hash mismatch: {digest}")
    value = load_json(path)
    if value["protocol_version"] != REFERENCE_PROTOCOL:
        raise ValueError("unexpected locked reference protocol")
    for budget in BUDGETS:
        for kind in KINDS:
            for seed in MODEL_SEEDS:
                entry = value["entries"][str(budget)][kind][str(seed)]
                if not entry["valid"]:
                    raise ValueError(
                        f"invalid locked reference: K{budget}/{kind}/seed{seed}"
                    )
    return value


def read_formal_result(
    path: Path,
    *,
    protocol: str,
    budget: int,
    seed: int,
    condition_kind: str,
    registry_sha256: str,
    subset_seed: int,
) -> dict[str, Any]:
    value = load_json(path)
    identity = {
        "protocol": value.get("protocol_version") == protocol,
        "mode": value.get("mode") == "formal",
        "budget": int(value.get("budget", -1)) == budget,
        "seed": int(value.get("seed", -1)) == seed,
        "condition_kind": value.get("condition_kind") == condition_kind,
        "registry_sha256": value.get("registry_config_sha256")
        == registry_sha256,
        "subset_seed": int(value.get("subset", {}).get("subset_seed", -1))
        == subset_seed,
        "subset_hash": value.get("subset", {}).get("subset_sha256")
        == value.get("subset", {}).get("expected_subset_sha256"),
        "semantic_validation_evaluated": value.get(
            "semantic_validation_evaluated"
        )
        is True,
        "test_evaluated": value.get("test_evaluated") is True,
    }
    decision = value.get("decision", {})
    validity = decision.get("validity", {})
    valid = bool(
        all(identity.values())
        and decision.get("verdict") == "relation_budget_condition_valid"
        and validity
        and all(validity.values())
    )
    training = value["training"]
    validation = value["post_lock_validation_semantics"]
    metrics = value["metrics"]
    return {
        "protocol_version": protocol,
        "budget": budget,
        "seed": seed,
        "condition_kind": condition_kind,
        "subset_seed": subset_seed,
        "valid": valid,
        "identity_checks": identity,
        "validity_checks": validity,
        "result_path": str(path),
        "result_sha256": base.sha256_file(path),
        "validation_rgb_mean_direct_abs_correlation": float(
            validation["mean_direct_abs_correlation"]
        ),
        "validation_rgb_mean_r2": float(validation["mean_r2"]),
        "test_rgb_mean_direct_abs_correlation": float(
            np.mean(metrics["direct_abs_correlation"][:3])
        ),
        "parameter_count": int(training["parameter_count"]),
        "final_supervision_weight": float(training["final_supervision_weight"]),
        "training_condition": training["condition"],
    }


def reference_entry(
    reference: dict[str, Any], budget: int, kind: str, seed: int
) -> dict[str, Any]:
    return reference["entries"][str(budget)][kind][str(seed)]


def paired_differences(
    left: dict[int, dict[str, Any]],
    right: dict[int, dict[str, Any]],
    field: str,
) -> dict[str, float]:
    if set(left) != set(right):
        raise ValueError("paired seed sets differ")
    return {
        str(seed): float(left[seed][field] - right[seed][field])
        for seed in sorted(left)
    }


def decide_raw(
    summaries: dict[int, dict[str, Any]], all_valid: bool
) -> dict[str, Any]:
    supported = [
        budget
        for budget in BUDGETS
        if all(
            delta > 0
            for delta in summaries[budget]["deltas"]["raw_minus_point"][
                "test_by_seed"
            ].values()
        )
    ]
    if not all_valid:
        verdict = "raw_ridge_followup_invalid"
    elif len(supported) >= 3:
        verdict = "generic_propagation_major_gain_supported"
    else:
        verdict = "generic_propagation_major_gain_not_supported"
    return {
        "verdict": verdict,
        "all_formal_runs_valid": all_valid,
        "budgets_with_raw_minus_point_test_3_of_3_positive": supported,
        "required_budget_count": 3,
        "performance_magnitude_threshold_used": False,
    }


def summarize_raw(
    entries: dict[int, dict[int, dict[str, Any]]],
    reference: dict[str, Any],
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    summaries: dict[int, dict[str, Any]] = {}
    all_valid = True
    for budget in BUDGETS:
        raw_by_seed = entries[budget]
        all_valid = all_valid and all(item["valid"] for item in raw_by_seed.values())
        refs = {
            kind: {
                seed: reference_entry(reference, budget, kind, seed)
                for seed in MODEL_SEEDS
            }
            for kind in KINDS
        }
        readout = {
            "raw_ridge": mean_sd(
                [
                    raw_by_seed[seed]["test_rgb_mean_direct_abs_correlation"]
                    for seed in MODEL_SEEDS
                ]
            ),
            **{
                kind: mean_sd(
                    [
                        refs[kind][seed]["test_rgb_mean_direct_abs_correlation"]
                        for seed in MODEL_SEEDS
                    ]
                )
                for kind in KINDS
            },
        }
        deltas: dict[str, Any] = {}
        for kind in KINDS:
            name = f"raw_minus_{kind}"
            validation = paired_differences(
                raw_by_seed,
                refs[kind],
                "validation_rgb_mean_direct_abs_correlation",
            )
            test = paired_differences(
                raw_by_seed,
                refs[kind],
                "test_rgb_mean_direct_abs_correlation",
            )
            deltas[name] = {
                "validation_by_seed": validation,
                "test_by_seed": test,
                "validation": mean_sd(list(validation.values())),
                "test": mean_sd(list(test.values())),
            }
        summaries[budget] = {"readout": readout, "deltas": deltas}
    return summaries, decide_raw(summaries, all_valid)


def series_robustness(
    cells: dict[int, dict[int, dict[str, float]]],
    original_block: dict[int, dict[str, float]],
) -> dict[str, Any]:
    block_test_means = {
        str(subset_seed): float(
            np.mean([cells[subset_seed][budget]["test"] for budget in BUDGETS])
        )
        for subset_seed in SUBSET_SEEDS
    }
    budget_test_means = {
        str(budget): float(
            np.mean(
                [cells[subset_seed][budget]["test"] for subset_seed in SUBSET_SEEDS]
            )
        )
        for budget in BUDGETS
    }
    test_values = [
        cells[subset_seed][budget]["test"]
        for subset_seed in SUBSET_SEEDS
        for budget in BUDGETS
    ]
    validation_values = [
        cells[subset_seed][budget]["validation"]
        for subset_seed in SUBSET_SEEDS
        for budget in BUDGETS
    ]
    positive_cells = sum(value > 0 for value in test_values)
    validation_mean = float(np.mean(validation_values))
    test_mean = float(np.mean(test_values))
    checks = {
        "each_new_subset_four_budget_test_mean_positive": all(
            value > 0 for value in block_test_means.values()
        ),
        "each_budget_three_subset_test_mean_positive": all(
            value > 0 for value in budget_test_means.values()
        ),
        "at_least_10_of_12_new_test_cells_positive": positive_cells >= 10,
        "validation_and_test_overall_direction_positive": validation_mean > 0
        and test_mean > 0,
    }
    all_block_test_means = {
        **block_test_means,
        "20260731": float(
            np.mean([original_block[budget]["test"] for budget in BUDGETS])
        ),
    }
    return {
        "new_subset_block_test_means": block_test_means,
        "new_budget_test_means": budget_test_means,
        "new_positive_test_cells": positive_cells,
        "new_validation_delta": mean_sd(validation_values),
        "new_test_delta": mean_sd(test_values),
        "original_subset_deltas": {
            str(budget): original_block[budget] for budget in BUDGETS
        },
        "all_subset_block_test_means": all_block_test_means,
        "negative_subset_block_count": sum(
            value < 0 for value in all_block_test_means.values()
        ),
        "checks": checks,
        "robust": bool(all(checks.values())),
        "nested_cells_treated_as_independent_for_p_value": False,
    }


def decide_subset(
    series: dict[str, dict[str, Any]], all_valid: bool
) -> dict[str, Any]:
    if not all_valid:
        verdict = "subset_robustness_invalid"
    elif any(item["negative_subset_block_count"] >= 2 for item in series.values()):
        verdict = "fixed_subset_claim_not_robust"
    elif all(item["robust"] for item in series.values()):
        verdict = "subset_robustness_supported"
    else:
        verdict = "subset_robustness_mixed"
    return {
        "verdict": verdict,
        "all_formal_runs_valid": all_valid,
        "series_robustness": {
            name: item["robust"] for name, item in series.items()
        },
        "nested_budget_cells_used_for_p_values": False,
    }


def summarize_subset(
    entries: dict[int, dict[int, dict[str, dict[str, Any]]]],
    reference: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    all_valid = all(
        item["valid"]
        for subset_seed in SUBSET_SEEDS
        for budget in BUDGETS
        for item in entries[subset_seed][budget].values()
    )
    series: dict[str, Any] = {}
    for kind in ("empirical", "physical"):
        cells: dict[int, dict[int, dict[str, float]]] = {}
        for subset_seed in SUBSET_SEEDS:
            cells[subset_seed] = {}
            for budget in BUDGETS:
                left = entries[subset_seed][budget][kind]
                point = entries[subset_seed][budget]["point"]
                cells[subset_seed][budget] = {
                    "validation": float(
                        left["validation_rgb_mean_direct_abs_correlation"]
                        - point["validation_rgb_mean_direct_abs_correlation"]
                    ),
                    "test": float(
                        left["test_rgb_mean_direct_abs_correlation"]
                        - point["test_rgb_mean_direct_abs_correlation"]
                    ),
                }
        original = {}
        for budget in BUDGETS:
            left = reference_entry(reference, budget, kind, 3407)
            point = reference_entry(reference, budget, "point", 3407)
            original[budget] = {
                "validation": float(
                    left["validation_rgb_mean_direct_abs_correlation"]
                    - point["validation_rgb_mean_direct_abs_correlation"]
                ),
                "test": float(
                    left["test_rgb_mean_direct_abs_correlation"]
                    - point["test_rgb_mean_direct_abs_correlation"]
                ),
            }
        series[f"{kind}_minus_point"] = {
            "new_cells": {
                str(subset_seed): {
                    str(budget): cells[subset_seed][budget] for budget in BUDGETS
                }
                for subset_seed in SUBSET_SEEDS
            },
            **series_robustness(cells, original),
        }
    return series, decide_subset(series, all_valid)


def markdown_report(
    raw_summaries: dict[int, dict[str, Any]],
    raw_decision: dict[str, Any],
    subset_series: dict[str, Any],
    subset_decision: dict[str, Any],
) -> str:
    lines = [
        "# Raw-ridge propagation study aggregate",
        "",
        "Machine-generated from locked JSON. Nested budget cells are not treated as independent replicates.",
        "",
        "## G1: raw-feature ridge propagation",
        "",
        "| K | Point | Raw ridge | Interaction OLS | Physical | Raw−Point | Raw−Interaction | Raw−Physical |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    cell = lambda value: f"{value['mean']:.6f}±{value['population_sd']:.6f}"
    for budget in BUDGETS:
        item = raw_summaries[budget]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(budget),
                    cell(item["readout"]["point"]),
                    cell(item["readout"]["raw_ridge"]),
                    cell(item["readout"]["empirical"]),
                    cell(item["readout"]["physical"]),
                    cell(item["deltas"]["raw_minus_point"]["test"]),
                    cell(item["deltas"]["raw_minus_empirical"]["test"]),
                    cell(item["deltas"]["raw_minus_physical"]["test"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            f"G1 verdict: `{raw_decision['verdict']}`.",
            "",
            "## S1: calibration-subset robustness",
            "",
            "| Series | Positive cells | Validation mean | Test mean | Robust |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for name, item in subset_series.items():
        lines.append(
            f"| {name} | {item['new_positive_test_cells']}/12 | "
            f"{item['new_validation_delta']['mean']:+.6f} | "
            f"{item['new_test_delta']['mean']:+.6f} | {item['robust']} |"
        )
    lines.extend(
        ["", f"S1 verdict: `{subset_decision['verdict']}`.", ""]
    )
    return "\n".join(lines)


def raw_markdown_report(
    raw_summaries: dict[int, dict[str, Any]], raw_decision: dict[str, Any]
) -> str:
    lines = [
        "# Raw-ridge propagation study G1 aggregate",
        "",
        "Machine-generated from locked JSON.",
        "",
        "| K | Point | Raw ridge | Interaction OLS | Physical | Raw−Point | Raw−Interaction | Raw−Physical |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    cell = lambda value: f"{value['mean']:.6f}±{value['population_sd']:.6f}"
    for budget in BUDGETS:
        item = raw_summaries[budget]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(budget),
                    cell(item["readout"]["point"]),
                    cell(item["readout"]["raw_ridge"]),
                    cell(item["readout"]["empirical"]),
                    cell(item["readout"]["physical"]),
                    cell(item["deltas"]["raw_minus_point"]["test"]),
                    cell(item["deltas"]["raw_minus_empirical"]["test"]),
                    cell(item["deltas"]["raw_minus_physical"]["test"]),
                ]
            )
            + " |"
        )
    lines.extend(["", f"G1 verdict: `{raw_decision['verdict']}`.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--reference-aggregate", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--subset-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("raw", "all"), default="all")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    reference_path = args.reference_aggregate.resolve()
    raw_root = args.raw_root.resolve()
    output_dir = args.output_dir.resolve()
    reference = load_reference(reference_path)

    raw_config_path = project_root / "configs" / "config_raw_ridge_propagation_v1a.yaml"
    raw_config, raw_config_sha = load_registry(raw_config_path, RAW_PROTOCOL)
    if tuple(raw_config["budget_curve"]["budgets"]) != BUDGETS:
        raise ValueError("raw-ridge budget registry changed")
    if tuple(raw_config["budget_curve"]["seeds"]) != MODEL_SEEDS:
        raise ValueError("raw-ridge model seeds changed")
    raw_entries: dict[int, dict[int, dict[str, Any]]] = {}
    for budget in BUDGETS:
        raw_entries[budget] = {}
        for seed in MODEL_SEEDS:
            path = raw_root / "formal" / f"k{budget}_seed{seed}" / "diagnostic_results.json"
            raw_entries[budget][seed] = read_formal_result(
                path,
                protocol=RAW_PROTOCOL,
                budget=budget,
                seed=seed,
                condition_kind="empirical",
                registry_sha256=raw_config_sha,
                subset_seed=20260731,
            )
    raw_summaries, raw_decision = summarize_raw(raw_entries, reference)

    if args.stage == "raw":
        result = {
            "protocol_version": "raw_ridge_raw_aggregate_v1",
            "reference_aggregate": {
                "path": str(reference_path),
                "sha256": REFERENCE_AGGREGATE_SHA256,
            },
            "raw_ridge": {
                "config_path": str(raw_config_path),
                "config_sha256": raw_config_sha,
                "entries": raw_entries,
                "budget_summaries": raw_summaries,
                "decision": raw_decision,
            },
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        result_path = output_dir / "raw_aggregate_results.json"
        v11.atomic_json(result_path, result)
        report_path = output_dir / "raw_followup_table.md"
        report_path.write_text(
            raw_markdown_report(raw_summaries, raw_decision), encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "result_path": str(result_path),
                    "sha256": base.sha256_file(result_path),
                    "report_path": str(report_path),
                    "raw_verdict": raw_decision["verdict"],
                },
                sort_keys=True,
            )
        )
        return

    if args.subset_root is None:
        raise ValueError("--subset-root is required for --stage all")
    subset_root = args.subset_root.resolve()

    subset_configs: dict[int, tuple[dict[str, Any], str]] = {}
    for subset_seed in SUBSET_SEEDS:
        path = project_root / f"config_raw_ridge_subset_{subset_seed}_v1.yaml"
        registry, digest = load_registry(path, SUBSET_PROTOCOL)
        if int(registry["budget_curve"]["subset_seed"]) != subset_seed:
            raise ValueError("subset registry identity changed")
        subset_configs[subset_seed] = (registry, digest)
    subset_entries: dict[int, dict[int, dict[str, dict[str, Any]]]] = {}
    for subset_seed in SUBSET_SEEDS:
        registry_sha = subset_configs[subset_seed][1]
        subset_entries[subset_seed] = {}
        for budget in BUDGETS:
            subset_entries[subset_seed][budget] = {}
            for kind in KINDS:
                path = (
                    subset_root
                    / "formal"
                    / f"subset{subset_seed}"
                    / f"k{budget}_{kind}"
                    / "diagnostic_results.json"
                )
                subset_entries[subset_seed][budget][kind] = read_formal_result(
                    path,
                    protocol=SUBSET_PROTOCOL,
                    budget=budget,
                    seed=3407,
                    condition_kind=kind,
                    registry_sha256=registry_sha,
                    subset_seed=subset_seed,
                )
    subset_series, subset_decision = summarize_subset(subset_entries, reference)

    result = {
        "protocol_version": "raw_ridge_propagation_aggregate_v1",
        "reference_aggregate": {
            "path": str(reference_path),
            "sha256": REFERENCE_AGGREGATE_SHA256,
        },
        "raw_ridge": {
            "config_path": str(raw_config_path),
            "config_sha256": raw_config_sha,
            "entries": raw_entries,
            "budget_summaries": raw_summaries,
            "decision": raw_decision,
        },
        "subset_robustness": {
            "configs": {
                str(seed): {
                    "path": str(
                        project_root
                        / f"config_raw_ridge_subset_{seed}_v1.yaml"
                    ),
                    "sha256": digest,
                }
                for seed, (_, digest) in subset_configs.items()
            },
            "entries": subset_entries,
            "series": subset_series,
            "decision": subset_decision,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "aggregate_results.json"
    v11.atomic_json(result_path, result)
    report_path = output_dir / "raw_ridge_table.md"
    report_path.write_text(
        markdown_report(
            raw_summaries, raw_decision, subset_series, subset_decision
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "result_path": str(result_path),
                "sha256": base.sha256_file(result_path),
                "report_path": str(report_path),
                "raw_verdict": raw_decision["verdict"],
                "subset_verdict": subset_decision["verdict"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
