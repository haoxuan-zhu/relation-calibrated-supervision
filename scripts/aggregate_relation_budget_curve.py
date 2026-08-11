"""Aggregate the frozen relation-supervision budget curve without hand copies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

import run_diagnostic as base
import run_relation_budget_condition as runner


COMMON_TRAINING_KEYS = [
    "epochs",
    "batch_size",
    "learning_rate",
    "scheduler_factor",
    "scheduler_patience",
    "validation_interval",
    "targets",
    "kappa",
    "eta",
    "mu",
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def k80_equivalence_checks(
    registry: dict[str, Any], kind: str, seed: int, value: dict[str, Any]
) -> dict[str, bool]:
    new = runner.build_run_config(registry, kind, 80, seed)
    old = value["config"]
    checks = {
        "model": new["model"] == old["model"],
        "split": new["split"] == old["split"],
        "initial_state": new["initialization"]["expected_state_dict_sha256"]
        == old["initialization"]["expected_state_dict_sha256"],
        "no_warm_start": "warm_start" not in old,
        "normalization_source": new["normalization"]["rgb_statistics_source"]
        == old["normalization"]["rgb_statistics_source"],
        "normalization_budget": int(new["normalization"]["rgb_statistics_budget"])
        == int(old["normalization"]["rgb_statistics_budget"]),
        "normalization_subset": new["normalization"]["subset_sha256"]
        == old["normalization"]["subset_sha256"],
        "floor_schedule": new["floor_schedule"] == old["floor_schedule"],
        "training": all(
            new["training"][key] == old["training"][key]
            for key in COMMON_TRAINING_KEYS
        ),
    }
    if kind == "point":
        checks["condition_definition"] = (
            new["point_anchor"] == old["point_anchor"]
            and old["training"]["conditions"] == [runner.CONDITIONS[kind]]
        )
    elif kind == "empirical":
        checks["condition_definition"] = (
            new["empirical_anchor"] == old["empirical_anchor"]
            and all(
                new["functional_anchor"][key]
                == old["functional_anchor"][key]
                for key in ["budget", "subset_seed", "permutation_seed"]
            )
            and old["training"]["conditions"] == [runner.CONDITIONS[kind]]
        )
    else:
        checks["condition_definition"] = (
            all(
                new["functional_anchor"][key]
                == old["functional_anchor"][key]
                for key in [
                    "budget",
                    "subset_seed",
                    "permutation_seed",
                    "pinv_rcond",
                    "clip_predictions_to_unit_interval",
                ]
            )
            and runner.CONDITIONS[kind] in old["training"]["conditions"]
        )
    return checks


def read_legacy_k80(
    project_root: Path,
    registry: dict[str, Any],
    kind: str,
    seed: int,
) -> dict[str, Any]:
    relative = Path(runner.keyed(registry["legacy_k80"][kind], seed))
    path = (project_root / relative).resolve()
    value = load_json(path)
    checks = k80_equivalence_checks(registry, kind, seed, value)
    if not all(checks.values()):
        raise ValueError(f"K80 {kind}/seed{seed} is not contract-equivalent: {checks}")
    if kind == "physical":
        training = value["training"][runner.CONDITIONS[kind]]
        validation = value["post_lock_validation_semantics"][runner.CONDITIONS[kind]]
        metrics = value["metrics"][runner.CONDITIONS[kind]]
        legacy_valid = all(value["decision"]["validity"].values())
    else:
        training = value["training"]
        validation = value["post_lock_validation_semantics"]
        metrics = value["metrics"]
        legacy_valid = all(value["decision"]["validity"].values())
    return make_entry(
        path,
        value,
        kind,
        80,
        seed,
        training,
        validation,
        metrics,
        legacy_valid,
        {"legacy_k80_contract_equivalence": checks},
    )


def make_entry(
    path: Path,
    raw: dict[str, Any],
    kind: str,
    budget: int,
    seed: int,
    training: dict[str, Any],
    validation: dict[str, Any],
    metrics: dict[str, Any],
    valid: bool,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    history = training["history"]
    final_validation = history[-1].get("validation", {})
    return {
        "condition_kind": kind,
        "budget": budget,
        "seed": seed,
        "valid": bool(valid),
        "result_path": str(path),
        "result_sha256": base.sha256_file(path),
        "validation_rgb_mean_direct_abs_correlation": float(
            validation["mean_direct_abs_correlation"]
        ),
        "validation_rgb_direct_abs_correlation": list(
            validation["direct_abs_correlation"]
        ),
        "validation_rgb_mean_r2": float(validation["mean_r2"]),
        "test_rgb_mean_direct_abs_correlation": float(
            np.mean(metrics["direct_abs_correlation"][:3])
        ),
        "test_direct_abs_correlation": list(metrics["direct_abs_correlation"]),
        "test_hungarian_rgb_mcc": float(metrics["unanchored_mcc"]),
        "test_anchor_mean_direct_r2": float(metrics["anchor_mean_direct_r2"]),
        "test_linear_readout_mean_r2": float(metrics["linear_readout_mean_r2"]),
        "graph": metrics["graph"],
        "final_ccrl_validation_total": (
            float(final_validation["total"])
            if "total" in final_validation
            else None
        ),
        "parameter_count": int(training["parameter_count"]),
        "final_supervision_weight": float(training["final_supervision_weight"]),
        "post_lock_teacher_validation": raw.get("post_lock_teacher_validation"),
        **(extra or {}),
    }


def read_new(
    new_root: Path, kind: str, budget: int, seed: int
) -> dict[str, Any]:
    path = (new_root / f"k{budget}" / f"seed{seed}" / kind / "diagnostic_results.json").resolve()
    value = load_json(path)
    if value["protocol_version"] != runner.PROTOCOL:
        raise ValueError(f"unexpected protocol in {path}")
    if (
        value["condition_kind"] != kind
        or int(value["budget"]) != budget
        or int(value["seed"]) != seed
    ):
        raise ValueError(f"identity mismatch in {path}")
    valid = value["decision"]["verdict"] == "relation_budget_condition_valid"
    valid = valid and all(value["decision"]["validity"].values())
    return make_entry(
        path,
        value,
        kind,
        budget,
        seed,
        value["training"],
        value["post_lock_validation_semantics"],
        value["metrics"],
        valid,
    )


def mean_sd(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "population_sd": float(np.std(array, ddof=0)),
    }


def summarize_budget(
    entries: dict[str, dict[int, dict[str, Any]]],
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    seeds = sorted(entries["point"])
    for kind in runner.CONDITIONS:
        if sorted(entries[kind]) != seeds:
            raise ValueError("condition seed sets differ")
        if not all(entries[kind][seed]["valid"] for seed in seeds):
            raise ValueError(f"invalid formal run in {kind}")
    readout = {
        kind: {
            "validation": mean_sd(
                [
                    entries[kind][seed][
                        "validation_rgb_mean_direct_abs_correlation"
                    ]
                    for seed in seeds
                ]
            ),
            "test": mean_sd(
                [
                    entries[kind][seed][
                        "test_rgb_mean_direct_abs_correlation"
                    ]
                    for seed in seeds
                ]
            ),
        }
        for kind in runner.CONDITIONS
    }
    deltas: dict[str, Any] = {}
    for left, right, name in [
        ("physical", "point", "physical_minus_point"),
        ("empirical", "point", "empirical_minus_point"),
        ("physical", "empirical", "physical_minus_empirical"),
    ]:
        validation = [
            entries[left][seed]["validation_rgb_mean_direct_abs_correlation"]
            - entries[right][seed]["validation_rgb_mean_direct_abs_correlation"]
            for seed in seeds
        ]
        test = [
            entries[left][seed]["test_rgb_mean_direct_abs_correlation"]
            - entries[right][seed]["test_rgb_mean_direct_abs_correlation"]
            for seed in seeds
        ]
        deltas[name] = {
            "validation_by_seed": dict(zip(map(str, seeds), validation)),
            "test_by_seed": dict(zip(map(str, seeds), test)),
            "validation": mean_sd(validation),
            "test": mean_sd(test),
        }
    threshold = float(evaluation["relation_gain_mean_min"])
    physical = deltas["physical_minus_point"]
    empirical = deltas["empirical_minus_point"]
    support = {
        "physical": bool(
            all(value > 0 for value in physical["validation_by_seed"].values())
            and all(value > 0 for value in physical["test_by_seed"].values())
            and physical["test"]["mean"] >= threshold
        ),
        "empirical": bool(
            all(value > 0 for value in empirical["validation_by_seed"].values())
            and all(value > 0 for value in empirical["test_by_seed"].values())
            and empirical["test"]["mean"] >= threshold
        ),
    }
    return {"readout": readout, "deltas": deltas, "support": support}


def decide_curve(
    summaries: dict[int, dict[str, Any]], evaluation: dict[str, Any]
) -> dict[str, Any]:
    new_budgets = [20, 40, 160]
    physical_supported = [
        budget for budget in new_budgets if summaries[budget]["support"]["physical"]
    ]
    empirical_supported = [
        budget for budget in new_budgets if summaries[budget]["support"]["empirical"]
    ]
    required = int(evaluation["supported_new_budget_count_min"])
    if len(physical_supported) >= required:
        verdict = "budget_curve_supported"
    elif len(physical_supported) == 1:
        verdict = "budget_support_mixed"
    else:
        verdict = "budget_support_not_confirmed"
    return {
        "verdict": verdict,
        "physical_supported_new_budgets": physical_supported,
        "physical_budget_curve_supported": len(physical_supported) >= required,
        "empirical_supported_new_budgets": empirical_supported,
        "empirical_relation_generalization_supported": len(empirical_supported)
        >= required,
        "required_supported_new_budget_count": required,
        "k80_counted_as_new_budget": False,
    }


def markdown_table(summaries: dict[int, dict[str, Any]]) -> str:
    lines = [
        "# Relation-supervision budget curve",
        "",
        "Machine-generated from locked JSON; values are test RGB mean direct absolute correlation (mean±population SD over three registered seeds).",
        "",
        "| K | Point | Empirical relation | Physical relation | Physical−Point | Empirical−Point |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for budget in sorted(summaries):
        item = summaries[budget]
        values = item["readout"]
        deltas = item["deltas"]
        cell = lambda x: f"{x['mean']:.6f}±{x['population_sd']:.6f}"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(budget),
                    cell(values["point"]["test"]),
                    cell(values["empirical"]["test"]),
                    cell(values["physical"]["test"]),
                    cell(deltas["physical_minus_point"]["test"]),
                    cell(deltas["empirical_minus_point"]["test"]),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--new-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    registry = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    runner.validate_registry(registry)
    project_root = args.project_root.resolve()
    new_root = args.new_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_entries: dict[int, dict[str, dict[int, dict[str, Any]]]] = {}
    summaries: dict[int, dict[str, Any]] = {}
    for budget in registry["budget_curve"]["budgets"]:
        by_kind: dict[str, dict[int, dict[str, Any]]] = {}
        for kind in runner.CONDITIONS:
            by_seed = {}
            for seed in registry["budget_curve"]["seeds"]:
                if budget == 80:
                    entry = read_legacy_k80(project_root, registry, kind, seed)
                else:
                    entry = read_new(new_root, kind, budget, seed)
                by_seed[seed] = entry
            by_kind[kind] = by_seed
        all_entries[budget] = by_kind
        summaries[budget] = summarize_budget(by_kind, registry["evaluation"])

    decision = decide_curve(summaries, registry["evaluation"])
    result = {
        "protocol_version": runner.PROTOCOL,
        "config_path": str(config_path),
        "config_sha256": base.sha256_file(config_path),
        "entries": all_entries,
        "budget_summaries": summaries,
        "decision": decision,
    }
    json_path = output_dir / "aggregate_results.json"
    runner.v11.atomic_json(json_path, result)
    table_path = output_dir / "budget_curve_table.md"
    table_path.write_text(markdown_table(summaries), encoding="utf-8")
    print(
        json.dumps(
            {
                "result_path": str(json_path),
                "sha256": base.sha256_file(json_path),
                "table_path": str(table_path),
                "verdict": decision["verdict"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
