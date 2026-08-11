"""Freeze historical matched-budget test correlations before K13 readout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import run_diagnostic as base
import run_isotropic_relation_tube_budget_k11 as k11
import run_physics_functional_anchor_training as v11


PROTOCOL = "heldout_baseline_comparator_k13_v1"
CONDITIONS = ("point", "empirical", "physical", "raw_ridge")
EXPECTED_PARAMETER_COUNT = 16902384


def extract(relation: dict[str, Any], raw: dict[str, Any]) -> dict[str, Any]:
    entries: dict[str, Any] = {}
    for budget in k11.BUDGETS:
        entries[str(budget)] = {condition: {} for condition in CONDITIONS}
        for seed in k11.SEEDS:
            for condition in CONDITIONS[:3]:
                item = relation["entries"][str(budget)][condition][str(seed)]
                if item["valid"] is not True:
                    raise ValueError(
                        f"K{budget} seed{seed} invalid historical {condition} run"
                    )
                if int(item["parameter_count"]) != EXPECTED_PARAMETER_COUNT:
                    raise ValueError(
                        f"K{budget} seed{seed} {condition} parameter-count mismatch"
                    )
                entries[str(budget)][condition][str(seed)] = {
                    "mean_direct_abs_rgb_correlation": float(
                        item["test_rgb_mean_direct_abs_correlation"]
                    ),
                    "source_result_sha256": item["result_sha256"],
                    "parameter_count": int(item["parameter_count"]),
                }
            item = raw["raw_ridge"]["entries"][str(budget)][str(seed)]
            if item["valid"] is not True or not all(item["validity_checks"].values()):
                raise ValueError(f"K{budget} seed{seed} invalid historical raw G1 run")
            if int(item["parameter_count"]) != EXPECTED_PARAMETER_COUNT:
                raise ValueError(
                    f"K{budget} seed{seed} raw G1 parameter-count mismatch"
                )
            entries[str(budget)]["raw_ridge"][str(seed)] = {
                "mean_direct_abs_rgb_correlation": float(
                    item["test_rgb_mean_direct_abs_correlation"]
                ),
                "source_result_sha256": item["result_sha256"],
                "parameter_count": int(item["parameter_count"]),
            }
    return {
        "protocol_version": PROTOCOL,
        "budgets": list(k11.BUDGETS),
        "seeds": list(k11.SEEDS),
        "conditions": list(CONDITIONS),
        "entries": entries,
        "historical_baseline_test_metrics_copied": True,
        "candidate_test_evaluated": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--relation-aggregate", type=Path, required=True)
    parser.add_argument("--raw-aggregate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite K13 comparator: {output}")
    relation_path = args.relation_aggregate.resolve()
    raw_path = args.raw_aggregate.resolve()
    relation = json.loads(relation_path.read_text(encoding="utf-8"))
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    result = {
        **extract(relation, raw),
        "sources": {
            "relation_aggregate": {
                "path": str(relation_path),
                "sha256": base.sha256_file(relation_path),
            },
            "raw_aggregate": {
                "path": str(raw_path),
                "sha256": base.sha256_file(raw_path),
            },
        },
        "source_files_sha256": {
            "extract_heldout_baseline_comparator_k13.py": base.sha256_file(
                Path(__file__).resolve()
            )
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    v11.atomic_json(output, result)


if __name__ == "__main__":
    main()
