"""Create a test-free comparator from the completed relation budget aggregate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import run_diagnostic as base
import run_physics_functional_anchor_training as v11


PROTOCOL = "relation_budget_validation_comparator_v1"
SOURCE_PROTOCOL = "relation_supervision_budget_curve_v1"
CONDITIONS = ("point", "empirical", "physical")
BUDGETS = (20, 40, 80, 160)
SEEDS = (0, 42, 3407)


def extract(source: dict[str, Any]) -> dict[str, Any]:
    if source["protocol_version"] != SOURCE_PROTOCOL:
        raise ValueError("unexpected source protocol")
    entries: dict[str, Any] = {}
    for budget in BUDGETS:
        budget_entries = source["entries"][str(budget)]
        entries[str(budget)] = {}
        for condition in CONDITIONS:
            entries[str(budget)][condition] = {}
            for seed in SEEDS:
                item = budget_entries[condition][str(seed)]
                if not bool(item["valid"]):
                    raise ValueError(f"invalid source entry K{budget} {condition} seed{seed}")
                entries[str(budget)][condition][str(seed)] = {
                    "validation_rgb_direct_abs_correlation": item[
                        "validation_rgb_direct_abs_correlation"
                    ],
                    "validation_rgb_mean_direct_abs_correlation": item[
                        "validation_rgb_mean_direct_abs_correlation"
                    ],
                    "validation_rgb_mean_r2": item["validation_rgb_mean_r2"],
                    "final_ccrl_validation_total": item[
                        "final_ccrl_validation_total"
                    ],
                    "graph": item["graph"],
                    "parameter_count": item["parameter_count"],
                }
    return {
        "protocol_version": PROTOCOL,
        "budgets": list(BUDGETS),
        "conditions": list(CONDITIONS),
        "seeds": list(SEEDS),
        "entries": entries,
        "copied_test_fields": False,
        "test_evaluated": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_path = args.source.resolve()
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite comparator: {output_path}")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    result = extract(source)
    result.update(
        source_path=str(source_path),
        source_sha256=base.sha256_file(source_path),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    v11.atomic_json(output_path, result)


if __name__ == "__main__":
    main()
