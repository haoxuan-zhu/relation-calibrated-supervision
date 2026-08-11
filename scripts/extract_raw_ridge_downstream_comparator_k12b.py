"""Extract a validation-only graph comparator from the locked G1 runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import run_diagnostic as base
import run_isotropic_relation_tube_budget_k11 as k11
import run_physics_functional_anchor_training as v11


PROTOCOL = "raw_ridge_downstream_validation_comparator_k12b_v1"
EXPECTED_PARAMETER_COUNT = 16902384


def extract(aggregate: dict[str, Any], source_root: Path) -> dict[str, Any]:
    raw = aggregate["raw_ridge"]
    entries: dict[str, Any] = {}
    sources: dict[str, Any] = {}
    for budget in k11.BUDGETS:
        entries[str(budget)] = {}
        sources[str(budget)] = {}
        for seed in k11.SEEDS:
            registry = raw["entries"][str(budget)][str(seed)]
            path = source_root / f"k{budget}_seed{seed}" / "diagnostic_results.json"
            digest = base.sha256_file(path)
            if digest != registry["result_sha256"]:
                raise ValueError(f"K{budget} seed{seed} G1 result hash mismatch")
            result = json.loads(path.read_text(encoding="utf-8"))
            validation = result["training"]["history"][-1]["validation"]
            graph = result["metrics"]["graph"]
            checks = {
                "budget": int(result["budget"]) == budget,
                "seed": int(result["seed"]) == seed,
                "valid": all(result["decision"]["validity"].values()),
                "epoch": int(result["training"]["final_epoch"]) == 100,
                "parameter_count": int(result["training"]["parameter_count"])
                == EXPECTED_PARAMETER_COUNT,
                "graph_threshold": float(
                    result["derived_config"]["evaluation"]["graph_threshold"]
                )
                == 0.3,
                "subset_seed": int(result["subset"]["subset_seed"])
                == int(registry["subset_seed"]),
                "registered_subset_hash_valid": registry["identity_checks"][
                    "subset_hash"
                ]
                is True,
                "registered_training_valid": all(registry["validity_checks"].values()),
            }
            if not all(checks.values()):
                raise ValueError(
                    f"K{budget} seed{seed} invalid G1 downstream source: {checks}"
                )
            entries[str(budget)][str(seed)] = {
                "edge_auroc": float(graph["edge_auroc"]),
                "fixed_threshold_shd": int(graph["fixed_threshold_shd"]),
                "ccrl_validation_total": float(validation["total"]),
                "parameter_count": int(result["training"]["parameter_count"]),
            }
            sources[str(budget)][str(seed)] = {
                "path": str(path.resolve()),
                "sha256": digest,
            }
    return {
        "protocol_version": PROTOCOL,
        "budgets": list(k11.BUDGETS),
        "seeds": list(k11.SEEDS),
        "entries": entries,
        "sources": sources,
        "copied_test_fields": False,
        "new_test_evaluation": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite K12b comparator: {output}")
    aggregate_path = args.aggregate.resolve()
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    result = {
        **extract(aggregate, args.source_root.resolve()),
        "aggregate_source": {
            "path": str(aggregate_path),
            "sha256": base.sha256_file(aggregate_path),
        },
        "source_files_sha256": {
            "extract_raw_ridge_downstream_comparator_k12b.py": base.sha256_file(
                Path(__file__).resolve()
            )
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    v11.atomic_json(output, result)


if __name__ == "__main__":
    main()
