"""Read-only scale-sensitive audit of anchor-preserving propagation runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

import audit_raw_ridge_locked_metrics as raw_audit
import run_diagnostic as base
import run_physics_functional_anchor_training as v11


PROTOCOL = "anchor_preserving_locked_metrics_v1"
BUDGETS = (20, 40, 80, 160)
SEEDS = (0, 42, 3407)


def summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("summary values must be finite and non-empty")
    return {
        "mean": float(array.mean()),
        "population_sd": float(array.std(ddof=0)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def manifest_matches(manifest: Path) -> bool:
    lines = manifest.read_text(encoding="utf-8").splitlines()
    if len(lines) != 3:
        return False
    for line in lines:
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            return False
        expected, raw_path = parts
        path = Path(raw_path.lstrip("*"))
        if not path.is_file() or base.sha256_file(path) != expected:
            return False
    return True


def summarize(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        str(budget): {
            split: {
                name: summary(
                    [
                        float(metrics[str(budget)][str(seed)][split][name])
                        for seed in SEEDS
                    ]
                )
                for name in ("mean_direct_abs_correlation", "mean_direct_r2")
            }
            for split in ("validation", "test")
        }
        for budget in BUDGETS
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    root = args.result_root.resolve()
    metrics: dict[str, Any] = {}
    source_integrity: dict[str, bool] = {}
    run_validity: dict[str, bool] = {}
    max_correlation_error = 0.0

    for budget in BUDGETS:
        metrics[str(budget)] = {}
        for seed in SEEDS:
            name = f"k{budget}/seed{seed}"
            path = root / "formal" / f"k{budget}_seed{seed}" / "diagnostic_results.json"
            run = json.loads(path.read_text(encoding="utf-8"))
            manifest = path.parent / "sha256_manifest.txt"
            source_integrity[name] = bool(
                path.exists() and manifest.exists() and manifest_matches(manifest)
            )
            run_validity[name] = bool(
                run["protocol_version"] == "anchor_preserving_raw_ridge_v1"
                and run["decision"]["verdict"] == "relation_budget_condition_valid"
                and all(run["decision"]["validity"].values())
                and run["training"]["anchor_preservation"]
                == {
                    "measured_row_count": budget,
                    "measured_environments": ["obs"],
                    "intervention_truth_read": False,
                    "unmeasured_targets": "raw_ridge_relation",
                }
                and run["test_evaluated"] is True
            )
            metrics[str(budget)][str(seed)] = {}
            for split in ("validation", "test"):
                item = raw_audit.recompute_raw_split(run, split, device)
                stored = (
                    float(run["post_lock_validation_semantics"]["mean_direct_abs_correlation"])
                    if split == "validation"
                    else float(np.mean(run["metrics"]["direct_abs_correlation"][:3]))
                )
                error = abs(item["mean_direct_abs_correlation"] - stored)
                item["stored_correlation_abs_error"] = error
                max_correlation_error = max(max_correlation_error, error)
                metrics[str(budget)][str(seed)][split] = item

    result = {
        "protocol_version": PROTOCOL,
        "fact_type": "post_lock_read_only_audit",
        "scope": "12_anchor_preserving_checkpoints_no_training_no_selection",
        "result_root": str(root),
        "source_integrity": source_integrity,
        "all_source_manifests_present": bool(all(source_integrity.values())),
        "run_validity": run_validity,
        "all_runs_valid": bool(all(run_validity.values())),
        "maximum_stored_correlation_abs_error": max_correlation_error,
        "stored_correlation_reproduced": raw_audit.correlation_reproduced(
            max_correlation_error
        ),
        "locked_metrics": metrics,
        "summaries": summarize(metrics),
        "implementation": {
            "entrypoint": str(Path(__file__).resolve()),
            "entrypoint_sha256": base.sha256_file(Path(__file__).resolve()),
        },
        "training_performed": False,
        "test_evaluated": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    v11.atomic_json(args.output.resolve(), result)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "sha256": base.sha256_file(args.output.resolve()),
                "all_runs_valid": result["all_runs_valid"],
                "stored_correlation_reproduced": result[
                    "stored_correlation_reproduced"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
