"""Independent aggregate audit for the K44 calibration-subset experiment.

This module deliberately does not import the formal K44 runner.  It rebuilds the
registered deltas and decision from the three saved paired readouts, then checks
them against the formal aggregate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


PROTOCOL = "relation_tube_vs_raw_subset_k44_independent_audit_v1"
FORMAL_PROTOCOL = "relation_tube_vs_raw_subset_k44_v1"
FORMAL_MODE = "paired_validation_only_readout"
SUBSET_SEEDS = (20260811, 20260821, 20260831)
METRICS = ("mean_direct_abs_correlation", "mean_r2")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _raw_metrics(readout: dict[str, Any], run_name: str) -> dict[str, float]:
    raw = readout["runs"][run_name]["semantic"]["raw"]
    return {metric: float(raw[metric]) for metric in METRICS}


def audit_readouts(
    readout_paths: list[Path], formal_aggregate: dict[str, Any]
) -> dict[str, Any]:
    if len(readout_paths) != len(SUBSET_SEEDS):
        raise ValueError("K44 audit expects exactly three paired readouts")

    cells: list[dict[str, Any]] = []
    deltas_by_metric: dict[str, list[float]] = {metric: [] for metric in METRICS}
    seen_subsets: list[int] = []

    for path in sorted(readout_paths):
        readout = json.loads(path.read_text(encoding="utf-8"))
        subset_seed = int(readout["subset_seed"])
        seen_subsets.append(subset_seed)
        if readout["protocol_version"] != FORMAL_PROTOCOL:
            raise ValueError(f"unexpected formal protocol in {path}")
        if readout["mode"] != FORMAL_MODE:
            raise ValueError(f"unexpected formal mode in {path}")
        if readout["semantic_validation_evaluated"] is not True:
            raise ValueError(f"validation was not evaluated in {path}")
        if readout["test_evaluated"] is not False:
            raise ValueError(f"test was read in {path}")
        if not all(value is True for value in readout["checks"].values()):
            raise ValueError(f"formal contract check failed in {path}")
        if not all(
            value is True for value in readout["locked_tube_reproduction"].values()
        ):
            raise ValueError(f"locked Tube reproduction failed in {path}")

        tube = _raw_metrics(readout, "isotropic_tube")
        propagation = _raw_metrics(readout, "raw_propagation")
        recomputed = {
            metric: float(tube[metric] - propagation[metric]) for metric in METRICS
        }
        stored = readout["isotropic_tube_minus_raw_propagation"]
        for metric in METRICS:
            if not np.isclose(
                recomputed[metric], float(stored[metric]), rtol=0.0, atol=1e-12
            ):
                raise ValueError(f"stored {metric} delta drifted in {path}")
            deltas_by_metric[metric].append(recomputed[metric])

        bootstrap = readout["paired_bootstrap"]
        if int(bootstrap["repetitions"]) != 10_000:
            raise ValueError(f"bootstrap repetition drift in {path}")
        cells.append(
            {
                "subset_seed": subset_seed,
                "readout_path": path.as_posix(),
                "readout_sha256": sha256_file(path),
                "tube": tube,
                "raw_propagation": propagation,
                "recomputed_delta": recomputed,
                "bootstrap_lower": {
                    metric: float(bootstrap[metric]["lower"]) for metric in METRICS
                },
                "all_formal_checks_true": True,
                "locked_tube_reproduction_true": True,
            }
        )

    if tuple(sorted(seen_subsets)) != SUBSET_SEEDS:
        raise ValueError(f"K44 subset registry drift: {sorted(seen_subsets)}")

    mean_deltas = {
        metric: float(np.mean(deltas_by_metric[metric])) for metric in METRICS
    }
    positive_counts = {
        metric: sum(value > 0.0 for value in deltas_by_metric[metric])
        for metric in METRICS
    }
    bootstrap_lower_positive_counts = {
        metric: sum(cell["bootstrap_lower"][metric] > 0.0 for cell in cells)
        for metric in METRICS
    }
    all_positive = all(
        positive_counts[metric] == len(SUBSET_SEEDS) for metric in METRICS
    )
    verdict = (
        "subset_stability_supported"
        if all_positive
        else "subset_stability_not_supported"
    )
    decision = {
        "positive_counts": positive_counts,
        "bootstrap_lower_positive_counts": bootstrap_lower_positive_counts,
        "mean_deltas": mean_deltas,
        "all_three_subsets_both_primary_metrics_positive": all_positive,
        "both_primary_metric_means_positive": all(
            mean_deltas[metric] > 0.0 for metric in METRICS
        ),
        "verdict": verdict,
    }

    formal_decision = formal_aggregate["decision"]
    if decision["positive_counts"] != formal_decision["positive_counts"]:
        raise ValueError("formal positive counts do not reproduce")
    if (
        decision["bootstrap_lower_positive_counts"]
        != formal_decision["bootstrap_lower_positive_counts"]
    ):
        raise ValueError("formal bootstrap counts do not reproduce")
    for metric in METRICS:
        if not np.isclose(
            decision["mean_deltas"][metric],
            float(formal_decision["mean_deltas"][metric]),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(f"formal mean {metric} delta does not reproduce")
    if decision["verdict"] != formal_decision["verdict"]:
        raise ValueError("formal K44 verdict does not reproduce")

    return {
        "protocol_version": PROTOCOL,
        "mode": "saved_readout_independent_recomputation",
        "semantic_validation_evaluated": True,
        "test_evaluated": False,
        "formal_aggregate_sha256": formal_aggregate["_sha256"],
        "cells": cells,
        "decision": decision,
        "formal_aggregate_reproduced": True,
        "note": (
            "Metrics and the decision were recomputed from saved Tube and matched "
            "raw-propagation readouts without importing the formal K44 runner. "
            "Bootstrap intervals were contract-checked, not resampled."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--formal-aggregate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite K44 independent audit: {output}")
    aggregate_path = args.formal_aggregate.resolve()
    formal_aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    formal_aggregate["_sha256"] = sha256_file(aggregate_path)
    readouts = sorted(
        args.input_root.resolve().glob("subset*/formal/paired_validation_readout.json")
    )
    result = audit_readouts(readouts, formal_aggregate)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["decision"], sort_keys=True))


if __name__ == "__main__":
    main()
