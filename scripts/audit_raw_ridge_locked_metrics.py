"""Scale-sensitive audit of the 12 locked raw-ridge follow-up checkpoints.

The audit is read-only: it never trains, selects a checkpoint, or changes the
preregistered correlation decision.  It recomputes named-coordinate metrics
from the locked checkpoints and compares them with the previously audited
point, interaction-OLS, and physical conditions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

import audit_relation_budget_locked_metrics as locked
import run_conflict_projected_physics_training as v13
import run_diagnostic as base
import run_physics_functional_anchor_training as v11
import run_supervised_continuation_diagnostic as v6


BUDGETS = (20, 40, 80, 160)
SEEDS = (0, 42, 3407)
REFERENCE_KINDS = ("point", "empirical", "physical")
RAW_AGGREGATE_SHA256 = (
    "59080b4c7791bc5295c48b885d01a9d478067000e8cc45ccb7e0261e1189babd"
)
REFERENCE_LOCKED_METRICS_SHA256 = (
    "9bec4733baa3f1bee5c90fabe1f7b58a9390d404895c2122b7d6b19263d17e17"
)
# Repeated CUDA evaluation in the prior locked-metric audit reached a maximum
# absolute correlation difference of 3.2631e-5.  The tolerance is therefore a
# numerical-reproduction bound, not a performance threshold.
CORRELATION_REPRODUCTION_ATOL = 5e-5


def signed_correlation(true: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    if true.shape != predicted.shape or true.ndim != 2:
        raise ValueError("true and predicted must be matching 2-D arrays")
    values = []
    for coordinate in range(true.shape[1]):
        value = np.corrcoef(true[:, coordinate], predicted[:, coordinate])[0, 1]
        values.append(value)
    result = np.asarray(values, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError("signed correlation is non-finite")
    return result


def recompute_raw_split(
    raw: dict[str, Any], split: str, device: torch.device
) -> dict[str, Any]:
    config = raw["derived_config"]
    training = raw["training"]
    checkpoint = Path(training["checkpoint"])
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    if base.sha256_file(checkpoint) != training["checkpoint_sha256"]:
        raise ValueError(f"checkpoint hash mismatch: {checkpoint}")

    images = np.load(Path(config["dataset"]["cache_path"]), mmap_mode="r")
    raw_latents = base.load_latents(config)
    mean, std = locked.normalization_arrays(raw)
    latents = (raw_latents - mean[None, None, :]) / std[None, None, :]
    train_end = int(config["split"]["train_end"])
    validation_end = int(config["split"]["validation_end"])
    test_end = int(config["split"]["test_end"])
    start, end = (
        (train_end, validation_end)
        if split == "validation"
        else (validation_end, test_end)
    )

    model = v13.load_model(checkpoint, config, device)
    prediction = v6.predict_rgb(
        model,
        images,
        latents,
        start,
        end,
        int(config["training"]["batch_size"]),
        device,
    )
    truth = latents[0, start:end, :3]
    correlation = signed_correlation(truth, prediction)
    r2 = locked.direct_r2(truth, prediction)
    rmse = np.sqrt(np.mean((truth - prediction) ** 2, axis=0))
    result = {
        "direct_signed_correlation": correlation.tolist(),
        "direct_abs_correlation": np.abs(correlation).tolist(),
        "mean_direct_abs_correlation": float(np.mean(np.abs(correlation))),
        "direct_r2": r2.tolist(),
        "mean_direct_r2": float(np.mean(r2)),
        "normalized_rmse_by_coordinate": rmse.tolist(),
        "mean_normalized_rmse": float(np.mean(rmse)),
        **locked.prediction_calibration(truth, prediction),
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def mean_sd(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("summary values must be finite and nonempty")
    return {
        "mean": float(np.mean(array)),
        "population_sd": float(np.std(array, ddof=0)),
    }


def correlation_reproduced(max_abs_error: float) -> bool:
    return bool(max_abs_error <= CORRELATION_REPRODUCTION_ATOL)


def summarize(
    raw_metrics: dict[str, dict[str, dict[str, Any]]],
    reference: dict[str, Any],
) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    reference_metrics = reference["locked_metrics"]
    for budget in BUDGETS:
        key = str(budget)
        item: dict[str, Any] = {
            "raw_ridge": {
                split: mean_sd(
                    [
                        raw_metrics[key][str(seed)][split]["mean_direct_r2"]
                        for seed in SEEDS
                    ]
                )
                for split in ("validation", "test")
            }
        }
        for kind in REFERENCE_KINDS:
            item[kind] = {
                split: mean_sd(
                    [
                        reference_metrics[key][kind][str(seed)][split][
                            "mean_direct_r2"
                        ]
                        for seed in SEEDS
                    ]
                )
                for split in ("validation", "test")
            }
            name = f"raw_minus_{kind}"
            item[name] = {}
            for split in ("validation", "test"):
                values = [
                    raw_metrics[key][str(seed)][split]["mean_direct_r2"]
                    - reference_metrics[key][kind][str(seed)][split][
                        "mean_direct_r2"
                    ]
                    for seed in SEEDS
                ]
                item[name][split] = {
                    "by_seed": dict(zip(map(str, SEEDS), values)),
                    **mean_sd(values),
                }
        item["raw_calibration"] = {
            split: {
                "mean_abs_slope_error_from_one": mean_sd(
                    [
                        raw_metrics[key][str(seed)][split][
                            "mean_abs_slope_error_from_one"
                        ]
                        for seed in SEEDS
                    ]
                ),
                "mean_abs_intercept": mean_sd(
                    [
                        raw_metrics[key][str(seed)][split]["mean_abs_intercept"]
                        for seed in SEEDS
                    ]
                ),
            }
            for split in ("validation", "test")
        }
        summaries[key] = item
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-aggregate", type=Path, required=True)
    parser.add_argument("--reference-locked-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    raw_aggregate_path = args.raw_aggregate.resolve()
    reference_path = args.reference_locked_metrics.resolve()
    if base.sha256_file(raw_aggregate_path) != RAW_AGGREGATE_SHA256:
        raise ValueError("raw aggregate hash mismatch")
    if base.sha256_file(reference_path) != REFERENCE_LOCKED_METRICS_SHA256:
        raise ValueError("reference locked-metric hash mismatch")
    raw_aggregate = locked.load_json(raw_aggregate_path)
    reference = locked.load_json(reference_path)
    if raw_aggregate["protocol_version"] != "raw_ridge_raw_aggregate_v1":
        raise ValueError("unexpected raw aggregate protocol")
    if reference["protocol_version"] != "relation_budget_locked_metric_integrity_audit_k0_v1":
        raise ValueError("unexpected reference metric protocol")

    device = torch.device(
        args.device
        if args.device != "cuda" or torch.cuda.is_available()
        else "cpu"
    )
    source_integrity: dict[str, bool] = {}
    run_validity: dict[str, bool] = {}
    raw_metrics: dict[str, dict[str, dict[str, Any]]] = {}
    max_stored_correlation_abs_error = 0.0
    for budget in BUDGETS:
        key = str(budget)
        raw_metrics[key] = {}
        for seed in SEEDS:
            entry = raw_aggregate["raw_ridge"]["entries"][key][str(seed)]
            path = Path(entry["result_path"])
            name = f"k{budget}/seed{seed}"
            source_integrity[name] = (
                path.exists() and base.sha256_file(path) == entry["result_sha256"]
            )
            raw = locked.load_json(path)
            run_validity[name] = bool(
                entry["valid"]
                and raw["decision"]["verdict"] == "relation_budget_condition_valid"
                and all(raw["decision"]["validity"].values())
            )
            raw_metrics[key][str(seed)] = {}
            for split in ("validation", "test"):
                metrics = recompute_raw_split(raw, split, device)
                stored = (
                    raw["post_lock_validation_semantics"][
                        "mean_direct_abs_correlation"
                    ]
                    if split == "validation"
                    else float(np.mean(raw["metrics"]["direct_abs_correlation"][:3]))
                )
                error = abs(metrics["mean_direct_abs_correlation"] - stored)
                metrics["stored_correlation_abs_error"] = error
                max_stored_correlation_abs_error = max(
                    max_stored_correlation_abs_error, error
                )
                raw_metrics[key][str(seed)][split] = metrics

    summaries = summarize(raw_metrics, reference)
    result = {
        "protocol_version": "raw_ridge_raw_locked_metric_audit_v1",
        "fact_type": "retrospective_audit",
        "scope": "12_locked_raw_ridge_checkpoints_no_training_no_selection",
        "formal_preregistered_decision_unchanged": True,
        "raw_aggregate": {
            "path": str(raw_aggregate_path),
            "sha256": RAW_AGGREGATE_SHA256,
        },
        "reference_locked_metrics": {
            "path": str(reference_path),
            "sha256": REFERENCE_LOCKED_METRICS_SHA256,
        },
        "implementation": {
            "entrypoint": str(Path(__file__).resolve()),
            "entrypoint_sha256": base.sha256_file(Path(__file__).resolve()),
        },
        "source_integrity": source_integrity,
        "all_source_hashes_match": bool(all(source_integrity.values())),
        "run_validity": run_validity,
        "all_runs_valid": bool(all(run_validity.values())),
        "max_stored_correlation_abs_error": max_stored_correlation_abs_error,
        "correlation_reproduction_atol": CORRELATION_REPRODUCTION_ATOL,
        "correlation_reproduction_tolerance_basis": (
            "The preceding 36-checkpoint locked-metric audit recorded a maximum "
            "same-evaluator absolute correlation difference of 3.2631e-5; 5e-5 "
            "is a conservative CUDA numerical-reproduction bound."
        ),
        "stored_correlation_reproduced": correlation_reproduced(
            max_stored_correlation_abs_error
        ),
        "locked_metrics": raw_metrics,
        "scale_sensitive_summaries": summaries,
        "interpretation_boundary": (
            "Direct R2 and calibration were not the preregistered primary endpoint. "
            "This read-only audit tests whether the correlation ordering is a scale artifact; "
            "it does not select or recalibrate a checkpoint."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    v11.atomic_json(args.output.resolve(), result)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "sha256": base.sha256_file(args.output.resolve()),
                "all_source_hashes_match": result["all_source_hashes_match"],
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
