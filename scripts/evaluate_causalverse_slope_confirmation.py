"""Evaluate the locked multi-seed CausalVerse Slope relation confirmation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

import causalverse_slope_preflight as slope
import run_causalverse_slope_preflight as preflight


PRIMARY_KEYS = (
    "mean_direct_abs_correlation_relation4",
    "mean_direct_r2_relation4",
    "mean_direct_abs_correlation_free3",
    "hungarian_mcc",
)


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError("summary values must be a non-empty finite vector")
    return {
        "mean": float(array.mean()),
        "std_population": float(array.std(ddof=0)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def aggregate_decision(valid: bool, seed_decisions: dict[str, str]) -> str:
    if not valid:
        return "invalid"
    decisions = list(seed_decisions.values())
    if decisions and all(value == "relation_content_signal" for value in decisions):
        return "multiseed_relation_content_confirmed"
    if any(value == "point_baseline_saturated_stop" for value in decisions):
        return "multiseed_point_saturation_stop"
    return "multiseed_relation_content_not_confirmed"


def finite_history(summary: dict[str, object]) -> bool:
    return all(
        np.isfinite(float(value))
        for row in summary["history"]
        for key, value in row.items()
        if key != "epoch"
    )


def run(config_path: Path, root: Path, audit_path: Path, output_path: Path, device: torch.device) -> None:
    config, grouped, splits, _, filter_audit = preflight.load_context(config_path)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    seeds = [int(value) for value in config["training"]["seeds"]]
    conditions = tuple(config["conditions"])
    if tuple(conditions) != slope.CONDITIONS:
        raise ValueError("condition order differs from the frozen Slope contract")

    expected_assets: dict[tuple[int, str], tuple[Path, Path, Path]] = {}
    for seed in seeds:
        for condition in conditions:
            condition_dir = root / f"seed{seed}" / condition
            assets = (
                condition_dir / "training_lock.json",
                condition_dir / "training_summary.json",
                condition_dir / "checkpoint.pt",
            )
            if not all(path.is_file() for path in assets):
                raise FileNotFoundError(f"locked assets are incomplete: seed={seed} condition={condition}")
            expected_assets[(seed, condition)] = assets

    # Semantic validation is selected only after every condition lock/checkpoint is present.
    validation = slope.select_ids(grouped, splits["validation"])
    config_sha = preflight.sha256_file(config_path)
    feature_sha = preflight.sha256_file(Path(config["runtime"]["feature_cache"]))
    source_sha = {
        "core": preflight.sha256_file(Path(slope.__file__)),
        "runner": preflight.sha256_file(Path(preflight.__file__)),
    }
    audit_contract = (
        bool(audit["valid"])
        and audit["config_sha256"] == config_sha
        and audit["feature_cache_sha256"] == feature_sha
    )

    metrics: dict[str, dict[str, dict[str, object]]] = {}
    deltas: dict[str, dict[str, dict[str, float]]] = {}
    seed_decisions: dict[str, str] = {}
    lock_validity: dict[str, bool] = {}
    initial_state_shared: dict[str, bool] = {}
    lock_sha256: dict[str, dict[str, str]] = {}

    for seed in seeds:
        seed_key = str(seed)
        seed_metrics: dict[str, dict[str, object]] = {}
        locks: dict[str, dict[str, object]] = {}
        seed_lock_checks: list[bool] = []
        lock_sha256[seed_key] = {}
        for condition in conditions:
            lock_path, summary_path, checkpoint_path = expected_assets[(seed, condition)]
            lock = json.loads(lock_path.read_text(encoding="utf-8"))
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            lock_sha256[seed_key][condition] = preflight.sha256_file(lock_path)
            locks[condition] = lock
            seed_lock_checks.append(
                lock["condition"] == condition
                and int(lock["seed"]) == seed
                and not bool(lock["smoke"])
                and int(lock["epochs"]) == int(config["training"]["epochs"])
                and lock["config_sha256"] == config_sha
                and lock["feature_cache_sha256"] == feature_sha
                and lock["checkpoint_sha256"] == preflight.sha256_file(checkpoint_path)
                and lock["source_sha256"] == source_sha
                and not bool(lock["semantic_validation_evaluated_during_training"])
                and not bool(lock["test_evaluated_during_training"])
                and finite_history(summary)
            )
            model = slope.build_head(
                int(config["backbone"]["feature_dim"]), config["model"]["hidden_dims"], seed, device
            )
            model.load_state_dict(checkpoint["state_dict"])
            predicted = slope.predict_physical(
                model,
                validation,
                int(config["backbone"]["feature_dim"]),
                np.asarray(checkpoint["normalization_mean"]),
                np.asarray(checkpoint["normalization_std"]),
                device,
            )
            seed_metrics[condition] = slope.evaluate_predictions(validation.latents, predicted)
        metrics[seed_key] = seed_metrics
        initial_hashes = {lock["initial_state_sha256"] for lock in locks.values()}
        initial_state_shared[seed_key] = len(initial_hashes) == 1
        lock_validity[seed_key] = bool(all(seed_lock_checks))
        seed_valid = bool(audit_contract and lock_validity[seed_key] and initial_state_shared[seed_key])
        seed_decisions[seed_key] = preflight.make_decision(
            seed_valid, seed_metrics, config["evaluation"]
        )
        point = seed_metrics["point"]
        correct = seed_metrics["correct_relation"]
        permuted = seed_metrics["coefficient_permuted_relation"]
        deltas[seed_key] = {
            "correct_minus_point": {
                "relation4_correlation": float(
                    correct["mean_direct_abs_correlation_relation4"]
                    - point["mean_direct_abs_correlation_relation4"]
                ),
                "relation4_r2": float(
                    correct["mean_direct_r2_relation4"] - point["mean_direct_r2_relation4"]
                ),
                "free3_correlation": float(
                    correct["mean_direct_abs_correlation_free3"]
                    - point["mean_direct_abs_correlation_free3"]
                ),
            },
            "correct_minus_coefficient_permuted_relation": {
                "relation4_correlation": float(
                    correct["mean_direct_abs_correlation_relation4"]
                    - permuted["mean_direct_abs_correlation_relation4"]
                ),
                "relation4_r2": float(
                    correct["mean_direct_r2_relation4"]
                    - permuted["mean_direct_r2_relation4"]
                ),
                "free3_correlation": float(
                    correct["mean_direct_abs_correlation_free3"]
                    - permuted["mean_direct_abs_correlation_free3"]
                ),
            },
        }

    aggregate_metrics: dict[str, dict[str, dict[str, float]]] = {}
    for condition in conditions:
        aggregate_metrics[condition] = {
            key: summarize([float(metrics[str(seed)][condition][key]) for seed in seeds])
            for key in PRIMARY_KEYS
        }
    aggregate_deltas: dict[str, dict[str, dict[str, float]]] = {}
    for comparison in ("correct_minus_point", "correct_minus_coefficient_permuted_relation"):
        aggregate_deltas[comparison] = {
            key: summarize([deltas[str(seed)][comparison][key] for seed in seeds])
            for key in ("relation4_correlation", "relation4_r2", "free3_correlation")
        }

    validity = {
        "audit_contract_matches": audit_contract,
        "all_nine_locks_present_before_validation_read": len(expected_assets) == len(seeds) * len(conditions),
        "all_lock_hash_and_history_contracts_match": bool(all(lock_validity.values())),
        "initial_state_shared_within_each_seed": bool(all(initial_state_shared.values())),
        "incomplete_product_filter_contract_matches": bool(filter_audit["contract_matches"]),
        "test_unread": True,
    }
    valid = bool(all(validity.values()))
    decision = aggregate_decision(valid, seed_decisions)
    payload = {
        "protocol_version": config["protocol_version"],
        "fact_type": "validation_only_multiseed_relation_confirmation",
        "valid": valid,
        "validity": validity,
        "seeds": seeds,
        "seed_decisions": seed_decisions,
        "machine_decision": decision,
        "metrics_by_seed": metrics,
        "deltas_by_seed": deltas,
        "aggregate_metrics": aggregate_metrics,
        "aggregate_deltas": aggregate_deltas,
        "condition_lock_sha256": lock_sha256,
        "evaluator_sha256": preflight.sha256_file(Path(__file__)),
        "test_evaluated": False,
    }
    preflight.write_json(output_path, payload)
    print(json.dumps({"valid": valid, "machine_decision": decision, "output": str(output_path)}))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    run(args.config, args.root, args.audit, args.output, device)
