"""Compare locked Slope relation training with a zero-training analytic closure."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import yaml

import causalverse_slope_preflight as slope
import run_causalverse_slope_preflight as preflight


def analytic_closure(predicted: np.ndarray, parameters: slope.RelationParameters) -> np.ndarray:
    values = np.asarray(predicted, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(slope.LATENT_COLUMNS):
        raise ValueError("predictions must have shape [N,7]")
    closed = values.copy()
    roughness = closed[:, 0]
    theta = np.deg2rad(closed[:, 1])
    v0 = closed[:, 2]
    mu1 = parameters.mu1_slope * roughness + parameters.mu1_intercept
    mu2 = parameters.mu2_slope * roughness + parameters.mu2_intercept
    v1 = v0 - parameters.deceleration * mu1
    denominator = parameters.incline * (np.sin(theta) + mu2 * np.cos(theta))
    if not np.isfinite(denominator).all() or np.any(np.abs(denominator) <= 1e-8):
        raise ValueError("analytic closure has a non-finite or degenerate denominator")
    closed[:, 3] = mu1
    closed[:, 4] = mu2
    closed[:, 5] = v1
    closed[:, 6] = v1**2 / denominator
    if not np.isfinite(closed).all():
        raise ValueError("analytic closure produced non-finite coordinates")
    return closed


def seed_decision(correct: dict[str, object], closure: dict[str, object]) -> str:
    correct_better = (
        correct["mean_direct_abs_correlation_relation4"]
        > closure["mean_direct_abs_correlation_relation4"]
        and correct["mean_direct_r2_relation4"] > closure["mean_direct_r2_relation4"]
    )
    closure_explains = (
        closure["mean_direct_abs_correlation_relation4"]
        >= correct["mean_direct_abs_correlation_relation4"]
        and closure["mean_direct_r2_relation4"] >= correct["mean_direct_r2_relation4"]
    )
    if correct_better:
        return "training_beyond_analytic_closure"
    if closure_explains:
        return "analytic_closure_at_least_as_strong"
    return "mixed_primary_metrics"


def aggregate_decision(valid: bool, decisions: dict[str, str]) -> str:
    if not valid:
        return "invalid"
    values = list(decisions.values())
    if values and all(value == "training_beyond_analytic_closure" for value in values):
        return "training_beyond_analytic_closure_supported"
    if values and all(value == "analytic_closure_at_least_as_strong" for value in values):
        return "analytic_closure_explains_relation_gain"
    return "mixed_training_vs_analytic_closure"


def run(config_path: Path, root: Path, audit_path: Path, output_path: Path, device: torch.device) -> None:
    config, grouped, splits, label_ids, filter_audit = preflight.load_context(config_path)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    seeds = [int(value) for value in config["training"]["seeds"]]
    config_sha = preflight.sha256_file(config_path)
    feature_sha = preflight.sha256_file(Path(config["runtime"]["feature_cache"]))
    source_sha = {
        "core": preflight.sha256_file(Path(slope.__file__)),
        "runner": preflight.sha256_file(Path(preflight.__file__)),
    }
    audit_valid = (
        bool(audit["valid"])
        and audit["config_sha256"] == config_sha
        and audit["feature_cache_sha256"] == feature_sha
    )

    assets: dict[tuple[int, str], tuple[Path, Path]] = {}
    lock_checks: list[bool] = []
    for seed in seeds:
        for condition in ("point", "correct_relation"):
            directory = root / f"seed{seed}" / condition
            lock_path = directory / "training_lock.json"
            checkpoint_path = directory / "checkpoint.pt"
            if not lock_path.is_file() or not checkpoint_path.is_file():
                raise FileNotFoundError(f"missing K23 lock: seed={seed} condition={condition}")
            lock = json.loads(lock_path.read_text(encoding="utf-8"))
            lock_checks.append(
                lock["condition"] == condition
                and int(lock["seed"]) == seed
                and lock["config_sha256"] == config_sha
                and lock["feature_cache_sha256"] == feature_sha
                and lock["checkpoint_sha256"] == preflight.sha256_file(checkpoint_path)
                and lock["source_sha256"] == source_sha
                and not bool(lock["semantic_validation_evaluated_during_training"])
                and not bool(lock["test_evaluated_during_training"])
            )
            assets[(seed, condition)] = (lock_path, checkpoint_path)

    train = slope.select_ids(grouped, splits["train"])
    label_indices = slope.ordered_label_indices(train.ids, label_ids)
    parameters = slope.fit_relation_parameters(train.latents[label_indices])
    validation = slope.select_ids(grouped, splits["validation"])
    metrics: dict[str, dict[str, dict[str, object]]] = {}
    deltas: dict[str, dict[str, dict[str, float]]] = {}
    decisions: dict[str, str] = {}

    for seed in seeds:
        predictions: dict[str, np.ndarray] = {}
        for condition in ("point", "correct_relation"):
            _, checkpoint_path = assets[(seed, condition)]
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model = slope.build_head(
                int(config["backbone"]["feature_dim"]), config["model"]["hidden_dims"], seed, device
            )
            model.load_state_dict(checkpoint["state_dict"])
            predictions[condition] = slope.predict_physical(
                model,
                validation,
                int(config["backbone"]["feature_dim"]),
                np.asarray(checkpoint["normalization_mean"]),
                np.asarray(checkpoint["normalization_std"]),
                device,
            )
        predictions["point_analytic_closure"] = analytic_closure(predictions["point"], parameters)
        seed_metrics = {
            name: slope.evaluate_predictions(validation.latents, predicted)
            for name, predicted in predictions.items()
        }
        seed_key = str(seed)
        metrics[seed_key] = seed_metrics
        point = seed_metrics["point"]
        closure = seed_metrics["point_analytic_closure"]
        correct = seed_metrics["correct_relation"]
        decisions[seed_key] = seed_decision(correct, closure)
        deltas[seed_key] = {
            "closure_minus_point": {
                "relation4_correlation": float(
                    closure["mean_direct_abs_correlation_relation4"]
                    - point["mean_direct_abs_correlation_relation4"]
                ),
                "relation4_r2": float(
                    closure["mean_direct_r2_relation4"] - point["mean_direct_r2_relation4"]
                ),
            },
            "correct_minus_closure": {
                "relation4_correlation": float(
                    correct["mean_direct_abs_correlation_relation4"]
                    - closure["mean_direct_abs_correlation_relation4"]
                ),
                "relation4_r2": float(
                    correct["mean_direct_r2_relation4"] - closure["mean_direct_r2_relation4"]
                ),
                "free3_correlation": float(
                    correct["mean_direct_abs_correlation_free3"]
                    - closure["mean_direct_abs_correlation_free3"]
                ),
            },
        }

    validity = {
        "k23_audit_matches": audit_valid,
        "all_six_locks_match": bool(all(lock_checks) and len(lock_checks) == 6),
        "incomplete_product_filter_contract_matches": bool(filter_audit["contract_matches"]),
        "all_closure_outputs_finite": True,
        "test_unread": True,
    }
    valid = bool(all(validity.values()))
    payload = {
        "protocol_version": "causalverse_slope_analytic_closure_k24",
        "fact_type": "locked_validation_posthoc_analytic_closure_audit",
        "valid": valid,
        "validity": validity,
        "relation_parameters": asdict(parameters),
        "metrics_by_seed": metrics,
        "deltas_by_seed": deltas,
        "seed_decisions": decisions,
        "machine_decision": aggregate_decision(valid, decisions),
        "auditor_sha256": preflight.sha256_file(Path(__file__)),
        "test_evaluated": False,
    }
    preflight.write_json(output_path, payload)
    print(json.dumps({"valid": valid, "machine_decision": payload["machine_decision"], "output": str(output_path)}))


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
