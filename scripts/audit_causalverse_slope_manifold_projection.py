"""Project locked Slope point predictions onto the calibrated physical manifold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import least_squares

import audit_causalverse_slope_analytic_closure as closure_audit
import causalverse_slope_preflight as slope
import run_causalverse_slope_preflight as preflight


def state_from_free(free: np.ndarray, parameters: slope.RelationParameters) -> np.ndarray:
    roughness, theta_degrees, v0 = np.asarray(free, dtype=np.float64)
    theta = np.deg2rad(theta_degrees)
    mu1 = parameters.mu1_slope * roughness + parameters.mu1_intercept
    mu2 = parameters.mu2_slope * roughness + parameters.mu2_intercept
    v1 = v0 - parameters.deceleration * mu1
    denominator = parameters.incline * (np.sin(theta) + mu2 * np.cos(theta))
    if not np.isfinite(denominator) or abs(denominator) <= 1e-10:
        return np.full(7, 1e12, dtype=np.float64)
    return np.asarray([roughness, theta_degrees, v0, mu1, mu2, v1, v1**2 / denominator])


def project_to_manifold(
    predicted: np.ndarray,
    parameters: slope.RelationParameters,
    coordinate_scale: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    max_nfev: int = 200,
) -> tuple[np.ndarray, dict[str, object]]:
    values = np.asarray(predicted, dtype=np.float64)
    scale = np.asarray(coordinate_scale, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 7 or scale.shape != (7,):
        raise ValueError("projection expects [N,7] predictions and seven coordinate scales")
    if np.any(scale <= 1e-8) or np.any(lower >= upper):
        raise ValueError("projection scale or free-coordinate bounds are degenerate")
    projected = np.empty_like(values)
    successes: list[bool] = []
    nfev: list[int] = []
    clipped_starts = 0
    objective_before: list[float] = []
    objective_after: list[float] = []
    for index, target in enumerate(values):
        raw_start = target[:3]
        start = np.clip(raw_start, lower, upper)
        clipped_starts += int(not np.array_equal(raw_start, start))

        def residual(free: np.ndarray) -> np.ndarray:
            return (state_from_free(free, parameters) - target) / scale

        objective_before.append(float(np.mean(residual(start) ** 2)))
        result = least_squares(
            residual,
            start,
            bounds=(lower, upper),
            max_nfev=max_nfev,
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
        )
        state = state_from_free(result.x, parameters)
        if not np.isfinite(state).all():
            raise ValueError(f"projection produced non-finite state at row {index}")
        projected[index] = state
        successes.append(bool(result.success))
        nfev.append(int(result.nfev))
        objective_after.append(float(np.mean(residual(result.x) ** 2)))
    diagnostics = {
        "row_count": len(values),
        "success_count": int(sum(successes)),
        "clipped_start_count": clipped_starts,
        "mean_nfev": float(np.mean(nfev)),
        "max_nfev_observed": int(max(nfev)),
        "mean_scaled_objective_before": float(np.mean(objective_before)),
        "mean_scaled_objective_after": float(np.mean(objective_after)),
    }
    return projected, diagnostics


def seed_decision(correct: dict[str, object], projected: dict[str, object]) -> str:
    correct_better = (
        correct["mean_direct_abs_correlation_relation4"]
        > projected["mean_direct_abs_correlation_relation4"]
        and correct["mean_direct_r2_relation4"] > projected["mean_direct_r2_relation4"]
    )
    projection_explains = (
        projected["mean_direct_abs_correlation_relation4"]
        >= correct["mean_direct_abs_correlation_relation4"]
        and projected["mean_direct_r2_relation4"] >= correct["mean_direct_r2_relation4"]
    )
    if correct_better:
        return "training_beyond_manifold_projection"
    if projection_explains:
        return "manifold_projection_at_least_as_strong"
    return "mixed_primary_metrics"


def aggregate_decision(valid: bool, decisions: dict[str, str]) -> str:
    if not valid:
        return "invalid"
    values = list(decisions.values())
    if values and all(value == "training_beyond_manifold_projection" for value in values):
        return "training_beyond_manifold_projection_supported"
    if values and all(value == "manifold_projection_at_least_as_strong" for value in values):
        return "manifold_projection_explains_relation_gain"
    return "mixed_training_vs_manifold_projection"


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

    train = slope.select_ids(grouped, splits["train"])
    label_indices = slope.ordered_label_indices(train.ids, label_ids)
    labels = train.latents[label_indices]
    parameters = slope.fit_relation_parameters(labels)
    coordinate_scale = labels.std(axis=0)
    lower = labels[:, :3].min(axis=0)
    upper = labels[:, :3].max(axis=0)

    assets: dict[tuple[int, str], Path] = {}
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
            assets[(seed, condition)] = checkpoint_path

    validation = slope.select_ids(grouped, splits["validation"])
    metrics: dict[str, dict[str, dict[str, object]]] = {}
    deltas: dict[str, dict[str, dict[str, float]]] = {}
    decisions: dict[str, str] = {}
    diagnostics: dict[str, dict[str, object]] = {}
    for seed in seeds:
        predictions: dict[str, np.ndarray] = {}
        for condition in ("point", "correct_relation"):
            checkpoint = torch.load(assets[(seed, condition)], map_location=device, weights_only=False)
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
        predictions["free_variable_substitution"] = closure_audit.analytic_closure(
            predictions["point"], parameters
        )
        predictions["all_coordinate_manifold_projection"], projection_diagnostics = project_to_manifold(
            predictions["point"], parameters, coordinate_scale, lower, upper
        )
        seed_key = str(seed)
        diagnostics[seed_key] = projection_diagnostics
        seed_metrics = {
            name: slope.evaluate_predictions(validation.latents, predicted)
            for name, predicted in predictions.items()
        }
        metrics[seed_key] = seed_metrics
        point = seed_metrics["point"]
        projected = seed_metrics["all_coordinate_manifold_projection"]
        correct = seed_metrics["correct_relation"]
        decisions[seed_key] = seed_decision(correct, projected)
        deltas[seed_key] = {
            "projection_minus_point": {
                "relation4_correlation": float(
                    projected["mean_direct_abs_correlation_relation4"]
                    - point["mean_direct_abs_correlation_relation4"]
                ),
                "relation4_r2": float(
                    projected["mean_direct_r2_relation4"] - point["mean_direct_r2_relation4"]
                ),
            },
            "correct_minus_projection": {
                "relation4_correlation": float(
                    correct["mean_direct_abs_correlation_relation4"]
                    - projected["mean_direct_abs_correlation_relation4"]
                ),
                "relation4_r2": float(
                    correct["mean_direct_r2_relation4"]
                    - projected["mean_direct_r2_relation4"]
                ),
                "free3_correlation": float(
                    correct["mean_direct_abs_correlation_free3"]
                    - projected["mean_direct_abs_correlation_free3"]
                ),
            },
        }

    validity = {
        "k23_audit_matches": audit_valid,
        "all_six_locks_match": bool(all(lock_checks) and len(lock_checks) == 6),
        "incomplete_product_filter_contract_matches": bool(filter_audit["contract_matches"]),
        "all_projection_rows_converged": bool(
            all(item["success_count"] == item["row_count"] for item in diagnostics.values())
        ),
        "test_unread": True,
    }
    valid = bool(all(validity.values()))
    payload = {
        "protocol_version": "causalverse_slope_manifold_projection_k25",
        "fact_type": "locked_validation_all_coordinate_manifold_projection_audit",
        "valid": valid,
        "validity": validity,
        "projection_contract": {
            "coordinate_scale": coordinate_scale.tolist(),
            "free_lower_k40": lower.tolist(),
            "free_upper_k40": upper.tolist(),
            "max_nfev": 200,
            "tolerance": 1e-10,
        },
        "projection_diagnostics": diagnostics,
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
