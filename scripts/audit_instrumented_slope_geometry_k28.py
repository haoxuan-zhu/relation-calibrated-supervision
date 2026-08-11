"""Train-only geometry screen for the constructed Instrumented Slope task."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.optimize import minimize_scalar

import audit_instrumented_slope_interface_k27 as k27
import causalverse_slope_preflight as slope
import run_causalverse_slope_preflight as preflight


def physical_state_from_roughness(
    side_state: np.ndarray,
    roughness: np.ndarray | float,
    parameters: slope.RelationParameters,
) -> np.ndarray:
    side = np.asarray(side_state, dtype=np.float64)
    if side.ndim != 2 or side.shape[1] != 2:
        raise ValueError("side_state must have shape [N,2]")
    r = np.broadcast_to(np.asarray(roughness, dtype=np.float64), (len(side),))
    theta = np.deg2rad(side[:, 0])
    v0 = side[:, 1]
    mu1 = parameters.mu1_slope * r + parameters.mu1_intercept
    mu2 = parameters.mu2_slope * r + parameters.mu2_intercept
    v1 = v0 - parameters.deceleration * mu1
    denominator = parameters.incline * (np.sin(theta) + mu2 * np.cos(theta))
    if not np.isfinite(denominator).all() or np.any(np.abs(denominator) <= 1e-10):
        raise ValueError("physical state denominator is degenerate")
    state = np.column_stack((r, mu1, mu2, v1, v1**2 / denominator))
    if not np.isfinite(state).all():
        raise ValueError("physical state contains non-finite values")
    return state


def physical_tangent(
    side_state: np.ndarray,
    roughness: float,
    parameters: slope.RelationParameters,
) -> np.ndarray:
    side = np.asarray(side_state, dtype=np.float64)
    r = float(roughness)
    theta = np.deg2rad(side[:, 0])
    v0 = side[:, 1]
    mu1 = parameters.mu1_slope * r + parameters.mu1_intercept
    mu2 = parameters.mu2_slope * r + parameters.mu2_intercept
    v1 = v0 - parameters.deceleration * mu1
    dv1 = -parameters.deceleration * parameters.mu1_slope
    denominator = parameters.incline * (np.sin(theta) + mu2 * np.cos(theta))
    denominator_derivative = parameters.incline * parameters.mu2_slope * np.cos(theta)
    dl = (
        2.0 * v1 * dv1 * denominator - v1**2 * denominator_derivative
    ) / denominator**2
    tangent = np.column_stack(
        (
            np.ones(len(side)),
            np.full(len(side), parameters.mu1_slope),
            np.full(len(side), parameters.mu2_slope),
            np.full(len(side), dv1),
            dl,
        )
    )
    if not np.isfinite(tangent).all():
        raise ValueError("physical tangent contains non-finite values")
    return tangent


def finite_sample_radius(scores: np.ndarray, coverage: float) -> tuple[float, int, float]:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("calibration scores must be a finite vector")
    if not 0.0 < coverage < 1.0:
        raise ValueError("coverage must lie in (0,1)")
    order_index = min(len(values), int(math.ceil((len(values) + 1) * coverage)))
    radius = float(np.sort(values)[order_index - 1])
    empirical = float(np.mean(values <= radius + 1e-12))
    return radius, order_index, empirical


def leave_one_out_geometry(
    labels: np.ndarray, coverage: float
) -> tuple[slope.RelationParameters, float, np.ndarray, dict[str, Any]]:
    values = np.asarray(labels, dtype=np.float64)
    targets = values[:, k27.TARGET_INDICES]
    target_scale = targets.std(axis=0)
    if np.any(target_scale <= 1e-8):
        raise ValueError("K40 target scale is degenerate")
    loo_residuals = np.empty_like(targets)
    for holdout in range(len(values)):
        keep = np.arange(len(values)) != holdout
        parameters = slope.fit_relation_parameters(values[keep])
        mean_roughness = float(values[keep, 0].mean())
        center = physical_state_from_roughness(
            values[holdout : holdout + 1, k27.SIDE_INDICES], mean_roughness, parameters
        )[0]
        loo_residuals[holdout] = targets[holdout] - center
    normalized = loo_residuals / target_scale
    isotropic_scores = np.linalg.norm(normalized, axis=1)
    scalar_scores = np.abs(normalized[:, 0])
    isotropic_radius, isotropic_index, isotropic_coverage = finite_sample_radius(
        isotropic_scores, coverage
    )
    scalar_radius, scalar_index, scalar_coverage = finite_sample_radius(scalar_scores, coverage)
    final_parameters = slope.fit_relation_parameters(values)
    mean_roughness = float(values[:, 0].mean())
    audit = {
        "label_count": int(len(values)),
        "target_scale": target_scale.tolist(),
        "loo_residual_mean": loo_residuals.mean(axis=0).tolist(),
        "loo_residual_std": loo_residuals.std(axis=0).tolist(),
        "isotropic": {
            "radius": isotropic_radius,
            "order_index_one_based": isotropic_index,
            "empirical_coverage": isotropic_coverage,
            "score_minimum": float(isotropic_scores.min()),
            "score_median": float(np.median(isotropic_scores)),
            "score_maximum": float(isotropic_scores.max()),
        },
        "roughness_scalar": {
            "radius": scalar_radius,
            "order_index_one_based": scalar_index,
            "empirical_coverage": scalar_coverage,
            "score_minimum": float(scalar_scores.min()),
            "score_median": float(np.median(scalar_scores)),
            "score_maximum": float(scalar_scores.max()),
        },
    }
    return final_parameters, mean_roughness, target_scale, audit


def clip_rows_to_radius(values: np.ndarray, radius: float) -> tuple[np.ndarray, dict[str, float]]:
    array = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(array, axis=1)
    scale = np.minimum(1.0, float(radius) / np.maximum(norms, 1e-12))
    clipped = array * scale[:, None]
    return clipped, {
        "boundary_active_fraction": float(np.mean(norms > float(radius))),
        "mean_projection_scale": float(np.mean(scale)),
        "minimum_projection_scale": float(np.min(scale)),
    }


def tangent_projection(
    normalized_residual: np.ndarray,
    side_state: np.ndarray,
    mean_roughness: float,
    parameters: slope.RelationParameters,
    target_scale: np.ndarray,
) -> np.ndarray:
    raw = np.asarray(normalized_residual, dtype=np.float64)
    tangent = physical_tangent(side_state, mean_roughness, parameters) / target_scale[None, :]
    denominator = np.sum(tangent**2, axis=1)
    coefficients = np.sum(raw * tangent, axis=1) / np.maximum(denominator, 1e-12)
    return tangent * coefficients[:, None]


def nonlinear_curve_projection(
    raw_prediction: np.ndarray,
    side_state: np.ndarray,
    parameters: slope.RelationParameters,
    target_scale: np.ndarray,
    lower: float,
    upper: float,
    tolerance: float,
    max_iterations: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not lower < upper:
        raise ValueError("roughness projection interval is empty")
    predictions = np.asarray(raw_prediction, dtype=np.float64)
    projected = np.empty_like(predictions)
    successes: list[bool] = []
    evaluations: list[int] = []
    for index, target in enumerate(predictions):
        side = side_state[index : index + 1]

        def objective(value: float) -> float:
            state = physical_state_from_roughness(side, value, parameters)[0]
            return float(np.sum(((state - target) / target_scale) ** 2))

        result = minimize_scalar(
            objective,
            bounds=(lower, upper),
            method="bounded",
            options={"xatol": float(tolerance), "maxiter": int(max_iterations)},
        )
        projected[index] = physical_state_from_roughness(side, float(result.x), parameters)[0]
        successes.append(bool(result.success))
        evaluations.append(int(result.nfev))
    return projected, {
        "row_count": int(len(predictions)),
        "success_count": int(sum(successes)),
        "mean_function_evaluations": float(np.mean(evaluations)),
        "max_function_evaluations": int(max(evaluations)),
        "roughness_lower": float(lower),
        "roughness_upper": float(upper),
    }


def better_both(candidate: dict[str, Any], baseline: dict[str, Any]) -> bool:
    return bool(
        candidate["mean_abs_correlation"] > baseline["mean_abs_correlation"]
        and candidate["mean_r2"] > baseline["mean_r2"]
    )


def run(config_path: Path, output_path: Path) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    interface_config_path = Path(config["source_contract"]["interface_config"])
    interface_result_path = Path(config["source_contract"]["interface_result"])
    interface_result = json.loads(interface_result_path.read_text(encoding="utf-8"))
    interface_config, grouped, splits, label_ids, filter_audit = preflight.load_context(
        interface_config_path
    )
    train = slope.select_ids(grouped, splits["train"])
    label_rows = slope.ordered_label_indices(train.ids, label_ids)
    evaluation_rows = np.flatnonzero(~np.isin(train.ids, label_ids))
    labels = train.latents[label_rows]
    coverage = float(config["calibration"]["coverage"])
    parameters, mean_roughness, target_scale, geometry = leave_one_out_geometry(labels, coverage)

    train_side = train.latents[:, k27.SIDE_INDICES]
    train_targets = train.latents[:, k27.TARGET_INDICES]
    center_all = physical_state_from_roughness(train_side, mean_roughness, parameters)
    residual_all = train_targets - center_all
    features = train.features.mean(axis=1, dtype=np.float64)
    ridge = k27.fit_standardized_dual_ridge(
        features[label_rows], residual_all[label_rows], float(config["raw_predictor"]["ridge_alpha"])
    )
    predicted_residual = k27.predict_standardized_dual_ridge(ridge, features[evaluation_rows])
    side = train_side[evaluation_rows]
    truth = train_targets[evaluation_rows]
    center = center_all[evaluation_rows]
    normalized_raw = predicted_residual / target_scale[None, :]

    isotropic_normalized, isotropic_audit = clip_rows_to_radius(
        normalized_raw, float(geometry["isotropic"]["radius"])
    )
    tangent_normalized = tangent_projection(
        normalized_raw, side, mean_roughness, parameters, target_scale
    )
    tangent_clipped, tangent_audit = clip_rows_to_radius(
        tangent_normalized, float(geometry["isotropic"]["radius"])
    )

    roughness_raw = mean_roughness + predicted_residual[:, 0]
    physical_lower, physical_upper = (
        float(value) for value in config["calibration"]["roughness_physical_bounds"]
    )
    scalar_span = float(geometry["roughness_scalar"]["radius"]) * float(target_scale[0])
    rank_lower = max(physical_lower, mean_roughness - scalar_span)
    rank_upper = min(physical_upper, mean_roughness + scalar_span)
    roughness_domain = np.clip(roughness_raw, physical_lower, physical_upper)
    roughness_rank = np.clip(roughness_raw, rank_lower, rank_upper)

    predictions = {
        "physical_center": center,
        "unbounded_five_residual": center + predicted_residual,
        "isotropic_rank_tube": center + isotropic_normalized * target_scale[None, :],
        "tangent_rank_tube": center + tangent_clipped * target_scale[None, :],
        "formula_from_predicted_roughness": physical_state_from_roughness(side, roughness_raw, parameters),
        "formula_physical_bounds": physical_state_from_roughness(side, roughness_domain, parameters),
        "formula_rank_tube": physical_state_from_roughness(side, roughness_rank, parameters),
    }
    raw_prediction = predictions["unbounded_five_residual"]
    curve_prediction, curve_audit = nonlinear_curve_projection(
        raw_prediction,
        side,
        parameters,
        target_scale,
        rank_lower,
        rank_upper,
        float(config["projection"]["nonlinear_tolerance"]),
        int(config["projection"]["nonlinear_max_iterations"]),
    )
    predictions["nonlinear_curve_rank_tube"] = curve_prediction

    if list(predictions) != list(config["candidates"]):
        raise ValueError("candidate registry drifted from the frozen config")
    metrics = {name: k27.direct_metrics(truth, prediction) for name, prediction in predictions.items()}
    unbounded = metrics["unbounded_five_residual"]
    isotropic_better = better_both(metrics["isotropic_rank_tube"], unbounded)
    nonlinear_names = ("tangent_rank_tube", "formula_rank_tube", "nonlinear_curve_rank_tube")
    nonlinear_better = {
        name: better_both(metrics[name], unbounded) for name in nonlinear_names
    }
    simple_formula_better = better_both(metrics["formula_physical_bounds"], unbounded)
    best_correlation = max(metrics, key=lambda name: metrics[name]["mean_abs_correlation"])
    best_r2 = max(metrics, key=lambda name: metrics[name]["mean_r2"])

    evaluation_ids = train.ids[evaluation_rows]
    validation_ids = set(int(value) for value in splits["validation"])
    test_ids = set(int(value) for value in splits["test"])
    validity = {
        "interface_config_sha_matches": preflight.sha256_file(interface_config_path)
        == str(config["source_contract"]["interface_config_sha256"]),
        "interface_result_sha_matches": preflight.sha256_file(interface_result_path)
        == str(config["source_contract"]["interface_result_sha256"]),
        "interface_result_supported": interface_result["machine_decision"]
        == "instrumented_slope_interface_supported",
        "interface_result_valid": bool(interface_result["valid"]),
        "feature_cache_sha_matches": preflight.sha256_file(Path(interface_config["runtime"]["feature_cache"]))
        == str(interface_config["runtime"]["feature_cache_sha256"]),
        "incomplete_product_filter_contract_matches": bool(filter_audit["contract_matches"]),
        "candidate_registry_exact": list(predictions) == list(config["candidates"]),
        "fit_and_evaluation_ids_disjoint": not bool(set(int(value) for value in label_ids) & set(int(value) for value in evaluation_ids)),
        "validation_ids_not_evaluated": not bool(set(int(value) for value in evaluation_ids) & validation_ids),
        "test_ids_not_evaluated": not bool(set(int(value) for value in evaluation_ids) & test_ids),
        "all_predictions_finite": all(np.isfinite(value).all() for value in predictions.values()),
        "calibration_rank_exact": geometry["isotropic"]["order_index_one_based"] == 39
        and geometry["roughness_scalar"]["order_index_one_based"] == 39,
        "nonlinear_projection_complete": curve_audit["success_count"] == len(evaluation_ids),
    }
    valid = bool(all(validity.values()))
    if not valid:
        decision = "invalid"
    elif isotropic_better:
        decision = "isotropic_tube_train_signal"
    elif any(nonlinear_better.values()):
        decision = "nonlinear_relation_geometry_train_signal"
    elif simple_formula_better:
        decision = "simple_formula_explains_train_signal"
    else:
        decision = "no_geometry_signal"

    payload = {
        "protocol_version": config["protocol_version"],
        "fact_type": "train_only_constructed_geometry_screen",
        "scope": "remaining_train_ids_only_no_validation_or_test_semantics",
        "config_sha256": preflight.sha256_file(config_path),
        "interface_config_sha256": preflight.sha256_file(interface_config_path),
        "interface_result_sha256": preflight.sha256_file(interface_result_path),
        "feature_cache_sha256": preflight.sha256_file(Path(interface_config["runtime"]["feature_cache"])),
        "label_count": int(len(label_ids)),
        "evaluation_count": int(len(evaluation_ids)),
        "label_ids_sha256": preflight.sha256_int_array(label_ids),
        "evaluation_ids_sha256": preflight.sha256_int_array(evaluation_ids),
        "physical_parameters": {
            name: float(value) for name, value in zip(parameters.__dataclass_fields__, parameters.as_array())
        },
        "mean_roughness_k40": mean_roughness,
        "calibration_geometry": geometry,
        "roughness_intervals": {
            "physical": [physical_lower, physical_upper],
            "rank_intersection": [rank_lower, rank_upper],
        },
        "projection_audits": {
            "isotropic_rank_tube": isotropic_audit,
            "tangent_rank_tube": tangent_audit,
            "nonlinear_curve_rank_tube": curve_audit,
        },
        "metrics": metrics,
        "deltas_vs_unbounded": {
            name: {
                "mean_abs_correlation": metrics[name]["mean_abs_correlation"]
                - unbounded["mean_abs_correlation"],
                "mean_r2": metrics[name]["mean_r2"] - unbounded["mean_r2"],
            }
            for name in metrics
            if name != "unbounded_five_residual"
        },
        "directional_checks": {
            "isotropic_better_both": isotropic_better,
            "nonlinear_better_both": nonlinear_better,
            "simple_formula_better_both": simple_formula_better,
        },
        "ranking": {
            "best_mean_abs_correlation": best_correlation,
            "best_mean_r2": best_r2,
        },
        "validity": validity,
        "valid": valid,
        "semantic_validation_evaluated": False,
        "test_evaluated": False,
        "machine_decision": decision,
    }
    preflight.write_json(output_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    run(args.config, args.output or Path(config["runtime"]["output"]))


if __name__ == "__main__":
    main()
