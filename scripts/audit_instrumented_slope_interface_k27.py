"""Train-only audit for the constructed Instrumented Slope interface."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

import causalverse_slope_preflight as slope
import run_causalverse_slope_preflight as preflight


SIDE_INDICES = (1, 2)
TARGET_INDICES = (0, 3, 4, 5, 6)
TARGET_NAMES = tuple(slope.LATENT_COLUMNS[index] for index in TARGET_INDICES)


def physical_target_center(
    side_state: np.ndarray,
    mean_roughness: float,
    parameters: slope.RelationParameters,
) -> np.ndarray:
    """Evaluate the five-target physical center from observed theta and v0."""
    side = np.asarray(side_state, dtype=np.float64)
    if side.ndim != 2 or side.shape[1] != 2:
        raise ValueError("side_state must have shape [N,2] for theta and v0")
    theta = np.deg2rad(side[:, 0])
    v0 = side[:, 1]
    roughness = np.full(len(side), float(mean_roughness), dtype=np.float64)
    mu1 = parameters.mu1_slope * roughness + parameters.mu1_intercept
    mu2 = parameters.mu2_slope * roughness + parameters.mu2_intercept
    v1 = v0 - parameters.deceleration * mu1
    denominator = parameters.incline * (np.sin(theta) + mu2 * np.cos(theta))
    if not np.isfinite(denominator).all() or np.any(np.abs(denominator) <= 1e-10):
        raise ValueError("physical center denominator is degenerate")
    length = v1**2 / denominator
    center = np.column_stack((roughness, mu1, mu2, v1, length))
    if not np.isfinite(center).all():
        raise ValueError("physical center contains non-finite values")
    return center


def fit_standardized_dual_ridge(
    features: np.ndarray, targets: np.ndarray, alpha: float
) -> dict[str, np.ndarray | float]:
    """Fit ridge with an unpenalized intercept using the sample-space solve."""
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0] or x.shape[0] < 2:
        raise ValueError("ridge features and targets must be matched matrices")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or alpha <= 0:
        raise ValueError("ridge inputs must be finite and alpha must be positive")
    x_mean = x.mean(axis=0)
    x_scale = x.std(axis=0)
    x_scale = np.where(x_scale > 1e-12, x_scale, 1.0)
    y_mean = y.mean(axis=0)
    standardized = (x - x_mean) / x_scale
    centered_targets = y - y_mean
    gram = standardized @ standardized.T
    dual = np.linalg.solve(gram + float(alpha) * np.eye(len(x)), centered_targets)
    coefficients = standardized.T @ dual
    return {
        "alpha": float(alpha),
        "feature_mean": x_mean,
        "feature_scale": x_scale,
        "target_mean": y_mean,
        "coefficients": coefficients,
    }


def predict_standardized_dual_ridge(
    model: dict[str, np.ndarray | float], features: np.ndarray
) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    mean = np.asarray(model["feature_mean"], dtype=np.float64)
    scale = np.asarray(model["feature_scale"], dtype=np.float64)
    target_mean = np.asarray(model["target_mean"], dtype=np.float64)
    coefficients = np.asarray(model["coefficients"], dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != mean.size:
        raise ValueError("ridge prediction feature dimension mismatch")
    return target_mean + ((x - mean) / scale) @ coefficients


def direct_metrics(
    truth: np.ndarray, prediction: np.ndarray, names: Sequence[str] = TARGET_NAMES
) -> dict[str, Any]:
    y = np.asarray(truth, dtype=np.float64)
    pred = np.asarray(prediction, dtype=np.float64)
    if y.shape != pred.shape or y.ndim != 2 or y.shape[1] != len(names):
        raise ValueError("metric arrays do not match the target registry")
    coordinates: dict[str, dict[str, float]] = {}
    correlations: list[float] = []
    r2_values: list[float] = []
    nrmse_values: list[float] = []
    for index, name in enumerate(names):
        target = y[:, index]
        estimate = pred[:, index]
        if np.std(target) > 1e-12 and np.std(estimate) > 1e-12:
            correlation = abs(float(np.corrcoef(target, estimate)[0, 1]))
        else:
            correlation = 0.0
        denominator = float(np.sum((target - target.mean()) ** 2))
        squared_error = float(np.sum((target - estimate) ** 2))
        r2 = 1.0 - squared_error / max(denominator, 1e-12)
        nrmse = float(np.sqrt(np.mean((target - estimate) ** 2)) / max(np.std(target), 1e-12))
        coordinates[str(name)] = {
            "abs_correlation": correlation,
            "r2": r2,
            "normalized_rmse": nrmse,
        }
        correlations.append(correlation)
        r2_values.append(r2)
        nrmse_values.append(nrmse)
    return {
        "coordinates": coordinates,
        "mean_abs_correlation": float(np.mean(correlations)),
        "mean_r2": float(np.mean(r2_values)),
        "mean_normalized_rmse": float(np.mean(nrmse_values)),
    }


def residual_geometry(residual: np.ndarray, roughness_delta: np.ndarray) -> dict[str, Any]:
    values = np.asarray(residual, dtype=np.float64)
    hidden = np.asarray(roughness_delta, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(TARGET_NAMES) or hidden.shape != (len(values),):
        raise ValueError("residual geometry inputs have incompatible shapes")
    centered = values - values.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    variance = singular_values**2
    fractions = variance / max(float(variance.sum()), 1e-12)
    positive = fractions[fractions > 0]
    effective_rank = float(np.exp(-np.sum(positive * np.log(positive))))
    cumulative = np.cumsum(fractions)
    rank_99 = int(np.searchsorted(cumulative, 0.99, side="left") + 1)
    signed_correlations = []
    for index in range(values.shape[1]):
        if np.std(values[:, index]) > 1e-12 and np.std(hidden) > 1e-12:
            signed_correlations.append(float(np.corrcoef(values[:, index], hidden)[0, 1]))
        else:
            signed_correlations.append(0.0)
    return {
        "target_names": list(TARGET_NAMES),
        "residual_mean": values.mean(axis=0).tolist(),
        "residual_std": values.std(axis=0).tolist(),
        "singular_values": singular_values.tolist(),
        "variance_fractions": fractions.tolist(),
        "effective_rank": effective_rank,
        "rank_for_99pct_variance": rank_99,
        "signed_correlation_with_hidden_roughness_delta": signed_correlations,
        "abs_correlation_with_hidden_roughness_delta": np.abs(signed_correlations).tolist(),
    }


def machine_decision(valid: bool, residual_non_degenerate: bool, gates: dict[str, bool]) -> str:
    if not valid:
        return "invalid"
    if not residual_non_degenerate:
        return "degenerate_center_no_tube"
    if all(gates.values()):
        return "instrumented_slope_interface_supported"
    return "residual_exists_but_not_visually_recoverable"


def run(config_path: Path, output_path: Path) -> None:
    config, grouped, splits, label_ids, filter_audit = preflight.load_context(config_path)
    train = slope.select_ids(grouped, splits["train"])
    label_rows = slope.ordered_label_indices(train.ids, label_ids)
    evaluation_mask = ~np.isin(train.ids, label_ids)
    evaluation_rows = np.flatnonzero(evaluation_mask)
    labels = train.latents[label_rows]
    parameters = slope.fit_relation_parameters(labels)
    mean_roughness = float(labels[:, 0].mean())

    train_targets = train.latents[:, TARGET_INDICES]
    train_side = train.latents[:, SIDE_INDICES]
    train_center = physical_target_center(train_side, mean_roughness, parameters)
    train_residual = train_targets - train_center
    geometry = residual_geometry(train_residual[evaluation_rows], train.latents[evaluation_rows, 0] - mean_roughness)

    product_features = train.features.mean(axis=1, dtype=np.float64)
    ridge = fit_standardized_dual_ridge(
        product_features[label_rows],
        train_residual[label_rows],
        float(config["probe"]["ridge_alpha"]),
    )
    predicted_residual = predict_standardized_dual_ridge(ridge, product_features[evaluation_rows])
    truth = train_targets[evaluation_rows]
    center = train_center[evaluation_rows]
    residual_truth = train_residual[evaluation_rows]
    center_residual = np.zeros_like(residual_truth)
    corrected = center + predicted_residual

    center_metrics = direct_metrics(truth, center)
    corrected_metrics = direct_metrics(truth, corrected)
    zero_residual_metrics = direct_metrics(residual_truth, center_residual)
    image_residual_metrics = direct_metrics(residual_truth, predicted_residual)

    parent_path = Path(config["source_contract"]["parent_config"])
    feature_cache_path = Path(config["runtime"]["feature_cache"])
    expected_parameters = np.asarray(config["relation"]["expected_parameters"], dtype=np.float64)
    actual_parameters = parameters.as_array()
    evaluation_ids = train.ids[evaluation_rows]
    split_sets = {name: set(int(value) for value in ids) for name, ids in splits.items()}
    validity = {
        "parent_config_sha_matches": preflight.sha256_file(parent_path)
        == str(config["source_contract"]["parent_config_sha256"]),
        "feature_cache_sha_matches": preflight.sha256_file(feature_cache_path)
        == str(config["runtime"]["feature_cache_sha256"]),
        "incomplete_product_filter_contract_matches": bool(filter_audit["contract_matches"]),
        "schema_matches_frozen_interface": tuple(config["task"]["observed_controls"])
        == ("theta", "v0")
        and tuple(config["task"]["hidden_visual_state"]) == ("roughness",)
        and tuple(config["task"]["prediction_targets"]) == TARGET_NAMES,
        "parameters_match_published_equations": bool(
            np.max(np.abs(actual_parameters - expected_parameters))
            <= float(config["relation"]["correct_parameter_max_abs_tolerance"])
        ),
        "fit_and_evaluation_ids_disjoint": not bool(set(int(value) for value in label_ids) & set(int(value) for value in evaluation_ids)),
        "evaluation_is_remaining_train_only": set(int(value) for value in evaluation_ids)
        == split_sets["train"] - set(int(value) for value in label_ids),
        "validation_ids_not_evaluated": not bool(set(int(value) for value in evaluation_ids) & split_sets["validation"]),
        "test_ids_not_evaluated": not bool(set(int(value) for value in evaluation_ids) & split_sets["test"]),
        "all_outputs_finite": bool(
            np.isfinite(train_center).all()
            and np.isfinite(predicted_residual).all()
            and np.isfinite(corrected).all()
        ),
    }
    valid = bool(all(validity.values()))
    residual_std = np.asarray(geometry["residual_std"], dtype=np.float64)
    residual_non_degenerate = bool(
        np.all(residual_std > float(config["probe"]["residual_std_minimum"]))
    )
    gates = {
        "corrected_target_correlation_above_center": corrected_metrics["mean_abs_correlation"]
        > center_metrics["mean_abs_correlation"],
        "corrected_target_r2_above_center": corrected_metrics["mean_r2"] > center_metrics["mean_r2"],
        "image_residual_r2_above_zero_residual": image_residual_metrics["mean_r2"]
        > zero_residual_metrics["mean_r2"],
        "roughness_residual_correlation_nonzero": image_residual_metrics["coordinates"]["roughness"]["abs_correlation"]
        > 0.0,
    }
    decision = machine_decision(valid, residual_non_degenerate, gates)
    payload = {
        "protocol_version": config["protocol_version"],
        "fact_type": "train_only_constructed_interface_audit",
        "scope": "remaining_train_ids_only_no_validation_or_test_semantics",
        "config_sha256": preflight.sha256_file(config_path),
        "parent_config_sha256": preflight.sha256_file(parent_path),
        "feature_cache_sha256": preflight.sha256_file(feature_cache_path),
        "task": config["task"],
        "target_names": list(TARGET_NAMES),
        "side_indices": list(SIDE_INDICES),
        "target_indices": list(TARGET_INDICES),
        "label_count": int(len(label_ids)),
        "evaluation_count": int(len(evaluation_ids)),
        "label_ids_sha256": preflight.sha256_int_array(label_ids),
        "evaluation_ids_sha256": preflight.sha256_int_array(evaluation_ids),
        "validation_ids_sha256": preflight.sha256_int_array(splits["validation"]),
        "test_ids_sha256": preflight.sha256_int_array(splits["test"]),
        "physical_parameters": asdict(parameters),
        "mean_roughness_k40": mean_roughness,
        "residual_geometry": geometry,
        "metrics": {
            "physical_center": center_metrics,
            "center_plus_k40_image_ridge": corrected_metrics,
            "zero_residual": zero_residual_metrics,
            "k40_image_residual_ridge": image_residual_metrics,
        },
        "deltas": {
            "target_mean_abs_correlation": corrected_metrics["mean_abs_correlation"]
            - center_metrics["mean_abs_correlation"],
            "target_mean_r2": corrected_metrics["mean_r2"] - center_metrics["mean_r2"],
            "residual_mean_abs_correlation": image_residual_metrics["mean_abs_correlation"]
            - zero_residual_metrics["mean_abs_correlation"],
            "residual_mean_r2": image_residual_metrics["mean_r2"] - zero_residual_metrics["mean_r2"],
        },
        "directional_gates": gates,
        "residual_non_degenerate": residual_non_degenerate,
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
    output = args.output or Path(config["runtime"]["output"])
    run(args.config, output)


if __name__ == "__main__":
    main()
