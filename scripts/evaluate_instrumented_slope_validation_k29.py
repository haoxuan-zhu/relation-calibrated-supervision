"""Locked validation read for Instrumented Slope K29."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

import audit_instrumented_slope_geometry_k28 as k28
import audit_instrumented_slope_interface_k27 as k27
import causalverse_slope_preflight as slope
import run_causalverse_slope_preflight as preflight


def paired_bootstrap(
    truth: np.ndarray,
    candidate: np.ndarray,
    baseline: np.ndarray,
    repetitions: int,
    seed: int,
    confidence: float,
) -> dict[str, Any]:
    y = np.asarray(truth, dtype=np.float64)
    first = np.asarray(candidate, dtype=np.float64)
    second = np.asarray(baseline, dtype=np.float64)
    if y.shape != first.shape or y.shape != second.shape or len(y) < 2:
        raise ValueError("bootstrap arrays must be matched and nontrivial")
    if repetitions < 100 or not 0.0 < confidence < 1.0:
        raise ValueError("invalid bootstrap contract")
    generator = np.random.default_rng(seed)
    correlation_deltas = np.empty(repetitions, dtype=np.float64)
    r2_deltas = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        rows = generator.integers(0, len(y), size=len(y))
        candidate_metrics = k27.direct_metrics(y[rows], first[rows])
        baseline_metrics = k27.direct_metrics(y[rows], second[rows])
        correlation_deltas[index] = (
            candidate_metrics["mean_abs_correlation"]
            - baseline_metrics["mean_abs_correlation"]
        )
        r2_deltas[index] = candidate_metrics["mean_r2"] - baseline_metrics["mean_r2"]
    tail = (1.0 - confidence) / 2.0
    return {
        "repetitions": int(repetitions),
        "seed": int(seed),
        "confidence": float(confidence),
        "mean_abs_correlation_difference": {
            "mean": float(correlation_deltas.mean()),
            "lower": float(np.quantile(correlation_deltas, tail)),
            "upper": float(np.quantile(correlation_deltas, 1.0 - tail)),
        },
        "mean_r2_difference": {
            "mean": float(r2_deltas.mean()),
            "lower": float(np.quantile(r2_deltas, tail)),
            "upper": float(np.quantile(r2_deltas, 1.0 - tail)),
        },
    }


def comparison_key(candidate: str, baseline: str) -> str:
    return f"{candidate}_minus_{baseline}"


def confirmed(delta: dict[str, float], bootstrap: dict[str, Any]) -> bool:
    return bool(
        delta["mean_abs_correlation"] > 0.0
        and delta["mean_r2"] > 0.0
        and bootstrap["mean_abs_correlation_difference"]["lower"] > 0.0
        and bootstrap["mean_r2_difference"]["lower"] > 0.0
    )


def run(config_path: Path, output_path: Path) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    geometry_config_path = Path(config["source_contract"]["geometry_config"])
    geometry_result_path = Path(config["source_contract"]["geometry_result"])
    geometry_config = yaml.safe_load(geometry_config_path.read_text(encoding="utf-8"))
    geometry_result = json.loads(geometry_result_path.read_text(encoding="utf-8"))
    interface_config_path = Path(geometry_config["source_contract"]["interface_config"])
    interface_config, grouped, splits, label_ids, filter_audit = preflight.load_context(
        interface_config_path
    )
    train = slope.select_ids(grouped, splits["train"])
    validation = slope.select_ids(grouped, splits["validation"])
    label_rows = slope.ordered_label_indices(train.ids, label_ids)
    labels = train.latents[label_rows]
    parameters, mean_roughness, target_scale, geometry = k28.leave_one_out_geometry(
        labels, float(geometry_config["calibration"]["coverage"])
    )

    train_side = train.latents[:, k27.SIDE_INDICES]
    train_targets = train.latents[:, k27.TARGET_INDICES]
    train_center = k28.physical_state_from_roughness(train_side, mean_roughness, parameters)
    train_residual = train_targets - train_center
    train_features = train.features.mean(axis=1, dtype=np.float64)
    ridge = k27.fit_standardized_dual_ridge(
        train_features[label_rows],
        train_residual[label_rows],
        float(geometry_config["raw_predictor"]["ridge_alpha"]),
    )

    side = validation.latents[:, k27.SIDE_INDICES]
    truth = validation.latents[:, k27.TARGET_INDICES]
    center = k28.physical_state_from_roughness(side, mean_roughness, parameters)
    features = validation.features.mean(axis=1, dtype=np.float64)
    predicted_residual = k27.predict_standardized_dual_ridge(ridge, features)
    normalized_raw = predicted_residual / target_scale[None, :]
    isotropic_normalized, isotropic_audit = k28.clip_rows_to_radius(
        normalized_raw, float(geometry["isotropic"]["radius"])
    )

    roughness_raw = mean_roughness + predicted_residual[:, 0]
    physical_lower, physical_upper = (
        float(value) for value in geometry_config["calibration"]["roughness_physical_bounds"]
    )
    roughness_domain = np.clip(roughness_raw, physical_lower, physical_upper)
    scalar_span = float(geometry["roughness_scalar"]["radius"]) * float(target_scale[0])
    rank_lower = max(physical_lower, mean_roughness - scalar_span)
    rank_upper = min(physical_upper, mean_roughness + scalar_span)

    predictions = {
        "physical_center": center,
        "unbounded_five_residual": center + predicted_residual,
        "isotropic_rank_tube": center + isotropic_normalized * target_scale[None, :],
        "formula_physical_bounds": k28.physical_state_from_roughness(
            side, roughness_domain, parameters
        ),
    }
    curve, curve_audit = k28.nonlinear_curve_projection(
        predictions["unbounded_five_residual"],
        side,
        parameters,
        target_scale,
        rank_lower,
        rank_upper,
        float(geometry_config["projection"]["nonlinear_tolerance"]),
        int(geometry_config["projection"]["nonlinear_max_iterations"]),
    )
    predictions["nonlinear_curve_rank_tube"] = curve
    if list(predictions) != list(config["candidates"]):
        raise ValueError("K29 candidate registry drifted")
    metrics = {name: k27.direct_metrics(truth, prediction) for name, prediction in predictions.items()}

    repetitions = int(config["bootstrap"]["repetitions"])
    bootstrap_seed = int(config["bootstrap"]["seed"])
    confidence = float(config["bootstrap"]["confidence"])
    deltas: dict[str, dict[str, float]] = {}
    bootstraps: dict[str, dict[str, Any]] = {}
    for candidate, baseline in config["comparisons"]:
        key = comparison_key(candidate, baseline)
        deltas[key] = {
            "mean_abs_correlation": metrics[candidate]["mean_abs_correlation"]
            - metrics[baseline]["mean_abs_correlation"],
            "mean_r2": metrics[candidate]["mean_r2"] - metrics[baseline]["mean_r2"],
        }
        bootstraps[key] = paired_bootstrap(
            truth,
            predictions[candidate],
            predictions[baseline],
            repetitions,
            bootstrap_seed,
            confidence,
        )

    isotropic_key = comparison_key("isotropic_rank_tube", "unbounded_five_residual")
    curve_formula_key = comparison_key("nonlinear_curve_rank_tube", "formula_physical_bounds")
    isotropic_confirmed = confirmed(deltas[isotropic_key], bootstraps[isotropic_key])
    curve_confirmed = confirmed(deltas[curve_formula_key], bootstraps[curve_formula_key])
    isotropic_directional = all(value > 0.0 for value in deltas[isotropic_key].values())
    curve_directional = all(value > 0.0 for value in deltas[curve_formula_key].values())

    validation_ids = set(int(value) for value in validation.ids)
    validity = {
        "geometry_config_sha_matches": preflight.sha256_file(geometry_config_path)
        == str(config["source_contract"]["geometry_config_sha256"]),
        "geometry_result_sha_matches": preflight.sha256_file(geometry_result_path)
        == str(config["source_contract"]["geometry_result_sha256"]),
        "geometry_result_valid": bool(geometry_result["valid"]),
        "geometry_result_train_only": not bool(geometry_result["semantic_validation_evaluated"])
        and not bool(geometry_result["test_evaluated"]),
        "feature_cache_sha_matches": preflight.sha256_file(Path(interface_config["runtime"]["feature_cache"]))
        == str(interface_config["runtime"]["feature_cache_sha256"]),
        "incomplete_product_filter_contract_matches": bool(filter_audit["contract_matches"]),
        "validation_ids_exact": validation_ids == set(int(value) for value in splits["validation"]),
        "train_ids_not_evaluated": not bool(validation_ids & set(int(value) for value in splits["train"])),
        "test_ids_not_evaluated": not bool(validation_ids & set(int(value) for value in splits["test"])),
        "candidate_registry_exact": list(predictions) == list(config["candidates"]),
        "all_predictions_finite": all(np.isfinite(value).all() for value in predictions.values()),
        "nonlinear_projection_complete": curve_audit["success_count"] == len(validation.ids),
    }
    valid = bool(all(validity.values()))
    if not valid:
        decision = "invalid"
    elif isotropic_confirmed and curve_confirmed:
        decision = "isotropic_tube_and_nonlinear_curve_validation_confirmed"
    elif isotropic_confirmed:
        decision = "isotropic_tube_validation_confirmed"
    elif curve_confirmed:
        decision = "nonlinear_curve_beyond_formula_validation_confirmed"
    elif isotropic_directional or curve_directional:
        decision = "directional_validation_signal_without_joint_ci_confirmation"
    else:
        decision = "constructed_geometry_validation_not_confirmed"

    payload = {
        "protocol_version": config["protocol_version"],
        "fact_type": "locked_constructed_interface_validation",
        "scope": "registered_validation_ids_only_test_sealed",
        "config_sha256": preflight.sha256_file(config_path),
        "geometry_config_sha256": preflight.sha256_file(geometry_config_path),
        "geometry_result_sha256": preflight.sha256_file(geometry_result_path),
        "feature_cache_sha256": preflight.sha256_file(Path(interface_config["runtime"]["feature_cache"])),
        "validation_count": int(len(validation.ids)),
        "validation_ids_sha256": preflight.sha256_int_array(validation.ids),
        "test_ids_sha256": preflight.sha256_int_array(splits["test"]),
        "calibration_geometry": geometry,
        "roughness_rank_interval": [rank_lower, rank_upper],
        "projection_audits": {
            "isotropic_rank_tube": isotropic_audit,
            "nonlinear_curve_rank_tube": curve_audit,
        },
        "metrics": metrics,
        "paired_deltas": deltas,
        "paired_bootstrap": bootstraps,
        "decision_checks": {
            "isotropic_directional_both": isotropic_directional,
            "isotropic_joint_ci_confirmed": isotropic_confirmed,
            "nonlinear_curve_directional_both_vs_formula": curve_directional,
            "nonlinear_curve_joint_ci_confirmed_vs_formula": curve_confirmed,
        },
        "validity": validity,
        "valid": valid,
        "semantic_validation_evaluated": True,
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
