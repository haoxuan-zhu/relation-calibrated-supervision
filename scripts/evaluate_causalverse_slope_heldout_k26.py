"""Perform the one-time held-out Slope evaluation from locked K23 assets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

import audit_causalverse_slope_analytic_closure as closure_audit
import audit_causalverse_slope_manifold_projection as projection_audit
import causalverse_slope_preflight as slope
import run_causalverse_slope_preflight as preflight


def primary_metrics(true: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    true = np.asarray(true, dtype=np.float64)[:, list(slope.RELATION_INDICES)]
    predicted = np.asarray(predicted, dtype=np.float64)[:, list(slope.RELATION_INDICES)]
    correlations = []
    for index in range(true.shape[1]):
        if np.std(true[:, index]) <= 1e-12 or np.std(predicted[:, index]) <= 1e-12:
            correlations.append(0.0)
        else:
            correlations.append(abs(float(np.corrcoef(true[:, index], predicted[:, index])[0, 1])))
    denominator = np.sum((true - true.mean(axis=0, keepdims=True)) ** 2, axis=0)
    r2 = 1.0 - np.sum((true - predicted) ** 2, axis=0) / np.maximum(denominator, 1e-12)
    return float(np.mean(correlations)), float(np.mean(r2))


def paired_bootstrap(
    true: np.ndarray,
    correct: np.ndarray,
    baseline: np.ndarray,
    resamples: int,
    seed: int,
    confidence: float,
) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    correlation_deltas = np.empty(resamples, dtype=np.float64)
    r2_deltas = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        rows = rng.integers(0, len(true), size=len(true))
        correct_corr, correct_r2 = primary_metrics(true[rows], correct[rows])
        baseline_corr, baseline_r2 = primary_metrics(true[rows], baseline[rows])
        correlation_deltas[index] = correct_corr - baseline_corr
        r2_deltas[index] = correct_r2 - baseline_r2
    alpha = (1.0 - confidence) / 2.0

    def summary(values: np.ndarray) -> dict[str, float]:
        return {
            "mean": float(values.mean()),
            "lower": float(np.quantile(values, alpha)),
            "upper": float(np.quantile(values, 1.0 - alpha)),
            "positive_fraction": float(np.mean(values > 0.0)),
        }

    return {"relation4_correlation": summary(correlation_deltas), "relation4_r2": summary(r2_deltas)}


def seed_flags(metrics: dict[str, dict[str, object]], maximum_free_drop: float) -> dict[str, bool]:
    correct = metrics["correct_relation"]
    point = metrics["point"]
    permuted = metrics["coefficient_permuted_relation"]
    projected = metrics["all_coordinate_manifold_projection"]

    def beats(baseline: dict[str, object]) -> bool:
        return bool(
            correct["mean_direct_abs_correlation_relation4"]
            > baseline["mean_direct_abs_correlation_relation4"]
            and correct["mean_direct_r2_relation4"] > baseline["mean_direct_r2_relation4"]
        )

    return {
        "correct_beats_point": beats(point),
        "correct_beats_permuted": beats(permuted),
        "correct_beats_projection": beats(projected),
        "free_safe_vs_point": bool(
            correct["mean_direct_abs_correlation_free3"]
            >= point["mean_direct_abs_correlation_free3"] - maximum_free_drop
        ),
    }


def aggregate_decision(valid: bool, flags: dict[str, dict[str, bool]]) -> str:
    if not valid:
        return "invalid"
    values = list(flags.values())
    point_all = all(item["correct_beats_point"] and item["free_safe_vs_point"] for item in values)
    permuted_all = all(item["correct_beats_permuted"] for item in values)
    projection_all = all(item["correct_beats_projection"] for item in values)
    if point_all and permuted_all and projection_all:
        return "heldout_relation_content_and_projection_confirmed"
    if point_all and permuted_all:
        return "heldout_relation_content_confirmed_projection_not_beaten"
    if point_all:
        return "heldout_relation_regularization_only"
    return "heldout_relation_effect_not_confirmed"


def run(protocol_path: Path, output_path: Path, device: torch.device) -> None:
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    parent = protocol["parent"]
    parent_config_path = Path(parent["config"])
    confirmation_path = Path(parent["confirmation_result"])
    manifold_path = Path(parent["manifold_validation_audit"])
    audit_path = Path(parent["contract_audit"])
    checkpoint_root = Path(parent["checkpoint_root"])
    dependencies = {
        "parent_config": preflight.sha256_file(parent_config_path) == parent["config_sha256"],
        "confirmation_result": preflight.sha256_file(confirmation_path)
        == parent["confirmation_result_sha256"],
        "manifold_validation_audit": preflight.sha256_file(manifold_path)
        == parent["manifold_validation_audit_sha256"],
    }
    confirmation = json.loads(confirmation_path.read_text(encoding="utf-8"))
    manifold = json.loads(manifold_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    dependencies.update(
        {
            "confirmation_valid_and_test_unread": bool(
                confirmation["valid"]
                and confirmation["machine_decision"] == "multiseed_relation_content_confirmed"
                and not confirmation["test_evaluated"]
            ),
            "manifold_valid_and_test_unread": bool(
                manifold["valid"]
                and manifold["machine_decision"] == "training_beyond_manifold_projection_supported"
                and not manifold["test_evaluated"]
            ),
            "parent_contract_valid": bool(audit["valid"]),
        }
    )
    if not all(dependencies.values()):
        raise ValueError(f"K26 dependency contract failed: {dependencies}")

    config, grouped, splits, label_ids, filter_audit = preflight.load_context(parent_config_path)
    seeds = [int(value) for value in protocol["evaluation"]["seeds"]]
    conditions = tuple(protocol["evaluation"]["conditions"])
    if seeds != [int(value) for value in config["training"]["seeds"]]:
        raise ValueError("K26 seeds differ from K23")
    if set(conditions) != set(slope.CONDITIONS):
        raise ValueError("K26 conditions differ from K23")

    config_sha = preflight.sha256_file(parent_config_path)
    feature_sha = preflight.sha256_file(Path(config["runtime"]["feature_cache"]))
    source_sha = {
        "core": preflight.sha256_file(Path(slope.__file__)),
        "runner": preflight.sha256_file(Path(preflight.__file__)),
    }
    assets: dict[tuple[int, str], Path] = {}
    lock_checks: list[bool] = []
    for seed in seeds:
        for condition in conditions:
            directory = checkpoint_root / f"seed{seed}" / condition
            lock_path = directory / "training_lock.json"
            checkpoint_path = directory / "checkpoint.pt"
            if not lock_path.is_file() or not checkpoint_path.is_file():
                raise FileNotFoundError(f"missing K23 asset: seed={seed} condition={condition}")
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
    if not all(lock_checks) or len(lock_checks) != 9:
        raise ValueError("K23 checkpoint lock contract failed before held-out read")

    train = slope.select_ids(grouped, splits["train"])
    label_indices = slope.ordered_label_indices(train.ids, label_ids)
    labels = train.latents[label_indices]
    parameters = slope.fit_relation_parameters(labels)
    coordinate_scale = labels.std(axis=0)
    lower = labels[:, :3].min(axis=0)
    upper = labels[:, :3].max(axis=0)

    # This is the single semantic read of the previously sealed Slope test split.
    heldout = slope.select_ids(grouped, splits[protocol["evaluation"]["split"]])
    metrics: dict[str, dict[str, dict[str, object]]] = {}
    deltas: dict[str, dict[str, dict[str, float]]] = {}
    flags: dict[str, dict[str, bool]] = {}
    bootstrap: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    projection_diagnostics: dict[str, dict[str, object]] = {}
    bootstrap_config = protocol["bootstrap"]

    for seed_index, seed in enumerate(seeds):
        predictions: dict[str, np.ndarray] = {}
        for condition in conditions:
            checkpoint = torch.load(assets[(seed, condition)], map_location=device, weights_only=False)
            model = slope.build_head(
                int(config["backbone"]["feature_dim"]), config["model"]["hidden_dims"], seed, device
            )
            model.load_state_dict(checkpoint["state_dict"])
            predictions[condition] = slope.predict_physical(
                model,
                heldout,
                int(config["backbone"]["feature_dim"]),
                np.asarray(checkpoint["normalization_mean"]),
                np.asarray(checkpoint["normalization_std"]),
                device,
            )
        predictions["free_variable_substitution"] = closure_audit.analytic_closure(
            predictions["point"], parameters
        )
        predictions["all_coordinate_manifold_projection"], projection_info = (
            projection_audit.project_to_manifold(
                predictions["point"], parameters, coordinate_scale, lower, upper
            )
        )
        seed_key = str(seed)
        projection_diagnostics[seed_key] = projection_info
        seed_metrics = {
            name: slope.evaluate_predictions(heldout.latents, predicted)
            for name, predicted in predictions.items()
        }
        metrics[seed_key] = seed_metrics
        flags[seed_key] = seed_flags(
            seed_metrics, float(protocol["evaluation"]["maximum_free_correlation_drop"])
        )
        correct = seed_metrics["correct_relation"]
        deltas[seed_key] = {}
        bootstrap[seed_key] = {}
        comparisons = {
            "correct_minus_point": "point",
            "correct_minus_coefficient_permuted_relation": "coefficient_permuted_relation",
            "correct_minus_all_coordinate_manifold_projection": "all_coordinate_manifold_projection",
        }
        for comparison_index, (comparison, baseline_name) in enumerate(comparisons.items()):
            baseline = seed_metrics[baseline_name]
            deltas[seed_key][comparison] = {
                "relation4_correlation": float(
                    correct["mean_direct_abs_correlation_relation4"]
                    - baseline["mean_direct_abs_correlation_relation4"]
                ),
                "relation4_r2": float(
                    correct["mean_direct_r2_relation4"] - baseline["mean_direct_r2_relation4"]
                ),
                "free3_correlation": float(
                    correct["mean_direct_abs_correlation_free3"]
                    - baseline["mean_direct_abs_correlation_free3"]
                ),
            }
            bootstrap[seed_key][comparison] = paired_bootstrap(
                heldout.latents,
                predictions["correct_relation"],
                predictions[baseline_name],
                int(bootstrap_config["resamples"]),
                int(bootstrap_config["seed"]) + 100 * seed_index + comparison_index,
                float(bootstrap_config["confidence"]),
            )

    validity = {
        **dependencies,
        "all_nine_checkpoint_locks_match": bool(all(lock_checks) and len(lock_checks) == 9),
        "incomplete_product_filter_contract_matches": bool(filter_audit["contract_matches"]),
        "all_heldout_projections_converged": bool(
            all(item["success_count"] == item["row_count"] for item in projection_diagnostics.values())
        ),
        "heldout_split_nonempty": len(heldout.ids) > 0,
    }
    valid = bool(all(validity.values()))
    decision = aggregate_decision(valid, flags)
    payload = {
        "protocol_version": protocol["protocol_version"],
        "fact_type": "one_time_slope_heldout_confirmation",
        "valid": valid,
        "validity": validity,
        "machine_decision": decision,
        "heldout_id_count": int(len(heldout.ids)),
        "heldout_ids_sha256": preflight.sha256_int_array(heldout.ids),
        "seed_flags": flags,
        "metrics_by_seed": metrics,
        "deltas_by_seed": deltas,
        "paired_bootstrap": bootstrap,
        "projection_diagnostics": projection_diagnostics,
        "protocol_sha256": preflight.sha256_file(protocol_path),
        "evaluator_sha256": preflight.sha256_file(Path(__file__)),
        "validation_evaluated_in_k26": False,
        "test_evaluated": True,
        "test_read_event": "K26_single_locked_read",
    }
    preflight.write_json(output_path, payload)
    print(json.dumps({"valid": valid, "machine_decision": decision, "output": str(output_path)}))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    run(args.protocol, args.output, device)
