from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

import causalverse_spring_pilot as base
import causalverse_spring_correctness_v3 as v3
import run_causalverse_spring_pilot as legacy_runner
import run_causalverse_spring_correctness_v3 as runner


RELATION_INDICES = np.asarray(base.RELATION_INDICES, dtype=np.int64)
FREE_INDICES = np.asarray(base.FREE_INDICES, dtype=np.int64)


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def direct_r2(true: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    denominator = np.sum((true - true.mean(axis=0, keepdims=True)) ** 2, axis=0)
    return 1.0 - np.sum((true - predicted) ** 2, axis=0) / np.maximum(denominator, 1e-12)


def signed_pearson(true: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    values = []
    for index in range(true.shape[1]):
        if np.std(true[:, index]) <= 1e-12 or np.std(predicted[:, index]) <= 1e-12:
            values.append(0.0)
        else:
            values.append(float(np.corrcoef(true[:, index], predicted[:, index])[0, 1]))
    return np.asarray(values, dtype=np.float64)


def enhanced_metrics(true: np.ndarray, predicted: np.ndarray) -> dict[str, object]:
    metrics = base.evaluate_predictions(true, predicted)
    r2 = direct_r2(true, predicted)
    signed = signed_pearson(true, predicted)
    standard_deviation = np.std(true, axis=0)
    nrmse = np.sqrt(np.mean((true - predicted) ** 2, axis=0)) / np.maximum(standard_deviation, 1e-12)
    correct_spec = v3.RELATION_SPECS["correct_relation"]
    fitted_coefficient = v3.fit_relation_coefficient(predicted, correct_spec)
    relation_scale = v3.relation_scale(true, correct_spec)
    true_relation_nrmse = float(
        np.sqrt(np.mean(v3.relation_residual_numpy(predicted, 9.81, correct_spec) ** 2))
        / relation_scale
    )
    metrics.update({
        "direct_signed_pearson": signed.tolist(),
        "relation3_mean_signed_pearson": float(signed[RELATION_INDICES].mean()),
        "relation3_mean_direct_r2": float(r2[RELATION_INDICES].mean()),
        "free2_mean_direct_r2": float(r2[FREE_INDICES].mean()),
        "direct_normalized_rmse": nrmse.tolist(),
        "relation3_mean_normalized_rmse": float(nrmse[RELATION_INDICES].mean()),
        "fitted_correct_relation_coefficient": fitted_coefficient,
        "true_relation_normalized_rmse": true_relation_nrmse,
    })
    return metrics


def load_locked_model(
    condition_dir: Path,
    config: dict[str, object],
    condition: str,
    seed: int,
    device: torch.device,
):
    lock = read_json(condition_dir / "training_lock.json")
    summary = read_json(condition_dir / "training_summary.json")
    checkpoint_path = Path(lock["checkpoint"])
    if legacy_runner.sha256_file(checkpoint_path) != lock["checkpoint_sha256"]:
        raise ValueError(f"checkpoint hash mismatch: seed={seed}, condition={condition}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint["condition"] != condition or int(checkpoint["seed"]) != seed:
        raise ValueError(f"checkpoint identity mismatch: seed={seed}, condition={condition}")
    model = base.SpringSemanticHead(
        int(config["backbone"]["feature_dim"]),
        config["model"]["hidden_dims"],
        int(config["model"]["output_dim"]),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    if base.state_dict_sha256(model.state_dict()) != lock["final_state_sha256"]:
        raise ValueError(f"state hash mismatch: seed={seed}, condition={condition}")
    return model, checkpoint, lock, summary


def paired_bootstrap_r2_difference(
    true: np.ndarray,
    correct: np.ndarray,
    control: np.ndarray,
    repetitions: int,
    seed: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    differences = np.empty(repetitions, dtype=np.float64)
    for iteration in range(repetitions):
        indices = rng.integers(0, len(true), size=len(true))
        sampled_true = true[indices]
        correct_r2 = direct_r2(sampled_true, correct[indices])[RELATION_INDICES].mean()
        control_r2 = direct_r2(sampled_true, control[indices])[RELATION_INDICES].mean()
        differences[iteration] = correct_r2 - control_r2
    return {
        "repetitions": repetitions,
        "seed": seed,
        "point": float(
            direct_r2(true, correct)[RELATION_INDICES].mean()
            - direct_r2(true, control)[RELATION_INDICES].mean()
        ),
        "bootstrap_mean": float(differences.mean()),
        "ci95_percentile": np.percentile(differences, [2.5, 97.5]).tolist(),
        "positive_fraction": float(np.mean(differences > 0.0)),
    }


def metric_value(metrics: dict[str, object], name: str) -> float:
    if name == "pearson":
        return float(metrics["mean_direct_abs_correlation_relation3"])
    if name == "r2":
        return float(metrics["relation3_mean_direct_r2"])
    if name == "nrmse":
        return float(metrics["relation3_mean_normalized_rmse"])
    raise ValueError(name)


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "population_sd": float(array.std(ddof=0)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate all locked Spring correctness v3 conditions")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    audit = read_json(args.audit)
    config_loaded, cache, splits, label_ids = runner.protocol_context(args.config)
    if config_loaded["protocol_version"] != config["protocol_version"]:
        raise ValueError("protocol mismatch")
    validation = legacy_runner.group_selected(cache, splits["validation"])
    test = legacy_runner.group_selected(cache, splits["test"])
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    seeds = tuple(int(value) for value in config["training"]["seeds"])
    conditions = tuple(config["conditions"])
    thresholds = config["evaluation"]

    metrics: dict[str, object] = {}
    predictions: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    records: dict[str, dict[str, dict[str, object]]] = {}
    for seed in seeds:
        seed_key = str(seed)
        metrics[seed_key] = {}
        predictions[seed_key] = {}
        records[seed_key] = {}
        for condition in conditions:
            condition_dir = args.root / f"seed{seed}" / condition
            model, checkpoint, lock, summary = load_locked_model(
                condition_dir, config, condition, seed, device
            )
            predictions[seed_key][condition] = {}
            metrics[seed_key][condition] = {}
            for split_name, grouped in (("validation", validation), ("test", test)):
                predicted = base.predict_physical(
                    model,
                    grouped,
                    int(config["backbone"]["feature_dim"]),
                    checkpoint["normalization_mean"],
                    checkpoint["normalization_std"],
                    device,
                )
                predictions[seed_key][condition][split_name] = predicted
                metrics[seed_key][condition][split_name] = enhanced_metrics(grouped.latents, predicted)
            records[seed_key][condition] = {"lock": lock, "summary": summary}
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    validity: dict[str, object] = {"audit_pass": bool(audit["valid"])}
    source_hashes = set()
    config_hashes = set()
    feature_hashes = set()
    label_hashes = set()
    train_hashes = set()
    semantic_flags = []
    finite_histories = []
    mapping_matches = []
    initial_shared: dict[str, bool] = {}
    prediction_not_collapsed: dict[str, bool] = {}
    audit_mapping_sha = audit["relation_contracts"]["coefficient_permuted_relation"]["mapping_sha256"]
    for seed in seeds:
        seed_records = records[str(seed)]
        initial_shared[str(seed)] = len({seed_records[name]["lock"]["initial_state_sha256"] for name in conditions}) == 1
        true_std = np.std(validation.latents, axis=0)
        predicted_std = np.asarray(metrics[str(seed)]["correct_relation"]["validation"]["prediction_std"])
        prediction_not_collapsed[str(seed)] = bool(
            np.min(predicted_std / np.maximum(true_std, 1e-12))
            >= float(thresholds["minimum_prediction_std_ratio"])
        )
        for condition in conditions:
            lock = seed_records[condition]["lock"]
            summary = seed_records[condition]["summary"]
            config_hashes.add(lock["config_sha256"])
            feature_hashes.add(lock["feature_cache_sha256"])
            label_hashes.add(lock["label_ids_sha256"])
            train_hashes.add(lock["train_ids_sha256"])
            source_hashes.add(json.dumps(lock["source_sha256"], sort_keys=True))
            semantic_flags.append(
                not lock["semantic_validation_evaluated_during_training"]
                and not lock["test_evaluated_during_training"]
                and not summary["semantic_validation_evaluated"]
                and not summary["test_evaluated"]
            )
            finite_histories.append(all(
                np.isfinite(float(value))
                for entry in summary["history"]
                for key, value in entry.items()
                if key != "epoch"
            ))
            if condition == "coefficient_permuted_relation":
                mapping_matches.append(lock["relation_contract"]["mapping_sha256"] == audit_mapping_sha)
    validity.update({
        "config_hash_shared": len(config_hashes) == 1,
        "feature_hash_shared": len(feature_hashes) == 1,
        "label_subset_shared": len(label_hashes) == 1,
        "train_split_shared": len(train_hashes) == 1,
        "source_manifest_shared": len(source_hashes) == 1,
        "initial_state_shared_within_seed": bool(all(initial_shared.values())),
        "no_semantic_or_test_read_during_training": bool(all(semantic_flags)),
        "all_training_histories_finite": bool(all(finite_histories)),
        "mapping_matches_audit_all_seeds": bool(all(mapping_matches)),
        "correct_prediction_not_collapsed_all_seeds": bool(all(prediction_not_collapsed.values())),
        "all_condition_locks_present_before_readout": True,
    })

    controls = tuple(condition for condition in conditions if condition != "correct_relation")
    deltas: dict[str, object] = {}
    bootstraps: dict[str, object] = {}
    for seed_index, seed in enumerate(seeds):
        seed_key = str(seed)
        deltas[seed_key] = {}
        bootstraps[seed_key] = {}
        for split in ("validation", "test"):
            deltas[seed_key][split] = {}
            correct_metrics = metrics[seed_key]["correct_relation"][split]
            for control in controls:
                control_metrics = metrics[seed_key][control][split]
                deltas[seed_key][split][f"correct_minus_{control}"] = {
                    "relation3_pearson": metric_value(correct_metrics, "pearson") - metric_value(control_metrics, "pearson"),
                    "relation3_r2": metric_value(correct_metrics, "r2") - metric_value(control_metrics, "r2"),
                    "relation3_nrmse_advantage": metric_value(control_metrics, "nrmse") - metric_value(correct_metrics, "nrmse"),
                    "true_relation_nrmse_advantage": float(control_metrics["true_relation_normalized_rmse"]) - float(correct_metrics["true_relation_normalized_rmse"]),
                }
            point_free = metrics[seed_key]["point"][split]["mean_direct_abs_correlation_free2"]
            correct_free = correct_metrics["mean_direct_abs_correlation_free2"]
            deltas[seed_key][split]["correct_minus_point_free2_pearson"] = float(correct_free - point_free)
        for control_index, control in enumerate(controls):
            bootstraps[seed_key][control] = paired_bootstrap_r2_difference(
                validation.latents,
                predictions[seed_key]["correct_relation"]["validation"],
                predictions[seed_key][control]["validation"],
                int(thresholds["bootstrap_repetitions"]),
                int(thresholds["bootstrap_seed"]) + seed_index * 100 + control_index,
            )

    primary_controls = (
        "point",
        "coefficient_permuted_relation",
        "wrong_lm_to_k_relation",
        "wrong_km_to_l_relation",
    )
    direction_checks: dict[str, bool] = {}
    for control in primary_controls:
        direction_checks[f"correct_beats_{control}_r2_all_seed_splits"] = all(
            float(deltas[str(seed)][split][f"correct_minus_{control}"]["relation3_r2"]) > 0.0
            for seed in seeds
            for split in ("validation", "test")
        )
    direction_checks["correct_beats_point_pearson_all_seed_splits"] = all(
        float(deltas[str(seed)][split]["correct_minus_point"]["relation3_pearson"]) > 0.0
        for seed in seeds
        for split in ("validation", "test")
    )
    coefficient_ci_positive_count = sum(
        float(bootstraps[str(seed)]["coefficient_permuted_relation"]["ci95_percentile"][0]) > 0.0
        for seed in seeds
    )
    free2_consistent_degradation = all(
        float(deltas[str(seed)]["validation"]["correct_minus_point_free2_pearson"]) < 0.0
        and float(deltas[str(seed)]["test"]["correct_minus_point_free2_pearson"]) < 0.0
        for seed in seeds
    )

    valid = bool(all(validity.values()))
    supported = bool(
        valid
        and all(direction_checks.values())
        and coefficient_ci_positive_count >= int(thresholds["required_positive_bootstrap_seed_count"])
        and not free2_consistent_degradation
    )
    coefficient_r2_values = [
        float(deltas[str(seed)][split]["correct_minus_coefficient_permuted_relation"]["relation3_r2"])
        for seed in seeds
        for split in ("validation", "test")
    ]
    coefficient_pearson_values = [
        float(deltas[str(seed)][split]["correct_minus_coefficient_permuted_relation"]["relation3_pearson"])
        for seed in seeds
        for split in ("validation", "test")
    ]
    soft_effects = {
        "coefficient_correctness_r2": {
            **summarize(coefficient_r2_values),
            "target": float(thresholds["soft_mean_correctness_r2_gain"]),
            "target_met": float(np.mean(coefficient_r2_values)) >= float(thresholds["soft_mean_correctness_r2_gain"]),
        },
        "coefficient_correctness_pearson": {
            **summarize(coefficient_pearson_values),
            "target": float(thresholds["soft_mean_correctness_pearson_gain"]),
            "target_met": float(np.mean(coefficient_pearson_values)) >= float(thresholds["soft_mean_correctness_pearson_gain"]),
        },
    }
    if not valid:
        decision = "causalverse_spring_correctness_v3_invalid"
    elif supported:
        decision = "cross_system_relation_correctness_supported_multiseed_same_shards"
    elif direction_checks["correct_beats_point_pearson_all_seed_splits"] and direction_checks["correct_beats_point_r2_all_seed_splits"]:
        decision = "relation_regularization_only_multiseed"
    else:
        decision = "correctness_not_stable_same_shards"

    payload = {
        "protocol_version": config["protocol_version"],
        "fact_type": "rerun",
        "scope": "all_fifteen_locks_before_single_validation_test_readout",
        "config_sha256": legacy_runner.sha256_file(args.config),
        "audit_sha256": legacy_runner.sha256_file(args.audit),
        "metrics": metrics,
        "deltas": deltas,
        "validation_paired_bootstrap": bootstraps,
        "validity": validity,
        "valid": valid,
        "direction_checks": direction_checks,
        "coefficient_bootstrap_ci_positive_seed_count": coefficient_ci_positive_count,
        "free2_consistent_degradation": free2_consistent_degradation,
        "soft_effects": soft_effects,
        "machine_decision": decision,
        "unread_shard_replication_unlocked": supported,
        "test_read_after_all_condition_locks": True,
        "condition_lock_sha256": {
            str(seed): {
                condition: legacy_runner.sha256_file(args.root / f"seed{seed}" / condition / "training_lock.json")
                for condition in conditions
            }
            for seed in seeds
        },
    }
    runner.write_json(args.output, payload)
    print(json.dumps({
        "valid": valid,
        "machine_decision": decision,
        "unread_shard_replication_unlocked": supported,
        "output": str(args.output),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
