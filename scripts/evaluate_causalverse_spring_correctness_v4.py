"""Formal one-shot readout for the frozen Spring unread-shard v4 protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

import causalverse_spring_pilot as base
import evaluate_causalverse_spring_correctness_v3 as metric_impl
import run_causalverse_spring_correctness_v3 as runner
import run_causalverse_spring_pilot as legacy_runner


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "population_sd": float(array.std(ddof=0)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def make_decision(
    valid: bool,
    point_positive_count: int,
    coefficient_positive_count: int,
    coefficient_split_means_positive: bool,
    coefficient_ci_positive_count: int,
    free2_consistent_degradation: bool,
    relation_residual_means_better: bool,
    topology_all_positive: bool,
    thresholds: dict[str, object],
) -> tuple[str, bool]:
    if not valid:
        return "causalverse_spring_correctness_v4_invalid", False
    primary = bool(
        point_positive_count
        >= int(thresholds["required_correct_minus_point_positive_seed_splits"])
        and coefficient_positive_count
        >= int(thresholds["required_correct_minus_coefficient_positive_seed_splits"])
        and coefficient_split_means_positive
        and coefficient_ci_positive_count
        >= int(thresholds["required_positive_coefficient_validation_bootstrap_seeds"])
        and not free2_consistent_degradation
        and relation_residual_means_better
    )
    if primary and topology_all_positive:
        return (
            "unread_shard_primary_replication_supported_topology_supported_pending_state_disjoint",
            True,
        )
    if primary:
        return "unread_shard_primary_replication_supported_pending_state_disjoint", True
    if point_positive_count >= int(
        thresholds["required_correct_minus_point_positive_seed_splits"]
    ):
        return "unread_shard_relation_regularization_only", False
    return "unread_shard_replication_not_supported", False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
    loaded_config, cache, splits, _ = runner.protocol_context(args.config)
    if loaded_config["protocol_version"] != config["protocol_version"]:
        raise ValueError("protocol version mismatch")
    validation = legacy_runner.group_selected(cache, splits["validation"])
    test = legacy_runner.group_selected(cache, splits["test"])
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    seeds = tuple(int(seed) for seed in config["training"]["seeds"])
    conditions = tuple(str(name) for name in config["conditions"])
    thresholds = config["evaluation"]

    metrics: dict[str, object] = {}
    predictions: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    records: dict[str, object] = {}
    for seed in seeds:
        seed_key = str(seed)
        metrics[seed_key] = {}
        predictions[seed_key] = {}
        records[seed_key] = {}
        for condition in conditions:
            condition_dir = args.root / f"seed{seed}" / condition
            model, checkpoint, lock, summary = metric_impl.load_locked_model(
                condition_dir, config, condition, seed, device
            )
            metrics[seed_key][condition] = {}
            predictions[seed_key][condition] = {}
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
                metrics[seed_key][condition][split_name] = metric_impl.enhanced_metrics(
                    grouped.latents, predicted
                )
            records[seed_key][condition] = {"lock": lock, "summary": summary}
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    validity: dict[str, object] = {"audit_pass": bool(audit["valid"])}
    source_hashes: set[str] = set()
    config_hashes: set[str] = set()
    feature_hashes: set[str] = set()
    label_hashes: set[str] = set()
    train_hashes: set[str] = set()
    semantic_flags: list[bool] = []
    finite_histories: list[bool] = []
    mapping_matches: list[bool] = []
    initial_shared: dict[str, bool] = {}
    prediction_not_collapsed: dict[str, bool] = {}
    audit_mapping_sha = audit["relation_contracts"]["coefficient_permuted_relation"][
        "mapping_sha256"
    ]
    for seed in seeds:
        seed_key = str(seed)
        seed_records = records[seed_key]
        initial_shared[seed_key] = len(
            {seed_records[name]["lock"]["initial_state_sha256"] for name in conditions}
        ) == 1
        true_std = np.std(validation.latents, axis=0)
        predicted_std = np.asarray(
            metrics[seed_key]["correct_relation"]["validation"]["prediction_std"]
        )
        prediction_not_collapsed[seed_key] = bool(
            np.min(predicted_std / np.maximum(true_std, 1e-12))
            >= float(thresholds["minimum_prediction_std_ratio"])
        )
        for condition in conditions:
            lock = seed_records[condition]["lock"]
            summary = seed_records[condition]["summary"]
            source_hashes.add(json.dumps(lock["source_sha256"], sort_keys=True))
            config_hashes.add(lock["config_sha256"])
            feature_hashes.add(lock["feature_cache_sha256"])
            label_hashes.add(lock["label_ids_sha256"])
            train_hashes.add(lock["train_ids_sha256"])
            semantic_flags.append(
                not lock["semantic_validation_evaluated_during_training"]
                and not lock["test_evaluated_during_training"]
                and not summary["semantic_validation_evaluated"]
                and not summary["test_evaluated"]
            )
            finite_histories.append(
                all(
                    np.isfinite(float(value))
                    for entry in summary["history"]
                    for key, value in entry.items()
                    if key != "epoch"
                )
            )
            if condition == "coefficient_permuted_relation":
                mapping_matches.append(
                    lock["relation_contract"]["mapping_sha256"] == audit_mapping_sha
                )
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
        "correct_prediction_not_collapsed_all_seeds": bool(
            all(prediction_not_collapsed.values())
        ),
        "all_condition_locks_present_before_readout": True,
    })

    controls = tuple(name for name in conditions if name != "correct_relation")
    deltas: dict[str, object] = {}
    bootstraps: dict[str, object] = {}
    for seed_index, seed in enumerate(seeds):
        seed_key = str(seed)
        deltas[seed_key] = {}
        bootstraps[seed_key] = {}
        for split_name in ("validation", "test"):
            deltas[seed_key][split_name] = {}
            correct = metrics[seed_key]["correct_relation"][split_name]
            for control in controls:
                baseline = metrics[seed_key][control][split_name]
                deltas[seed_key][split_name][f"correct_minus_{control}"] = {
                    "relation3_pearson": float(
                        correct["relation3_mean_signed_pearson"]
                        - baseline["relation3_mean_signed_pearson"]
                    ),
                    "relation3_r2": float(
                        correct["relation3_mean_direct_r2"]
                        - baseline["relation3_mean_direct_r2"]
                    ),
                    "relation3_nrmse_advantage": float(
                        baseline["relation3_mean_normalized_rmse"]
                        - correct["relation3_mean_normalized_rmse"]
                    ),
                    "true_relation_nrmse_advantage": float(
                        baseline["true_relation_normalized_rmse"]
                        - correct["true_relation_normalized_rmse"]
                    ),
                }
            deltas[seed_key][split_name]["correct_minus_point_free2_pearson"] = float(
                correct["mean_direct_abs_correlation_free2"]
                - metrics[seed_key]["point"][split_name][
                    "mean_direct_abs_correlation_free2"
                ]
            )
        for control_index, control in enumerate(controls):
            bootstraps[seed_key][control] = metric_impl.paired_bootstrap_r2_difference(
                validation.latents,
                predictions[seed_key]["correct_relation"]["validation"],
                predictions[seed_key][control]["validation"],
                int(thresholds["bootstrap_repetitions"]),
                int(thresholds["bootstrap_seed"]) + seed_index * 100 + control_index,
            )

    comparisons = [
        (str(seed), split_name)
        for seed in seeds
        for split_name in ("validation", "test")
    ]
    point_positive_count = sum(
        deltas[seed][split_name]["correct_minus_point"]["relation3_r2"] > 0.0
        for seed, split_name in comparisons
    )
    coefficient_positive_count = sum(
        deltas[seed][split_name]["correct_minus_coefficient_permuted_relation"][
            "relation3_r2"
        ]
        > 0.0
        for seed, split_name in comparisons
    )
    coefficient_split_means = {
        split_name: float(np.mean([
            deltas[str(seed)][split_name]["correct_minus_coefficient_permuted_relation"][
                "relation3_r2"
            ]
            for seed in seeds
        ]))
        for split_name in ("validation", "test")
    }
    coefficient_ci_positive_count = sum(
        bootstraps[str(seed)]["coefficient_permuted_relation"]["ci95_percentile"][0]
        > 0.0
        for seed in seeds
    )
    free2_consistent_degradation = all(
        deltas[str(seed)][split_name]["correct_minus_point_free2_pearson"] < 0.0
        for seed in seeds
        for split_name in ("validation", "test")
    )
    residual_means: dict[str, dict[str, float]] = {}
    for split_name in ("validation", "test"):
        residual_means[split_name] = {
            condition: float(np.mean([
                metrics[str(seed)][condition][split_name]["true_relation_normalized_rmse"]
                for seed in seeds
            ]))
            for condition in ("point", "correct_relation", "coefficient_permuted_relation")
        }
    relation_residual_means_better = all(
        residual_means[split_name]["correct_relation"]
        < residual_means[split_name][control]
        for split_name in ("validation", "test")
        for control in ("point", "coefficient_permuted_relation")
    )
    topology_all_positive = all(
        deltas[str(seed)][split_name][f"correct_minus_{control}"]["relation3_r2"]
        > 0.0
        for seed in seeds
        for split_name in ("validation", "test")
        for control in ("wrong_lm_to_k_relation", "wrong_km_to_l_relation")
    )
    valid = bool(all(validity.values()))
    decision, primary_supported = make_decision(
        valid,
        point_positive_count,
        coefficient_positive_count,
        all(value > 0.0 for value in coefficient_split_means.values()),
        coefficient_ci_positive_count,
        free2_consistent_degradation,
        relation_residual_means_better,
        topology_all_positive,
        thresholds,
    )
    coefficient_r2 = [
        float(deltas[seed][split_name]["correct_minus_coefficient_permuted_relation"]["relation3_r2"])
        for seed, split_name in comparisons
    ]
    coefficient_pearson = [
        float(deltas[seed][split_name]["correct_minus_coefficient_permuted_relation"]["relation3_pearson"])
        for seed, split_name in comparisons
    ]
    soft_effects = {
        "coefficient_correctness_r2": {
            **summarize(coefficient_r2),
            "target": float(thresholds["soft_mean_correctness_r2_gain"]),
            "target_met": float(np.mean(coefficient_r2))
            >= float(thresholds["soft_mean_correctness_r2_gain"]),
        },
        "coefficient_correctness_pearson": {
            **summarize(coefficient_pearson),
            "target": float(thresholds["soft_mean_correctness_pearson_gain"]),
            "target_met": float(np.mean(coefficient_pearson))
            >= float(thresholds["soft_mean_correctness_pearson_gain"]),
        },
    }
    payload = {
        "protocol_version": config["protocol_version"],
        "fact_type": "unread_shard_external_replication",
        "scope": "all_fifteen_locks_before_single_validation_test_readout",
        "config_sha256": legacy_runner.sha256_file(args.config),
        "audit_sha256": legacy_runner.sha256_file(args.audit),
        "metrics": metrics,
        "deltas": deltas,
        "validation_paired_bootstrap": bootstraps,
        "validity": validity,
        "valid": valid,
        "point_positive_seed_split_count": point_positive_count,
        "coefficient_positive_seed_split_count": coefficient_positive_count,
        "coefficient_split_mean_r2_delta": coefficient_split_means,
        "coefficient_validation_bootstrap_ci_positive_seed_count": (
            coefficient_ci_positive_count
        ),
        "free2_consistent_degradation": free2_consistent_degradation,
        "true_relation_nrmse_seed_means": residual_means,
        "true_relation_nrmse_means_better": relation_residual_means_better,
        "topology_r2_all_seed_splits_positive": topology_all_positive,
        "soft_effects": soft_effects,
        "machine_decision": decision,
        "primary_replication_supported": primary_supported,
        "strong_replication_pending_independent_state_disjoint_audit": primary_supported,
        "test_read_after_all_condition_locks": True,
        "condition_lock_sha256": {
            str(seed): {
                condition: legacy_runner.sha256_file(
                    args.root / f"seed{seed}" / condition / "training_lock.json"
                )
                for condition in conditions
            }
            for seed in seeds
        },
    }
    runner.write_json(args.output, payload)
    print(json.dumps({
        "valid": valid,
        "machine_decision": decision,
        "primary_replication_supported": primary_supported,
        "output": str(args.output),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
