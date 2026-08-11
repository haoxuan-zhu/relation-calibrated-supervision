"""Run and evaluate the validation-only CausalVerse Slope K22 preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
import yaml

import causalverse_slope_preflight as slope


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_int_array(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_context(config_path: Path):
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cache_path = Path(config["runtime"]["feature_cache"])
    with np.load(cache_path, allow_pickle=False) as archive:
        required = {"features", "ids", "views", "latents", "latent_columns"}
        if not required.issubset(archive.files):
            raise ValueError(f"feature cache is missing {sorted(required - set(archive.files))}")
        cache = {name: archive[name] for name in required}
    expected_columns = np.asarray(config["dataset"]["latent_columns"], dtype=np.str_)
    if not np.array_equal(cache["latent_columns"], expected_columns):
        raise ValueError("feature-cache latent columns differ from config")
    required_views = config["dataset"]["required_views"]
    incomplete_ids = slope.incomplete_product_ids(cache["ids"], cache["views"], required_views)
    expected_incomplete = np.asarray(
        config["dataset"].get("expected_incomplete_product_ids", []), dtype=np.int64
    )
    policy = config["dataset"].get("incomplete_product_policy", "reject")
    if policy == "drop_exact_ids_before_split":
        if not np.array_equal(incomplete_ids, expected_incomplete):
            raise ValueError(
                "incomplete product IDs differ from the frozen shard-boundary contract: "
                f"actual={incomplete_ids.tolist()} expected={expected_incomplete.tolist()}"
            )
        keep = ~np.isin(cache["ids"], incomplete_ids)
        source_row_count = int(len(cache["ids"]))
        for name in ("features", "ids", "views", "latents"):
            cache[name] = cache[name][keep]
    elif len(incomplete_ids):
        raise ValueError(f"incomplete product IDs found under reject policy: {incomplete_ids.tolist()}")
    else:
        source_row_count = int(len(cache["ids"]))
    grouped = slope.group_features_by_id(
        cache["ids"], cache["views"], cache["features"], cache["latents"],
        required_views,
    )
    split_config = config["split"]
    splits = slope.make_id_splits(
        grouped.ids,
        int(split_config["id_permutation_seed"]),
        float(split_config["train_fraction"]),
        float(split_config["validation_fraction"]),
    )
    label_config = config["labels"]
    label_ids = np.random.default_rng(int(label_config["id_permutation_seed"])).permutation(
        splits["train"]
    )[: int(label_config["count"])]
    filter_audit = {
        "policy": policy,
        "source_row_count": source_row_count,
        "retained_row_count": int(len(cache["ids"])),
        "dropped_row_count": source_row_count - int(len(cache["ids"])),
        "dropped_incomplete_product_ids": incomplete_ids.tolist(),
        "expected_incomplete_product_ids": expected_incomplete.tolist(),
        "contract_matches": bool(np.array_equal(incomplete_ids, expected_incomplete)),
    }
    return config, grouped, splits, label_ids.astype(np.int64), filter_audit


def relation_contract(train: slope.GroupedFeatures, label_ids: np.ndarray, permutation_seed: int):
    label_indices = slope.ordered_label_indices(train.ids, label_ids)
    labels = train.latents[label_indices]
    mapped_ids = slope.explicit_derangement(label_ids, permutation_seed)
    mapped_indices = slope.ordered_label_indices(train.ids, mapped_ids)
    correct = slope.fit_relation_parameters(labels)
    permuted = slope.fit_relation_parameters(labels, train.latents[mapped_indices])
    scales = slope.relation_scales(labels)
    correct_nrmse = np.sqrt(np.mean((slope.relation_residuals_numpy(labels, correct) / scales) ** 2, axis=0))
    return labels, mapped_ids, correct, permuted, scales, correct_nrmse


def run_audit(config_path: Path, output_path: Path) -> None:
    config, grouped, splits, label_ids, filter_audit = load_context(config_path)
    train = slope.select_ids(grouped, splits["train"])
    labels, mapped_ids, correct, permuted, scales, correct_nrmse = relation_contract(
        train, label_ids, int(config["relation"]["permutation_seed"])
    )
    expected = np.asarray(config["relation"]["expected_parameters"], dtype=np.float64)
    actual = correct.as_array()
    tolerance = float(config["relation"]["correct_parameter_max_abs_tolerance"])
    expected_cache_sha = str(config["runtime"]["feature_cache_sha256"])
    cache_sha = sha256_file(Path(config["runtime"]["feature_cache"]))
    validity = {
        "feature_cache_sha_matches": cache_sha == expected_cache_sha,
        "incomplete_product_filter_contract_matches": bool(filter_audit["contract_matches"]),
        "all_rows_finite": bool(np.isfinite(grouped.features).all() and np.isfinite(grouped.latents).all()),
        "each_id_has_four_views": bool(grouped.features.shape[1] == 4),
        "split_ids_disjoint": bool(
            not (set(splits["train"]) & set(splits["validation"]))
            and not (set(splits["train"]) & set(splits["test"]))
            and not (set(splits["validation"]) & set(splits["test"]))
        ),
        "label_count_exact": len(label_ids) == int(config["labels"]["count"]),
        "mapping_is_complete_derangement": bool(
            np.all(mapped_ids != label_ids) and set(mapped_ids.tolist()) == set(label_ids.tolist())
        ),
        "correct_parameters_match_published_equations": bool(np.max(np.abs(actual - expected)) <= tolerance),
        "correct_label_mean_nrmse_small": bool(
            float(np.mean(correct_nrmse)) <= float(config["relation"]["correct_label_mean_nrmse_max"])
        ),
        "permuted_parameters_differ": bool(np.linalg.norm(actual - permuted.as_array()) > 0.05),
    }
    payload = {
        "protocol_version": config["protocol_version"],
        "fact_type": "content_blind_system_preflight_contract",
        "scope": "train_metadata_and_feature_contract_no_validation_or_test_semantics",
        "config_sha256": sha256_file(config_path),
        "feature_cache_sha256": cache_sha,
        "incomplete_product_filter": filter_audit,
        "row_count": int(len(grouped.ids) * 4),
        "unique_id_count": int(len(grouped.ids)),
        "split_counts": {name: int(len(values)) for name, values in splits.items()},
        "split_sha256": {name: sha256_int_array(values) for name, values in splits.items()},
        "label_ids_sha256": sha256_int_array(label_ids),
        "correct_parameters": asdict(correct),
        "coefficient_permuted_parameters": asdict(permuted),
        "parameter_difference_l2": float(np.linalg.norm(actual - permuted.as_array())),
        "relation_scales": scales.tolist(),
        "correct_label_relation_nrmse": correct_nrmse.tolist(),
        "mapping_sha256": slope.relation_mapping_sha256(label_ids, mapped_ids),
        "validity": validity,
        "valid": bool(all(validity.values())),
        "semantic_validation_evaluated": False,
        "test_evaluated": False,
        "machine_decision": "slope_k22_contract_pass" if all(validity.values()) else "slope_k22_contract_fail",
    }
    write_json(output_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


def save_training_assets(
    output_dir: Path,
    config_path: Path,
    config: dict[str, object],
    splits: dict[str, np.ndarray],
    label_ids: np.ndarray,
    condition: str,
    seed: int,
    result: slope.TrainingResult,
    epochs: int,
    smoke: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    parameters = None if result.relation_parameters is None else asdict(result.relation_parameters)
    checkpoint_path = output_dir / "checkpoint.pt"
    torch.save(
        {
            "protocol_version": config["protocol_version"],
            "condition": condition,
            "seed": seed,
            "state_dict": result.model.state_dict(),
            "normalization_mean": result.normalization_mean,
            "normalization_std": result.normalization_std,
            "relation_parameters": parameters,
            "relation_scales": result.relation_scales,
            "epochs": epochs,
        },
        checkpoint_path,
    )
    source_sha = {
        "core": sha256_file(Path(slope.__file__)),
        "runner": sha256_file(Path(__file__)),
    }
    lock = {
        "protocol_version": config["protocol_version"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "condition": condition,
        "seed": seed,
        "smoke": smoke,
        "epochs": epochs,
        "config_sha256": sha256_file(config_path),
        "source_sha256": source_sha,
        "feature_cache_sha256": sha256_file(Path(config["runtime"]["feature_cache"])),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "initial_state_sha256": result.initial_state_sha256,
        "final_state_sha256": result.final_state_sha256,
        "train_ids_sha256": sha256_int_array(splits["train"]),
        "validation_ids_sha256": sha256_int_array(splits["validation"]),
        "test_ids_sha256": sha256_int_array(splits["test"]),
        "label_ids_sha256": sha256_int_array(label_ids),
        "label_count": int(len(label_ids)),
        "relation_parameters": parameters,
        "relation_scales": result.relation_scales.tolist(),
        "mapping_sha256": result.mapping_sha256,
        "semantic_validation_evaluated_during_training": False,
        "test_evaluated_during_training": False,
    }
    write_json(output_dir / "training_lock.json", lock)
    write_json(
        output_dir / "training_summary.json",
        {
            "protocol_version": config["protocol_version"],
            "condition": condition,
            "seed": seed,
            "smoke": smoke,
            "history": result.history,
            "relation_parameters": parameters,
            "mapping_sha256": result.mapping_sha256,
            "semantic_validation_evaluated": False,
            "test_evaluated": False,
            "machine_decision": "smoke_completed_no_semantic_read" if smoke else "training_locked_no_semantic_read",
        },
    )


def run_train(
    config_path: Path, output_dir: Path, condition: str, seed: int, device: torch.device, smoke: bool
) -> None:
    config, grouped, splits, label_ids, _ = load_context(config_path)
    train = slope.select_ids(grouped, splits["train"])
    epochs = 2 if smoke else int(config["training"]["epochs"])
    result = slope.train_condition(
        train=train,
        label_ids=label_ids,
        condition=condition,
        feature_dim=int(config["backbone"]["feature_dim"]),
        hidden_dims=config["model"]["hidden_dims"],
        seed=seed,
        epochs=epochs,
        learning_rate=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
        loss_weights=config["loss_weights"],
        range_limit=float(config["loss_weights"]["range_zscore_limit"]),
        permutation_seed=int(config["relation"]["permutation_seed"]),
        device=device,
    )
    save_training_assets(
        output_dir, config_path, config, splits, label_ids, condition, seed, result, epochs, smoke
    )
    print(json.dumps({"condition": condition, "seed": seed, "smoke": smoke, "output": str(output_dir)}))


def make_decision(valid: bool, metrics: dict[str, dict[str, object]], evaluation: dict[str, object]) -> str:
    if not valid:
        return "invalid"
    point = metrics["point"]
    if point["mean_direct_abs_correlation_relation4"] >= float(
        evaluation["point_saturation_mean_correlation"]
    ):
        return "point_baseline_saturated_stop"
    correct = metrics["correct_relation"]
    permuted = metrics["coefficient_permuted_relation"]
    correct_over_point = (
        correct["mean_direct_abs_correlation_relation4"] > point["mean_direct_abs_correlation_relation4"]
        and correct["mean_direct_r2_relation4"] > point["mean_direct_r2_relation4"]
    )
    correct_over_permuted = (
        correct["mean_direct_abs_correlation_relation4"]
        > permuted["mean_direct_abs_correlation_relation4"]
        and correct["mean_direct_r2_relation4"] > permuted["mean_direct_r2_relation4"]
    )
    free_safe = (
        correct["mean_direct_abs_correlation_free3"]
        >= point["mean_direct_abs_correlation_free3"]
        - float(evaluation["maximum_free_correlation_drop"])
    )
    if correct_over_point and correct_over_permuted and free_safe:
        return "relation_content_signal"
    if correct_over_point:
        return "relation_regularization_only"
    return "no_preflight_signal"


def run_evaluate(config_path: Path, root: Path, audit_path: Path, output_path: Path, device: torch.device) -> None:
    config, grouped, splits, label_ids, _ = load_context(config_path)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    validation = slope.select_ids(grouped, splits["validation"])
    seed = int(config["training"]["seeds"][0])
    metrics: dict[str, dict[str, object]] = {}
    locks = {}
    finite_histories = []
    lock_integrity = []
    current_config_sha = sha256_file(config_path)
    current_feature_sha = sha256_file(Path(config["runtime"]["feature_cache"]))
    current_source_sha = {
        "core": sha256_file(Path(slope.__file__)),
        "runner": sha256_file(Path(__file__)),
    }
    for condition in slope.CONDITIONS:
        condition_dir = root / f"seed{seed}" / condition
        lock_path = condition_dir / "training_lock.json"
        summary_path = condition_dir / "training_summary.json"
        checkpoint_path = condition_dir / "checkpoint.pt"
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
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
        metrics[condition] = slope.evaluate_predictions(validation.latents, predicted)
        locks[condition] = lock
        lock_integrity.append(
            lock["condition"] == condition
            and int(lock["seed"]) == seed
            and lock["config_sha256"] == current_config_sha
            and lock["feature_cache_sha256"] == current_feature_sha
            and lock["checkpoint_sha256"] == sha256_file(checkpoint_path)
            and lock["source_sha256"] == current_source_sha
        )
        finite_histories.append(
            all(np.isfinite(float(value)) for row in summary["history"] for key, value in row.items() if key != "epoch")
        )
    initial_hashes = {lock["initial_state_sha256"] for lock in locks.values()}
    validity = {
        "audit_valid": bool(audit["valid"]),
        "config_matches_audit": current_config_sha == audit["config_sha256"],
        "feature_cache_matches_audit": current_feature_sha == audit["feature_cache_sha256"],
        "all_three_locks_present_before_validation_read": len(locks) == 3,
        "all_lock_hash_contracts_match": bool(all(lock_integrity)),
        "initial_state_shared": len(initial_hashes) == 1,
        "all_training_histories_finite": bool(all(finite_histories)),
        "no_semantic_validation_during_training": all(
            not lock["semantic_validation_evaluated_during_training"] for lock in locks.values()
        ),
        "test_unread": True,
    }
    valid = bool(all(validity.values()))
    decision = make_decision(valid, metrics, config["evaluation"])
    correct = metrics["correct_relation"]
    deltas = {}
    for control in ("point", "coefficient_permuted_relation"):
        baseline = metrics[control]
        deltas[f"correct_minus_{control}"] = {
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
    payload = {
        "protocol_version": config["protocol_version"],
        "fact_type": "validation_only_system_preflight",
        "metrics": metrics,
        "deltas": deltas,
        "validity": validity,
        "diagnostics": {
            "prediction_not_collapsed": {
                condition: min(record["prediction_std"]) > 1e-8
                for condition, record in metrics.items()
            }
        },
        "valid": valid,
        "machine_decision": decision,
        "test_evaluated": False,
        "three_seed_unlocked": decision == "relation_content_signal",
        "condition_lock_sha256": {
            condition: sha256_file(root / f"seed{seed}" / condition / "training_lock.json")
            for condition in slope.CONDITIONS
        },
    }
    write_json(output_path, payload)
    print(json.dumps({"valid": valid, "machine_decision": decision, "output": str(output_path)}))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--mode", choices=("audit", "train", "evaluate"), required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--condition", choices=slope.CONDITIONS)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "audit":
        run_audit(args.config, args.output)
        return
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    if args.mode == "train":
        if args.condition is None:
            raise ValueError("--condition is required for training")
        run_train(args.config, args.output, args.condition, args.seed, device, args.smoke)
        return
    if args.root is None or args.audit is None:
        raise ValueError("--root and --audit are required for evaluation")
    run_evaluate(args.config, args.root, args.audit, args.output, device)


if __name__ == "__main__":
    main()
