from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
import yaml

import causalverse_spring_pilot as spring


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_int_array(values: np.ndarray) -> str:
    array = np.asarray(values, dtype="<i8")
    return hashlib.sha256(array.tobytes()).hexdigest()


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_cache(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        required = {"features", "ids", "views", "latents", "latent_columns"}
        if not required.issubset(archive.files):
            raise ValueError(f"feature cache is missing fields: {sorted(required - set(archive.files))}")
        return {name: archive[name] for name in required}


def group_selected(cache: dict[str, np.ndarray], selected_ids: np.ndarray) -> spring.GroupedFeatures:
    mask = np.isin(cache["ids"], selected_ids)
    return spring.group_features_by_id(
        cache["ids"][mask], cache["views"][mask], cache["features"][mask], cache["latents"][mask]
    )


def protocol_context(config_path: Path) -> tuple[dict[str, object], dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cache_path = Path(config["runtime"]["feature_cache"])
    cache = load_cache(cache_path)
    expected_columns = np.asarray(config["dataset"]["latent_columns"], dtype=np.str_)
    if not np.array_equal(cache["latent_columns"], expected_columns):
        raise ValueError("feature-cache latent column order does not match config")
    unique_ids = np.unique(cache["ids"]).astype(np.int64)
    split_config = config["split"]
    splits = spring.make_id_splits(
        unique_ids,
        int(split_config["id_permutation_seed"]),
        float(split_config["train_fraction"]),
        float(split_config["validation_fraction"]),
    )
    label_config = config["labels"]
    label_ids = np.random.default_rng(int(label_config["id_permutation_seed"])).permutation(
        splits["train"]
    )[: int(label_config["count"])]
    return config, cache, splits, label_ids.astype(np.int64)


def run_audit(config_path: Path, output_path: Path) -> None:
    config, cache, splits, label_ids = protocol_context(config_path)
    grouped = spring.group_features_by_id(cache["ids"], cache["views"], cache["features"], cache["latents"])
    train = spring.select_ids(grouped, splits["train"])
    label_group = spring.select_ids(train, label_ids)
    correct = spring.fit_relation_coefficient(label_group.latents)
    permutation = np.random.default_rng(int(config["relation"]["permutation_seed"])).permutation(len(label_ids))
    permuted = spring.fit_relation_coefficient(label_group.latents, permutation)
    correct_residual = spring.relation_residual_numpy(label_group.latents, correct)
    ratios = grouped.latents[:, 4] * grouped.latents[:, 3] / grouped.latents[:, 2]
    validity = {
        "all_rows_finite": bool(np.isfinite(grouped.features).all() and np.isfinite(grouped.latents).all()),
        "each_id_has_four_consistent_views": bool(grouped.features.shape[1] == 4),
        "split_ids_disjoint": bool(
            not (set(splits["train"]) & set(splits["validation"]))
            and not (set(splits["train"]) & set(splits["test"]))
            and not (set(splits["validation"]) & set(splits["test"]))
        ),
        "label_count_exact": bool(len(label_ids) == int(config["labels"]["count"])),
        "correct_residual_max_lt_1e_8": bool(np.max(np.abs(correct_residual)) < 1e-8),
        "permuted_coefficient_separated": bool(abs(correct - permuted) >= 0.05),
    }
    payload = {
        "protocol_version": config["protocol_version"],
        "fact_type": "rerun",
        "audit": "metadata_and_feature_contract",
        "feature_cache": str(config["runtime"]["feature_cache"]),
        "feature_cache_sha256": sha256_file(Path(config["runtime"]["feature_cache"])),
        "row_count": int(len(cache["ids"])),
        "unique_id_count": int(len(grouped.ids)),
        "split_counts": {name: int(len(values)) for name, values in splits.items()},
        "split_sha256": {name: sha256_int_array(values) for name, values in splits.items()},
        "label_ids_sha256": sha256_int_array(label_ids),
        "label_count": int(len(label_ids)),
        "correct_relation_coefficient": correct,
        "permuted_relation_coefficient": permuted,
        "coefficient_absolute_difference": abs(correct - permuted),
        "label_subset_correct_residual_max_abs": float(np.max(np.abs(correct_residual))),
        "all_pilot_lk_over_m_min": float(np.min(ratios)),
        "all_pilot_lk_over_m_max": float(np.max(ratios)),
        "validity": validity,
        "valid": bool(all(validity.values())),
        "machine_decision": "metadata_audit_pass" if all(validity.values()) else "metadata_audit_fail",
    }
    if int(config["labels"]["count"]) == 80:
        payload["k80_correct_residual_max_abs"] = payload["label_subset_correct_residual_max_abs"]
    write_json(output_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


def save_training_assets(
    output_dir: Path,
    config_path: Path,
    config: dict[str, object],
    splits: dict[str, np.ndarray],
    label_ids: np.ndarray,
    condition: str,
    result: spring.TrainingResult,
    epochs: int,
    smoke: bool,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint.pt"
    torch.save(
        {
            "protocol_version": config["protocol_version"],
            "condition": condition,
            "state_dict": result.model.state_dict(),
            "normalization_mean": result.normalization_mean,
            "normalization_std": result.normalization_std,
            "relation_coefficient": result.relation_coefficient,
            "relation_scale": result.relation_scale,
            "epochs": epochs,
        },
        checkpoint_path,
    )
    checkpoint_sha = sha256_file(checkpoint_path)
    lock = {
        "protocol_version": config["protocol_version"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "condition": condition,
        "smoke": smoke,
        "epochs": epochs,
        "config_sha256": sha256_file(config_path),
        "core_source_sha256": sha256_file(Path(spring.__file__)),
        "runner_source_sha256": sha256_file(Path(__file__)),
        "feature_cache_sha256": sha256_file(Path(config["runtime"]["feature_cache"])),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "initial_state_sha256": result.initial_state_sha256,
        "final_state_sha256": result.final_state_sha256,
        "train_ids_sha256": sha256_int_array(splits["train"]),
        "validation_ids_sha256": sha256_int_array(splits["validation"]),
        "test_ids_sha256": sha256_int_array(splits["test"]),
        "label_ids_sha256": sha256_int_array(label_ids),
        "label_count": int(len(label_ids)),
        "semantic_validation_evaluated_during_training": False,
        "test_evaluated_during_training": False,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    lock_path = output_dir / "training_lock.json"
    write_json(lock_path, lock)
    summary = {
        "protocol_version": config["protocol_version"],
        "fact_type": "rerun",
        "condition": condition,
        "smoke": smoke,
        "history": result.history,
        "relation_coefficient": result.relation_coefficient,
        "relation_scale": result.relation_scale,
        "normalization_mean": result.normalization_mean.tolist(),
        "normalization_std": result.normalization_std.tolist(),
        "training_lock": str(lock_path),
        "training_lock_sha256": sha256_file(lock_path),
        "checkpoint_sha256": checkpoint_sha,
        "semantic_validation_evaluated": False,
        "test_evaluated": False,
        "machine_decision": "smoke_completed_no_semantic_read" if smoke else "training_locked_no_semantic_read",
    }
    write_json(output_dir / "training_summary.json", summary)
    return summary


def run_upper_bound(config_path: Path, output_dir: Path, device: torch.device) -> None:
    config, cache, splits, label_ids = protocol_context(config_path)
    train = group_selected(cache, splits["train"])
    validation = group_selected(cache, splits["validation"])
    result = spring.train_supervised_upper_bound(
        train=train,
        feature_dim=int(config["backbone"]["feature_dim"]),
        hidden_dims=config["model"]["hidden_dims"],
        seed=int(config["training"]["seed"]),
        epochs=int(config["training"]["epochs"]),
        learning_rate=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
        device=device,
    )
    save_training_assets(
        output_dir, config_path, config, splits, label_ids, "supervised_upper_bound", result,
        int(config["training"]["epochs"]), False,
    )
    predicted = spring.predict_physical(
        result.model,
        validation,
        int(config["backbone"]["feature_dim"]),
        result.normalization_mean,
        result.normalization_std,
        device,
    )
    metrics = spring.evaluate_predictions(validation.latents, predicted)
    threshold = config["evaluation"]
    each = np.asarray(metrics["direct_abs_correlation"])
    passed = bool(
        metrics["mean_direct_abs_correlation_all5"] >= float(threshold["upper_bound_mean_corr_min"])
        and np.min(each) >= float(threshold["upper_bound_each_corr_min"])
    )
    payload = {
        "protocol_version": config["protocol_version"],
        "fact_type": "rerun",
        "condition": "supervised_upper_bound",
        "validation": metrics,
        "test_evaluated": False,
        "valid": passed,
        "machine_decision": "upper_bound_pass" if passed else "upper_bound_fail",
    }
    write_json(output_dir / "upper_bound_results.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


def run_condition(
    config_path: Path,
    output_dir: Path,
    condition: str,
    device: torch.device,
    smoke: bool,
) -> None:
    config, cache, splits, label_ids = protocol_context(config_path)
    train = group_selected(cache, splits["train"])
    epochs = 2 if smoke else int(config["training"]["epochs"])
    result = spring.train_condition(
        train=train,
        label_ids=label_ids,
        condition=condition,
        feature_dim=int(config["backbone"]["feature_dim"]),
        hidden_dims=config["model"]["hidden_dims"],
        seed=int(config["training"]["seed"]),
        epochs=epochs,
        learning_rate=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
        loss_weights=config["loss_weights"],
        range_limit=float(config["loss_weights"]["range_zscore_limit"]),
        permutation_seed=int(config["relation"]["permutation_seed"]),
        device=device,
    )
    summary = save_training_assets(
        output_dir, config_path, config, splits, label_ids, condition, result, epochs, smoke
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the frozen CausalVerse Spring external pilot")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--mode", choices=("audit", "upper-bound", "train"), required=True)
    parser.add_argument("--condition", choices=("point", "correct_relation", "permuted_relation"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "audit":
        run_audit(args.config, args.output)
        return
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    if args.mode == "upper-bound":
        run_upper_bound(args.config, args.output, device)
        return
    if args.condition is None:
        raise ValueError("--condition is required for --mode train")
    run_condition(args.config, args.output, args.condition, device, args.smoke)


if __name__ == "__main__":
    main()
