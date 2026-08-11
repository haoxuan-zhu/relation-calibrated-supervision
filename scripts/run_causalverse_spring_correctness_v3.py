from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
import yaml

import causalverse_spring_pilot as base
import causalverse_spring_correctness_v3 as v3
import run_causalverse_spring_pilot as legacy_runner


def read_config(path: Path) -> dict[str, object]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def protocol_context(config_path: Path):
    config, cache, splits, label_ids = legacy_runner.protocol_context(config_path)
    if tuple(config["conditions"]) != v3.CONDITIONS:
        raise ValueError("config conditions do not match the frozen v3 condition order")
    return config, cache, splits, label_ids


def _relation_contracts(train: base.GroupedFeatures, label_ids: np.ndarray, permutation_seed: int):
    label_indices = v3.ordered_label_indices(train.ids, label_ids)
    label_latents = train.latents[label_indices].astype(np.float64)
    mapped_ids = v3.explicit_derangement(label_ids, permutation_seed)
    mapped_indices = v3.ordered_label_indices(train.ids, mapped_ids)
    mapped_latents = train.latents[mapped_indices].astype(np.float64)
    contracts: dict[str, dict[str, object]] = {}
    for condition, spec in v3.RELATION_SPECS.items():
        rhs_latents = mapped_latents if condition == "coefficient_permuted_relation" else None
        coefficient = v3.fit_relation_coefficient(label_latents, spec, rhs_latents)
        label_residual = v3.relation_residual_numpy(label_latents, coefficient, spec)
        train_residual = v3.relation_residual_numpy(train.latents, coefficient, spec)
        scale = v3.relation_scale(label_latents, spec)
        contracts[condition] = {
            "equation": spec.equation,
            "coefficient": coefficient,
            "relation_scale": scale,
            "label_nrmse": float(np.sqrt(np.mean(label_residual**2)) / scale),
            "train_nrmse": float(np.sqrt(np.mean(train_residual**2)) / scale),
        }
    contracts["coefficient_permuted_relation"]["mapping_sha256"] = v3.relation_mapping_sha256(
        label_ids, mapped_ids
    )
    contracts["coefficient_permuted_relation"]["explicit_id_mapping"] = v3.json_mapping(
        label_ids, mapped_ids
    )
    return contracts, mapped_ids


def run_audit(config_path: Path, output_path: Path) -> None:
    config, cache, splits, label_ids = protocol_context(config_path)
    train = legacy_runner.group_selected(cache, splits["train"])
    contracts, mapped_ids = _relation_contracts(
        train, label_ids, int(config["relation"]["permutation_seed"])
    )
    expected_cache_sha = str(config["runtime"]["feature_cache_sha256"])
    cache_sha = legacy_runner.sha256_file(Path(config["runtime"]["feature_cache"]))
    expected_correct = float(config["relation"]["expected_correct_coefficient"])
    correct = float(contracts["correct_relation"]["coefficient"])
    validity = {
        "feature_cache_sha_matches": cache_sha == expected_cache_sha,
        "label_count_exact": len(label_ids) == int(config["labels"]["count"]),
        "label_ids_unique": len(np.unique(label_ids)) == len(label_ids),
        "mapping_is_complete_derangement": bool(
            set(mapped_ids.tolist()) == set(label_ids.tolist()) and np.all(mapped_ids != label_ids)
        ),
        "correct_coefficient_matches_expected": bool(abs(correct - expected_correct) <= 1e-10),
        "correct_label_relation_exact": bool(contracts["correct_relation"]["label_nrmse"] <= 1e-10),
        "all_contract_values_finite": bool(
            all(
                np.isfinite(float(record[key]))
                for record in contracts.values()
                for key in ("coefficient", "relation_scale", "label_nrmse", "train_nrmse")
            )
        ),
        "wrong_relations_nontrivial_on_train": bool(
            all(
                contracts[name]["train_nrmse"] >= float(config["relation"]["minimum_wrong_train_nrmse"])
                for name in (
                    "coefficient_permuted_relation",
                    "wrong_lm_to_k_relation",
                    "wrong_km_to_l_relation",
                )
            )
        ),
    }
    payload = {
        "protocol_version": config["protocol_version"],
        "fact_type": "internally_frozen_preflight",
        "scope": "train_metadata_and_feature_contract_no_validation_or_test_semantics",
        "config_sha256": legacy_runner.sha256_file(config_path),
        "feature_cache_sha256": cache_sha,
        "split_sha256": {name: legacy_runner.sha256_int_array(ids) for name, ids in splits.items()},
        "label_ids": label_ids.tolist(),
        "label_ids_sha256": legacy_runner.sha256_int_array(label_ids),
        "relation_contracts": contracts,
        "validity": validity,
        "valid": bool(all(validity.values())),
        "semantic_validation_evaluated": False,
        "test_evaluated": False,
        "machine_decision": "v3_contract_audit_pass" if all(validity.values()) else "v3_contract_audit_fail",
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
    result: v3.TrainingResult,
    epochs: int,
    smoke: bool,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    relation_contract = v3.contract_json(result, label_ids)
    checkpoint_path = output_dir / "checkpoint.pt"
    torch.save({
        "protocol_version": config["protocol_version"],
        "condition": condition,
        "seed": seed,
        "state_dict": result.model.state_dict(),
        "normalization_mean": result.normalization_mean,
        "normalization_std": result.normalization_std,
        "relation_contract": relation_contract,
        "epochs": epochs,
    }, checkpoint_path)
    checkpoint_sha = legacy_runner.sha256_file(checkpoint_path)
    source_manifest = {
        "base_core": legacy_runner.sha256_file(Path(base.__file__)),
        "v3_core": legacy_runner.sha256_file(Path(v3.__file__)),
        "runner": legacy_runner.sha256_file(Path(__file__)),
    }
    lock = {
        "protocol_version": config["protocol_version"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "condition": condition,
        "seed": seed,
        "smoke": smoke,
        "epochs": epochs,
        "config_sha256": legacy_runner.sha256_file(config_path),
        "source_sha256": source_manifest,
        "feature_cache_sha256": legacy_runner.sha256_file(Path(config["runtime"]["feature_cache"])),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "initial_state_sha256": result.initial_state_sha256,
        "final_state_sha256": result.final_state_sha256,
        "train_ids_sha256": legacy_runner.sha256_int_array(splits["train"]),
        "validation_ids_sha256": legacy_runner.sha256_int_array(splits["validation"]),
        "test_ids_sha256": legacy_runner.sha256_int_array(splits["test"]),
        "label_ids_sha256": legacy_runner.sha256_int_array(label_ids),
        "label_count": int(len(label_ids)),
        "relation_contract": relation_contract,
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
        "seed": seed,
        "smoke": smoke,
        "history": result.history,
        "relation_contract": relation_contract,
        "normalization_mean": result.normalization_mean.tolist(),
        "normalization_std": result.normalization_std.tolist(),
        "training_lock": str(lock_path),
        "training_lock_sha256": legacy_runner.sha256_file(lock_path),
        "checkpoint_sha256": checkpoint_sha,
        "semantic_validation_evaluated": False,
        "test_evaluated": False,
        "machine_decision": "smoke_completed_no_semantic_read" if smoke else "training_locked_no_semantic_read",
    }
    write_json(output_dir / "training_summary.json", summary)
    return summary


def run_condition(
    config_path: Path,
    output_dir: Path,
    condition: str,
    seed: int,
    device: torch.device,
    smoke: bool,
) -> None:
    config, cache, splits, label_ids = protocol_context(config_path)
    allowed_seeds = tuple(int(value) for value in config["training"]["seeds"])
    if seed not in allowed_seeds:
        raise ValueError(f"seed {seed} is not registered: {allowed_seeds}")
    train = legacy_runner.group_selected(cache, splits["train"])
    epochs = 2 if smoke else int(config["training"]["epochs"])
    result = v3.train_condition(
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
    summary = save_training_assets(
        output_dir, config_path, config, splits, label_ids, condition, seed, result, epochs, smoke
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run internally frozen CausalVerse Spring correctness v3")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--mode", choices=("audit", "train"), required=True)
    parser.add_argument("--condition", choices=v3.CONDITIONS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "audit":
        run_audit(args.config, args.output)
        return
    if args.condition is None or args.seed is None:
        raise ValueError("--condition and --seed are required for training")
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    run_condition(args.config, args.output, args.condition, args.seed, device, args.smoke)


if __name__ == "__main__":
    main()
