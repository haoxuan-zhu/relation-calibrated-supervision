"""Run the independent-seed K35 replication of projected point supervision."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml

import audit_instrumented_slope_interface_k27 as k27
import causalverse_slope_preflight as slope
import evaluate_instrumented_slope_validation_k29 as k29
import run_causalverse_slope_preflight as preflight
import run_instrumented_slope_neural_tube_k30 as k30
import run_instrumented_slope_projected_training_k34 as k34


CONDITIONS = ("raw_baseline", "projected_only")
SEEDS = (123, 2026, 31415)


def source_hashes() -> dict[str, str]:
    paths = {
        "k35": Path(__file__),
        "k34": Path(k34.__file__),
        "k30": Path(k30.__file__),
        "k29": Path(k29.__file__),
        "k27": Path(k27.__file__),
        "slope": Path(slope.__file__),
        "preflight": Path(preflight.__file__),
    }
    return {name: preflight.sha256_file(path) for name, path in paths.items()}


def load_protocol(config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["protocol_version"] != "instrumented_slope_projected_replication_k35":
        raise ValueError("unexpected K35 protocol")
    if tuple(config["model"]["conditions"]) != CONDITIONS:
        raise ValueError("K35 condition registry changed")
    if tuple(int(value) for value in config["training"]["seeds"]) != SEEDS:
        raise ValueError("K35 seed registry changed")
    if int(config["training"]["epochs"]) != 500:
        raise ValueError("K35 epoch registry changed")
    if tuple(config["evaluation"]["candidates"]) != (
        "raw_baseline_curve",
        "projected_only_curve",
    ):
        raise ValueError("K35 evaluation registry changed")
    for condition in CONDITIONS:
        weights = config["point_loss"][condition]
        if abs(float(weights["raw_weight"]) + float(weights["projected_weight"]) - 1.0) > 1e-12:
            raise ValueError("K35 point-loss weights must sum to one")
    return config


def load_context(config: Mapping[str, Any]):
    context = k30.load_context_chain(config)
    result_path = Path(config["source_contract"]["k34_result"])
    result = json.loads(result_path.read_text(encoding="utf-8"))
    return (*context, result, preflight.sha256_file(result_path))


def build_model(
    config: Mapping[str, Any], frozen: Mapping[str, Any], seed: int, device: torch.device
) -> k30.InstrumentedSlopeTubeHead:
    return k34.build_model(config, frozen, seed, device)


def preflight_main(config_path: Path, output_root: Path) -> None:
    config = load_protocol(config_path)
    (
        _, validation_result, geometry_config, interface_config, grouped, splits, label_ids,
        filter_audit, hashes, k34_result, k34_result_hash,
    ) = load_context(config)
    train = slope.select_ids(grouped, splits["train"])
    frozen = k30.frozen_components(train, label_ids, geometry_config)
    initial_hashes: dict[str, str] = {}
    parameter_counts: dict[str, int] = {}
    for seed in SEEDS:
        first = build_model(config, frozen, seed, torch.device("cpu"))
        second = build_model(config, frozen, seed, torch.device("cpu"))
        first_hash = slope.state_dict_sha256(first.state_dict())
        if first_hash != slope.state_dict_sha256(second.state_dict()):
            raise RuntimeError("K35 paired conditions do not share initial state")
        initial_hashes[str(seed)] = first_hash
        parameter_counts[str(seed)] = sum(parameter.numel() for parameter in first.parameters())
    projection_audit = k34.train_only_projection_audit(
        config, frozen, geometry_config, train
    )
    source = config["source_contract"]
    validity = {
        "validation_config_sha_matches": hashes["validation_config"]
        == str(source["validation_config_sha256"]),
        "validation_result_sha_matches": hashes["validation_result"]
        == str(source["validation_result_sha256"]),
        "validation_result_valid": bool(validation_result["valid"]),
        "k34_result_sha_matches": k34_result_hash == str(source["k34_result_sha256"]),
        "k34_result_valid": bool(k34_result["valid"]),
        "k34_external_and_test_unread": not bool(k34_result["external_evaluated"])
        and not bool(k34_result["test_evaluated"]),
        "feature_cache_sha_matches": hashes["feature_cache"]
        == str(interface_config["runtime"]["feature_cache_sha256"]),
        "incomplete_product_filter_contract_matches": bool(filter_audit["contract_matches"]),
        "initial_states_unique": len(set(initial_hashes.values())) == len(SEEDS),
        "parameter_counts_matched": len(set(parameter_counts.values())) == 1,
        "projection_exact_on_train_audit": projection_audit[
            "max_normalized_state_l2_difference"
        ]
        <= 1.0e-4,
    }
    payload = {
        "protocol_version": config["protocol_version"],
        "fact_type": "new_seed_replication_preflight_after_known_development_validation",
        "scope": "new_initializations_training_ids_k40_labels_external_and_test_sealed",
        "config_sha256": preflight.sha256_file(config_path),
        "source_files_sha256": source_hashes(),
        "source_contract_hashes": {**hashes, "k34_result": k34_result_hash},
        "label_ids_sha256": preflight.sha256_int_array(label_ids),
        "train_ids_sha256": preflight.sha256_int_array(splits["train"]),
        "validation_ids_sha256": preflight.sha256_int_array(splits["validation"]),
        "test_ids_sha256": preflight.sha256_int_array(splits["test"]),
        "initial_state_sha256": initial_hashes,
        "parameter_counts": parameter_counts,
        "train_only_projection_audit": projection_audit,
        "development_validation_used_for_design": True,
        "external_evaluated": False,
        "test_evaluated": False,
        "validity": validity,
        "valid": bool(all(validity.values())),
        "machine_decision": "projected_replication_preflight_pass"
        if all(validity.values())
        else "invalid",
    }
    output = output_root / "preflight" / "preflight.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    preflight.write_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


def load_preflight(output_root: Path, config_path: Path) -> dict[str, Any]:
    payload = json.loads(
        (output_root / "preflight" / "preflight.json").read_text(encoding="utf-8")
    )
    checks = {
        "valid": bool(payload["valid"]),
        "config_current": payload["config_sha256"] == preflight.sha256_file(config_path),
        "sources_current": payload["source_files_sha256"] == source_hashes(),
        "external_unread": not bool(payload["external_evaluated"]),
        "test_unread": not bool(payload["test_evaluated"]),
    }
    if not all(checks.values()):
        raise ValueError(f"invalid K35 preflight: {checks}")
    return payload


def train_main(config_path: Path, output_root: Path, condition: str, seed: int) -> None:
    config = load_protocol(config_path)
    if condition not in CONDITIONS or seed not in SEEDS:
        raise ValueError("unregistered K35 train request")
    preflight_payload = load_preflight(output_root, config_path)
    (
        _, _, geometry_config, _, grouped, splits, label_ids, _, _, _, _,
    ) = load_context(config)
    train = slope.select_ids(grouped, splits["train"])
    frozen = k30.frozen_components(train, label_ids, geometry_config)
    torch.set_num_threads(int(config["runtime"]["num_threads"]))
    torch.set_float32_matmul_precision("high")
    device = torch.device(config["runtime"]["device"] if torch.cuda.is_available() else "cpu")
    model, history, initial_hash, final_hash = k34.train_one(
        config, frozen, train, condition, seed, device
    )
    if initial_hash != preflight_payload["initial_state_sha256"][str(seed)]:
        raise RuntimeError("K35 initial state drifted after preflight")
    output_dir = output_root / "formal" / f"seed{seed}" / condition
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "training_results.json"
    checkpoint_path = output_dir / f"{condition}_seed{seed}.pt"
    lock_path = output_dir / "training_lock.json"
    if result_path.exists() or checkpoint_path.exists() or lock_path.exists():
        raise FileExistsError(f"refusing to overwrite K35 seed{seed} {condition}")
    torch.save(
        {
            "protocol_version": config["protocol_version"],
            "seed": seed,
            "condition": condition,
            "state_dict": model.state_dict(),
        },
        checkpoint_path,
    )
    checkpoint_sha = preflight.sha256_file(checkpoint_path)
    validity = {
        "preflight_valid": bool(preflight_payload["valid"]),
        "initial_state_exact": initial_hash
        == preflight_payload["initial_state_sha256"][str(seed)],
        "epoch_exact": history[-1]["epoch"] == int(config["training"]["epochs"]),
        "history_finite": all(
            np.isfinite(value)
            for record in history
            for key, value in record.items()
            if key != "epoch"
        ),
        "parameter_count_exact": sum(parameter.numel() for parameter in model.parameters())
        == int(preflight_payload["parameter_counts"][str(seed)]),
    }
    lock = {
        "status": "locked_before_joint_replication_readout",
        "protocol_version": config["protocol_version"],
        "config_sha256": preflight.sha256_file(config_path),
        "source_files_sha256": source_hashes(),
        "seed": seed,
        "condition": condition,
        "initial_state_sha256": initial_hash,
        "final_state_sha256": final_hash,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "validity": validity,
        "external_evaluated": False,
        "test_evaluated": False,
    }
    result = {
        "protocol_version": config["protocol_version"],
        "fact_type": "formal_new_seed_replication_train",
        "seed": seed,
        "condition": condition,
        "config_sha256": preflight.sha256_file(config_path),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "initial_state_sha256": initial_hash,
        "final_state_sha256": final_hash,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "history": history,
        "validity": validity,
        "valid": bool(all(validity.values())),
        "external_evaluated": False,
        "test_evaluated": False,
    }
    preflight.write_json(lock_path, lock)
    preflight.write_json(result_path, result)
    print(json.dumps({"result": str(result_path), "lock": str(lock_path)}, sort_keys=True))


def load_model_checkpoint(
    config: Mapping[str, Any],
    frozen: Mapping[str, Any],
    output_root: Path,
    condition: str,
    seed: int,
    device: torch.device,
) -> k30.InstrumentedSlopeTubeHead:
    output_dir = output_root / "formal" / f"seed{seed}" / condition
    lock = json.loads((output_dir / "training_lock.json").read_text(encoding="utf-8"))
    checkpoint_path = Path(lock["checkpoint"])
    checks = {
        "lock_valid": all(lock["validity"].values()),
        "identity": lock["condition"] == condition and int(lock["seed"]) == seed,
        "checkpoint_current": preflight.sha256_file(checkpoint_path)
        == lock["checkpoint_sha256"],
        "sources_current": lock["source_files_sha256"] == source_hashes(),
        "external_unread": not bool(lock["external_evaluated"]),
        "test_unread": not bool(lock["test_evaluated"]),
    }
    if not all(checks.values()):
        raise ValueError(f"invalid K35 checkpoint lock: {checks}")
    model = build_model(config, frozen, seed, device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    if slope.state_dict_sha256(model.state_dict()) != lock["final_state_sha256"]:
        raise ValueError("K35 final state hash mismatch")
    return model


def evaluate_main(config_path: Path, output_root: Path) -> None:
    config = load_protocol(config_path)
    preflight_payload = load_preflight(output_root, config_path)
    (
        _, _, geometry_config, _, grouped, splits, label_ids, _, _, _, _,
    ) = load_context(config)
    train = slope.select_ids(grouped, splits["train"])
    validation = slope.select_ids(grouped, splits["validation"])
    frozen = k30.frozen_components(train, label_ids, geometry_config)
    truth = validation.latents[:, k27.TARGET_INDICES]
    side = validation.latents[:, k27.SIDE_INDICES]
    device = torch.device(config["runtime"]["device"] if torch.cuda.is_available() else "cpu")
    seed_results: dict[str, Any] = {}
    baseline_predictions: list[np.ndarray] = []
    projected_predictions: list[np.ndarray] = []
    directional: list[bool] = []
    for seed in SEEDS:
        predictions: dict[str, np.ndarray] = {}
        audits: dict[str, Any] = {}
        for condition in CONDITIONS:
            model = load_model_checkpoint(
                config, frozen, output_root, condition, seed, device
            )
            raw, raw_audit = k30.predict_model(model, validation, frozen, device)
            curve, curve_audit = k34.exact_curve_prediction(
                raw, side, frozen, geometry_config
            )
            name = f"{condition}_curve"
            predictions[name] = curve
            audits[name] = {"raw_model": raw_audit, "exact_curve": curve_audit}
            del model
        metrics = {name: k27.direct_metrics(truth, value) for name, value in predictions.items()}
        delta = {
            "mean_abs_correlation": metrics["projected_only_curve"]["mean_abs_correlation"]
            - metrics["raw_baseline_curve"]["mean_abs_correlation"],
            "mean_r2": metrics["projected_only_curve"]["mean_r2"]
            - metrics["raw_baseline_curve"]["mean_r2"],
        }
        bootstrap = k29.paired_bootstrap(
            truth,
            predictions["projected_only_curve"],
            predictions["raw_baseline_curve"],
            int(config["evaluation"]["bootstrap_repetitions"]),
            int(config["evaluation"]["bootstrap_seed"]),
            float(config["evaluation"]["bootstrap_confidence"]),
        )
        directional.append(all(value > 0.0 for value in delta.values()))
        baseline_predictions.append(predictions["raw_baseline_curve"])
        projected_predictions.append(predictions["projected_only_curve"])
        seed_results[str(seed)] = {
            "metrics": metrics,
            "delta_projected_minus_raw": delta,
            "paired_bootstrap": bootstrap,
            "projection_audits": audits,
            "directional_both": directional[-1],
        }
        if device.type == "cuda":
            torch.cuda.empty_cache()
    baseline_mean = np.mean(np.stack(baseline_predictions, axis=0), axis=0)
    projected_mean = np.mean(np.stack(projected_predictions, axis=0), axis=0)
    aggregate_metrics = {
        "raw_baseline_curve": k27.direct_metrics(truth, baseline_mean),
        "projected_only_curve": k27.direct_metrics(truth, projected_mean),
    }
    aggregate_delta = {
        "mean_abs_correlation": aggregate_metrics["projected_only_curve"]["mean_abs_correlation"]
        - aggregate_metrics["raw_baseline_curve"]["mean_abs_correlation"],
        "mean_r2": aggregate_metrics["projected_only_curve"]["mean_r2"]
        - aggregate_metrics["raw_baseline_curve"]["mean_r2"],
    }
    aggregate_bootstrap = k29.paired_bootstrap(
        truth,
        projected_mean,
        baseline_mean,
        int(config["evaluation"]["bootstrap_repetitions"]),
        int(config["evaluation"]["bootstrap_seed"]),
        float(config["evaluation"]["bootstrap_confidence"]),
    )
    aggregate_ci = k29.confirmed(aggregate_delta, aggregate_bootstrap)
    replication_pass = all(directional) and aggregate_ci
    validity = {
        "preflight_valid": bool(preflight_payload["valid"]),
        "all_six_training_locks_loaded": len(seed_results) == len(SEEDS),
        "validation_ids_exact": set(int(value) for value in validation.ids)
        == set(int(value) for value in splits["validation"]),
        "test_ids_not_evaluated": not bool(
            set(int(value) for value in validation.ids)
            & set(int(value) for value in splits["test"])
        ),
        "all_metrics_finite": all(
            np.isfinite(value)
            for seed_result in seed_results.values()
            for condition_metrics in seed_result["metrics"].values()
            for value in (
                condition_metrics["mean_abs_correlation"],
                condition_metrics["mean_r2"],
                condition_metrics["mean_normalized_rmse"],
            )
        ),
    }
    valid = bool(all(validity.values()))
    if not valid:
        decision = "invalid"
    elif replication_pass:
        decision = "projected_objective_new_seed_replication_pass_external_unlocked"
    elif all(directional):
        decision = "projected_objective_directional_only_external_locked"
    else:
        decision = "projected_objective_new_seed_replication_failed_external_locked"
    payload = {
        "protocol_version": config["protocol_version"],
        "fact_type": "locked_projected_objective_new_seed_replication",
        "scope": "known_development_validation_new_initializations_external_and_test_sealed",
        "config_sha256": preflight.sha256_file(config_path),
        "source_files_sha256": source_hashes(),
        "validation_count": int(len(validation.ids)),
        "validation_ids_sha256": preflight.sha256_int_array(validation.ids),
        "test_ids_sha256": preflight.sha256_int_array(splits["test"]),
        "seed_results": seed_results,
        "aggregate_seed_ensemble": {
            "metrics": aggregate_metrics,
            "delta_projected_minus_raw": aggregate_delta,
            "paired_bootstrap": aggregate_bootstrap,
        },
        "decision_checks": {
            "directional_seed_count": int(sum(directional)),
            "aggregate_ci": aggregate_ci,
            "replication_pass": replication_pass,
            "external_evaluation_unlocked": replication_pass,
        },
        "validity": validity,
        "valid": valid,
        "development_validation_used_for_design": True,
        "external_evaluated": False,
        "test_evaluated": False,
        "machine_decision": decision,
    }
    output = output_root / "evaluation" / "validation_results.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    preflight.write_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "train", "evaluate"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--condition", choices=CONDITIONS)
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_protocol(args.config)
    output_root = args.output_root or Path(config["runtime"]["output_root"])
    if args.mode == "preflight":
        preflight_main(args.config, output_root)
    elif args.mode == "train":
        if args.condition is None or args.seed is None:
            raise ValueError("train mode requires --condition and --seed")
        train_main(args.config, output_root, args.condition, args.seed)
    else:
        evaluate_main(args.config, output_root)


if __name__ == "__main__":
    main()
