"""Evaluate K35 projected supervision on the frozen K31 external partition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

import audit_instrumented_slope_interface_k27 as k27
import causalverse_slope_preflight as slope
import evaluate_instrumented_slope_external_k31 as k31
import evaluate_instrumented_slope_validation_k29 as k29
import run_causalverse_slope_preflight as preflight
import run_instrumented_slope_neural_tube_k30 as k30
import run_instrumented_slope_projected_replication_k35 as k35
import run_instrumented_slope_projected_training_k34 as k34


SEEDS = (123, 2026, 31415)


def source_hashes() -> dict[str, str]:
    paths = {
        "k36": Path(__file__),
        "k35": Path(k35.__file__),
        "k34": Path(k34.__file__),
        "k31": Path(k31.__file__),
        "k30": Path(k30.__file__),
        "k29": Path(k29.__file__),
        "k27": Path(k27.__file__),
        "slope": Path(slope.__file__),
        "preflight": Path(preflight.__file__),
    }
    return {name: preflight.sha256_file(path) for name, path in paths.items()}


def load_protocol(config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["protocol_version"] != "instrumented_slope_projected_external_k36":
        raise ValueError("unexpected K36 protocol")
    if tuple(int(value) for value in config["evaluation"]["seeds"]) != SEEDS:
        raise ValueError("K36 seed registry changed")
    if tuple(config["evaluation"]["candidates"]) != (
        "raw_baseline_curve",
        "projected_only_curve",
    ):
        raise ValueError("K36 candidate registry changed")
    return config


def run(config_path: Path, output_path: Path) -> None:
    config = load_protocol(config_path)
    source = config["source_contract"]
    external_config_path = Path(source["external_config"])
    k35_config_path = Path(source["k35_config"])
    k35_result_path = Path(source["k35_result"])
    external_config = yaml.safe_load(external_config_path.read_text(encoding="utf-8"))
    k35_config = k35.load_protocol(k35_config_path)
    k35_result = json.loads(k35_result_path.read_text(encoding="utf-8"))
    source_contract_hashes = {
        "external_config": preflight.sha256_file(external_config_path),
        "k35_config": preflight.sha256_file(k35_config_path),
        "k35_result": preflight.sha256_file(k35_result_path),
    }
    external, external_filter_audit = k31.load_external(external_config)
    (
        _, _, geometry_config, _, grouped, splits, label_ids, _, _, _, _,
    ) = k35.load_context(k35_config)
    train = slope.select_ids(grouped, splits["train"])
    frozen = k30.frozen_components(train, label_ids, geometry_config)
    truth = external.latents[:, k27.TARGET_INDICES]
    side = external.latents[:, k27.SIDE_INDICES]
    device = torch.device(config["runtime"]["device"] if torch.cuda.is_available() else "cpu")
    output_root = Path(source["k35_output_root"])
    seed_results: dict[str, Any] = {}
    raw_predictions: list[np.ndarray] = []
    projected_predictions: list[np.ndarray] = []
    directional: list[bool] = []
    for seed in SEEDS:
        predictions: dict[str, np.ndarray] = {}
        audits: dict[str, Any] = {}
        for condition in k35.CONDITIONS:
            model = k35.load_model_checkpoint(
                k35_config, frozen, output_root, condition, seed, device
            )
            raw, raw_audit = k30.predict_model(model, external, frozen, device)
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
        raw_predictions.append(predictions["raw_baseline_curve"])
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
    raw_mean = np.mean(np.stack(raw_predictions, axis=0), axis=0)
    projected_mean = np.mean(np.stack(projected_predictions, axis=0), axis=0)
    aggregate_metrics = {
        "raw_baseline_curve": k27.direct_metrics(truth, raw_mean),
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
        raw_mean,
        int(config["evaluation"]["bootstrap_repetitions"]),
        int(config["evaluation"]["bootstrap_seed"]),
        float(config["evaluation"]["bootstrap_confidence"]),
    )
    aggregate_ci = k29.confirmed(aggregate_delta, aggregate_bootstrap)
    external_pass = all(directional) and aggregate_ci
    development_ids = set(int(value) for value in grouped.ids)
    external_ids = set(int(value) for value in external.ids)
    validity = {
        "external_config_sha_matches": source_contract_hashes["external_config"]
        == str(source["external_config_sha256"]),
        "k35_config_sha_matches": source_contract_hashes["k35_config"]
        == str(source["k35_config_sha256"]),
        "k35_result_sha_matches": source_contract_hashes["k35_result"]
        == str(source["k35_result_sha256"]),
        "k35_result_valid": bool(k35_result["valid"]),
        "k35_external_unlock_true": bool(
            k35_result["decision_checks"]["external_evaluation_unlocked"]
        ),
        "external_feature_cache_sha_matches": preflight.sha256_file(
            Path(external_config["runtime"]["feature_cache"])
        )
        == str(external_config["runtime"]["feature_cache_sha256"]),
        "external_filter_contract_matches": external_filter_audit[
            "incomplete_product_ids"
        ]
        == list(external_config["dataset"]["expected_incomplete_product_ids"]),
        "external_ids_disjoint_from_development": not bool(external_ids & development_ids),
        "all_external_products_have_four_views": external.features.shape[1] == 4,
        "all_six_locked_models_loaded": len(seed_results) == len(SEEDS),
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
    elif external_pass:
        decision = "projected_objective_external_confirmation_pass"
    elif all(directional):
        decision = "projected_objective_external_directional_only"
    else:
        decision = "projected_objective_external_confirmation_failed"
    payload = {
        "protocol_version": config["protocol_version"],
        "fact_type": "locked_unread_source_shard_projected_objective_external_evaluation",
        "scope": "frozen_k31_external_products_only_k26_test_sealed",
        "config_sha256": preflight.sha256_file(config_path),
        "source_files_sha256": source_hashes(),
        "source_contract_hashes": source_contract_hashes,
        "feature_cache_sha256": preflight.sha256_file(
            Path(external_config["runtime"]["feature_cache"])
        ),
        "external_product_count": int(len(external.ids)),
        "external_ids_sha256": preflight.sha256_int_array(external.ids),
        "seed_results": seed_results,
        "aggregate_seed_ensemble": {
            "metrics": aggregate_metrics,
            "delta_projected_minus_raw": aggregate_delta,
            "paired_bootstrap": aggregate_bootstrap,
        },
        "decision_checks": {
            "directional_seed_count": int(sum(directional)),
            "aggregate_ci": aggregate_ci,
            "external_confirmation_pass": external_pass,
        },
        "validity": validity,
        "valid": valid,
        "external_source_shards_evaluated": True,
        "sealed_k26_test_evaluated": False,
        "machine_decision": decision,
    }
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    preflight.write_json(output_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_protocol(args.config)
    run(args.config, args.output or Path(config["runtime"]["output"]))


if __name__ == "__main__":
    main()
