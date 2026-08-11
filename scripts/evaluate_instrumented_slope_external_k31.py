"""Evaluate locked K30 models on the unread Instrumented Slope K31 shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

import audit_instrumented_slope_geometry_k28 as k28
import audit_instrumented_slope_interface_k27 as k27
import causalverse_slope_preflight as slope
import evaluate_instrumented_slope_validation_k29 as k29
import run_causalverse_slope_preflight as preflight
import run_instrumented_slope_neural_tube_k30 as k30


def load_external(config: dict[str, Any]) -> tuple[slope.GroupedFeatures, dict[str, Any]]:
    cache_path = Path(config["runtime"]["feature_cache"])
    with np.load(cache_path, allow_pickle=False) as archive:
        required = {"features", "ids", "views", "latents", "latent_columns"}
        if not required.issubset(archive.files):
            raise ValueError(f"external cache is missing {sorted(required - set(archive.files))}")
        cache = {name: archive[name] for name in required}
    expected_columns = np.asarray(config["dataset"]["latent_columns"], dtype=np.str_)
    if not np.array_equal(cache["latent_columns"], expected_columns):
        raise ValueError("external latent-column registry drifted")
    incomplete = slope.incomplete_product_ids(
        cache["ids"], cache["views"], config["dataset"]["required_views"]
    )
    expected_incomplete = np.asarray(
        config["dataset"]["expected_incomplete_product_ids"], dtype=np.int64
    )
    if not np.array_equal(incomplete, expected_incomplete):
        raise ValueError(
            f"external incomplete IDs differ: {incomplete.tolist()} != {expected_incomplete.tolist()}"
        )
    keep = ~np.isin(cache["ids"], incomplete)
    grouped = slope.group_features_by_id(
        cache["ids"][keep],
        cache["views"][keep],
        cache["features"][keep],
        cache["latents"][keep],
        config["dataset"]["required_views"],
    )
    return grouped, {
        "source_row_count": int(len(cache["ids"])),
        "retained_row_count": int(np.sum(keep)),
        "dropped_row_count": int(np.sum(~keep)),
        "incomplete_product_ids": incomplete.tolist(),
        "incomplete_product_ids_sha256": preflight.sha256_int_array(incomplete),
    }


def aggregate_decision(
    valid: bool,
    curve_formula_directional: list[bool],
    curve_formula_ci: list[bool],
    curve_unbounded_directional: list[bool],
    curve_unbounded_ci: list[bool],
) -> str:
    if not valid:
        return "invalid"
    if (
        all(curve_formula_directional)
        and all(curve_formula_ci)
        and all(curve_unbounded_directional)
        and all(curve_unbounded_ci)
    ):
        return "nonlinear_curve_unread_shards_multiseed_ci_confirmed"
    if all(curve_formula_directional) and all(curve_unbounded_directional):
        return "nonlinear_curve_unread_shards_multiseed_directional"
    return "nonlinear_curve_unread_shards_mixed"


def run(config_path: Path, output_path: Path) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    preextract_config_path = Path(config["source_contract"]["preextract_config"])
    neural_config_path = Path(config["source_contract"]["neural_config"])
    neural_result_path = Path(config["source_contract"]["neural_validation_result"])
    neural_config = k30.load_protocol(neural_config_path)
    neural_result = json.loads(neural_result_path.read_text(encoding="utf-8"))
    (
        _, _, geometry_config, interface_config, old_grouped, old_splits, label_ids, _, _,
    ) = k30.load_context_chain(neural_config)
    old_train = slope.select_ids(old_grouped, old_splits["train"])
    frozen = k30.frozen_components(old_train, label_ids, geometry_config)
    external, filter_audit = load_external(config)

    truth = external.latents[:, k27.TARGET_INDICES]
    side = external.latents[:, k27.SIDE_INDICES]
    center = k28.physical_state_from_roughness(
        side, float(frozen["mean_roughness"]), frozen["parameters"]
    )
    lower, upper = (
        float(value) for value in geometry_config["calibration"]["roughness_physical_bounds"]
    )
    scalar_span = float(frozen["geometry"]["roughness_scalar"]["radius"]) * float(
        frozen["target_std"][0]
    )
    rank_lower = max(lower, float(frozen["mean_roughness"]) - scalar_span)
    rank_upper = min(upper, float(frozen["mean_roughness"]) + scalar_span)
    device = torch.device(neural_config["runtime"]["device"] if torch.cuda.is_available() else "cpu")
    output_root = Path(neural_config["runtime"]["output_root"])

    seed_results: dict[str, Any] = {}
    curve_formula_directional: list[bool] = []
    curve_formula_ci: list[bool] = []
    curve_unbounded_directional: list[bool] = []
    curve_unbounded_ci: list[bool] = []
    isotropic_directional: list[bool] = []
    isotropic_ci: list[bool] = []
    for seed_value in config["evaluation"]["seeds"]:
        seed = int(seed_value)
        unbounded_model = k30.load_model_checkpoint(
            neural_config, frozen, output_root, "unbounded", seed, device
        )
        isotropic_model = k30.load_model_checkpoint(
            neural_config, frozen, output_root, "isotropic", seed, device
        )
        unbounded, unbounded_audit = k30.predict_model(
            unbounded_model, external, frozen, device
        )
        isotropic, isotropic_audit = k30.predict_model(
            isotropic_model, external, frozen, device
        )
        formula = k28.physical_state_from_roughness(
            side, np.clip(unbounded[:, 0], lower, upper), frozen["parameters"]
        )
        curve, curve_audit = k28.nonlinear_curve_projection(
            unbounded,
            side,
            frozen["parameters"],
            frozen["target_std"],
            rank_lower,
            rank_upper,
            float(geometry_config["projection"]["nonlinear_tolerance"]),
            int(geometry_config["projection"]["nonlinear_max_iterations"]),
        )
        predictions = {
            "physical_center": center,
            "unbounded": unbounded,
            "isotropic": isotropic,
            "formula_from_unbounded_roughness": formula,
            "nonlinear_curve_from_unbounded": curve,
        }
        if list(predictions) != list(config["evaluation"]["candidates"]):
            raise ValueError("K31 candidate registry drifted")
        metrics = {name: k27.direct_metrics(truth, value) for name, value in predictions.items()}
        pairs = {
            "isotropic_minus_unbounded": ("isotropic", "unbounded"),
            "curve_minus_unbounded": ("nonlinear_curve_from_unbounded", "unbounded"),
            "curve_minus_formula": (
                "nonlinear_curve_from_unbounded",
                "formula_from_unbounded_roughness",
            ),
        }
        deltas: dict[str, dict[str, float]] = {}
        bootstrap: dict[str, Any] = {}
        for name, (candidate, baseline) in pairs.items():
            deltas[name] = {
                "mean_abs_correlation": metrics[candidate]["mean_abs_correlation"]
                - metrics[baseline]["mean_abs_correlation"],
                "mean_r2": metrics[candidate]["mean_r2"] - metrics[baseline]["mean_r2"],
            }
            bootstrap[name] = k29.paired_bootstrap(
                truth,
                predictions[candidate],
                predictions[baseline],
                int(config["evaluation"]["bootstrap_repetitions"]),
                int(config["evaluation"]["bootstrap_seed"]),
                float(config["evaluation"]["bootstrap_confidence"]),
            )
        cf_direction = all(value > 0.0 for value in deltas["curve_minus_formula"].values())
        cf_ci = k29.confirmed(deltas["curve_minus_formula"], bootstrap["curve_minus_formula"])
        cu_direction = all(value > 0.0 for value in deltas["curve_minus_unbounded"].values())
        cu_ci = k29.confirmed(deltas["curve_minus_unbounded"], bootstrap["curve_minus_unbounded"])
        iso_direction = all(value > 0.0 for value in deltas["isotropic_minus_unbounded"].values())
        iso_strong = k29.confirmed(
            deltas["isotropic_minus_unbounded"], bootstrap["isotropic_minus_unbounded"]
        )
        curve_formula_directional.append(cf_direction)
        curve_formula_ci.append(cf_ci)
        curve_unbounded_directional.append(cu_direction)
        curve_unbounded_ci.append(cu_ci)
        isotropic_directional.append(iso_direction)
        isotropic_ci.append(iso_strong)
        seed_results[str(seed)] = {
            "metrics": metrics,
            "deltas": deltas,
            "paired_bootstrap": bootstrap,
            "projection_audits": {
                "unbounded": unbounded_audit,
                "isotropic": isotropic_audit,
                "nonlinear_curve": curve_audit,
            },
            "decision_checks": {
                "curve_directional_both_vs_formula": cf_direction,
                "curve_joint_ci_confirmed_vs_formula": cf_ci,
                "curve_directional_both_vs_unbounded": cu_direction,
                "curve_joint_ci_confirmed_vs_unbounded": cu_ci,
                "isotropic_directional_both": iso_direction,
                "isotropic_joint_ci_confirmed": iso_strong,
            },
        }
        del unbounded_model, isotropic_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    metadata_audit_path = Path(config["source_contract"]["metadata_audit"])
    metadata_audit = json.loads(metadata_audit_path.read_text(encoding="utf-8"))
    external_ids = set(int(value) for value in external.ids)
    old_ids = set(int(value) for value in old_grouped.ids)
    validity = {
        "preextract_config_sha_matches": preflight.sha256_file(preextract_config_path)
        == str(config["source_contract"]["preextract_config_sha256"]),
        "neural_config_sha_matches": preflight.sha256_file(neural_config_path)
        == str(config["source_contract"]["neural_config_sha256"]),
        "neural_result_sha_matches": preflight.sha256_file(neural_result_path)
        == str(config["source_contract"]["neural_validation_result_sha256"]),
        "neural_result_valid": bool(neural_result["valid"]),
        "metadata_audit_sha_matches": preflight.sha256_file(metadata_audit_path)
        == str(config["source_contract"]["metadata_audit_sha256"]),
        "metadata_audit_presemantic": not bool(metadata_audit["semantic_target_values_read"])
        and not bool(metadata_audit["images_decoded"]),
        "feature_cache_sha_matches": preflight.sha256_file(Path(config["runtime"]["feature_cache"]))
        == str(config["runtime"]["feature_cache_sha256"]),
        "incomplete_ids_match_metadata_audit": filter_audit["incomplete_product_ids"]
        == metadata_audit["incomplete_product_ids"],
        "external_ids_disjoint_from_k22_k30": not bool(external_ids & old_ids),
        "all_external_products_have_four_views": external.features.shape[1] == 4,
        "all_six_locks_loaded": len(seed_results) == 3,
        "all_metrics_finite": all(
            np.isfinite(metric)
            for seed_result in seed_results.values()
            for condition_metrics in seed_result["metrics"].values()
            for metric in (
                condition_metrics["mean_abs_correlation"],
                condition_metrics["mean_r2"],
                condition_metrics["mean_normalized_rmse"],
            )
        ),
    }
    valid = bool(all(validity.values()))
    decision = aggregate_decision(
        valid,
        curve_formula_directional,
        curve_formula_ci,
        curve_unbounded_directional,
        curve_unbounded_ci,
    )
    payload = {
        "protocol_version": config["protocol_version"],
        "fact_type": "locked_unread_source_shard_external_evaluation",
        "config_sha256": preflight.sha256_file(config_path),
        "feature_cache_sha256": preflight.sha256_file(Path(config["runtime"]["feature_cache"])),
        "metadata_audit_sha256": preflight.sha256_file(metadata_audit_path),
        "external_product_count": int(len(external.ids)),
        "external_ids_sha256": preflight.sha256_int_array(external.ids),
        "filter_audit": filter_audit,
        "seed_results": seed_results,
        "aggregate_checks": {
            "curve_formula_directional_seed_count": int(sum(curve_formula_directional)),
            "curve_formula_ci_seed_count": int(sum(curve_formula_ci)),
            "curve_unbounded_directional_seed_count": int(sum(curve_unbounded_directional)),
            "curve_unbounded_ci_seed_count": int(sum(curve_unbounded_ci)),
            "isotropic_directional_seed_count": int(sum(isotropic_directional)),
            "isotropic_ci_seed_count": int(sum(isotropic_ci)),
        },
        "validity": validity,
        "valid": valid,
        "external_source_shards_evaluated": True,
        "sealed_k26_test_evaluated": False,
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
