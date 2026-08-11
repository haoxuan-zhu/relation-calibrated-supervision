"""Validation-only three-seed aggregate for the calibrated relation tube K3."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

import audit_calibrated_relation_tube_gauge_k1 as k1
import run_calibrated_relation_tube_k0 as tube
import run_conflict_projected_physics_k80_closed as v17
import run_conditioned_film_diagnostic as v3
import run_diagnostic as base
import run_physics_functional_anchor_training as v11
import run_supervised_continuation_diagnostic as v6


PROTOCOL = "calibrated_relation_tube_multiseed_k3_aggregate_v1"
SEEDS = (0, 42, 3407)
FRESH_SEEDS = (0, 42)
SOURCE_FILES = (
    "aggregate_calibrated_relation_tube_multiseed_k3.py",
    "audit_calibrated_relation_tube_gauge_k1.py",
    "run_calibrated_relation_tube_k0.py",
    "run_sparse_alignment_audit.py",
    "run_conflict_projected_physics_k80_closed.py",
    "run_conditioned_film_diagnostic.py",
    "run_supervised_continuation_diagnostic.py",
    "run_diagnostic.py",
)


def source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    return {name: base.sha256_file(root / name) for name in SOURCE_FILES}


def validate_config(config: dict[str, Any]) -> None:
    if config["protocol_version"] != PROTOCOL:
        raise ValueError("unexpected K3 aggregate protocol")
    calibration = config["calibration"]
    if int(calibration["budget"]) != 80:
        raise ValueError("K3 aggregate is locked to K80")
    if not np.isclose(float(calibration["ridge"]), 0.001):
        raise ValueError("K3 aggregate ridge changed")
    if tuple(int(seed) for seed in config["evaluation"]["seeds"]) != SEEDS:
        raise ValueError("K3 seed registry changed")
    if not np.isclose(
        float(config["evaluation"]["minimum_tube_full_affine_r2"]), 0.50
    ):
        raise ValueError("K3 full-affine threshold changed")
    if set(int(seed) for seed in config["upstream"]["fresh_seeds"]) != set(
        FRESH_SEEDS
    ):
        raise ValueError("K3 fresh-seed registry changed")
    if int(config["upstream"]["legacy_seed"]["seed"]) != 3407:
        raise ValueError("K3 legacy seed changed")


def load_json_with_hash(path: Path, expected_sha256: str) -> dict[str, Any]:
    if base.sha256_file(path) != expected_sha256:
        raise ValueError(f"upstream JSON hash mismatch: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_fresh_tube_context(
    entry: dict[str, Any], expected_seed: int
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Path]:
    config_path = Path(entry["tube_config_path"])
    tube_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    tube.validate_config(tube_config)
    if int(tube_config["training"]["seed"]) != expected_seed:
        raise ValueError("fresh tube seed/config mismatch")
    root = Path(entry["tube_output_root"])
    preflight, _, preflight_lock_path = tube.load_preflight(
        root, config_path, tube_config
    )
    readout_path = root / "formal" / "validation_readout.json"
    readout = json.loads(readout_path.read_text(encoding="utf-8"))
    readout_checks = {
        "protocol": readout["protocol_version"] == tube.CONFIRMATION_PROTOCOL,
        "config": readout["config_sha256"] == base.sha256_file(config_path),
        "preflight": readout["preflight_lock_sha256"]
        == base.sha256_file(preflight_lock_path),
        "four_training_locks": set(readout["training_locks"])
        == set(tube.CONDITIONS),
        "matched_locks": all(readout["matched_lock_checks"].values()),
        "semantic_read": readout["semantic_validation_evaluated"] is True,
        "test_unread": readout["test_evaluated"] is False,
    }
    if not all(readout_checks.values()):
        raise ValueError(f"invalid fresh tube readout: {readout_checks}")
    return tube_config, preflight, readout, readout_path


def load_fresh_tube_model(
    condition: str,
    entry: dict[str, Any],
    tube_config: dict[str, Any],
    preflight: dict[str, Any],
    readout: dict[str, Any],
    device: torch.device,
) -> torch.nn.Module:
    root = Path(entry["tube_output_root"])
    config_path = Path(entry["tube_config_path"])
    preflight_lock_path = root / "preflight" / "preflight_lock.json"
    lock_path = root / "formal" / condition / "training_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    checks = {
        "registered_lock_hash": base.sha256_file(lock_path)
        == readout["training_locks"][condition],
        "formal": lock["mode"] == "formal",
        "status": lock["status"]
        == "locked_before_joint_semantic_validation_readout",
        "protocol": lock["protocol_version"] == tube.CONFIRMATION_PROTOCOL,
        "condition": lock["condition"] == condition,
        "config": lock["config_sha256"] == base.sha256_file(config_path),
        "source": lock["source_files_sha256"] == tube.source_hashes(),
        "preflight": lock["preflight_lock_sha256"]
        == base.sha256_file(preflight_lock_path),
        "geometry": lock["geometry_sha256"] == preflight["geometry_sha256"],
        "validity": all(lock["validity"].values()),
        "semantic_unread_at_lock": lock["semantic_validation_evaluated"] is False,
        "test_unread": lock["test_evaluated"] is False,
    }
    if not all(checks.values()):
        raise ValueError(f"invalid fresh tube lock {condition}: {checks}")
    return tube.load_locked_model(condition, lock, tube_config, preflight, device)


def load_projected_model(
    entry: dict[str, Any], device: torch.device
) -> torch.nn.Module:
    projected_config = yaml.safe_load(
        Path(entry["projected_config_path"]).read_text(encoding="utf-8")
    )
    checkpoint_path = Path(entry["projected_checkpoint_path"])
    if base.sha256_file(checkpoint_path) != entry["projected_checkpoint_sha256"]:
        raise ValueError("projected checkpoint hash mismatch")
    model_cfg = projected_config["model"]
    model = v3.FilmConditionedModel(
        int(model_cfg["latent_dim"]),
        list(model_cfg["learned_indices"]),
        list(model_cfg["anchor_indices"]),
        int(model_cfg["hidden_channels"]),
        int(model_cfg["conv_layers"]),
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    return model


def full_affine_bundle(
    calibration_prediction: np.ndarray,
    validation_prediction: np.ndarray,
    calibration_truth: np.ndarray,
    validation_truth: np.ndarray,
    ridge: float,
) -> dict[str, Any]:
    bundle = k1.metric_bundle(
        calibration_prediction,
        validation_prediction,
        calibration_truth,
        validation_truth,
        ridge,
    )
    return {
        "raw": bundle["raw"],
        "full_affine": bundle["full_affine"],
        "full_weights": bundle["full_weights"],
        "full_input_stats": bundle["full_input_stats"],
    }


def fresh_seed_result(
    seed: int,
    entry: dict[str, Any],
    config: dict[str, Any],
    images: np.ndarray,
    latents: np.ndarray,
    subset: np.ndarray,
    calibration_truth: np.ndarray,
    validation_truth: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    tube_config, preflight, readout, readout_path = validate_fresh_tube_context(
        entry, seed
    )
    start = int(config["split"]["train_end"])
    end = int(config["split"]["validation_end"])
    ridge = float(config["calibration"]["ridge"])
    runs: dict[str, Any] = {}
    for condition in tube.CONDITIONS:
        model = load_fresh_tube_model(
            condition, entry, tube_config, preflight, readout, device
        )
        calibration_prediction = k1.predict_selected(
            model, images, latents, subset, 512, device
        )
        validation_prediction = v6.predict_rgb(
            model, images, latents, start, end, 512, device
        )
        runs[condition] = full_affine_bundle(
            calibration_prediction,
            validation_prediction,
            calibration_truth,
            validation_truth,
            ridge,
        )
        expected = readout["semantic_validation"][condition]
        if not (
            np.isclose(
                runs[condition]["raw"]["mean_direct_abs_correlation"],
                expected["mean_direct_abs_correlation"],
                atol=5e-6,
            )
            and np.isclose(
                runs[condition]["raw"]["mean_r2"], expected["mean_r2"], atol=5e-6
            )
        ):
            raise RuntimeError(f"raw tube metric drift: seed{seed}/{condition}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    projected = load_projected_model(entry, device)
    projected_calibration = k1.predict_selected(
        projected, images, latents, subset, 512, device
    )
    projected_validation = v6.predict_rgb(
        projected, images, latents, start, end, 512, device
    )
    runs["projected_physical_correct"] = full_affine_bundle(
        projected_calibration,
        projected_validation,
        calibration_truth,
        validation_truth,
        ridge,
    )
    projected_raw = runs["projected_physical_correct"]["raw"]
    projected_checks = {
        "correlation_exact": bool(
            np.isclose(
                projected_raw["mean_direct_abs_correlation"],
                float(entry["projected_raw_validation_correlation"]),
                atol=5e-6,
            )
        ),
        "r2_exact": bool(
            np.isclose(
                projected_raw["mean_r2"],
                float(entry["projected_raw_validation_r2"]),
                atol=5e-6,
            )
        ),
    }
    if not all(projected_checks.values()):
        raise RuntimeError(f"projected comparator drift: seed{seed}/{projected_checks}")
    return {
        "seed": seed,
        "tube_readout": str(readout_path),
        "tube_readout_sha256": base.sha256_file(readout_path),
        "tube_machine_decision": readout["decision"],
        "projected_raw_checks": projected_checks,
        "runs": runs,
        "test_evaluated": False,
    }


def legacy_seed_result(entry: dict[str, Any]) -> dict[str, Any]:
    readout_path = Path(entry["tube_validation_readout_path"])
    readout = load_json_with_hash(
        readout_path, entry["tube_validation_readout_sha256"]
    )
    gauge_path = Path(entry["gauge_result_path"])
    gauge = load_json_with_hash(gauge_path, entry["gauge_result_sha256"])
    checks = {
        "readout_test_unread": readout["test_evaluated"] is False,
        "gauge_test_unread": gauge["test_evaluated"] is False,
        "gauge_protocol": gauge["protocol_version"] == k1.PROTOCOL,
    }
    if not all(checks.values()):
        raise ValueError(f"invalid legacy seed3407 assets: {checks}")
    runs = {
        name: {
            "raw": gauge["runs"][name]["raw"],
            "full_affine": gauge["runs"][name]["full_affine"],
        }
        for name in (*tube.CONDITIONS, "projected_physical_correct")
    }
    return {
        "seed": 3407,
        "tube_readout": str(readout_path),
        "tube_readout_sha256": entry["tube_validation_readout_sha256"],
        "gauge_result": str(gauge_path),
        "gauge_result_sha256": entry["gauge_result_sha256"],
        "tube_machine_decision": readout["decision"],
        "runs": runs,
        "test_evaluated": False,
    }


def decide(results: dict[str, dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    threshold = float(config["evaluation"]["minimum_tube_full_affine_r2"])
    per_seed: dict[str, Any] = {}
    for seed in SEEDS:
        result = results[str(seed)]
        runs = result["runs"]
        tube_run = runs["tube_correct"]
        projected = runs["projected_physical_correct"]
        tube_full = tube_run["full_affine"]
        projected_full = projected["full_affine"]
        checks = {
            "raw_structure_mechanism": all(
                result["tube_machine_decision"]["mechanism_checks"].values()
            ),
            "full_r2_above_projected": tube_full["mean_r2"]
            > projected_full["mean_r2"],
            "full_r2_above_minimum": tube_full["mean_r2"] > threshold,
            "full_r2_above_unbounded": tube_full["mean_r2"]
            > runs["unbounded_correct"]["full_affine"]["mean_r2"],
            "full_r2_above_permuted": tube_full["mean_r2"]
            > runs["tube_permuted_matched"]["full_affine"]["mean_r2"],
            "test_unread": result["test_evaluated"] is False,
        }
        per_seed[str(seed)] = {
            "checks": checks,
            "all_checks_pass": all(checks.values()),
            "deltas": {
                "tube_minus_projected_full_r2": tube_full["mean_r2"]
                - projected_full["mean_r2"],
                "tube_minus_projected_full_correlation": tube_full[
                    "mean_direct_abs_correlation"
                ]
                - projected_full["mean_direct_abs_correlation"],
                "tube_minus_unbounded_full_r2": tube_full["mean_r2"]
                - runs["unbounded_correct"]["full_affine"]["mean_r2"],
                "tube_minus_permuted_full_r2": tube_full["mean_r2"]
                - runs["tube_permuted_matched"]["full_affine"]["mean_r2"],
            },
        }
    all_seed_checks = all(item["all_checks_pass"] for item in per_seed.values())
    positive_projected_seeds = sum(
        item["checks"]["full_r2_above_projected"] for item in per_seed.values()
    )
    if all_seed_checks:
        verdict = "relation_tube_multiseed_structure_supported"
    elif positive_projected_seeds == 2:
        verdict = "relation_tube_multiseed_mixed_two_of_three"
    else:
        verdict = "relation_tube_multiseed_not_supported"
    return {
        "verdict": verdict,
        "per_seed": per_seed,
        "positive_tube_minus_projected_seed_count": positive_projected_seeds,
        "all_seed_checks_pass": all_seed_checks,
        "test_evaluated": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    output_root = Path(config["runtime"]["output_root"])
    output_path = output_root / "multiseed_aggregate.json"
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite K3 aggregate: {output_path}")

    torch.set_num_threads(int(config["runtime"]["num_threads"]))
    torch.set_float32_matmul_precision("high")
    device = torch.device(
        config["runtime"]["device"] if torch.cuda.is_available() else "cpu"
    )
    images = np.load(Path(config["dataset"]["cache_path"]), mmap_mode="r")
    raw_latents = base.load_latents(config)
    subset, subset_audit = tube.dynamic.build_subset(config)
    latents, latent_mean, latent_std = v17.normalize_with_k80_rgb_statistics(
        raw_latents, int(config["split"]["train_end"]), subset
    )
    calibration_truth = latents[0, subset, :3]
    validation_truth = latents[
        0,
        int(config["split"]["train_end"]) : int(config["split"]["validation_end"]),
        :3,
    ]
    results: dict[str, Any] = {}
    for seed in FRESH_SEEDS:
        entry = config["upstream"]["fresh_seeds"][seed]
        results[str(seed)] = fresh_seed_result(
            seed,
            entry,
            config,
            images,
            latents,
            subset,
            calibration_truth,
            validation_truth,
            device,
        )
    results["3407"] = legacy_seed_result(config["upstream"]["legacy_seed"])
    result = {
        "protocol_version": PROTOCOL,
        "mode": "three_seed_k80_full_affine_validation_only_aggregate",
        "config_sha256": base.sha256_file(config_path),
        "source_files_sha256": source_hashes(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "subset": subset_audit,
        "normalization": {"latent_mean": latent_mean, "latent_std": latent_std},
        "seeds": results,
        "decision": decide(results, config),
        "fit_rows": "registered_k80_observational_train_only",
        "evaluation_rows": "observational_validation_8000_9000",
        "test_evaluated": False,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    v11.atomic_json(output_path, result)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "sha256": base.sha256_file(output_path),
                "verdict": result["decision"]["verdict"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
