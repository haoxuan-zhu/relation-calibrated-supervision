"""K44 validation-only subset check for Tube versus matched propagation."""

from __future__ import annotations

import argparse
import copy
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

import audit_relation_tube_geometry_downstream_k6 as k6
import run_conflict_projected_physics_training as v13
import run_diagnostic as base
import run_dynamic_residual_propagation_k0 as dynamic
import run_physics_functional_anchor_training as v11
import run_raw_ridge_propagation_condition as raw_condition
import run_raw_ridge_propagation_subset_robustness as raw_subset
import run_relation_tube_calibration_subset_k8 as k8
import run_supervised_continuation_diagnostic as v6


PROTOCOL = "relation_tube_vs_raw_subset_k44_v1"
SUBSET_SEEDS = (20260811, 20260821, 20260831)
ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = tuple(
    dict.fromkeys(
        [
            "run_relation_tube_vs_raw_subset_k44.py",
            *raw_subset.budget.SOURCE_FILES,
            *k8.SOURCE_FILES,
        ]
    )
)


def source_hashes() -> dict[str, str]:
    script_root = Path(__file__).resolve().parent
    return {name: base.sha256_file(script_root / name) for name in SOURCE_FILES}


def project_path(value: str) -> Path:
    return (ROOT / value).resolve()


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def subset_entry(config: dict[str, Any], subset_seed: int) -> dict[str, Any]:
    try:
        return config["subsets"][subset_seed]
    except KeyError:
        return config["subsets"][str(subset_seed)]


def validate_config(config: dict[str, Any]) -> None:
    if config["protocol_version"] != PROTOCOL:
        raise ValueError("unexpected K44 protocol")
    scope = config["scope"]
    if tuple(int(value) for value in scope["subset_seeds"]) != SUBSET_SEEDS:
        raise ValueError("K44 subset registry changed")
    if int(scope["budget"]) != 80 or int(scope["model_seed"]) != 3407:
        raise ValueError("K44 budget or model seed changed")
    if not np.isclose(float(scope["raw_ridge_alpha"]), 0.1):
        raise ValueError("K44 center alpha changed")
    if scope["test_evaluated"] is not False:
        raise ValueError("K44 must not read test")
    if int(config["bootstrap"]["repetitions"]) < 1000:
        raise ValueError("K44 bootstrap is too small")
    if not np.isclose(float(config["bootstrap"]["confidence"]), 0.95):
        raise ValueError("K44 confidence level changed")

    expected_entrypoint = config["implementation"]["entrypoint_sha256"]
    actual_entrypoint = base.sha256_file(Path(__file__).resolve())
    if actual_entrypoint != expected_entrypoint:
        raise ValueError("K44 entrypoint hash mismatch")

    for subset_seed in SUBSET_SEEDS:
        entry = subset_entry(config, subset_seed)
        raw_path = project_path(entry["raw_registry"])
        tube_path = project_path(entry["tube_config"])
        readout_path = project_path(entry["locked_tube_readout"])
        hashes = {
            "raw_registry": (raw_path, entry["raw_registry_sha256"]),
            "tube_config": (tube_path, entry["tube_config_sha256"]),
            "tube_readout": (readout_path, entry["locked_tube_readout_sha256"]),
        }
        for name, (path, expected) in hashes.items():
            if base.sha256_file(path) != expected:
                raise ValueError(f"K44 {name} hash mismatch for subset {subset_seed}")

        raw_registry = load_yaml(raw_path)
        tube_config = load_yaml(tube_path)
        locked_readout = load_json(readout_path)
        raw_subset.validate_registry(raw_registry)
        k8.validate_config(tube_config)
        raw_run = raw_subset.budget.build_run_config(raw_registry, "empirical", 80, 3407)
        raw_rows, raw_audit = raw_subset.budget.build_subset(raw_registry, 80)
        tube_rows, tube_audit = dynamic.build_subset(tube_config)

        checks = {
            "subset_seed": int(tube_config["calibration"]["subset_seed"]) == subset_seed,
            "raw_subset_seed": int(raw_registry["budget_curve"]["subset_seed"]) == subset_seed,
            "subset_rows": np.array_equal(raw_rows, tube_rows),
            "subset_hash": raw_audit["subset_sha256"] == tube_audit["subset_sha256"],
            "dataset": raw_run["dataset"] == tube_config["dataset"],
            "split": raw_run["split"] == tube_config["split"],
            "model": raw_run["model"] == tube_config["model"],
            "training": all(
                raw_run["training"][name] == tube_config["training"][name]
                for name in (
                    "shuffle_seed",
                    "seed",
                    "epochs",
                    "batch_size",
                    "learning_rate",
                    "scheduler_factor",
                    "scheduler_patience",
                    "validation_interval",
                    "targets",
                    "kappa",
                    "eta",
                    "mu",
                )
            ),
            "initial_state": raw_run["initialization"]["expected_state_dict_sha256"]
            == tube_config["initialization"]["expected_state_dict_sha256"],
            "alpha": np.isclose(float(tube_config["calibration"]["raw_ridge_alpha"]), 0.1),
            "locked_readout_protocol": locked_readout["protocol_version"] == k8.PROTOCOL,
            "locked_readout_subset": int(locked_readout["subset_seed"]) == subset_seed,
            "locked_readout_seed": int(tube_config["training"]["seed"]) == 3407,
            "locked_readout_validation": locked_readout["semantic_validation_evaluated"] is True,
            "locked_readout_test": locked_readout["test_evaluated"] is False,
        }
        if not all(checks.values()):
            raise ValueError(f"K44 upstream contract mismatch for subset {subset_seed}: {checks}")


def subset_root(config: dict[str, Any], subset_seed: int, override: Path | None) -> Path:
    root = override or Path(config["runtime"]["output_root"])
    return root.resolve() / f"subset{subset_seed}"


def load_upstream(
    config: dict[str, Any], subset_seed: int
) -> tuple[dict[str, Any], Path, dict[str, Any], Path, dict[str, Any], dict[str, Any]]:
    entry = subset_entry(config, subset_seed)
    raw_path = project_path(entry["raw_registry"])
    tube_path = project_path(entry["tube_config"])
    raw_registry = load_yaml(raw_path)
    tube_config = load_yaml(tube_path)
    tube_root = Path(entry["tube_artifact_root"]).resolve()
    tube_payload, tube_preflight_lock = k8.load_preflight(tube_root, tube_path)
    return raw_registry, raw_path, tube_config, tube_path, tube_payload, {
        "root": tube_root,
        "preflight_lock": tube_preflight_lock,
    }


def preflight_main(
    config: dict[str, Any], config_path: Path, subset_seed: int, output_override: Path | None
) -> None:
    root = subset_root(config, subset_seed, output_override)
    output_dir = root / "preflight"
    payload_path = output_dir / "preflight.json"
    lock_path = output_dir / "preflight_lock.json"
    if payload_path.exists() or lock_path.exists():
        raise FileExistsError(f"refusing to overwrite K44 preflight: {output_dir}")

    raw_registry, raw_path, tube_config, tube_path, tube_payload, tube_assets = load_upstream(
        config, subset_seed
    )
    derived = raw_subset.budget.build_run_config(raw_registry, "empirical", 80, 3407)
    images = np.load(Path(derived["dataset"]["cache_path"]), mmap_mode="r")
    raw_latents = base.load_latents(derived).astype(np.float64)
    raw_rows, raw_audit = raw_subset.budget.build_subset(raw_registry, 80)
    tube_rows, tube_audit = dynamic.build_subset(tube_config)
    recomputed = k8.build_preflight_payload(tube_config, images, raw_latents)

    teacher = tube_payload["teachers"]["correct"]
    checks = {
        "rows_exact": bool(np.array_equal(raw_rows, tube_rows)),
        "subset_hash_exact": raw_audit["subset_sha256"] == tube_audit["subset_sha256"],
        "teacher_alpha_exact": bool(np.isclose(float(teacher["audit"]["alpha"]), 0.1)),
        "teacher_coefficients_exact": teacher["audit"]["coefficients_sha256"]
        == recomputed["teachers"]["correct"]["audit"]["coefficients_sha256"],
        "normalization_mean_exact": bool(
            np.allclose(
                tube_payload["normalization"]["latent_mean"],
                recomputed["normalization"]["latent_mean"],
                atol=0.0,
                rtol=0.0,
            )
        ),
        "normalization_std_exact": bool(
            np.allclose(
                tube_payload["normalization"]["latent_std"],
                recomputed["normalization"]["latent_std"],
                atol=0.0,
                rtol=0.0,
            )
        ),
        "initial_state_exact": derived["initialization"]["expected_state_dict_sha256"]
        == tube_config["initialization"]["expected_state_dict_sha256"],
        "test_unread": True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"K44 preflight failed: {checks}")

    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol_version": PROTOCOL,
        "mode": "matched_center_train_preflight",
        "subset_seed": subset_seed,
        "config_sha256": base.sha256_file(config_path),
        "source_files_sha256": source_hashes(),
        "raw_registry": {"path": str(raw_path), "sha256": base.sha256_file(raw_path)},
        "tube_config": {"path": str(tube_path), "sha256": base.sha256_file(tube_path)},
        "tube_preflight_lock_sha256": base.sha256_file(tube_assets["preflight_lock"]),
        "subset": raw_audit,
        "teacher": teacher,
        "normalization": tube_payload["normalization"],
        "checks": checks,
        "semantic_validation_evaluated": False,
        "test_evaluated": False,
    }
    v11.atomic_json(payload_path, payload)
    v11.atomic_json(
        lock_path,
        {
            "status": "locked_k44_before_training",
            "protocol_version": PROTOCOL,
            "config_sha256": base.sha256_file(config_path),
            "source_files_sha256": source_hashes(),
            "payload_sha256": base.sha256_file(payload_path),
            "semantic_validation_evaluated": False,
            "test_evaluated": False,
        },
    )


def load_preflight(
    root: Path, config_path: Path, subset_seed: int
) -> tuple[dict[str, Any], Path]:
    payload_path = root / "preflight" / "preflight.json"
    lock_path = root / "preflight" / "preflight_lock.json"
    payload = load_json(payload_path)
    lock = load_json(lock_path)
    checks = {
        "status": lock["status"] == "locked_k44_before_training",
        "protocol": lock["protocol_version"] == PROTOCOL,
        "subset": int(payload["subset_seed"]) == subset_seed,
        "config": lock["config_sha256"] == base.sha256_file(config_path),
        "source": lock["source_files_sha256"] == source_hashes(),
        "payload": lock["payload_sha256"] == base.sha256_file(payload_path),
        "valid": all(payload["checks"].values()),
        "test_unread": lock["test_evaluated"] is False,
    }
    if not all(checks.values()):
        raise ValueError(f"invalid K44 preflight: {checks}")
    return payload, lock_path


def train_main(
    config: dict[str, Any],
    config_path: Path,
    subset_seed: int,
    output_override: Path | None,
    smoke: bool,
) -> None:
    root = subset_root(config, subset_seed, output_override)
    preflight, preflight_lock_path = load_preflight(root, config_path, subset_seed)
    raw_registry, _, _, _, _, _ = load_upstream(config, subset_seed)
    derived = raw_subset.budget.build_run_config(raw_registry, "empirical", 80, 3407)
    derived["training"]["epochs"] = 2 if smoke else 100
    mode = "smoke" if smoke else "formal"
    output_dir = root / mode / "raw_propagation"
    result_path = output_dir / ("smoke_results.json" if smoke else "training_results.json")
    lock_path = output_dir / "training_lock.json"
    if result_path.exists() or lock_path.exists():
        raise FileExistsError(f"refusing to overwrite K44 training: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(int(config["runtime"]["num_threads"]))
    torch.set_float32_matmul_precision("high")
    device = torch.device(config["runtime"]["device"] if torch.cuda.is_available() else "cpu")
    images = np.load(Path(derived["dataset"]["cache_path"]), mmap_mode="r")
    raw_latents = base.load_latents(derived)
    subset, subset_audit = raw_subset.budget.build_subset(raw_registry, 80)
    latents, latent_mean_np, latent_std_np = raw_subset.budget.v17.normalize_with_k80_rgb_statistics(
        raw_latents, int(derived["split"]["train_end"]), subset
    )
    if not np.allclose(latent_mean_np, preflight["normalization"]["latent_mean"]):
        raise ValueError("K44 latent mean mismatch")
    if not np.allclose(latent_std_np, preflight["normalization"]["latent_std"]):
        raise ValueError("K44 latent std mismatch")

    anchor_maps, anchor_audit = raw_subset.budget.v2.build_anchor_maps(derived)
    initial_state, initial_hash = raw_subset.budget.v16.make_initial_state(derived, device)
    expected_initial = derived["initialization"]["expected_state_dict_sha256"]
    if initial_hash != expected_initial:
        raise ValueError("K44 initial state mismatch")
    initial_path = output_dir / "initial_state_seed3407.pt"
    torch.save(
        {"seed": 3407, "state_dict_sha256": initial_hash, "state_dict": copy.deepcopy(initial_state)},
        initial_path,
    )
    teacher_bundle = {
        "empirical_coefficients": preflight["teacher"]["model"],
        "empirical_fit": preflight["teacher"]["audit"],
    }
    started = time.time()
    training = raw_condition.train_one(
        "empirical",
        initial_state,
        initial_hash,
        images,
        latents,
        raw_latents,
        anchor_maps,
        subset,
        derived,
        device,
        output_dir,
        teacher_bundle,
        torch.from_numpy(latent_mean_np).to(device),
        torch.from_numpy(latent_std_np).to(device),
    )
    checks = {
        "subset_hash_exact": subset_audit["subset_sha256"] == preflight["subset"]["subset_sha256"],
        "initial_state_exact": initial_hash == expected_initial,
        "parameter_count_exact": training["parameter_count"]
        == int(config["implementation"]["expected_parameter_count"]),
        "epoch_exact": training["final_epoch"] == (2 if smoke else 100),
        "history_finite": dynamic.numeric_history_is_finite(training["history"]),
        "semantic_unread": not training["semantic_truth_read_during_training"],
        "test_unread": True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"K44 training validity failed: {checks}")

    lock = {
        "status": "locked_k44_before_joint_validation_readout",
        "mode": mode,
        "protocol_version": PROTOCOL,
        "subset_seed": subset_seed,
        "config_sha256": base.sha256_file(config_path),
        "source_files_sha256": source_hashes(),
        "preflight_lock_sha256": base.sha256_file(preflight_lock_path),
        "initial_state": {
            "path": str(initial_path),
            "file_sha256": base.sha256_file(initial_path),
            "state_dict_sha256": initial_hash,
        },
        "checkpoint": {
            "path": training["checkpoint"],
            "sha256": training["checkpoint_sha256"],
            "epoch": training["final_epoch"],
        },
        "validity": checks,
        "semantic_validation_evaluated": False,
        "test_evaluated": False,
    }
    v11.atomic_json(lock_path, lock)
    v11.atomic_json(
        result_path,
        {
            "protocol_version": PROTOCOL,
            "mode": f"{mode}_train",
            "subset_seed": subset_seed,
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "numpy": np.__version__,
                "device": str(device),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            },
            "anchor_map_audit": anchor_audit,
            "training": training,
            "training_lock": str(lock_path),
            "training_lock_sha256": base.sha256_file(lock_path),
            "duration_seconds": time.time() - started,
            "semantic_validation_evaluated": False,
            "test_evaluated": False,
        },
    )


def load_raw_model(
    root: Path,
    config_path: Path,
    subset_seed: int,
    derived: dict[str, Any],
    device: torch.device,
) -> tuple[torch.nn.Module, str]:
    lock_path = root / "formal" / "raw_propagation" / "training_lock.json"
    lock = load_json(lock_path)
    checks = {
        "status": lock["status"] == "locked_k44_before_joint_validation_readout",
        "formal": lock["mode"] == "formal",
        "protocol": lock["protocol_version"] == PROTOCOL,
        "subset": int(lock["subset_seed"]) == subset_seed,
        "config": lock["config_sha256"] == base.sha256_file(config_path),
        "source": lock["source_files_sha256"] == source_hashes(),
        "valid": all(lock["validity"].values()),
        "test_unread": lock["test_evaluated"] is False,
    }
    if not all(checks.values()):
        raise ValueError(f"invalid K44 raw model lock: {checks}")
    checkpoint_path = Path(lock["checkpoint"]["path"])
    if base.sha256_file(checkpoint_path) != lock["checkpoint"]["sha256"]:
        raise ValueError("K44 raw checkpoint hash mismatch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    identity = {
        "condition": checkpoint["condition"] == raw_condition.raw.CONDITION,
        "seed": int(checkpoint["seed"]) == 3407,
        "epoch": int(checkpoint["epoch"]) == 100,
        "initial": checkpoint["initial_state_sha256"]
        == derived["initialization"]["expected_state_dict_sha256"],
    }
    if not all(identity.values()):
        raise ValueError(f"K44 raw checkpoint identity mismatch: {identity}")
    return v13.load_model(checkpoint_path, derived, device), base.sha256_file(lock_path)


def primary_delta(tube: dict[str, Any], raw: dict[str, Any]) -> dict[str, float]:
    return {
        "mean_direct_abs_correlation": float(
            tube["semantic"]["raw"]["mean_direct_abs_correlation"]
            - raw["semantic"]["raw"]["mean_direct_abs_correlation"]
        ),
        "mean_r2": float(
            tube["semantic"]["raw"]["mean_r2"] - raw["semantic"]["raw"]["mean_r2"]
        ),
    }


def paired_bootstrap(
    tube_prediction: np.ndarray,
    raw_prediction: np.ndarray,
    truth: np.ndarray,
    repetitions: int,
    seed: int,
    confidence: float,
) -> dict[str, Any]:
    tube_prediction = np.asarray(tube_prediction, dtype=np.float64)
    raw_prediction = np.asarray(raw_prediction, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if tube_prediction.shape != raw_prediction.shape or truth.shape != tube_prediction.shape:
        raise ValueError("paired bootstrap arrays must have the same shape")
    if truth.ndim != 2 or truth.shape[1] != 3:
        raise ValueError("paired bootstrap expects N-by-3 RGB coordinates")
    rng = np.random.default_rng(seed)
    correlation = np.empty(repetitions, dtype=np.float64)
    r2 = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        rows = rng.integers(0, truth.shape[0], size=truth.shape[0])
        tube_metrics = v6.regression_metrics(tube_prediction[rows], truth[rows])
        raw_metrics = v6.regression_metrics(raw_prediction[rows], truth[rows])
        correlation[index] = (
            tube_metrics["mean_direct_abs_correlation"]
            - raw_metrics["mean_direct_abs_correlation"]
        )
        r2[index] = tube_metrics["mean_r2"] - raw_metrics["mean_r2"]
    if not np.all(np.isfinite(correlation)) or not np.all(np.isfinite(r2)):
        raise ValueError("non-finite K44 bootstrap replicate")
    tail = (1.0 - confidence) / 2.0

    def summary(values: np.ndarray) -> dict[str, float]:
        return {
            "mean": float(values.mean()),
            "lower": float(np.quantile(values, tail)),
            "upper": float(np.quantile(values, 1.0 - tail)),
        }

    return {
        "repetitions": repetitions,
        "seed": seed,
        "confidence": confidence,
        "mean_direct_abs_correlation": summary(correlation),
        "mean_r2": summary(r2),
    }


def readout_main(
    config: dict[str, Any], config_path: Path, subset_seed: int, output_override: Path | None
) -> None:
    root = subset_root(config, subset_seed, output_override)
    preflight, preflight_lock_path = load_preflight(root, config_path, subset_seed)
    raw_registry, _, tube_config, tube_path, tube_payload, tube_assets = load_upstream(
        config, subset_seed
    )
    derived = raw_subset.budget.build_run_config(raw_registry, "empirical", 80, 3407)
    output_path = root / "formal" / "paired_validation_readout.json"
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite K44 readout: {output_path}")

    torch.set_num_threads(int(config["runtime"]["num_threads"]))
    torch.set_float32_matmul_precision("high")
    device = torch.device(config["runtime"]["device"] if torch.cuda.is_available() else "cpu")
    images = np.load(Path(derived["dataset"]["cache_path"]), mmap_mode="r")
    raw_latents = base.load_latents(derived)
    subset, subset_audit = raw_subset.budget.build_subset(raw_registry, 80)
    latents, _, _ = raw_subset.budget.v17.normalize_with_k80_rgb_statistics(
        raw_latents, int(derived["split"]["train_end"]), subset
    )
    raw_model, raw_lock_hash = load_raw_model(root, config_path, subset_seed, derived, device)
    tube_model, tube_lock_hash = k8.load_model(
        "isotropic",
        tube_assets["root"],
        tube_path,
        tube_config,
        tube_payload,
        tube_assets["preflight_lock"],
        device,
    )
    start = int(derived["split"]["train_end"])
    end = int(derived["split"]["validation_end"])
    batch_size = int(derived["training"]["batch_size"])
    truth = latents[0, start:end, :3]
    calibration_truth = latents[0, subset, :3]
    validation_full_truth = latents[0, start:end]

    runs: dict[str, Any] = {}
    predictions: dict[str, np.ndarray] = {}
    for name, model in (("raw_propagation", raw_model), ("isotropic_tube", tube_model)):
        calibration_prediction = k6.gauge.predict_selected(
            model, images, latents, subset, batch_size, device
        )
        validation_prediction = v6.predict_rgb(model, images, latents, start, end, batch_size, device)
        predictions[name] = validation_prediction
        runs[name] = {
            "semantic": k6.semantic_bundle(
                calibration_prediction,
                validation_prediction,
                calibration_truth,
                truth,
                validation_full_truth,
                list(derived["model"]["learned_indices"]),
                list(derived["model"]["anchor_indices"]),
                float(tube_config["evaluation"]["full_affine_ridge"]),
            )
        }

    locked_tube = load_json(project_path(subset_entry(config, subset_seed)["locked_tube_readout"]))
    observed = runs["isotropic_tube"]["semantic"]["raw"]
    expected = locked_tube["runs"]["isotropic"]["semantic"]["raw"]
    reproduction = {
        metric: bool(np.isclose(float(observed[metric]), float(expected[metric]), atol=1e-6, rtol=0.0))
        for metric in ("mean_direct_abs_correlation", "mean_r2", "mse")
    }
    if not all(reproduction.values()):
        raise RuntimeError(f"K44 failed to reproduce locked Tube readout: {reproduction}")

    delta = primary_delta(runs["isotropic_tube"], runs["raw_propagation"])
    bootstrap_cfg = config["bootstrap"]
    bootstrap = paired_bootstrap(
        predictions["isotropic_tube"],
        predictions["raw_propagation"],
        truth,
        int(bootstrap_cfg["repetitions"]),
        int(bootstrap_cfg["seed"]) + SUBSET_SEEDS.index(subset_seed),
        float(bootstrap_cfg["confidence"]),
    )
    checks = {
        "preflight_valid": all(preflight["checks"].values()),
        "subset_hash_exact": subset_audit["subset_sha256"] == preflight["subset"]["subset_sha256"],
        "tube_readout_reproduced": all(reproduction.values()),
        "same_initial_state": preflight["checks"]["initial_state_exact"],
        "same_center": preflight["checks"]["teacher_coefficients_exact"],
        "metrics_finite": all(np.isfinite(value) for value in delta.values()),
        "test_unread": True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"K44 readout validity failed: {checks}")
    output = {
        "protocol_version": PROTOCOL,
        "mode": "paired_validation_only_readout",
        "subset_seed": subset_seed,
        "model_seed": 3407,
        "config_sha256": base.sha256_file(config_path),
        "source_files_sha256": source_hashes(),
        "preflight_lock_sha256": base.sha256_file(preflight_lock_path),
        "training_locks": {
            "raw_propagation": raw_lock_hash,
            "isotropic_tube": tube_lock_hash,
        },
        "runs": runs,
        "isotropic_tube_minus_raw_propagation": delta,
        "paired_bootstrap": bootstrap,
        "locked_tube_reproduction": reproduction,
        "checks": checks,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": str(device),
        },
        "semantic_validation_evaluated": True,
        "test_evaluated": False,
    }
    v11.atomic_json(output_path, output)
    del raw_model, tube_model
    if device.type == "cuda":
        torch.cuda.empty_cache()


def aggregate_decision(deltas: list[dict[str, float]]) -> dict[str, Any]:
    correlation = np.asarray(
        [item["mean_direct_abs_correlation"] for item in deltas], dtype=np.float64
    )
    r2 = np.asarray([item["mean_r2"] for item in deltas], dtype=np.float64)
    all_positive = bool(np.all(correlation > 0.0) and np.all(r2 > 0.0))
    means_positive = bool(correlation.mean() > 0.0 and r2.mean() > 0.0)
    if all_positive:
        verdict = "subset_stability_supported"
    elif means_positive:
        verdict = "mixed_subset_signal_crossed_initializations_unlocked"
    else:
        verdict = "subset_stability_contradicted_stop"
    return {
        "verdict": verdict,
        "all_three_subsets_both_primary_metrics_positive": all_positive,
        "both_primary_metric_means_positive": means_positive,
        "positive_counts": {
            "mean_direct_abs_correlation": int(np.sum(correlation > 0.0)),
            "mean_r2": int(np.sum(r2 > 0.0)),
        },
        "mean_deltas": {
            "mean_direct_abs_correlation": float(correlation.mean()),
            "mean_r2": float(r2.mean()),
        },
    }


def aggregate_main(config: dict[str, Any], config_path: Path, output_override: Path | None) -> None:
    root = (output_override or Path(config["runtime"]["output_root"])).resolve()
    output_dir = root / "aggregate"
    output_path = output_dir / "subset_aggregate.json"
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite K44 aggregate: {output_path}")
    cells: list[dict[str, Any]] = []
    deltas: list[dict[str, float]] = []
    for subset_seed in SUBSET_SEEDS:
        path = root / f"subset{subset_seed}" / "formal" / "paired_validation_readout.json"
        result = load_json(path)
        checks = {
            "protocol": result["protocol_version"] == PROTOCOL,
            "mode": result["mode"] == "paired_validation_only_readout",
            "subset": int(result["subset_seed"]) == subset_seed,
            "seed": int(result["model_seed"]) == 3407,
            "config": result["config_sha256"] == base.sha256_file(config_path),
            "source": result["source_files_sha256"] == source_hashes(),
            "valid": all(result["checks"].values()),
            "validation": result["semantic_validation_evaluated"] is True,
            "test_unread": result["test_evaluated"] is False,
        }
        if not all(checks.values()):
            raise ValueError(f"invalid K44 cell {subset_seed}: {checks}")
        delta = result["isotropic_tube_minus_raw_propagation"]
        deltas.append(delta)
        cells.append(
            {
                "subset_seed": subset_seed,
                "readout_path": str(path),
                "readout_sha256": base.sha256_file(path),
                "delta": delta,
                "paired_bootstrap": result["paired_bootstrap"],
            }
        )
    decision = aggregate_decision(deltas)
    decision["bootstrap_lower_positive_counts"] = {
        metric: sum(
            cell["paired_bootstrap"][metric]["lower"] > 0.0 for cell in cells
        )
        for metric in ("mean_direct_abs_correlation", "mean_r2")
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    v11.atomic_json(
        output_path,
        {
            "protocol_version": PROTOCOL,
            "mode": "three_subset_validation_only_aggregate",
            "config_sha256": base.sha256_file(config_path),
            "source_files_sha256": source_hashes(),
            "cells": cells,
            "decision": decision,
            "semantic_validation_evaluated": True,
            "test_evaluated": False,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("preflight", "smoke", "train", "readout", "aggregate"), required=True
    )
    parser.add_argument("--subset-seed", type=int, choices=SUBSET_SEEDS)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_yaml(config_path)
    validate_config(config)
    if args.mode == "aggregate":
        if args.subset_seed is not None:
            raise ValueError("K44 aggregate does not accept a subset seed")
        aggregate_main(config, config_path, args.output_root)
        return
    if args.subset_seed is None:
        raise ValueError("K44 cell mode requires a subset seed")
    if args.mode == "preflight":
        preflight_main(config, config_path, args.subset_seed, args.output_root)
    elif args.mode in ("smoke", "train"):
        train_main(config, config_path, args.subset_seed, args.output_root, args.mode == "smoke")
    else:
        readout_main(config, config_path, args.subset_seed, args.output_root)


if __name__ == "__main__":
    main()
