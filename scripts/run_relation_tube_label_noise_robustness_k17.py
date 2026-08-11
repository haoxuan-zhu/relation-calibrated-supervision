"""Validation-only paired label-noise robustness audit for the relation tube."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

import audit_relation_tube_geometry_downstream_k6 as k6
import raw_ridge_propagation as raw
import run_calibrated_relation_tube_k0 as tube
import run_conflict_projected_physics_training as v13
import run_conflict_projected_physics_k80_closed as v17
import run_diagnostic as base
import run_dynamic_residual_propagation_k0 as dynamic
import run_isotropic_relation_tube_budget_k11 as k11
import run_physics_functional_anchor_training as v11
import run_supervised_continuation_diagnostic as v6


PROTOCOL = "relation_tube_label_noise_robustness_k17_v1"
SIGMAS = (0.01, 0.025, 0.05, 0.10)
SEEDS = (0, 42, 3407)
CONDITIONS = ("isotropic_tube", "raw_propagation")
NOISE_SEEDS = {0: 2026080300, 42: 2026080342, 3407: 2026083407}
SOURCE_FILES = tuple(
    dict.fromkeys(
        (
            "run_relation_tube_label_noise_robustness_k17.py",
            "run_isotropic_relation_tube_budget_k11.py",
            "run_calibrated_relation_tube_k0.py",
            "raw_ridge_propagation.py",
            "audit_relation_tube_geometry_downstream_k6.py",
            "run_conflict_projected_physics_k80_closed.py",
            "run_dynamic_residual_propagation_k0.py",
            "run_physics_functional_anchor_training.py",
            "run_supervised_continuation_diagnostic.py",
            "run_conflict_projected_physics_from_scratch.py",
            "run_conflict_projected_physics_training.py",
            "run_conditioned_film_diagnostic.py",
            "run_clamped_anchor_diagnostic.py",
            "run_diagnostic.py",
        )
    )
)


def keyed(mapping: dict[Any, Any], key: int) -> Any:
    if key in mapping:
        return mapping[key]
    if str(key) in mapping:
        return mapping[str(key)]
    raise KeyError(key)


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.tobytes()).hexdigest()


def source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    return {name: base.sha256_file(root / name) for name in SOURCE_FILES}


def sigma_key(sigma: float) -> str:
    registered = {
        0.01: "sigma_0p01",
        0.025: "sigma_0p025",
        0.05: "sigma_0p05",
        0.10: "sigma_0p10",
    }
    for value, name in registered.items():
        if np.isclose(sigma, value, atol=1e-12, rtol=0.0):
            return name
    raise ValueError(f"unregistered K17 sigma: {sigma}")


def validate_master(config: dict[str, Any]) -> None:
    if config["protocol_version"] != PROTOCOL:
        raise ValueError("unexpected K17 protocol")
    if tuple(float(value) for value in config["noise"]["sigma_full_scale"]) != SIGMAS:
        raise ValueError("K17 noise levels changed")
    if tuple(int(value) for value in config["noise"]["model_seeds"]) != SEEDS:
        raise ValueError("K17 model seeds changed")
    if tuple(config["conditions"]) != CONDITIONS:
        raise ValueError("K17 conditions changed")
    observed_noise_seeds = {
        seed: int(keyed(config["noise"]["noise_seed_by_model_seed"], seed))
        for seed in SEEDS
    }
    if observed_noise_seeds != NOISE_SEEDS:
        raise ValueError("K17 noise seeds changed")
    if config["noise"]["family"] != "clipped_additive_gaussian_on_unit_rgb":
        raise ValueError("K17 noise family changed")
    if config["noise"]["nested_across_sigma"] is not True:
        raise ValueError("K17 noise must be nested across sigma")
    if config["noise"]["shared_between_conditions"] is not True:
        raise ValueError("K17 conditions must share noisy labels")
    if [float(value) for value in config["noise"]["clip_interval"]] != [0.0, 1.0]:
        raise ValueError("K17 clipping interval changed")
    calibration = config["calibration"]
    if int(calibration["budget"]) != 80:
        raise ValueError("K17 is frozen to K80")
    if int(calibration["subset_seed"]) != 20260731:
        raise ValueError("K17 subset seed changed")
    if calibration["subset_sha256"] != "4bf1156bbf3f8f8ed0b5e63c530efd171e36395cd9d7d59003a8465ba43ddc84":
        raise ValueError("K17 subset hash changed")
    if not np.isclose(float(calibration["raw_ridge_alpha"]), 0.1):
        raise ValueError("K17 center ridge changed")
    if not np.isclose(float(calibration["coverage"]), 0.95):
        raise ValueError("K17 coverage changed")
    if int(calibration["expected_score_order_index_one_based"]) != 77:
        raise ValueError("K17 radius order statistic changed")
    if not np.isclose(float(calibration["expected_empirical_coverage"]), 0.9625):
        raise ValueError("K17 empirical coverage changed")
    if int(config["training"]["epochs"]) != 100:
        raise ValueError("K17 epoch contract changed")
    if int(config["training"]["shuffle_seed"]) != 20260730:
        raise ValueError("K17 training order changed")
    if int(config["evaluation"]["bootstrap_replicates"]) != 2000:
        raise ValueError("K17 bootstrap count changed")
    if int(config["implementation"]["expected_parameter_count"]) != 16902384:
        raise ValueError("K17 parameter-count contract changed")
    if config["audit"]["test_evaluated"] is not False:
        raise ValueError("K17 test must remain closed")


def build_run_config(master: dict[str, Any], sigma: float, seed: int) -> dict[str, Any]:
    validate_master(master)
    sigma_key(sigma)
    if seed not in SEEDS:
        raise ValueError("unregistered K17 seed")
    config = copy.deepcopy(master)
    config["selected_run"] = {
        "sigma_full_scale": float(sigma),
        "sigma_key": sigma_key(sigma),
        "model_seed": int(seed),
        "noise_seed": NOISE_SEEDS[seed],
    }
    config["training"]["seed"] = int(seed)
    config["initialization"] = {
        "mode": "same_seed_random_no_warm",
        "seed": int(seed),
        "expected_state_dict_sha256": keyed(
            master["initialization"]["expected_state_dict_sha256"], seed
        ),
    }
    return config


def noisy_observations(
    raw_latents: np.ndarray,
    subset: np.ndarray,
    sigma: float,
    noise_seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    values = np.asarray(raw_latents, dtype=np.float64)
    subset = np.asarray(subset, dtype=np.int64)
    clean = values[0, subset, :3] / 255.0
    if clean.shape != (80, 3) or np.any(clean < 0.0) or np.any(clean > 1.0):
        raise ValueError("K17 expected 80 physical unit-RGB labels")
    standard_noise = np.random.default_rng(noise_seed).standard_normal(clean.shape)
    unbounded = clean + float(sigma) * standard_noise
    noisy = np.clip(unbounded, 0.0, 1.0)
    observed = values.copy()
    observed[0, subset, :3] = noisy * 255.0
    # Hash the effective stored labels, not the pre-scaling temporary.  The
    # multiply/divide round trip can differ by one float64 ULP even though it
    # represents the same physical label.
    effective_noisy = observed[0, subset, :3] / 255.0
    error = effective_noisy - clean
    audit = {
        "family": "clipped_additive_gaussian_on_unit_rgb",
        "sigma_full_scale": float(sigma),
        "noise_seed": int(noise_seed),
        "standard_noise_shape": list(standard_noise.shape),
        "standard_noise_float64_sha256": array_sha256(standard_noise.astype(np.float64)),
        "clean_unit_rgb_float64_sha256": array_sha256(clean.astype(np.float64)),
        "noisy_unit_rgb_float64_sha256": array_sha256(
            effective_noisy.astype(np.float64)
        ),
        "noisy_unit_rgb": effective_noisy.tolist(),
        "realized_rmse_unit_rgb": float(np.sqrt(np.mean(error**2))),
        "realized_mae_unit_rgb": float(np.mean(np.abs(error))),
        "clipped_low_count": int(np.sum(unbounded < 0.0)),
        "clipped_high_count": int(np.sum(unbounded > 1.0)),
        "clipped_total_count": int(np.sum((unbounded < 0.0) | (unbounded > 1.0))),
    }
    return observed, audit


def build_preflight(
    config: dict[str, Any], images: np.ndarray, raw_latents: np.ndarray
) -> dict[str, Any]:
    subset, subset_audit = dynamic.build_subset(config)
    selected = config["selected_run"]
    observed_raw, noise_audit = noisy_observations(
        raw_latents,
        subset,
        float(selected["sigma_full_scale"]),
        int(selected["noise_seed"]),
    )
    train_end = int(config["split"]["train_end"])
    features = tube.calibration_features(images, observed_raw, subset, train_end)
    targets = observed_raw[0, subset, :3] / 255.0
    teacher, loo_predictions, teacher_audit = tube.fit_teacher(
        features,
        targets,
        float(config["calibration"]["raw_ridge_alpha"]),
        bool(config["calibration"]["clip_predictions_to_unit_interval"]),
    )
    _, latent_mean, latent_std = v17.normalize_with_k80_rgb_statistics(
        observed_raw, train_end, subset
    )
    residuals = (targets - loo_predictions) * 255.0 / latent_std[:3]
    geometry, geometry_audit = k11.build_isotropic_geometry(
        residuals,
        float(config["calibration"]["coverage"]),
        float(config["calibration"]["covariance_diagonal_ridge"]),
    )
    checks = {
        "subset_hash_exact": subset_audit["subset_sha256"]
        == config["calibration"]["subset_sha256"],
        "subset_budget_exact": int(subset_audit["unique_rows"]) == 80,
        "noise_shape_exact": noise_audit["standard_noise_shape"] == [80, 3],
        "noise_nonzero": float(noise_audit["realized_rmse_unit_rgb"]) > 0.0,
        "teacher_alpha_fixed": bool(
            np.isclose(
                float(teacher_audit["alpha"]),
                float(config["calibration"]["raw_ridge_alpha"]),
            )
        ),
        "teacher_finite": bool(teacher_audit["finite"]),
        "loo_complete": int(teacher_audit["loo_count"]) == 80,
        "normalization_finite": bool(
            np.all(np.isfinite(latent_mean))
            and np.all(np.isfinite(latent_std))
            and np.all(latent_std > 0.0)
        ),
        "geometry_finite": bool(geometry_audit["finite"]),
        "isotropic_eigenvalues_exact": bool(
            np.allclose(
                geometry_audit["second_moment_eigenvalues"],
                geometry_audit["second_moment_eigenvalues"][0],
                atol=1e-12,
                rtol=0.0,
            )
        ),
        "score_index_exact": int(geometry_audit["score_order_index_one_based"])
        == int(config["calibration"]["expected_score_order_index_one_based"]),
        "finite_sample_coverage_exact": bool(
            np.isclose(
                float(geometry_audit["empirical_coverage"]),
                float(config["calibration"]["expected_empirical_coverage"]),
                atol=1e-12,
                rtol=0.0,
            )
        ),
        "semantic_validation_unread": True,
        "test_unread": True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"K17 preflight failed: {checks}")
    teacher_json = tube.teacher_json(teacher)
    return {
        "selected_run": selected,
        "subset": subset_audit,
        "noise": noise_audit,
        "normalization": {
            "latent_mean": latent_mean,
            "latent_std": latent_std,
            "source": "noisy_k80_rgb_all_known_angle_anchors",
        },
        "shared_center": {"model": teacher_json, "audit": teacher_audit},
        "geometry": geometry,
        "geometry_audit": geometry_audit,
        "geometry_sha256": dynamic.canonical_sha256(geometry),
        "checks": checks,
    }


def run_root(master: dict[str, Any], output_root: Path | None, sigma: float, seed: int) -> Path:
    root = Path(output_root or master["runtime"]["output_root"]).resolve()
    return root / sigma_key(sigma) / f"seed{seed}"


def preflight_main(args: argparse.Namespace, master: dict[str, Any]) -> None:
    if args.seed not in SEEDS or args.condition is not None:
        raise ValueError("K17 preflight requires a registered seed and no condition")
    config = build_run_config(master, args.sigma, args.seed)
    output_dir = run_root(master, args.output_root, args.sigma, args.seed) / "preflight"
    output_dir.mkdir(parents=True, exist_ok=True)
    payload_path = output_dir / "noise_preflight.json"
    lock_path = output_dir / "preflight_lock.json"
    if payload_path.exists() or lock_path.exists():
        raise FileExistsError(f"refusing to overwrite K17 preflight: {output_dir}")
    images = np.load(Path(config["dataset"]["cache_path"]), mmap_mode="r")
    raw_latents = base.load_latents(config).astype(np.float64)
    payload = build_preflight(config, images, raw_latents)
    payload.update(
        protocol_version=PROTOCOL,
        master_config_path=str(args.config.resolve()),
        master_config_sha256=base.sha256_file(args.config.resolve()),
        source_files_sha256=source_hashes(),
        semantic_validation_evaluated=False,
        test_evaluated=False,
    )
    v11.atomic_json(payload_path, payload)
    v11.atomic_json(
        lock_path,
        {
            "status": "locked_k17_noisy_relation_preflight",
            "protocol_version": PROTOCOL,
            "selected_run": config["selected_run"],
            "master_config_sha256": base.sha256_file(args.config.resolve()),
            "source_files_sha256": source_hashes(),
            "payload_sha256": base.sha256_file(payload_path),
            "noisy_unit_rgb_float64_sha256": payload["noise"][
                "noisy_unit_rgb_float64_sha256"
            ],
            "semantic_validation_evaluated": False,
            "test_evaluated": False,
        },
    )


def load_preflight(
    master: dict[str, Any], config_path: Path, output_root: Path | None, sigma: float, seed: int
) -> tuple[dict[str, Any], Path]:
    root = run_root(master, output_root, sigma, seed) / "preflight"
    payload_path = root / "noise_preflight.json"
    lock_path = root / "preflight_lock.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    expected = build_run_config(master, sigma, seed)["selected_run"]
    checks = {
        "status": lock["status"] == "locked_k17_noisy_relation_preflight",
        "protocol": lock["protocol_version"] == PROTOCOL == payload["protocol_version"],
        "selected_run": lock["selected_run"] == expected == payload["selected_run"],
        "config": lock["master_config_sha256"] == base.sha256_file(config_path),
        "source": lock["source_files_sha256"] == source_hashes(),
        "payload": lock["payload_sha256"] == base.sha256_file(payload_path),
        "noise": lock["noisy_unit_rgb_float64_sha256"]
        == payload["noise"]["noisy_unit_rgb_float64_sha256"],
        "payload_valid": all(payload["checks"].values()),
        "semantic_unread": lock["semantic_validation_evaluated"] is False,
        "test_unread": lock["test_evaluated"] is False,
    }
    if not all(checks.values()):
        raise ValueError(f"invalid K17 preflight: {checks}")
    return payload, lock_path


def reconstruct_observed_latents(
    config: dict[str, Any], payload: dict[str, Any], raw_latents: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    subset, _ = dynamic.build_subset(config)
    observed_raw, audit = noisy_observations(
        raw_latents,
        subset,
        float(config["selected_run"]["sigma_full_scale"]),
        int(config["selected_run"]["noise_seed"]),
    )
    if audit["standard_noise_float64_sha256"] != payload["noise"]["standard_noise_float64_sha256"]:
        raise ValueError("K17 standard-noise reconstruction mismatch")
    if audit["noisy_unit_rgb_float64_sha256"] != payload["noise"]["noisy_unit_rgb_float64_sha256"]:
        raise ValueError("K17 noisy-label reconstruction mismatch")
    latents, latent_mean, latent_std = v17.normalize_with_k80_rgb_statistics(
        observed_raw, int(config["split"]["train_end"]), subset
    )
    if not np.allclose(latent_mean, payload["normalization"]["latent_mean"]):
        raise ValueError("K17 latent mean mismatch")
    if not np.allclose(latent_std, payload["normalization"]["latent_std"]):
        raise ValueError("K17 latent std mismatch")
    return observed_raw, latents, latent_mean, latent_std


def tube_preflight(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "teachers": {"correct": payload["shared_center"]},
        "normalization": payload["normalization"],
        "geometry": payload["geometry"],
    }


def train_raw_propagation(
    config: dict[str, Any],
    payload: dict[str, Any],
    images: np.ndarray,
    latents: np.ndarray,
    raw_latents: np.ndarray,
    anchor_maps: dict[str, np.ndarray],
    initial_state: dict[str, torch.Tensor],
    initial_hash: str,
    latent_mean: np.ndarray,
    latent_std: np.ndarray,
    device: torch.device,
    output_dir: Path,
) -> dict[str, Any]:
    condition = tube.v2.Condition(raw.CONDITION, "oracle")
    teacher = {
        key: torch.from_numpy(np.asarray(value, dtype=np.float32)).to(device)
        for key, value in payload["shared_center"]["model"].items()
        if key in ("feature_mean", "feature_scale", "coefficients")
    }
    original = v13.loss_components
    v13.loss_components = raw.loss_components
    try:
        result = tube.v16.train_condition(
            condition,
            initial_state,
            initial_hash,
            images,
            latents,
            raw_latents.astype(np.float32, copy=False),
            anchor_maps,
            config,
            device,
            output_dir,
            teacher,
            torch.as_tensor(latent_mean, dtype=torch.float32, device=device),
            torch.as_tensor(latent_std, dtype=torch.float32, device=device),
        )
    finally:
        v13.loss_components = original
    result["test_evaluated"] = False
    return result


def train_main(args: argparse.Namespace, master: dict[str, Any]) -> None:
    if args.seed not in SEEDS or args.condition not in CONDITIONS:
        raise ValueError("K17 train requires a registered seed and condition")
    smoke = args.mode == "smoke"
    epochs = int(args.epochs or (2 if smoke else 100))
    if (smoke and epochs != 2) or (not smoke and epochs != 100):
        raise ValueError("K17 smoke/formal epoch contract changed")
    config = build_run_config(master, args.sigma, args.seed)
    config["training"]["epochs"] = epochs
    root = run_root(master, args.output_root, args.sigma, args.seed)
    payload, preflight_lock_path = load_preflight(
        master, args.config.resolve(), args.output_root, args.sigma, args.seed
    )
    output_dir = root / args.condition / ("smoke" if smoke else "formal")
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / ("smoke_results.json" if smoke else "training_results.json")
    lock_path = output_dir / "training_lock.json"
    if result_path.exists() or lock_path.exists():
        raise FileExistsError(f"refusing to overwrite K17 run: {output_dir}")
    torch.set_num_threads(int(config["runtime"]["num_threads"]))
    torch.set_float32_matmul_precision("high")
    device = torch.device(config["runtime"]["device"] if torch.cuda.is_available() else "cpu")
    images = np.load(Path(config["dataset"]["cache_path"]), mmap_mode="r")
    raw_latents = base.load_latents(config).astype(np.float64)
    observed_raw, latents, latent_mean, latent_std = reconstruct_observed_latents(
        config, payload, raw_latents
    )
    anchor_maps, anchor_audit = tube.v2.build_anchor_maps(config)
    initial_state, initial_hash = tube.v16.make_initial_state(config, device)
    if initial_hash != config["initialization"]["expected_state_dict_sha256"]:
        raise ValueError("K17 initial state mismatch")
    if args.condition == "isotropic_tube":
        training = tube.train_condition(
            "tube_correct",
            config,
            tube_preflight(payload),
            images,
            latents,
            anchor_maps,
            initial_state,
            initial_hash,
            device,
            output_dir,
            epochs,
        )
    else:
        training = train_raw_propagation(
            config,
            payload,
            images,
            latents,
            observed_raw,
            anchor_maps,
            initial_state,
            initial_hash,
            latent_mean,
            latent_std,
            device,
            output_dir,
        )
    checks = {
        "preflight_valid": all(payload["checks"].values()),
        "shared_noisy_labels_exact": payload["noise"]["noisy_unit_rgb_float64_sha256"]
        == array_sha256(np.asarray(payload["noise"]["noisy_unit_rgb"], dtype=np.float64)),
        "initial_state_exact": initial_hash
        == config["initialization"]["expected_state_dict_sha256"],
        "parameter_count_exact": int(training["parameter_count"])
        == int(config["implementation"]["expected_parameter_count"]),
        "epoch_exact": int(training["final_epoch"]) == epochs,
        "history_finite": dynamic.numeric_history_is_finite(training["history"]),
        "semantic_unread": not bool(training["semantic_truth_read_during_training"]),
        "test_unread": not bool(training.get("test_evaluated", False)),
    }
    if not all(checks.values()):
        raise RuntimeError(f"K17 training validity failed: {checks}")
    lock = {
        "status": "locked_before_k17_paired_validation_readout",
        "mode": "smoke" if smoke else "formal",
        "protocol_version": PROTOCOL,
        "selected_run": config["selected_run"],
        "condition": args.condition,
        "master_config_sha256": base.sha256_file(args.config.resolve()),
        "source_files_sha256": source_hashes(),
        "preflight_lock_sha256": base.sha256_file(preflight_lock_path),
        "noisy_unit_rgb_float64_sha256": payload["noise"]["noisy_unit_rgb_float64_sha256"],
        "geometry_sha256": payload["geometry_sha256"],
        "initial_state_sha256": initial_hash,
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
            "mode": "smoke_train" if smoke else "formal_train",
            "selected_run": config["selected_run"],
            "condition": args.condition,
            "master_config_sha256": base.sha256_file(args.config.resolve()),
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
            "semantic_validation_evaluated": False,
            "test_evaluated": False,
        },
    )


def load_model(
    master: dict[str, Any],
    config_path: Path,
    output_root: Path | None,
    config: dict[str, Any],
    payload: dict[str, Any],
    preflight_lock_path: Path,
    condition: str,
    device: torch.device,
) -> tuple[torch.nn.Module, str]:
    root = run_root(
        master,
        output_root,
        float(config["selected_run"]["sigma_full_scale"]),
        int(config["selected_run"]["model_seed"]),
    )
    lock_path = root / condition / "formal" / "training_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    checks = {
        "status": lock["status"] == "locked_before_k17_paired_validation_readout",
        "formal": lock["mode"] == "formal",
        "protocol": lock["protocol_version"] == PROTOCOL,
        "selected_run": lock["selected_run"] == config["selected_run"],
        "condition": lock["condition"] == condition,
        "config": lock["master_config_sha256"] == base.sha256_file(config_path),
        "source": lock["source_files_sha256"] == source_hashes(),
        "preflight": lock["preflight_lock_sha256"] == base.sha256_file(preflight_lock_path),
        "noise": lock["noisy_unit_rgb_float64_sha256"]
        == payload["noise"]["noisy_unit_rgb_float64_sha256"],
        "geometry": lock["geometry_sha256"] == payload["geometry_sha256"],
        "validity": all(lock["validity"].values()),
        "test_unread": lock["test_evaluated"] is False,
    }
    if not all(checks.values()):
        raise ValueError(f"invalid K17 model lock {condition}: {checks}")
    checkpoint_path = Path(lock["checkpoint"]["path"])
    if base.sha256_file(checkpoint_path) != lock["checkpoint"]["sha256"]:
        raise ValueError("K17 checkpoint hash mismatch")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if condition == "isotropic_tube":
        if checkpoint["condition"] != "tube_correct" or checkpoint["protocol_version"] != PROTOCOL:
            raise ValueError("K17 tube checkpoint identity mismatch")
        model = tube.model_for_condition("tube_correct", config, tube_preflight(payload), device)
    else:
        if checkpoint["condition"] != raw.CONDITION:
            raise ValueError("K17 raw checkpoint identity mismatch")
        model_cfg = config["model"]
        model = tube.v3.FilmConditionedModel(
            int(model_cfg["latent_dim"]),
            list(model_cfg["learned_indices"]),
            list(model_cfg["anchor_indices"]),
            int(model_cfg["hidden_channels"]),
            int(model_cfg["conv_layers"]),
        ).to(device)
    if int(checkpoint["epoch"]) != 100:
        raise ValueError("K17 checkpoint epoch mismatch")
    if checkpoint["initial_state_sha256"] != config["initialization"]["expected_state_dict_sha256"]:
        raise ValueError("K17 checkpoint initialization mismatch")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, base.sha256_file(lock_path)


def raw_metrics(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    prediction = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if prediction.shape != truth.shape or prediction.ndim != 2 or prediction.shape[1] != 3:
        raise ValueError("K17 raw metrics require matched N-by-3 arrays")
    correlations = []
    r2 = []
    for channel in range(3):
        correlations.append(
            abs(float(np.corrcoef(prediction[:, channel], truth[:, channel])[0, 1]))
        )
        residual = np.sum((truth[:, channel] - prediction[:, channel]) ** 2)
        total = np.sum((truth[:, channel] - truth[:, channel].mean()) ** 2)
        r2.append(float(1.0 - residual / max(total, 1e-12)))
    return {
        "mean_direct_abs_correlation": float(np.mean(correlations)),
        "mean_r2": float(np.mean(r2)),
    }


def paired_bootstrap(
    tube_prediction: np.ndarray,
    raw_prediction: np.ndarray,
    truth: np.ndarray,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    observed_tube = raw_metrics(tube_prediction, truth)
    observed_raw = raw_metrics(raw_prediction, truth)
    metric_names = tuple(observed_tube)
    values = {name: np.empty(replicates, dtype=np.float64) for name in metric_names}
    rng = np.random.default_rng(seed)
    count = len(truth)
    for index in range(replicates):
        rows = rng.integers(0, count, size=count)
        tube_value = raw_metrics(tube_prediction[rows], truth[rows])
        raw_value = raw_metrics(raw_prediction[rows], truth[rows])
        for name in metric_names:
            values[name][index] = tube_value[name] - raw_value[name]
    result: dict[str, Any] = {
        "replicates": int(replicates),
        "seed": int(seed),
        "sampling_unit": "observational_validation_row",
        "metrics": {},
    }
    for name in metric_names:
        delta = observed_tube[name] - observed_raw[name]
        result["metrics"][name] = {
            "observed_delta": float(delta),
            "ci95": [
                float(np.quantile(values[name], 0.025)),
                float(np.quantile(values[name], 0.975)),
            ],
            "positive_fraction": float(np.mean(values[name] > 0.0)),
        }
    return result


@torch.no_grad()
def boundary_audit(
    model: torch.nn.Module,
    images: np.ndarray,
    latents: np.ndarray,
    start: int,
    end: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    active = 0
    count = 0
    scales = []
    for begin in range(start, end, batch_size):
        rows = np.arange(begin, min(begin + batch_size, end), dtype=np.int64)
        envs = np.zeros(len(rows), dtype=np.int64)
        x = base.image_batch(images, envs, rows, device)
        anchors = torch.from_numpy(np.array(latents[0, rows, 3:5], copy=True)).to(device)
        _, components = model.embedding.components(x, anchors)
        active += int(components["boundary_active"].sum().cpu())
        count += len(rows)
        scales.append(components["projection_scale"].cpu().numpy())
    scale = np.concatenate(scales)
    return {
        "count": int(count),
        "boundary_active_fraction": float(active / count),
        "projection_scale_mean": float(np.mean(scale)),
        "projection_scale_minimum": float(np.min(scale)),
    }


def readout_main(args: argparse.Namespace, master: dict[str, Any]) -> None:
    if args.seed not in SEEDS or args.condition is not None:
        raise ValueError("K17 readout requires a registered seed and no condition")
    config = build_run_config(master, args.sigma, args.seed)
    root = run_root(master, args.output_root, args.sigma, args.seed)
    output_path = root / "paired_validation_readout.json"
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite K17 readout: {output_path}")
    payload, preflight_lock_path = load_preflight(
        master, args.config.resolve(), args.output_root, args.sigma, args.seed
    )
    torch.set_num_threads(int(config["runtime"]["num_threads"]))
    torch.set_float32_matmul_precision("high")
    device = torch.device(config["runtime"]["device"] if torch.cuda.is_available() else "cpu")
    images = np.load(Path(config["dataset"]["cache_path"]), mmap_mode="r")
    raw_latents = base.load_latents(config).astype(np.float64)
    observed_raw, latents, _, _ = reconstruct_observed_latents(config, payload, raw_latents)
    subset, subset_audit = dynamic.build_subset(config)
    anchor_maps, _ = tube.v2.build_anchor_maps(config)
    models: dict[str, torch.nn.Module] = {}
    training_locks: dict[str, str] = {}
    for condition in CONDITIONS:
        models[condition], training_locks[condition] = load_model(
            master,
            args.config.resolve(),
            args.output_root,
            config,
            payload,
            preflight_lock_path,
            condition,
            device,
        )
    start = int(config["split"]["train_end"])
    end = int(config["split"]["validation_end"])
    if (start, end, int(config["split"]["test_end"])) != (8000, 9000, 10000):
        raise ValueError("K17 dataset split changed")
    calibration_truth = latents[0, subset, :3]
    validation_truth = latents[0, start:end, :3]
    semantic: dict[str, Any] = {}
    validation_predictions: dict[str, np.ndarray] = {}
    for condition, model in models.items():
        calibration_prediction = k6.gauge.predict_selected(
            model,
            images,
            latents,
            subset,
            int(config["training"]["batch_size"]),
            device,
        )
        validation_prediction = v6.predict_rgb(
            model,
            images,
            latents,
            start,
            end,
            int(config["training"]["batch_size"]),
            device,
        )
        validation_predictions[condition] = validation_prediction
        semantic[condition] = k6.semantic_bundle(
            calibration_prediction,
            validation_prediction,
            calibration_truth,
            validation_truth,
            latents[0, start:end],
            list(config["model"]["learned_indices"]),
            list(config["model"]["anchor_indices"]),
            float(config["evaluation"]["full_affine_ridge"]),
        )
    bootstrap_seed = (
        int(config["evaluation"]["bootstrap_seed"])
        + int(args.seed)
        + int(round(float(args.sigma) * 100000))
    )
    bootstrap = paired_bootstrap(
        validation_predictions["isotropic_tube"],
        validation_predictions["raw_propagation"],
        validation_truth,
        int(config["evaluation"]["bootstrap_replicates"]),
        bootstrap_seed,
    )
    teacher_validation = raw.validation_metrics(
        payload["shared_center"]["model"],
        images,
        observed_raw,
        start,
        end,
        bool(config["calibration"]["clip_predictions_to_unit_interval"]),
    )
    ccrl_validation = {
        "isotropic_tube": tube.v2.validate(
            models["isotropic_tube"],
            images,
            latents,
            anchor_maps,
            tube.v2.Condition("tube_correct", "oracle"),
            config,
            device,
        ),
        "raw_propagation": tube.v2.validate(
            models["raw_propagation"],
            images,
            latents,
            anchor_maps,
            tube.v2.Condition(raw.CONDITION, "oracle"),
            config,
            device,
        ),
    }
    graph = {
        condition: base.graph_metrics(
            model.parametric_part.A.detach().cpu().numpy(),
            float(config["evaluation"]["graph_threshold"]),
        )
        for condition, model in models.items()
    }
    checks = {
        "both_formal_locks_loaded": set(training_locks) == set(CONDITIONS),
        "shared_noisy_calibration_truth": array_sha256(
            np.asarray(observed_raw[0, subset, :3] / 255.0, dtype=np.float64)
        )
        == payload["noise"]["noisy_unit_rgb_float64_sha256"],
        "validation_truth_is_clean": bool(
            np.array_equal(observed_raw[0, start:end, :3], raw_latents[0, start:end, :3])
        ),
        "bootstrap_complete": int(bootstrap["replicates"])
        == int(config["evaluation"]["bootstrap_replicates"]),
        "semantic_finite": bool(
            all(
                np.isfinite(semantic[name]["raw"]["mean_r2"])
                and np.isfinite(semantic[name]["raw"]["mean_direct_abs_correlation"])
                for name in CONDITIONS
            )
        ),
        "test_unread": True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"K17 readout validity failed: {checks}")
    result = {
        "protocol_version": PROTOCOL,
        "mode": "paired_noisy_relation_validation_only_readout",
        "selected_run": config["selected_run"],
        "master_config_sha256": base.sha256_file(args.config.resolve()),
        "source_files_sha256": source_hashes(),
        "preflight_lock_sha256": base.sha256_file(preflight_lock_path),
        "training_lock_sha256": training_locks,
        "subset": subset_audit,
        "noise": payload["noise"],
        "normalization": payload["normalization"],
        "shared_center_audit": payload["shared_center"]["audit"],
        "geometry_audit": payload["geometry_audit"],
        "semantic": semantic,
        "paired_bootstrap": bootstrap,
        "teacher_validation": teacher_validation,
        "tube_boundary": boundary_audit(
            models["isotropic_tube"],
            images,
            latents,
            start,
            end,
            int(config["training"]["batch_size"]),
            device,
        ),
        "graph": graph,
        "ccrl_validation": ccrl_validation,
        "validity": checks,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "semantic_validation_evaluated": True,
        "test_evaluated": False,
    }
    v11.atomic_json(output_path, result)


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "population_std": float(np.std(array)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def aggregate_verdict(entries: dict[str, Any]) -> str:
    def all_positive(max_sigma: float) -> bool:
        selected = [sigma for sigma in SIGMAS if sigma <= max_sigma + 1e-12]
        return all(
            entries[sigma_key(sigma)][str(seed)]["deltas"][metric] > 0.0
            for sigma in selected
            for seed in SEEDS
            for metric in ("raw_correlation", "raw_r2")
        )

    if all_positive(0.10):
        return "paired_advantage_through_10pct"
    if all_positive(0.05):
        return "paired_advantage_through_5pct"
    if all_positive(0.025):
        return "paired_advantage_through_2p5pct"
    means_positive = all(
        np.mean(
            [entries[sigma_key(sigma)][str(seed)]["deltas"][metric] for seed in SEEDS]
        )
        > 0.0
        for sigma in SIGMAS
        if sigma <= 0.05 + 1e-12
        for metric in ("raw_correlation", "raw_r2")
    )
    if means_positive:
        return "mixed_seedwise_positive_mean"
    return "no_stable_paired_noise_advantage"


def resolve_project_path(config_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return config_path.resolve().parent.parent / path


def aggregate_main(args: argparse.Namespace, master: dict[str, Any]) -> None:
    if args.seed is not None or args.condition is not None:
        raise ValueError("K17 aggregate does not accept seed or condition")
    root = Path(args.output_root or master["runtime"]["output_root"]).resolve()
    output_dir = root / "aggregate"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "noise_robustness_aggregate.json"
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite K17 aggregate: {output_path}")
    clean_path = resolve_project_path(args.config, master["clean_reference"]["path"])
    if base.sha256_file(clean_path) != master["clean_reference"]["sha256"]:
        raise ValueError("K17 clean K12 reference hash mismatch")
    clean = json.loads(clean_path.read_text(encoding="utf-8"))
    if clean["protocol_version"] != "isotropic_tube_vs_raw_propagation_k12_v1":
        raise ValueError("K17 clean reference protocol mismatch")
    entries: dict[str, Any] = {}
    summary: dict[str, Any] = {}
    for sigma in SIGMAS:
        key = sigma_key(sigma)
        entries[key] = {}
        corr_deltas = []
        r2_deltas = []
        for seed in SEEDS:
            path = run_root(master, args.output_root, sigma, seed) / "paired_validation_readout.json"
            item = json.loads(path.read_text(encoding="utf-8"))
            expected = build_run_config(master, sigma, seed)["selected_run"]
            checks = {
                "protocol": item["protocol_version"] == PROTOCOL,
                "selected_run": item["selected_run"] == expected,
                "config": item["master_config_sha256"] == base.sha256_file(args.config.resolve()),
                "source": item["source_files_sha256"] == source_hashes(),
                "validity": all(item["validity"].values()),
                "semantic_read": item["semantic_validation_evaluated"] is True,
                "test_unread": item["test_evaluated"] is False,
            }
            if not all(checks.values()):
                raise ValueError(f"invalid K17 readout {key} seed{seed}: {checks}")
            tube_raw = item["semantic"]["isotropic_tube"]["raw"]
            propagation_raw = item["semantic"]["raw_propagation"]["raw"]
            deltas = {
                "raw_correlation": float(
                    tube_raw["mean_direct_abs_correlation"]
                    - propagation_raw["mean_direct_abs_correlation"]
                ),
                "raw_r2": float(tube_raw["mean_r2"] - propagation_raw["mean_r2"]),
                "full_affine_r2": float(
                    item["semantic"]["isotropic_tube"]["full_affine"]["mean_r2"]
                    - item["semantic"]["raw_propagation"]["full_affine"]["mean_r2"]
                ),
            }
            corr_deltas.append(deltas["raw_correlation"])
            r2_deltas.append(deltas["raw_r2"])
            clean_entry = clean["budgets"]["80"]["seeds"][str(seed)]
            entries[key][str(seed)] = {
                "input_path": str(path),
                "input_sha256": base.sha256_file(path),
                "noise": {
                    "realized_rmse_unit_rgb": item["noise"]["realized_rmse_unit_rgb"],
                    "clipped_total_count": item["noise"]["clipped_total_count"],
                },
                "tube": {
                    "raw_correlation": tube_raw["mean_direct_abs_correlation"],
                    "raw_r2": tube_raw["mean_r2"],
                    "full_affine_r2": item["semantic"]["isotropic_tube"]["full_affine"]["mean_r2"],
                },
                "raw_propagation": {
                    "raw_correlation": propagation_raw["mean_direct_abs_correlation"],
                    "raw_r2": propagation_raw["mean_r2"],
                    "full_affine_r2": item["semantic"]["raw_propagation"]["full_affine"]["mean_r2"],
                },
                "deltas": deltas,
                "bootstrap": item["paired_bootstrap"],
                "clean_reference": clean_entry,
                "degradation_from_clean": {
                    "tube_raw_correlation": float(
                        tube_raw["mean_direct_abs_correlation"]
                        - clean_entry["tube"]["raw_correlation"]
                    ),
                    "tube_raw_r2": float(tube_raw["mean_r2"] - clean_entry["tube"]["raw_r2"]),
                    "raw_propagation_raw_correlation": float(
                        propagation_raw["mean_direct_abs_correlation"]
                        - clean_entry["raw_representation"]["raw_correlation"]
                    ),
                    "raw_propagation_raw_r2": float(
                        propagation_raw["mean_r2"]
                        - clean_entry["raw_representation"]["raw_r2"]
                    ),
                },
            }
        summary[key] = {
            "sigma_full_scale": sigma,
            "tube_minus_raw_correlation": summarize(corr_deltas),
            "tube_minus_raw_r2": summarize(r2_deltas),
            "raw_correlation_strict_positive_count": int(sum(value > 0.0 for value in corr_deltas)),
            "raw_r2_strict_positive_count": int(sum(value > 0.0 for value in r2_deltas)),
            "bootstrap_correlation_ci_lower_positive_count": int(
                sum(
                    entries[key][str(seed)]["bootstrap"]["metrics"]
                    ["mean_direct_abs_correlation"]["ci95"][0]
                    > 0.0
                    for seed in SEEDS
                )
            ),
            "bootstrap_r2_ci_lower_positive_count": int(
                sum(
                    entries[key][str(seed)]["bootstrap"]["metrics"]["mean_r2"]["ci95"][0]
                    > 0.0
                    for seed in SEEDS
                )
            ),
        }
    result = {
        "protocol_version": PROTOCOL,
        "mode": "four_level_three_seed_paired_validation_aggregate",
        "master_config_path": str(args.config.resolve()),
        "master_config_sha256": base.sha256_file(args.config.resolve()),
        "source_files_sha256": source_hashes(),
        "clean_reference": {
            "path": str(clean_path),
            "sha256": base.sha256_file(clean_path),
            "test_evaluated": clean["test_evaluated"],
        },
        "entries": entries,
        "summary": summary,
        "decision": {
            "verdict": aggregate_verdict(entries),
            "method_selection_reopened": False,
            "test_evaluated": False,
        },
        "semantic_validation_evaluated": True,
        "test_evaluated": False,
    }
    v11.atomic_json(output_path, result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("preflight", "smoke", "train", "readout", "aggregate"), required=True
    )
    parser.add_argument("--sigma", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--condition", choices=CONDITIONS)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    master = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    validate_master(master)
    if args.mode == "aggregate":
        if args.sigma is not None:
            raise ValueError("K17 aggregate does not accept sigma")
        aggregate_main(args, master)
        return
    if args.sigma is None:
        raise ValueError("K17 non-aggregate mode requires sigma")
    sigma_key(args.sigma)
    if args.mode == "preflight":
        preflight_main(args, master)
    elif args.mode in ("smoke", "train"):
        train_main(args, master)
    else:
        readout_main(args, master)


if __name__ == "__main__":
    main()
