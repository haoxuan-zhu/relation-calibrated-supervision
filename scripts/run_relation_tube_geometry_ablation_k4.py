"""Equal-coverage geometry ablation for the calibrated relation tube K4."""

from __future__ import annotations

import argparse
import json
import math
import platform
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

import audit_calibrated_relation_tube_gauge_k1 as gauge
import run_calibrated_relation_tube_k0 as tube
import run_conflict_projected_physics_k80_closed as v17
import run_diagnostic as base
import run_dynamic_residual_propagation_k0 as dynamic
import run_physics_functional_anchor_training as v11
import run_supervised_continuation_diagnostic as v6


PROTOCOL = "relation_tube_geometry_ablation_k4_v1"
GEOMETRIES = ("full_anisotropic", "diagonal", "isotropic", "rotated")
TRAIN_GEOMETRIES = ("diagonal", "isotropic", "rotated")
REGISTERED_SEEDS = (0, 42, 3407)
EXPECTED_FULL_GEOMETRY_SHA256 = (
    "d60717f6e09f1902491e759de6fda51c2cdc5f8ab1627b8e1cd2e09a7ebc7197"
)
EXPECTED_K3_AGGREGATE_SHA256 = (
    "d728a6a598b3528eae79dc446d7b9b8c01e9e359a518c659310ccde9c7e1a007"
)
SOURCE_FILES = (
    "run_relation_tube_geometry_ablation_k4.py",
    "run_calibrated_relation_tube_k0.py",
    "audit_calibrated_relation_tube_gauge_k1.py",
    "run_dynamic_residual_propagation_k0.py",
    "audit_physics_functional_anchor.py",
    "raw_ridge_propagation.py",
    "run_physics_functional_anchor_training.py",
    "run_supervised_continuation_diagnostic.py",
    "run_conflict_projected_physics_from_scratch.py",
    "run_conflict_projected_physics_k80_closed.py",
    "run_conditioned_film_diagnostic.py",
    "run_clamped_anchor_diagnostic.py",
    "run_diagnostic.py",
)


def source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    return {name: base.sha256_file(root / name) for name in SOURCE_FILES}


def validate_config(config: dict[str, Any]) -> None:
    if config["protocol_version"] != PROTOCOL:
        raise ValueError("unexpected geometry-ablation protocol")
    seed = int(config["training"]["seed"])
    if seed not in REGISTERED_SEEDS:
        raise ValueError("unregistered geometry-ablation seed")
    if int(config["initialization"]["seed"]) != seed:
        raise ValueError("initialization/training seed mismatch")
    if (
        config["initialization"]["expected_state_dict_sha256"]
        != tube.REGISTERED_INITIAL_STATE_HASHES[seed]
    ):
        raise ValueError("registered initial-state hash changed")
    if tuple(config["geometry_ablation"]["modes"]) != GEOMETRIES:
        raise ValueError("geometry registry changed")
    if tuple(config["geometry_ablation"]["train_modes"]) != TRAIN_GEOMETRIES:
        raise ValueError("geometry training registry changed")
    axis = np.asarray(config["geometry_ablation"]["rotation_axis"], dtype=np.float64)
    if not np.allclose(axis, np.ones(3) / np.sqrt(3.0), atol=1e-12):
        raise ValueError("registered rotation axis changed")
    if not np.isclose(
        float(config["geometry_ablation"]["rotation_angle_radians"]),
        math.pi / 4.0,
        atol=1e-15,
    ):
        raise ValueError("registered rotation angle changed")
    calibration = config["calibration"]
    if int(calibration["budget"]) != 80:
        raise ValueError("K4 is locked to K80")
    if int(calibration["subset_seed"]) != 20260731:
        raise ValueError("K4 subset changed")
    if calibration["subset_sha256"] != (
        "4bf1156bbf3f8f8ed0b5e63c530efd171e36395cd9d7d59003a8465ba43ddc84"
    ):
        raise ValueError("K4 subset hash changed")
    if not np.isclose(float(calibration["raw_ridge_alpha"]), 0.1):
        raise ValueError("raw-ridge alpha changed")
    if not np.isclose(float(calibration["coverage"]), 0.95):
        raise ValueError("coverage changed")
    if not np.isclose(float(calibration["covariance_diagonal_ridge"]), 1e-6):
        raise ValueError("second-moment ridge changed")
    expected_index = math.ceil(
        (int(calibration["budget"]) + 1) * float(calibration["coverage"])
    )
    if expected_index != int(calibration["expected_score_order_index_one_based"]):
        raise ValueError("finite-sample score index changed")
    training = config["training"]
    if int(training["epochs"]) != 100:
        raise ValueError("formal geometry training is locked to 100 epochs")
    if int(training["shuffle_seed"]) != 20260730:
        raise ValueError("anchor-map shuffle seed changed")
    if int(config["implementation"]["expected_parameter_count"]) != 16902384:
        raise ValueError("model parameter-count contract changed")
    upstream = config["upstream"]
    if upstream["full_geometry_sha256"] != EXPECTED_FULL_GEOMETRY_SHA256:
        raise ValueError("upstream full geometry changed")
    if upstream["k3_aggregate_sha256"] != EXPECTED_K3_AGGREGATE_SHA256:
        raise ValueError("upstream K3 aggregate changed")
    if int(upstream["seed"]) != seed:
        raise ValueError("upstream seed mismatch")


def rodrigues_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    vector = np.asarray(axis, dtype=np.float64)
    vector = vector / np.linalg.norm(vector)
    x, y, z = vector
    cross = np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64
    )
    return (
        np.eye(3, dtype=np.float64) * math.cos(angle)
        + (1.0 - math.cos(angle)) * np.outer(vector, vector)
        + math.sin(angle) * cross
    )


def calibrated_geometry(
    residuals: np.ndarray,
    second_moment: np.ndarray,
    coverage: float,
    diagonal_ridge: float,
    name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    values = np.asarray(residuals, dtype=np.float64)
    matrix = np.asarray(second_moment, dtype=np.float64)
    eigenvalues = np.linalg.eigvalsh(matrix)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("geometry residuals must be K-by-3")
    if matrix.shape != (3, 3) or not np.allclose(matrix, matrix.T, atol=1e-12):
        raise ValueError("geometry matrix must be symmetric 3-by-3")
    if not np.all(np.isfinite(eigenvalues)) or float(eigenvalues.min()) <= 0:
        raise ValueError("geometry matrix is not positive definite")
    precision = np.linalg.inv(matrix)
    scores = np.einsum("ni,ij,nj->n", values, precision, values)
    order_index = min(len(scores), math.ceil((len(scores) + 1) * coverage))
    radius_squared = float(np.sort(scores)[order_index - 1])
    empirical_coverage = float(np.mean(scores <= radius_squared + 1e-12))
    sign, logdet = np.linalg.slogdet(matrix)
    if sign <= 0:
        raise ValueError("geometry log determinant is invalid")
    geometry = {
        # Historical model/schema key; mathematically this is a second moment.
        "covariance": matrix,
        "precision": precision,
        "radius_squared": radius_squared,
    }
    audit = {
        "name": name,
        "residual_count": int(len(values)),
        "coverage_target": float(coverage),
        "score_order_index_one_based": int(order_index),
        "empirical_coverage": empirical_coverage,
        "diagonal_ridge": float(diagonal_ridge),
        "second_moment_eigenvalues": eigenvalues,
        "second_moment_condition_number": float(eigenvalues.max() / eigenvalues.min()),
        "score_minimum": float(scores.min()),
        "score_median": float(np.median(scores)),
        "score_maximum": float(scores.max()),
        "radius_squared": radius_squared,
        "log_volume_without_4pi_over_3": float(
            0.5 * logdet + 1.5 * math.log(radius_squared)
        ),
        "finite": bool(
            np.all(np.isfinite(matrix))
            and np.all(np.isfinite(precision))
            and np.all(np.isfinite(scores))
        ),
    }
    return geometry, audit


def build_geometry_registry(
    normalized_loo_residuals: np.ndarray,
    coverage: float,
    diagonal_ridge: float,
    rotation_axis: np.ndarray,
    rotation_angle: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    residuals = np.asarray(normalized_loo_residuals, dtype=np.float64)
    base_second_moment = residuals.T @ residuals / len(residuals)
    base_second_moment += diagonal_ridge * np.eye(3, dtype=np.float64)
    rotation = rodrigues_rotation(rotation_axis, rotation_angle)
    matrices = {
        "full_anisotropic": base_second_moment,
        "diagonal": np.diag(np.diag(base_second_moment)),
        "isotropic": np.eye(3, dtype=np.float64)
        * (np.trace(base_second_moment) / 3.0),
        "rotated": rotation @ base_second_moment @ rotation.T,
    }
    registry: dict[str, Any] = {}
    audits: dict[str, Any] = {}
    for name in GEOMETRIES:
        geometry, audit = calibrated_geometry(
            residuals,
            matrices[name],
            coverage,
            diagonal_ridge,
            name,
        )
        registry[name] = geometry
        audits[name] = audit
    audits["rotation"] = {
        "matrix": rotation,
        "orthogonal": bool(np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)),
        "determinant": float(np.linalg.det(rotation)),
        "full_rotated_eigenvalues_match": bool(
            np.allclose(
                audits["full_anisotropic"]["second_moment_eigenvalues"],
                audits["rotated"]["second_moment_eigenvalues"],
                atol=1e-12,
            )
        ),
    }
    return registry, audits


def build_preflight_payload(
    config: dict[str, Any], images: np.ndarray, raw_latents: np.ndarray
) -> dict[str, Any]:
    subset, subset_audit = dynamic.build_subset(config)
    train_end = int(config["split"]["train_end"])
    features = tube.calibration_features(images, raw_latents, subset, train_end)
    targets = raw_latents[0, subset, :3].astype(np.float64) / 255.0
    teacher, loo_predictions, teacher_audit = tube.fit_teacher(
        features,
        targets,
        float(config["calibration"]["raw_ridge_alpha"]),
        bool(config["calibration"]["clip_predictions_to_unit_interval"]),
    )
    _, latent_mean, latent_std = v17.normalize_with_k80_rgb_statistics(
        raw_latents, train_end, subset
    )
    residuals = (targets - loo_predictions) * 255.0 / latent_std[:3]
    geometry_registry, geometry_audits = build_geometry_registry(
        residuals,
        float(config["calibration"]["coverage"]),
        float(config["calibration"]["covariance_diagonal_ridge"]),
        np.asarray(config["geometry_ablation"]["rotation_axis"], dtype=np.float64),
        float(config["geometry_ablation"]["rotation_angle_radians"]),
    )
    geometry_hashes = {
        name: dynamic.canonical_sha256(geometry_registry[name]) for name in GEOMETRIES
    }
    coverages = [geometry_audits[name]["empirical_coverage"] for name in GEOMETRIES]
    order_indices = [
        geometry_audits[name]["score_order_index_one_based"] for name in GEOMETRIES
    ]
    checks = {
        "subset_exact": subset_audit["subset_sha256"]
        == config["calibration"]["subset_sha256"],
        "subset_unique": subset_audit["unique_rows"] == 80,
        "teacher_finite": teacher_audit["finite"],
        "loo_complete": teacher_audit["loo_count"] == 80,
        "all_geometry_finite": all(
            geometry_audits[name]["finite"] for name in GEOMETRIES
        ),
        "all_score_indices_exact": all(
            value
            == int(config["calibration"]["expected_score_order_index_one_based"])
            for value in order_indices
        ),
        "coverage_matched": bool(np.allclose(coverages, coverages[0], atol=0.0)),
        "coverage_at_least_target": all(
            value >= float(config["calibration"]["coverage"])
            for value in coverages
        ),
        "full_geometry_matches_k3": geometry_hashes["full_anisotropic"]
        == config["upstream"]["full_geometry_sha256"],
        "rotation_orthogonal": geometry_audits["rotation"]["orthogonal"],
        "rotation_proper": bool(
            np.isclose(geometry_audits["rotation"]["determinant"], 1.0, atol=1e-12)
        ),
        "rotated_eigenspectrum_exact": geometry_audits["rotation"][
            "full_rotated_eigenvalues_match"
        ],
        "test_not_read": True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"geometry-ablation preflight failed: {checks}")
    return {
        "subset": subset_audit,
        "normalization": {"latent_mean": latent_mean, "latent_std": latent_std},
        "teachers": {
            "correct": {"model": tube.teacher_json(teacher), "audit": teacher_audit}
        },
        "geometries": geometry_registry,
        "geometry_audits": geometry_audits,
        "geometry_sha256": geometry_hashes,
        "checks": checks,
    }


def preflight_main(args: argparse.Namespace, config: dict[str, Any]) -> None:
    if args.geometry is not None or args.epochs is not None:
        raise ValueError("preflight accepts neither geometry nor epochs")
    root = Path(args.output_root or config["runtime"]["output_root"]).resolve()
    output_dir = root / "preflight"
    output_dir.mkdir(parents=True, exist_ok=True)
    payload_path = output_dir / "geometry_preflight.json"
    lock_path = output_dir / "preflight_lock.json"
    if payload_path.exists() or lock_path.exists():
        raise FileExistsError("refusing to overwrite geometry preflight")
    images = np.load(Path(config["dataset"]["cache_path"]), mmap_mode="r")
    raw_latents = base.load_latents(config).astype(np.float64)
    payload = build_preflight_payload(config, images, raw_latents)
    payload.update(
        protocol_version=PROTOCOL,
        mode="train_only_equal_coverage_geometry_preflight",
        config_path=str(args.config.resolve()),
        config_sha256=base.sha256_file(args.config.resolve()),
        source_files_sha256=source_hashes(),
        semantic_validation_evaluated=False,
        test_evaluated=False,
    )
    v11.atomic_json(payload_path, payload)
    lock = {
        "status": "locked_train_only_equal_coverage_geometry",
        "protocol_version": PROTOCOL,
        "config_sha256": base.sha256_file(args.config.resolve()),
        "source_files_sha256": source_hashes(),
        "payload_sha256": base.sha256_file(payload_path),
        "semantic_validation_evaluated": False,
        "test_evaluated": False,
    }
    v11.atomic_json(lock_path, lock)


def load_preflight(
    root: Path, config_path: Path, config: dict[str, Any]
) -> tuple[dict[str, Any], Path]:
    payload_path = root / "preflight" / "geometry_preflight.json"
    lock_path = root / "preflight" / "preflight_lock.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    checks = {
        "status": lock["status"] == "locked_train_only_equal_coverage_geometry",
        "protocol": lock["protocol_version"] == PROTOCOL,
        "config": lock["config_sha256"] == base.sha256_file(config_path),
        "source": lock["source_files_sha256"] == source_hashes(),
        "payload": lock["payload_sha256"] == base.sha256_file(payload_path),
        "payload_valid": all(payload["checks"].values()),
        "test_unread": lock["test_evaluated"] is False,
    }
    if not all(checks.values()):
        raise ValueError(f"invalid geometry preflight: {checks}")
    return payload, lock_path


def geometry_preflight(payload: dict[str, Any], name: str) -> dict[str, Any]:
    return {
        "teachers": payload["teachers"],
        "normalization": payload["normalization"],
        "geometry": payload["geometries"][name],
    }


def train_main(args: argparse.Namespace, config: dict[str, Any]) -> None:
    if args.geometry not in TRAIN_GEOMETRIES:
        raise ValueError("train/smoke requires diagonal, isotropic, or rotated")
    formal = args.mode == "train"
    epochs = int(args.epochs or (100 if formal else 2))
    if formal and epochs != 100:
        raise ValueError("formal geometry training is locked to 100 epochs")
    if not formal and epochs != 2:
        raise ValueError("geometry smoke is locked to two epochs")
    root = Path(args.output_root or config["runtime"]["output_root"]).resolve()
    payload, preflight_lock_path = load_preflight(root, args.config.resolve(), config)
    output_dir = root / ("formal" if formal else "smoke") / args.geometry
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / ("training_results.json" if formal else "smoke_results.json")
    lock_path = output_dir / "training_lock.json"
    if result_path.exists() or lock_path.exists():
        raise FileExistsError(f"refusing to overwrite geometry run: {output_dir}")
    torch.set_num_threads(int(config["runtime"]["num_threads"]))
    torch.set_float32_matmul_precision("high")
    device = torch.device(config["runtime"]["device"] if torch.cuda.is_available() else "cpu")
    images = np.load(Path(config["dataset"]["cache_path"]), mmap_mode="r")
    raw_latents = base.load_latents(config)
    subset, _ = dynamic.build_subset(config)
    latents, latent_mean, latent_std = v17.normalize_with_k80_rgb_statistics(
        raw_latents, int(config["split"]["train_end"]), subset
    )
    if not np.allclose(latent_mean, payload["normalization"]["latent_mean"]):
        raise ValueError("preflight/train latent mean mismatch")
    if not np.allclose(latent_std, payload["normalization"]["latent_std"]):
        raise ValueError("preflight/train latent std mismatch")
    anchor_maps, anchor_audit = tube.v2.build_anchor_maps(config)
    initial_state, initial_hash = tube.v16.make_initial_state(config, device)
    if initial_hash != config["initialization"]["expected_state_dict_sha256"]:
        raise ValueError("geometry initial state mismatch")
    training = tube.train_condition(
        "tube_correct",
        config,
        geometry_preflight(payload, args.geometry),
        images,
        latents,
        anchor_maps,
        initial_state,
        initial_hash,
        device,
        output_dir,
        epochs,
    )
    checks = {
        "preflight_valid": all(payload["checks"].values()),
        "initial_state_exact": initial_hash
        == config["initialization"]["expected_state_dict_sha256"],
        "parameter_count_exact": training["parameter_count"]
        == int(config["implementation"]["expected_parameter_count"]),
        "epoch_exact": training["final_epoch"] == epochs,
        "history_finite": dynamic.numeric_history_is_finite(training["history"]),
        "semantic_unread": not training["semantic_truth_read_during_training"],
        "test_unread": not training["test_evaluated"],
    }
    if not all(checks.values()):
        raise RuntimeError(f"geometry training validity failed: {checks}")
    lock = {
        "status": "locked_before_joint_geometry_validation_readout",
        "mode": "formal" if formal else "smoke",
        "protocol_version": PROTOCOL,
        "geometry": args.geometry,
        "config_sha256": base.sha256_file(args.config.resolve()),
        "source_files_sha256": source_hashes(),
        "preflight_lock_sha256": base.sha256_file(preflight_lock_path),
        "geometry_sha256": payload["geometry_sha256"][args.geometry],
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
    result = {
        "protocol_version": PROTOCOL,
        "mode": "formal_train" if formal else "smoke_train",
        "geometry": args.geometry,
        "config_sha256": base.sha256_file(args.config.resolve()),
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
    }
    v11.atomic_json(result_path, result)


def load_new_model(
    name: str,
    root: Path,
    config_path: Path,
    config: dict[str, Any],
    payload: dict[str, Any],
    preflight_lock_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, str]:
    lock_path = root / "formal" / name / "training_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    checks = {
        "status": lock["status"] == "locked_before_joint_geometry_validation_readout",
        "formal": lock["mode"] == "formal",
        "protocol": lock["protocol_version"] == PROTOCOL,
        "geometry_name": lock["geometry"] == name,
        "config": lock["config_sha256"] == base.sha256_file(config_path),
        "source": lock["source_files_sha256"] == source_hashes(),
        "preflight": lock["preflight_lock_sha256"] == base.sha256_file(preflight_lock_path),
        "geometry": lock["geometry_sha256"] == payload["geometry_sha256"][name],
        "validity": all(lock["validity"].values()),
        "test_unread": lock["test_evaluated"] is False,
    }
    if not all(checks.values()):
        raise ValueError(f"invalid geometry lock {name}: {checks}")
    checkpoint_path = Path(lock["checkpoint"]["path"])
    if base.sha256_file(checkpoint_path) != lock["checkpoint"]["sha256"]:
        raise ValueError(f"geometry checkpoint hash mismatch: {name}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if (
        checkpoint["protocol_version"] != PROTOCOL
        or checkpoint["condition"] != "tube_correct"
        or int(checkpoint["epoch"]) != 100
    ):
        raise ValueError(f"geometry checkpoint identity mismatch: {name}")
    model = tube.model_for_condition(
        "tube_correct", config, geometry_preflight(payload, name), device
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model, base.sha256_file(lock_path)


def load_upstream_full_model(
    config: dict[str, Any], payload: dict[str, Any], device: torch.device
) -> tuple[torch.nn.Module, dict[str, Any]]:
    upstream = config["upstream"]
    lock_path = Path(upstream["full_training_lock_path"])
    if base.sha256_file(lock_path) != upstream["full_training_lock_sha256"]:
        raise ValueError("upstream full training-lock hash mismatch")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    checks = {
        "condition": lock["condition"] == "tube_correct",
        "formal": lock["mode"] == "formal",
        "seed": lock["initial_state_sha256"]
        == config["initialization"]["expected_state_dict_sha256"],
        "geometry": lock["geometry_sha256"]
        == payload["geometry_sha256"]["full_anisotropic"],
        "checkpoint_registered": lock["checkpoint"]["sha256"]
        == upstream["full_checkpoint_sha256"],
        "test_unread": lock["test_evaluated"] is False,
        "validity": all(lock["validity"].values()),
    }
    if not all(checks.values()):
        raise ValueError(f"invalid upstream full lock: {checks}")
    checkpoint_path = Path(upstream["full_checkpoint_path"])
    if base.sha256_file(checkpoint_path) != upstream["full_checkpoint_sha256"]:
        raise ValueError("upstream full checkpoint hash mismatch")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if (
        checkpoint["condition"] != "tube_correct"
        or int(checkpoint["epoch"]) != 100
        or checkpoint["initial_state_sha256"]
        != config["initialization"]["expected_state_dict_sha256"]
    ):
        raise ValueError("upstream full checkpoint identity mismatch")
    model = tube.model_for_condition(
        "tube_correct",
        config,
        geometry_preflight(payload, "full_anisotropic"),
        device,
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model, checks


def metric_bundle(
    model: torch.nn.Module,
    images: np.ndarray,
    latents: np.ndarray,
    subset: np.ndarray,
    start: int,
    end: int,
    batch_size: int,
    ridge: float,
    anchor_maps: dict[str, Any],
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    calibration_prediction = gauge.predict_selected(
        model, images, latents, subset, batch_size, device
    )
    validation_prediction = v6.predict_rgb(
        model, images, latents, start, end, batch_size, device
    )
    bundle = gauge.metric_bundle(
        calibration_prediction,
        validation_prediction,
        latents[0, subset, :3],
        latents[0, start:end, :3],
        ridge,
    )
    return {
        "raw": bundle["raw"],
        "full_affine": bundle["full_affine"],
        "ccrl_validation": tube.v2.validate(
            model,
            images,
            latents,
            anchor_maps,
            tube.v2.Condition("tube_correct", "oracle"),
            config,
            device,
        ),
        "representation_audit": tube.representation_audit(
            model, images, latents, start, end, batch_size, device
        ),
    }


def decide(runs: dict[str, Any]) -> dict[str, Any]:
    full = runs["full_anisotropic"]
    full_r2 = float(full["full_affine"]["mean_r2"])
    controls = TRAIN_GEOMETRIES
    r2_deltas = {
        f"full_minus_{name}": full_r2
        - float(runs[name]["full_affine"]["mean_r2"])
        for name in controls
    }
    raw_correlation_deltas = {
        f"full_minus_{name}": float(full["raw"]["mean_direct_abs_correlation"])
        - float(runs[name]["raw"]["mean_direct_abs_correlation"])
        for name in controls
    }
    full_best_r2 = all(value > 0 for value in r2_deltas.values())
    if full_best_r2:
        verdict = "anisotropic_geometry_development_signal_unlock_confirmation"
    elif (
        float(runs["diagonal"]["full_affine"]["mean_r2"]) >= full_r2
        and full_r2 > float(runs["isotropic"]["full_affine"]["mean_r2"])
    ):
        verdict = "axis_scaled_clipping_sufficient_no_offdiagonal_claim"
    elif float(runs["isotropic"]["full_affine"]["mean_r2"]) >= full_r2:
        verdict = "generic_isotropic_clipping_sufficient_stop_geometry_claim"
    elif float(runs["rotated"]["full_affine"]["mean_r2"]) >= full_r2:
        verdict = "data_aligned_orientation_not_supported_stop_geometry_claim"
    else:
        verdict = "geometry_ablation_mixed_no_upgrade"
    return {
        "verdict": verdict,
        "full_affine_r2_deltas": r2_deltas,
        "raw_correlation_deltas": raw_correlation_deltas,
        "full_best_full_affine_r2": full_best_r2,
        "full_best_raw_correlation": all(
            value > 0 for value in raw_correlation_deltas.values()
        ),
        "confirmation_unlocked": full_best_r2,
        "test_evaluated": False,
    }


def readout_main(args: argparse.Namespace, config: dict[str, Any]) -> None:
    if args.geometry is not None or args.epochs is not None:
        raise ValueError("readout accepts neither geometry nor epochs")
    root = Path(args.output_root or config["runtime"]["output_root"]).resolve()
    payload, preflight_lock_path = load_preflight(root, args.config.resolve(), config)
    torch.set_num_threads(int(config["runtime"]["num_threads"]))
    torch.set_float32_matmul_precision("high")
    device = torch.device(config["runtime"]["device"] if torch.cuda.is_available() else "cpu")
    images = np.load(Path(config["dataset"]["cache_path"]), mmap_mode="r")
    raw_latents = base.load_latents(config)
    subset, _ = dynamic.build_subset(config)
    latents, _, _ = v17.normalize_with_k80_rgb_statistics(
        raw_latents, int(config["split"]["train_end"]), subset
    )
    anchor_maps, _ = tube.v2.build_anchor_maps(config)
    start = int(config["split"]["train_end"])
    end = int(config["split"]["validation_end"])
    batch_size = int(config["training"]["batch_size"])
    ridge = float(config["evaluation"]["full_affine_ridge"])
    models: dict[str, torch.nn.Module] = {}
    lock_audits: dict[str, Any] = {}
    models["full_anisotropic"], lock_audits["full_anisotropic"] = (
        load_upstream_full_model(config, payload, device)
    )
    for name in TRAIN_GEOMETRIES:
        models[name], lock_audits[name] = load_new_model(
            name,
            root,
            args.config.resolve(),
            config,
            payload,
            preflight_lock_path,
            device,
        )
    runs: dict[str, Any] = {}
    for name in GEOMETRIES:
        runs[name] = metric_bundle(
            models[name],
            images,
            latents,
            subset,
            start,
            end,
            batch_size,
            ridge,
            anchor_maps,
            config,
            device,
        )
        del models[name]
        if device.type == "cuda":
            torch.cuda.empty_cache()
    aggregate_path = Path(config["upstream"]["k3_aggregate_path"])
    if base.sha256_file(aggregate_path) != config["upstream"]["k3_aggregate_sha256"]:
        raise ValueError("K3 aggregate hash mismatch")
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    expected = aggregate["seeds"][str(config["training"]["seed"])]["runs"]["tube_correct"]
    full_reproduction = {
        "raw_correlation": bool(
            np.isclose(
                runs["full_anisotropic"]["raw"]["mean_direct_abs_correlation"],
                expected["raw"]["mean_direct_abs_correlation"],
                atol=5e-6,
            )
        ),
        "raw_r2": bool(
            np.isclose(
                runs["full_anisotropic"]["raw"]["mean_r2"],
                expected["raw"]["mean_r2"],
                atol=5e-6,
            )
        ),
        "full_affine_correlation": bool(
            np.isclose(
                runs["full_anisotropic"]["full_affine"]["mean_direct_abs_correlation"],
                expected["full_affine"]["mean_direct_abs_correlation"],
                atol=5e-6,
            )
        ),
        "full_affine_r2": bool(
            np.isclose(
                runs["full_anisotropic"]["full_affine"]["mean_r2"],
                expected["full_affine"]["mean_r2"],
                atol=5e-6,
            )
        ),
    }
    if not all(full_reproduction.values()):
        raise RuntimeError(f"upstream full metric drift: {full_reproduction}")
    result = {
        "protocol_version": PROTOCOL,
        "mode": "equal_coverage_four_geometry_joint_validation_only_readout",
        "seed": int(config["training"]["seed"]),
        "config_sha256": base.sha256_file(args.config.resolve()),
        "source_files_sha256": source_hashes(),
        "preflight_lock_sha256": base.sha256_file(preflight_lock_path),
        "geometry_sha256": payload["geometry_sha256"],
        "geometry_audits": payload["geometry_audits"],
        "lock_audits": lock_audits,
        "upstream_full_reproduction": full_reproduction,
        "runs": runs,
        "decision": decide(runs),
        "semantic_validation_evaluated": True,
        "test_evaluated": False,
    }
    output_path = root / "formal" / "geometry_validation_readout.json"
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite geometry readout: {output_path}")
    v11.atomic_json(output_path, result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("preflight", "smoke", "train", "readout"), required=True
    )
    parser.add_argument("--geometry", choices=TRAIN_GEOMETRIES)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    validate_config(config)
    if args.mode == "preflight":
        preflight_main(args, config)
    elif args.mode in ("smoke", "train"):
        train_main(args, config)
    else:
        readout_main(args, config)


if __name__ == "__main__":
    main()
