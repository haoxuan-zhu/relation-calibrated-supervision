"""Calibration-subset robustness for full, diagonal, and isotropic relation tubes."""

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

import audit_relation_tube_geometry_downstream_k6 as k6
import run_calibrated_relation_tube_k0 as tube
import run_conflict_projected_physics_k80_closed as v17
import run_diagnostic as base
import run_dynamic_residual_propagation_k0 as dynamic
import run_physics_functional_anchor_training as v11
import run_relation_tube_geometry_ablation_k4 as k4
import run_supervised_continuation_diagnostic as v6


PROTOCOL = "relation_tube_calibration_subset_k8_v1"
SUBSET_SEEDS = (20260811, 20260821, 20260831)
SUBSET_HASHES = {
    20260811: "d43642e32abf1992d5d82cac599df2a279c46f9fc5c09fd334e0f236b2e958b5",
    20260821: "0e91398680e1ae3bb9fa8c2c6f582edc02caebcbbabbd290b8932e7577e52f3f",
    20260831: "2e111fb98d11949a0101147279bcda28d0ae841fbc6720e057713fdb33f7d935",
}
GEOMETRIES = ("full_anisotropic", "diagonal", "isotropic")
SOURCE_FILES = (
    "run_relation_tube_calibration_subset_k8.py",
    "audit_relation_tube_geometry_downstream_k6.py",
    *k4.SOURCE_FILES,
)


def source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    return {name: base.sha256_file(root / name) for name in SOURCE_FILES}


def validate_config(config: dict[str, Any]) -> None:
    if config["protocol_version"] != PROTOCOL:
        raise ValueError("unexpected K8 protocol")
    subset_seed = int(config["calibration"]["subset_seed"])
    if subset_seed not in SUBSET_SEEDS:
        raise ValueError("unregistered K8 subset seed")
    if config["calibration"]["subset_sha256"] != SUBSET_HASHES[subset_seed]:
        raise ValueError("K8 subset hash changed")
    if int(config["calibration"]["budget"]) != 80:
        raise ValueError("K8 is locked to K80")
    if tuple(config["geometry_subset"]["modes"]) != GEOMETRIES:
        raise ValueError("K8 geometry registry changed")
    if int(config["training"]["seed"]) != 3407:
        raise ValueError("K8 fixes model initialization to 3407")
    if int(config["initialization"]["seed"]) != 3407:
        raise ValueError("K8 initialization metadata changed")
    if (
        config["initialization"]["expected_state_dict_sha256"]
        != tube.REGISTERED_INITIAL_STATE_HASHES[3407]
    ):
        raise ValueError("K8 initial state hash changed")
    if int(config["training"]["epochs"]) != 100:
        raise ValueError("K8 formal training is locked to 100 epochs")
    if int(config["training"]["shuffle_seed"]) != 20260730:
        raise ValueError("K8 training order changed")
    if not np.isclose(float(config["calibration"]["coverage"]), 0.95):
        raise ValueError("K8 coverage changed")
    if not np.isclose(float(config["calibration"]["raw_ridge_alpha"]), 0.1):
        raise ValueError("K8 teacher ridge changed")
    if not np.isclose(float(config["calibration"]["covariance_diagonal_ridge"]), 1e-6):
        raise ValueError("K8 second-moment ridge changed")
    expected_index = math.ceil(
        (int(config["calibration"]["budget"]) + 1)
        * float(config["calibration"]["coverage"])
    )
    if expected_index != int(config["calibration"]["expected_score_order_index_one_based"]):
        raise ValueError("K8 score order statistic changed")
    if int(config["implementation"]["expected_parameter_count"]) != 16902384:
        raise ValueError("K8 parameter-count contract changed")
    if config["audit"]["test_evaluated"] is not False:
        raise ValueError("K8 test must remain closed")


def build_geometry_registry(
    residuals: np.ndarray, coverage: float, diagonal_ridge: float
) -> tuple[dict[str, Any], dict[str, Any]]:
    values = np.asarray(residuals, dtype=np.float64)
    full = values.T @ values / len(values)
    full += diagonal_ridge * np.eye(3, dtype=np.float64)
    matrices = {
        "full_anisotropic": full,
        "diagonal": np.diag(np.diag(full)),
        "isotropic": np.eye(3, dtype=np.float64) * (np.trace(full) / 3.0),
    }
    registry: dict[str, Any] = {}
    audits: dict[str, Any] = {}
    for name in GEOMETRIES:
        registry[name], audits[name] = k4.calibrated_geometry(
            values, matrices[name], coverage, diagonal_ridge, name
        )
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
    geometries, geometry_audits = build_geometry_registry(
        residuals,
        float(config["calibration"]["coverage"]),
        float(config["calibration"]["covariance_diagonal_ridge"]),
    )
    geometry_hashes = {
        name: dynamic.canonical_sha256(geometries[name]) for name in GEOMETRIES
    }
    coverages = [geometry_audits[name]["empirical_coverage"] for name in GEOMETRIES]
    checks = {
        "subset_exact": subset_audit["subset_sha256"]
        == config["calibration"]["subset_sha256"],
        "subset_unique": subset_audit["unique_rows"] == 80,
        "teacher_finite": teacher_audit["finite"],
        "loo_complete": teacher_audit["loo_count"] == 80,
        "all_geometry_finite": all(geometry_audits[name]["finite"] for name in GEOMETRIES),
        "all_score_indices_exact": all(
            geometry_audits[name]["score_order_index_one_based"]
            == int(config["calibration"]["expected_score_order_index_one_based"])
            for name in GEOMETRIES
        ),
        "coverage_matched": bool(np.allclose(coverages, coverages[0], atol=0.0)),
        "coverage_at_least_target": all(
            value >= float(config["calibration"]["coverage"]) for value in coverages
        ),
        "test_not_read": True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"K8 preflight failed: {checks}")
    return {
        "subset": subset_audit,
        "normalization": {"latent_mean": latent_mean, "latent_std": latent_std},
        "teachers": {
            "correct": {"model": tube.teacher_json(teacher), "audit": teacher_audit}
        },
        "geometries": geometries,
        "geometry_audits": geometry_audits,
        "geometry_sha256": geometry_hashes,
        "checks": checks,
    }


def geometry_preflight(payload: dict[str, Any], name: str) -> dict[str, Any]:
    return {
        "teachers": payload["teachers"],
        "normalization": payload["normalization"],
        "geometry": payload["geometries"][name],
    }


def preflight_main(args: argparse.Namespace, config: dict[str, Any]) -> None:
    if args.geometry is not None or args.epochs is not None:
        raise ValueError("K8 preflight accepts neither geometry nor epochs")
    root = Path(args.output_root or config["runtime"]["output_root"]).resolve()
    output_dir = root / "preflight"
    output_dir.mkdir(parents=True, exist_ok=True)
    payload_path = output_dir / "geometry_preflight.json"
    lock_path = output_dir / "preflight_lock.json"
    if payload_path.exists() or lock_path.exists():
        raise FileExistsError("refusing to overwrite K8 preflight")
    images = np.load(Path(config["dataset"]["cache_path"]), mmap_mode="r")
    raw_latents = base.load_latents(config).astype(np.float64)
    payload = build_preflight_payload(config, images, raw_latents)
    payload.update(
        protocol_version=PROTOCOL,
        config_path=str(args.config.resolve()),
        config_sha256=base.sha256_file(args.config.resolve()),
        source_files_sha256=source_hashes(),
        semantic_validation_evaluated=False,
        test_evaluated=False,
    )
    v11.atomic_json(payload_path, payload)
    v11.atomic_json(
        lock_path,
        {
            "status": "locked_k8_train_only_geometry_preflight",
            "protocol_version": PROTOCOL,
            "config_sha256": base.sha256_file(args.config.resolve()),
            "source_files_sha256": source_hashes(),
            "payload_sha256": base.sha256_file(payload_path),
            "semantic_validation_evaluated": False,
            "test_evaluated": False,
        },
    )


def load_preflight(
    root: Path, config_path: Path
) -> tuple[dict[str, Any], Path]:
    payload_path = root / "preflight" / "geometry_preflight.json"
    lock_path = root / "preflight" / "preflight_lock.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    checks = {
        "status": lock["status"] == "locked_k8_train_only_geometry_preflight",
        "protocol": lock["protocol_version"] == PROTOCOL,
        "config": lock["config_sha256"] == base.sha256_file(config_path),
        "source": lock["source_files_sha256"] == source_hashes(),
        "payload": lock["payload_sha256"] == base.sha256_file(payload_path),
        "payload_valid": all(payload["checks"].values()),
        "test_unread": lock["test_evaluated"] is False,
    }
    if not all(checks.values()):
        raise ValueError(f"invalid K8 preflight: {checks}")
    return payload, lock_path


def train_main(args: argparse.Namespace, config: dict[str, Any]) -> None:
    if args.geometry not in GEOMETRIES:
        raise ValueError("K8 train requires a registered geometry")
    smoke = args.mode == "smoke"
    epochs = int(args.epochs or (2 if smoke else 100))
    if (smoke and epochs != 2) or (not smoke and epochs != 100):
        raise ValueError("K8 smoke/formal epoch contract changed")
    root = Path(args.output_root or config["runtime"]["output_root"]).resolve()
    payload, preflight_lock_path = load_preflight(root, args.config.resolve())
    output_dir = root / ("smoke" if smoke else "formal") / args.geometry
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / ("smoke_results.json" if smoke else "training_results.json")
    lock_path = output_dir / "training_lock.json"
    if result_path.exists() or lock_path.exists():
        raise FileExistsError(f"refusing to overwrite K8 run: {output_dir}")
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
        raise ValueError("K8 latent mean mismatch")
    if not np.allclose(latent_std, payload["normalization"]["latent_std"]):
        raise ValueError("K8 latent std mismatch")
    anchor_maps, anchor_audit = tube.v2.build_anchor_maps(config)
    initial_state, initial_hash = tube.v16.make_initial_state(config, device)
    if initial_hash != config["initialization"]["expected_state_dict_sha256"]:
        raise ValueError("K8 initial state mismatch")
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
        raise RuntimeError(f"K8 training validity failed: {checks}")
    lock = {
        "status": "locked_before_k8_joint_validation_readout",
        "mode": "smoke" if smoke else "formal",
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
    v11.atomic_json(
        result_path,
        {
            "protocol_version": PROTOCOL,
            "mode": "smoke_train" if smoke else "formal_train",
            "subset_seed": int(config["calibration"]["subset_seed"]),
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
        },
    )


def load_model(
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
        "status": lock["status"] == "locked_before_k8_joint_validation_readout",
        "formal": lock["mode"] == "formal",
        "protocol": lock["protocol_version"] == PROTOCOL,
        "geometry": lock["geometry"] == name,
        "config": lock["config_sha256"] == base.sha256_file(config_path),
        "source": lock["source_files_sha256"] == source_hashes(),
        "preflight": lock["preflight_lock_sha256"] == base.sha256_file(preflight_lock_path),
        "geometry_hash": lock["geometry_sha256"] == payload["geometry_sha256"][name],
        "validity": all(lock["validity"].values()),
        "test_unread": lock["test_evaluated"] is False,
    }
    if not all(checks.values()):
        raise ValueError(f"invalid K8 model lock {name}: {checks}")
    checkpoint_path = Path(lock["checkpoint"]["path"])
    if base.sha256_file(checkpoint_path) != lock["checkpoint"]["sha256"]:
        raise ValueError(f"K8 checkpoint hash mismatch: {name}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if (
        checkpoint["protocol_version"] != PROTOCOL
        or checkpoint["condition"] != "tube_correct"
        or int(checkpoint["epoch"]) != 100
    ):
        raise ValueError(f"K8 checkpoint identity mismatch: {name}")
    model = tube.model_for_condition(
        "tube_correct", config, geometry_preflight(payload, name), device
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model, base.sha256_file(lock_path)


def comparison(full: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    semantic_deltas = {
        "raw_correlation": float(
            candidate["semantic"]["raw"]["mean_direct_abs_correlation"]
            - full["semantic"]["raw"]["mean_direct_abs_correlation"]
        ),
        "raw_r2": float(
            candidate["semantic"]["raw"]["mean_r2"]
            - full["semantic"]["raw"]["mean_r2"]
        ),
        "rgb_mcc": float(
            candidate["semantic"]["rgb_hungarian_mcc"]
            - full["semantic"]["rgb_hungarian_mcc"]
        ),
        "coordinatewise_r2": float(
            candidate["semantic"]["coordinatewise_affine"]["mean_r2"]
            - full["semantic"]["coordinatewise_affine"]["mean_r2"]
        ),
        "full_affine_r2": float(
            candidate["semantic"]["full_affine"]["mean_r2"]
            - full["semantic"]["full_affine"]["mean_r2"]
        ),
    }
    edge_delta = float(candidate["graph"]["edge_auroc"] - full["graph"]["edge_auroc"])
    full_minus_candidate_shd = int(
        full["graph"]["fixed_threshold_shd"]
        - candidate["graph"]["fixed_threshold_shd"]
    )
    return {
        "semantic_deltas": semantic_deltas,
        "complete_semantic_advantage": all(value > 0 for value in semantic_deltas.values()),
        "edge_auroc_delta": edge_delta,
        "full_minus_candidate_shd": full_minus_candidate_shd,
        "graph_advantage": edge_delta > 0 and full_minus_candidate_shd >= 0,
        "candidate_minus_full_ccrl_total": float(
            candidate["ccrl_validation"]["total"]
            - full["ccrl_validation"]["total"]
        ),
    }


def readout_main(args: argparse.Namespace, config: dict[str, Any]) -> None:
    if args.geometry is not None or args.epochs is not None:
        raise ValueError("K8 readout accepts neither geometry nor epochs")
    root = Path(args.output_root or config["runtime"]["output_root"]).resolve()
    payload, preflight_lock_path = load_preflight(root, args.config.resolve())
    torch.set_num_threads(int(config["runtime"]["num_threads"]))
    torch.set_float32_matmul_precision("high")
    device = torch.device(config["runtime"]["device"] if torch.cuda.is_available() else "cpu")
    images = np.load(Path(config["dataset"]["cache_path"]), mmap_mode="r")
    raw_latents = base.load_latents(config)
    subset, subset_audit = dynamic.build_subset(config)
    latents, _, _ = v17.normalize_with_k80_rgb_statistics(
        raw_latents, int(config["split"]["train_end"]), subset
    )
    anchor_maps, _ = tube.v2.build_anchor_maps(config)
    start = int(config["split"]["train_end"])
    end = int(config["split"]["validation_end"])
    if (start, end, int(config["split"]["test_end"])) != (8000, 9000, 10000):
        raise ValueError("K8 dataset split changed")
    runs: dict[str, Any] = {}
    locks: dict[str, str] = {}
    for name in GEOMETRIES:
        model, locks[name] = load_model(
            name,
            root,
            args.config.resolve(),
            config,
            payload,
            preflight_lock_path,
            device,
        )
        calibration_prediction = k6.gauge.predict_selected(
            model, images, latents, subset, int(config["training"]["batch_size"]), device
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
        semantic = k6.semantic_bundle(
            calibration_prediction,
            validation_prediction,
            latents[0, subset, :3],
            latents[0, start:end, :3],
            latents[0, start:end],
            list(config["model"]["learned_indices"]),
            list(config["model"]["anchor_indices"]),
            float(config["evaluation"]["full_affine_ridge"]),
        )
        runs[name] = {
            "semantic": semantic,
            "graph": base.graph_metrics(
                model.parametric_part.A.detach().cpu().numpy(),
                float(config["evaluation"]["graph_threshold"]),
            ),
            "estimated_A": model.parametric_part.A.detach().cpu().numpy(),
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
                model,
                images,
                latents,
                start,
                end,
                int(config["training"]["batch_size"]),
                device,
            ),
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    comparisons = {
        name: comparison(runs["full_anisotropic"], runs[name])
        for name in ("diagonal", "isotropic")
    }
    result = {
        "protocol_version": PROTOCOL,
        "mode": "three_geometry_single_subset_validation_only_readout",
        "subset_seed": int(config["calibration"]["subset_seed"]),
        "config_sha256": base.sha256_file(args.config.resolve()),
        "source_files_sha256": source_hashes(),
        "preflight_lock_sha256": base.sha256_file(preflight_lock_path),
        "subset": subset_audit,
        "geometry_sha256": payload["geometry_sha256"],
        "geometry_audits": payload["geometry_audits"],
        "training_locks": locks,
        "runs": runs,
        "comparisons": comparisons,
        "decision": {
            "verdict": (
                "axis_separable_semantic_advantage_both_candidates_subset"
                if all(
                    comparisons[name]["complete_semantic_advantage"]
                    for name in ("diagonal", "isotropic")
                )
                else "axis_separable_subset_result_mixed"
            ),
            "test_evaluated": False,
        },
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
    output = root / "formal" / "subset_validation_readout.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite K8 readout: {output}")
    v11.atomic_json(output, result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("preflight", "smoke", "train", "readout"), required=True)
    parser.add_argument("--geometry", choices=GEOMETRIES)
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
