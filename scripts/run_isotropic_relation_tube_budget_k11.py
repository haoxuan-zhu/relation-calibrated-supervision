"""Validation-only budget characterization for the fixed isotropic relation tube."""

from __future__ import annotations

import argparse
import copy
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


PROTOCOL = "isotropic_relation_tube_budget_k11_v1"
BUDGETS = (20, 40, 80, 160)
TRAIN_BUDGETS = (20, 40, 160)
SEEDS = (0, 42, 3407)
SOURCE_FILES = tuple(
    dict.fromkeys(
        (
            "run_isotropic_relation_tube_budget_k11.py",
            "audit_relation_tube_geometry_downstream_k6.py",
            "run_relation_tube_geometry_ablation_k4.py",
            *tube.SOURCE_FILES,
        )
    )
)


def keyed(mapping: dict[Any, Any], key: int) -> Any:
    if key in mapping:
        return mapping[key]
    if str(key) in mapping:
        return mapping[str(key)]
    raise KeyError(key)


def source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    return {name: base.sha256_file(root / name) for name in SOURCE_FILES}


def validate_master(config: dict[str, Any]) -> None:
    if config["protocol_version"] != PROTOCOL:
        raise ValueError("unexpected K11 protocol")
    curve = config["budget_curve"]
    if tuple(int(value) for value in curve["budgets"]) != BUDGETS:
        raise ValueError("K11 budget registry changed")
    if tuple(int(value) for value in curve["train_budgets"]) != TRAIN_BUDGETS:
        raise ValueError("K11 train-budget registry changed")
    if tuple(int(value) for value in curve["seeds"]) != SEEDS:
        raise ValueError("K11 seed registry changed")
    if int(curve["subset_seed"]) != 20260731:
        raise ValueError("K11 subset seed changed")
    if not np.isclose(float(config["calibration"]["coverage"]), 0.95):
        raise ValueError("K11 coverage changed")
    if not np.isclose(float(config["calibration"]["raw_ridge_alpha"]), 0.1):
        raise ValueError("K11 center ridge changed")
    if not np.isclose(
        float(config["calibration"]["covariance_diagonal_ridge"]), 1e-6
    ):
        raise ValueError("K11 second-moment ridge changed")
    if int(config["training"]["epochs"]) != 100:
        raise ValueError("K11 formal epoch contract changed")
    if int(config["training"]["shuffle_seed"]) != 20260730:
        raise ValueError("K11 training order changed")
    if int(config["implementation"]["expected_parameter_count"]) != 16902384:
        raise ValueError("K11 parameter-count contract changed")
    if config["audit"]["test_evaluated"] is not False:
        raise ValueError("K11 test must remain closed")
    for budget in BUDGETS:
        expected_index = min(
            budget,
            math.ceil((budget + 1) * float(config["calibration"]["coverage"])),
        )
        if expected_index != int(
            keyed(curve["expected_score_order_index_one_based"], budget)
        ):
            raise ValueError(f"K11 score index changed at K{budget}")
        expected_coverage = expected_index / budget
        if not np.isclose(
            expected_coverage,
            float(keyed(curve["expected_empirical_coverage"], budget)),
        ):
            raise ValueError(f"K11 empirical coverage changed at K{budget}")


def build_run_config(master: dict[str, Any], budget: int, seed: int) -> dict[str, Any]:
    validate_master(master)
    if budget not in BUDGETS or seed not in SEEDS:
        raise ValueError("unregistered K11 budget or seed")
    config = copy.deepcopy(master)
    curve = master["budget_curve"]
    config["initialization"] = {
        "mode": "same_seed_random_no_warm",
        "seed": seed,
        "expected_state_dict_sha256": keyed(
            master["initialization"]["expected_state_dict_sha256"], seed
        ),
    }
    config["calibration"] = {
        **master["calibration"],
        "budget": budget,
        "subset_seed": int(curve["subset_seed"]),
        "subset_sha256": keyed(curve["subset_sha256"], budget),
        "expected_score_order_index_one_based": int(
            keyed(curve["expected_score_order_index_one_based"], budget)
        ),
    }
    config["training"]["seed"] = seed
    config["selected_run"] = {"budget": budget, "seed": seed}
    return config


def build_isotropic_geometry(
    residuals: np.ndarray, coverage: float, diagonal_ridge: float
) -> tuple[dict[str, Any], dict[str, Any]]:
    values = np.asarray(residuals, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("K11 residuals must be K-by-3")
    second_moment = values.T @ values / len(values)
    second_moment += diagonal_ridge * np.eye(3, dtype=np.float64)
    isotropic_second_moment = np.eye(3, dtype=np.float64) * (
        np.trace(second_moment) / 3.0
    )
    return k4.calibrated_geometry(
        values,
        isotropic_second_moment,
        coverage,
        diagonal_ridge,
        "isotropic",
    )


def build_isotropic_preflight(
    config: dict[str, Any], images: np.ndarray, raw_latents: np.ndarray
) -> dict[str, Any]:
    subset, subset_audit = dynamic.build_subset(config)
    budget = int(config["calibration"]["budget"])
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
    ridge = float(config["calibration"]["covariance_diagonal_ridge"])
    geometry, geometry_audit = build_isotropic_geometry(
        residuals,
        float(config["calibration"]["coverage"]),
        ridge,
    )
    expected_coverage = float(
        keyed(
            config["budget_curve"]["expected_empirical_coverage"],
            budget,
        )
    )
    checks = {
        "subset_hash_exact": subset_audit["subset_sha256"]
        == config["calibration"]["subset_sha256"],
        "subset_budget_exact": subset_audit["unique_rows"] == budget,
        "teacher_finite": bool(teacher_audit["finite"]),
        "loo_complete": int(teacher_audit["loo_count"]) == budget,
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
                expected_coverage,
                atol=1e-12,
                rtol=0.0,
            )
        ),
        "test_not_read": True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"K11 preflight failed: {checks}")
    return {
        "subset": subset_audit,
        "normalization": {"latent_mean": latent_mean, "latent_std": latent_std},
        "teachers": {
            "correct": {"model": tube.teacher_json(teacher), "audit": teacher_audit}
        },
        "geometry": geometry,
        "geometry_audit": geometry_audit,
        "geometry_sha256": dynamic.canonical_sha256(geometry),
        "checks": checks,
    }


def budget_root(master: dict[str, Any], output_root: Path | None, budget: int) -> Path:
    root = Path(output_root or master["runtime"]["output_root"]).resolve()
    return root / f"k{budget}"


def preflight_main(args: argparse.Namespace, master: dict[str, Any]) -> None:
    if args.budget not in TRAIN_BUDGETS or args.seed is not None:
        raise ValueError("K11 preflight requires a new train budget and no seed")
    config = build_run_config(master, args.budget, 3407)
    root = budget_root(master, args.output_root, args.budget)
    output_dir = root / "preflight"
    output_dir.mkdir(parents=True, exist_ok=True)
    payload_path = output_dir / "isotropic_preflight.json"
    lock_path = output_dir / "preflight_lock.json"
    if payload_path.exists() or lock_path.exists():
        raise FileExistsError(f"refusing to overwrite K11 preflight K{args.budget}")
    images = np.load(Path(config["dataset"]["cache_path"]), mmap_mode="r")
    raw_latents = base.load_latents(config).astype(np.float64)
    payload = build_isotropic_preflight(config, images, raw_latents)
    payload.update(
        protocol_version=PROTOCOL,
        budget=args.budget,
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
            "status": "locked_k11_isotropic_budget_preflight",
            "protocol_version": PROTOCOL,
            "budget": args.budget,
            "master_config_sha256": base.sha256_file(args.config.resolve()),
            "source_files_sha256": source_hashes(),
            "payload_sha256": base.sha256_file(payload_path),
            "semantic_validation_evaluated": False,
            "test_evaluated": False,
        },
    )


def load_preflight(
    master: dict[str, Any], config_path: Path, output_root: Path | None, budget: int
) -> tuple[dict[str, Any], Path]:
    root = budget_root(master, output_root, budget)
    payload_path = root / "preflight" / "isotropic_preflight.json"
    lock_path = root / "preflight" / "preflight_lock.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    checks = {
        "status": lock["status"] == "locked_k11_isotropic_budget_preflight",
        "protocol": lock["protocol_version"] == PROTOCOL,
        "budget": int(lock["budget"]) == budget == int(payload["budget"]),
        "config": lock["master_config_sha256"] == base.sha256_file(config_path),
        "source": lock["source_files_sha256"] == source_hashes(),
        "payload": lock["payload_sha256"] == base.sha256_file(payload_path),
        "payload_valid": all(payload["checks"].values()),
        "test_unread": lock["test_evaluated"] is False,
    }
    if not all(checks.values()):
        raise ValueError(f"invalid K11 preflight K{budget}: {checks}")
    return payload, lock_path


def tube_preflight(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "teachers": payload["teachers"],
        "normalization": payload["normalization"],
        "geometry": payload["geometry"],
    }


def train_main(args: argparse.Namespace, master: dict[str, Any]) -> None:
    if args.budget not in TRAIN_BUDGETS or args.seed not in SEEDS:
        raise ValueError("K11 train requires a registered new budget and seed")
    smoke = args.mode == "smoke"
    epochs = int(args.epochs or (2 if smoke else 100))
    if (smoke and epochs != 2) or (not smoke and epochs != 100):
        raise ValueError("K11 smoke/formal epoch contract changed")
    config = build_run_config(master, args.budget, args.seed)
    root = budget_root(master, args.output_root, args.budget)
    payload, preflight_lock_path = load_preflight(
        master, args.config.resolve(), args.output_root, args.budget
    )
    output_dir = root / f"seed{args.seed}" / ("smoke" if smoke else "formal")
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / ("smoke_results.json" if smoke else "training_results.json")
    lock_path = output_dir / "training_lock.json"
    if result_path.exists() or lock_path.exists():
        raise FileExistsError(f"refusing to overwrite K11 run: {output_dir}")
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
        raise ValueError("K11 latent mean mismatch")
    if not np.allclose(latent_std, payload["normalization"]["latent_std"]):
        raise ValueError("K11 latent std mismatch")
    anchor_maps, anchor_audit = tube.v2.build_anchor_maps(config)
    initial_state, initial_hash = tube.v16.make_initial_state(config, device)
    if initial_hash != config["initialization"]["expected_state_dict_sha256"]:
        raise ValueError("K11 initial state mismatch")
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
        raise RuntimeError(f"K11 training validity failed: {checks}")
    lock = {
        "status": "locked_before_k11_budget_validation_readout",
        "mode": "smoke" if smoke else "formal",
        "protocol_version": PROTOCOL,
        "budget": args.budget,
        "seed": args.seed,
        "master_config_sha256": base.sha256_file(args.config.resolve()),
        "source_files_sha256": source_hashes(),
        "preflight_lock_sha256": base.sha256_file(preflight_lock_path),
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
            "budget": args.budget,
            "seed": args.seed,
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
    budget: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.nn.Module, str]:
    root = budget_root(master, output_root, budget)
    lock_path = root / f"seed{seed}" / "formal" / "training_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    checks = {
        "status": lock["status"] == "locked_before_k11_budget_validation_readout",
        "formal": lock["mode"] == "formal",
        "protocol": lock["protocol_version"] == PROTOCOL,
        "budget": int(lock["budget"]) == budget,
        "seed": int(lock["seed"]) == seed,
        "config": lock["master_config_sha256"] == base.sha256_file(config_path),
        "source": lock["source_files_sha256"] == source_hashes(),
        "preflight": lock["preflight_lock_sha256"] == base.sha256_file(preflight_lock_path),
        "geometry": lock["geometry_sha256"] == payload["geometry_sha256"],
        "validity": all(lock["validity"].values()),
        "test_unread": lock["test_evaluated"] is False,
    }
    if not all(checks.values()):
        raise ValueError(f"invalid K11 model lock K{budget} seed{seed}: {checks}")
    checkpoint_path = Path(lock["checkpoint"]["path"])
    if base.sha256_file(checkpoint_path) != lock["checkpoint"]["sha256"]:
        raise ValueError("K11 checkpoint hash mismatch")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if (
        checkpoint["protocol_version"] != PROTOCOL
        or checkpoint["condition"] != "tube_correct"
        or int(checkpoint["epoch"]) != 100
        or checkpoint["initial_state_sha256"]
        != config["initialization"]["expected_state_dict_sha256"]
    ):
        raise ValueError("K11 checkpoint identity mismatch")
    model = tube.model_for_condition("tube_correct", config, tube_preflight(payload), device)
    model.load_state_dict(checkpoint["state_dict"])
    return model, base.sha256_file(lock_path)


def readout_main(args: argparse.Namespace, master: dict[str, Any]) -> None:
    if args.budget not in TRAIN_BUDGETS or args.seed not in SEEDS:
        raise ValueError("K11 readout requires a registered new budget and seed")
    config = build_run_config(master, args.budget, args.seed)
    root = budget_root(master, args.output_root, args.budget)
    payload, preflight_lock_path = load_preflight(
        master, args.config.resolve(), args.output_root, args.budget
    )
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
    model, training_lock_sha256 = load_model(
        master,
        args.config.resolve(),
        args.output_root,
        config,
        payload,
        preflight_lock_path,
        args.budget,
        args.seed,
        device,
    )
    start = int(config["split"]["train_end"])
    end = int(config["split"]["validation_end"])
    if (start, end, int(config["split"]["test_end"])) != (8000, 9000, 10000):
        raise ValueError("K11 dataset split changed")
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
    result = {
        "protocol_version": PROTOCOL,
        "mode": "single_budget_seed_validation_only_readout",
        "budget": args.budget,
        "seed": args.seed,
        "master_config_sha256": base.sha256_file(args.config.resolve()),
        "source_files_sha256": source_hashes(),
        "preflight_lock_sha256": base.sha256_file(preflight_lock_path),
        "training_lock_sha256": training_lock_sha256,
        "subset": subset_audit,
        "geometry_sha256": payload["geometry_sha256"],
        "geometry_audit": payload["geometry_audit"],
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
    output_path = root / f"seed{args.seed}" / "formal" / "validation_readout.json"
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite K11 readout: {output_path}")
    v11.atomic_json(output_path, result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("preflight", "smoke", "train", "readout"), required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    master = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    validate_master(master)
    if args.mode == "preflight":
        preflight_main(args, master)
    elif args.mode in ("smoke", "train"):
        train_main(args, master)
    else:
        readout_main(args, master)


if __name__ == "__main__":
    main()
