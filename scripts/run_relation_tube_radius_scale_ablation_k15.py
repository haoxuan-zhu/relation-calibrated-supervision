"""Train and evaluate K80 isotropic relation tubes under frozen radius alternatives."""

from __future__ import annotations

import argparse
import copy
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

import audit_calibrated_relation_tube_gauge_k1 as gauge
import audit_relation_tube_geometry_downstream_k6 as k6
import run_calibrated_relation_tube_k0 as tube
import run_conflict_projected_physics_k80_closed as v17
import run_diagnostic as base
import run_dynamic_residual_propagation_k0 as dynamic
import run_physics_functional_anchor_training as v11
import run_relation_tube_geometry_ablation_k4 as k4
import run_relation_tube_geometry_tradeoff_k5 as k5
import run_supervised_continuation_diagnostic as v6


PROTOCOL = "relation_tube_radius_scale_ablation_k15_v1"
K4_PROTOCOL = "relation_tube_geometry_ablation_k4_v1"
K5_PROTOCOL = "relation_tube_geometry_tradeoff_k5_v1"
SEEDS = (3407, 0, 42)
TRAIN_MODES = ("half_empirical", "fixed_unit", "double_empirical")
ALL_MODES = (
    "center_only",
    "half_empirical",
    "empirical_rank",
    "fixed_unit",
    "double_empirical",
    "unbounded",
)
SOURCE_FILES = tuple(
    dict.fromkeys(
        (
            "run_relation_tube_radius_scale_ablation_k15.py",
            "run_relation_tube_geometry_ablation_k4.py",
            "run_relation_tube_geometry_tradeoff_k5.py",
            "audit_relation_tube_geometry_downstream_k6.py",
            *tube.SOURCE_FILES,
        )
    )
)


def source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    return {name: base.sha256_file(root / name) for name in SOURCE_FILES}


def keyed(mapping: dict[Any, Any], key: int) -> Any:
    if key in mapping:
        return mapping[key]
    return mapping[str(key)]


def validate_master(config: dict[str, Any]) -> None:
    if config["protocol_version"] != PROTOCOL:
        raise ValueError("unexpected K15 protocol")
    if int(config["scope"]["budget"]) != 80:
        raise ValueError("K15 budget changed")
    if tuple(int(value) for value in config["scope"]["seeds"]) != SEEDS:
        raise ValueError("K15 seeds changed")
    if tuple(int(value) for value in config["scope"]["validation_rows"]) != (8000, 9000):
        raise ValueError("K15 validation split changed")
    if config["scope"]["test_evaluated"] is not False:
        raise ValueError("K15 test must remain closed")
    if tuple(config["radius_modes"]["train_modes"]) != TRAIN_MODES:
        raise ValueError("K15 radius modes changed")
    multipliers = config["radius_modes"]["empirical_multipliers"]
    if not np.isclose(float(multipliers["half_empirical"]), 0.5):
        raise ValueError("K15 half radius changed")
    if not np.isclose(float(multipliers["double_empirical"]), 2.0):
        raise ValueError("K15 double radius changed")
    if not np.isclose(float(config["radius_modes"]["fixed_unit_euclidean_radius"]), 1.0):
        raise ValueError("K15 fixed radius changed")
    if int(config["training"]["formal_epochs"]) != 100 or int(config["training"]["smoke_epochs"]) != 2:
        raise ValueError("K15 epoch contract changed")
    if not np.isclose(float(config["evaluation"]["full_affine_ridge"]), 0.001):
        raise ValueError("K15 readout ridge changed")
    if not np.isclose(float(config["evaluation"]["fixed_unit_equivalence_r2"]), 0.01):
        raise ValueError("K15 equivalence tolerance changed")
    for seed in SEEDS:
        entry = keyed(config["runs"], seed)
        expected = K4_PROTOCOL if seed == 3407 else K5_PROTOCOL
        if entry["geometry_protocol"] != expected:
            raise ValueError(f"K15 parent protocol changed for seed {seed}")


def configure_parent(protocol: str) -> Any:
    if protocol == K4_PROTOCOL:
        return k4
    if protocol == K5_PROTOCOL:
        k5.configure_k4_engine()
        return k4
    raise ValueError(f"unsupported K15 parent protocol: {protocol}")


def load_parent(
    master: dict[str, Any], seed: int
) -> tuple[dict[str, Any], dict[str, Any], Path, Path, Any]:
    entry = keyed(master["runs"], seed)
    config_path = Path(entry["geometry_config_path"]).resolve()
    if base.sha256_file(config_path) != entry["geometry_config_sha256"]:
        raise ValueError(f"K15 parent config hash mismatch for seed {seed}")
    parent = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    engine = configure_parent(entry["geometry_protocol"])
    engine.validate_config(parent)
    root = Path(parent["runtime"]["output_root"]).resolve()
    payload, lock_path = engine.load_preflight(root, config_path, parent)
    return parent, payload, lock_path, config_path, engine


def scaled_geometry(
    empirical: dict[str, Any], mode: str, master: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    if mode not in TRAIN_MODES:
        raise ValueError(mode)
    geometry = copy.deepcopy(empirical)
    precision = np.asarray(geometry["precision"], dtype=np.float64)
    diagonal = np.diag(precision)
    if not np.allclose(precision, np.eye(3) * diagonal[0], atol=1e-12, rtol=0.0):
        raise ValueError("K15 parent geometry is not isotropic")
    empirical_radius = float(np.sqrt(float(geometry["radius_squared"]) / diagonal[0]))
    if mode == "fixed_unit":
        effective_radius = float(master["radius_modes"]["fixed_unit_euclidean_radius"])
    else:
        multiplier = float(master["radius_modes"]["empirical_multipliers"][mode])
        effective_radius = empirical_radius * multiplier
    geometry["radius_squared"] = float(diagonal[0] * effective_radius**2)
    audit = {
        "mode": mode,
        "empirical_effective_euclidean_radius": empirical_radius,
        "effective_euclidean_radius": effective_radius,
        "radius_squared": geometry["radius_squared"],
        "precision_scalar": float(diagonal[0]),
        "finite": bool(np.isfinite(effective_radius) and effective_radius > 0.0),
    }
    return geometry, audit


def output_root(master: dict[str, Any], override: Path | None) -> Path:
    return Path(override or master["runtime"]["output_root"]).resolve()


def mode_dir(mode: str) -> str:
    return mode


def preflight_main(args: argparse.Namespace, master: dict[str, Any]) -> None:
    if args.seed not in SEEDS or args.radius_mode is not None:
        raise ValueError("K15 preflight requires one registered seed and no radius mode")
    parent, payload, parent_lock_path, parent_config_path, _ = load_parent(master, args.seed)
    root = output_root(master, args.output_root)
    directory = root / "preflight" / f"seed{args.seed}"
    directory.mkdir(parents=True, exist_ok=True)
    payload_path = directory / "scaled_radius_preflight.json"
    lock_path = directory / "preflight_lock.json"
    if payload_path.exists() or lock_path.exists():
        raise FileExistsError(f"refusing to overwrite K15 preflight seed {args.seed}")
    geometries: dict[str, Any] = {}
    audits: dict[str, Any] = {}
    hashes: dict[str, str] = {}
    for mode in TRAIN_MODES:
        geometry, audit = scaled_geometry(payload["geometries"]["isotropic"], mode, master)
        geometries[mode] = geometry
        audits[mode] = audit
        hashes[mode] = dynamic.canonical_sha256(geometry)
    checks = {
        "parent_valid": all(payload["checks"].values()),
        "parent_isotropic_registered": "isotropic" in payload["geometries"],
        "all_scaled_geometry_finite": all(audits[mode]["finite"] for mode in TRAIN_MODES),
        "half_exact": bool(
            np.isclose(
                audits["half_empirical"]["effective_euclidean_radius"],
                0.5 * audits["half_empirical"]["empirical_effective_euclidean_radius"],
            )
        ),
        "fixed_unit_exact": bool(
            np.isclose(audits["fixed_unit"]["effective_euclidean_radius"], 1.0)
        ),
        "double_exact": bool(
            np.isclose(
                audits["double_empirical"]["effective_euclidean_radius"],
                2.0 * audits["double_empirical"]["empirical_effective_euclidean_radius"],
            )
        ),
        "test_not_read": True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"K15 preflight failed seed {args.seed}: {checks}")
    result = {
        "protocol_version": PROTOCOL,
        "seed": args.seed,
        "master_config_sha256": base.sha256_file(args.config.resolve()),
        "source_files_sha256": source_hashes(),
        "parent_config_path": str(parent_config_path),
        "parent_config_sha256": base.sha256_file(parent_config_path),
        "parent_preflight_lock_sha256": base.sha256_file(parent_lock_path),
        "normalization": payload["normalization"],
        "teachers": payload["teachers"],
        "geometries": geometries,
        "geometry_audits": audits,
        "geometry_sha256": hashes,
        "checks": checks,
        "semantic_validation_evaluated": False,
        "test_evaluated": False,
    }
    v11.atomic_json(payload_path, result)
    v11.atomic_json(
        lock_path,
        {
            "status": "locked_k15_radius_preflight",
            "protocol_version": PROTOCOL,
            "seed": args.seed,
            "master_config_sha256": base.sha256_file(args.config.resolve()),
            "source_files_sha256": source_hashes(),
            "payload_sha256": base.sha256_file(payload_path),
            "parent_preflight_lock_sha256": base.sha256_file(parent_lock_path),
            "test_evaluated": False,
        },
    )


def load_preflight(
    master: dict[str, Any], config_path: Path, override: Path | None, seed: int
) -> tuple[dict[str, Any], Path]:
    directory = output_root(master, override) / "preflight" / f"seed{seed}"
    payload_path = directory / "scaled_radius_preflight.json"
    lock_path = directory / "preflight_lock.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    checks = {
        "status": lock["status"] == "locked_k15_radius_preflight",
        "protocol": lock["protocol_version"] == PROTOCOL,
        "seed": int(lock["seed"]) == seed == int(payload["seed"]),
        "config": lock["master_config_sha256"] == base.sha256_file(config_path),
        "source": lock["source_files_sha256"] == source_hashes(),
        "payload": lock["payload_sha256"] == base.sha256_file(payload_path),
        "payload_valid": all(payload["checks"].values()),
        "test_unread": lock["test_evaluated"] is False,
    }
    if not all(checks.values()):
        raise ValueError(f"invalid K15 preflight seed {seed}: {checks}")
    return payload, lock_path


def run_config(parent: dict[str, Any], master: dict[str, Any], seed: int, mode: str) -> dict[str, Any]:
    config = copy.deepcopy(parent)
    config["protocol_version"] = PROTOCOL
    config["preregistration"] = master["preregistration"]
    config["selected_radius_run"] = {"seed": seed, "mode": mode}
    config["training"]["conditions"] = ["tube_correct"]
    config["training"]["epochs"] = int(master["training"]["formal_epochs"])
    config["runtime"]["output_root"] = str(output_root(master, None))
    return config


def tube_preflight(payload: dict[str, Any], mode: str) -> dict[str, Any]:
    return {
        "teachers": payload["teachers"],
        "normalization": payload["normalization"],
        "geometry": payload["geometries"][mode],
    }


def run_directory(root: Path, seed: int, mode: str, formal: bool) -> Path:
    return root / f"seed{seed}" / mode_dir(mode) / ("formal" if formal else "smoke")


def train_main(args: argparse.Namespace, master: dict[str, Any]) -> None:
    if args.seed not in SEEDS or args.radius_mode not in TRAIN_MODES:
        raise ValueError("K15 train requires a registered seed and radius mode")
    formal = args.mode == "train"
    epochs = int(master["training"]["formal_epochs"] if formal else master["training"]["smoke_epochs"])
    if not formal and args.seed != 3407:
        raise ValueError("K15 smoke is restricted to seed3407")
    parent, _, _, _, _ = load_parent(master, args.seed)
    payload, preflight_lock_path = load_preflight(
        master, args.config.resolve(), args.output_root, args.seed
    )
    config = run_config(parent, master, args.seed, args.radius_mode)
    root = output_root(master, args.output_root)
    directory = run_directory(root, args.seed, args.radius_mode, formal)
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory / ("training_results.json" if formal else "smoke_results.json")
    lock_path = directory / "training_lock.json"
    if result_path.exists() or lock_path.exists():
        raise FileExistsError(f"refusing to overwrite K15 run: {directory}")
    torch.set_num_threads(int(master["runtime"]["num_threads"]))
    torch.set_float32_matmul_precision("high")
    device = torch.device(master["runtime"]["device"] if torch.cuda.is_available() else "cpu")
    images = np.load(Path(config["dataset"]["cache_path"]), mmap_mode="r")
    raw_latents = base.load_latents(config)
    subset, _ = dynamic.build_subset(config)
    latents, latent_mean, latent_std = v17.normalize_with_k80_rgb_statistics(
        raw_latents, int(config["split"]["train_end"]), subset
    )
    if not np.allclose(latent_mean, payload["normalization"]["latent_mean"]):
        raise ValueError("K15 latent mean mismatch")
    if not np.allclose(latent_std, payload["normalization"]["latent_std"]):
        raise ValueError("K15 latent std mismatch")
    anchor_maps, anchor_audit = tube.v2.build_anchor_maps(config)
    initial_state, initial_hash = tube.v16.make_initial_state(config, device)
    if initial_hash != config["initialization"]["expected_state_dict_sha256"]:
        raise ValueError("K15 initial state mismatch")
    training = tube.train_condition(
        "tube_correct",
        config,
        tube_preflight(payload, args.radius_mode),
        images,
        latents,
        anchor_maps,
        initial_state,
        initial_hash,
        device,
        directory,
        epochs,
    )
    checks = {
        "preflight_valid": all(payload["checks"].values()),
        "initial_state_exact": initial_hash == config["initialization"]["expected_state_dict_sha256"],
        "parameter_count_exact": training["parameter_count"] == int(config["implementation"]["expected_parameter_count"]),
        "epoch_exact": training["final_epoch"] == epochs,
        "history_finite": dynamic.numeric_history_is_finite(training["history"]),
        "semantic_unread": not training["semantic_truth_read_during_training"],
        "test_unread": not training["test_evaluated"],
    }
    if not all(checks.values()):
        raise RuntimeError(f"K15 training validity failed: {checks}")
    lock = {
        "status": "locked_before_k15_radius_validation_readout",
        "mode": "formal" if formal else "smoke",
        "protocol_version": PROTOCOL,
        "seed": args.seed,
        "radius_mode": args.radius_mode,
        "master_config_sha256": base.sha256_file(args.config.resolve()),
        "source_files_sha256": source_hashes(),
        "preflight_lock_sha256": base.sha256_file(preflight_lock_path),
        "geometry_sha256": payload["geometry_sha256"][args.radius_mode],
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
            "mode": "formal_train" if formal else "smoke_train",
            "seed": args.seed,
            "radius_mode": args.radius_mode,
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
    master: dict[str, Any], config_path: Path, override: Path | None, seed: int, mode: str, device: torch.device
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any], dict[str, Any]]:
    parent, _, _, _, _ = load_parent(master, seed)
    payload, preflight_lock_path = load_preflight(master, config_path, override, seed)
    config = run_config(parent, master, seed, mode)
    directory = run_directory(output_root(master, override), seed, mode, True)
    lock_path = directory / "training_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    checks = {
        "status": lock["status"] == "locked_before_k15_radius_validation_readout",
        "formal": lock["mode"] == "formal",
        "protocol": lock["protocol_version"] == PROTOCOL,
        "seed": int(lock["seed"]) == seed,
        "mode": lock["radius_mode"] == mode,
        "config": lock["master_config_sha256"] == base.sha256_file(config_path),
        "source": lock["source_files_sha256"] == source_hashes(),
        "preflight": lock["preflight_lock_sha256"] == base.sha256_file(preflight_lock_path),
        "geometry": lock["geometry_sha256"] == payload["geometry_sha256"][mode],
        "validity": all(lock["validity"].values()),
        "test_unread": lock["test_evaluated"] is False,
    }
    if not all(checks.values()):
        raise ValueError(f"invalid K15 model lock seed {seed} mode {mode}: {checks}")
    checkpoint_path = Path(lock["checkpoint"]["path"])
    if base.sha256_file(checkpoint_path) != lock["checkpoint"]["sha256"]:
        raise ValueError("K15 checkpoint hash mismatch")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint["protocol_version"] != PROTOCOL or checkpoint["condition"] != "tube_correct" or int(checkpoint["epoch"]) != 100:
        raise ValueError("K15 checkpoint identity mismatch")
    model = tube.model_for_condition("tube_correct", config, tube_preflight(payload, mode), device)
    model.load_state_dict(checkpoint["state_dict"])
    return model, config, payload, {"lock_path": str(lock_path), "lock_sha256": base.sha256_file(lock_path), "checkpoint_sha256": lock["checkpoint"]["sha256"]}


def readout_main(args: argparse.Namespace, master: dict[str, Any]) -> None:
    if args.seed not in SEEDS or args.radius_mode not in TRAIN_MODES:
        raise ValueError("K15 readout requires a registered seed and radius mode")
    torch.set_num_threads(int(master["runtime"]["num_threads"]))
    torch.set_float32_matmul_precision("high")
    device = torch.device(master["runtime"]["device"] if torch.cuda.is_available() else "cpu")
    model, config, payload, identity = load_model(
        master, args.config.resolve(), args.output_root, args.seed, args.radius_mode, device
    )
    images = np.load(Path(config["dataset"]["cache_path"]), mmap_mode="r")
    raw_latents = base.load_latents(config)
    subset, subset_audit = dynamic.build_subset(config)
    latents, _, _ = v17.normalize_with_k80_rgb_statistics(
        raw_latents, int(config["split"]["train_end"]), subset
    )
    start, end = tuple(int(value) for value in master["scope"]["validation_rows"])
    batch_size = int(config["training"]["batch_size"])
    calibration_prediction = gauge.predict_selected(model, images, latents, subset, batch_size, device)
    validation_prediction = v6.predict_rgb(model, images, latents, start, end, batch_size, device)
    semantic = k6.semantic_bundle(
        calibration_prediction,
        validation_prediction,
        latents[0, subset, :3],
        latents[0, start:end, :3],
        latents[0, start:end],
        list(config["model"]["learned_indices"]),
        list(config["model"]["anchor_indices"]),
        float(master["evaluation"]["full_affine_ridge"]),
    )
    anchor_maps, _ = tube.v2.build_anchor_maps(config)
    result = {
        "protocol_version": PROTOCOL,
        "mode": "single_radius_validation_only_readout",
        "seed": args.seed,
        "radius_mode": args.radius_mode,
        "master_config_sha256": base.sha256_file(args.config.resolve()),
        "source_files_sha256": source_hashes(),
        "identity": identity,
        "subset": subset_audit,
        "geometry_audit": payload["geometry_audits"][args.radius_mode],
        "semantic": semantic,
        "graph": base.graph_metrics(model.parametric_part.A.detach().cpu().numpy(), float(master["evaluation"]["graph_threshold"])),
        "ccrl_validation": tube.v2.validate(model, images, latents, anchor_maps, tube.v2.Condition("tube_correct", "oracle"), config, device),
        "representation_audit": tube.representation_audit(model, images, latents, start, end, batch_size, device),
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
    directory = run_directory(output_root(master, args.output_root), args.seed, args.radius_mode, True)
    path = directory / "validation_readout.json"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite K15 readout: {path}")
    v11.atomic_json(path, result)


def compact_run(semantic: dict[str, Any], graph: dict[str, Any] | None, ccrl: dict[str, Any] | None, audit: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "raw": semantic["raw"],
        "coordinatewise_affine": semantic.get("coordinatewise_affine"),
        "full_affine": semantic["full_affine"],
        "rgb_hungarian_mcc": semantic.get("rgb_hungarian_mcc"),
        "graph": graph,
        "ccrl_validation": ccrl,
        "representation_audit": audit,
    }


def aggregate_main(args: argparse.Namespace, master: dict[str, Any]) -> None:
    root = output_root(master, args.output_root)
    k3_path = Path(master["k3_aggregate"]["path"])
    if base.sha256_file(k3_path) != master["k3_aggregate"]["sha256"]:
        raise ValueError("K15 K3 aggregate hash mismatch")
    k3 = json.loads(k3_path.read_text(encoding="utf-8"))
    if k3["test_evaluated"] is not False:
        raise ValueError("K15 K3 comparator touched test")
    results: dict[int, dict[str, Any]] = {}
    inputs: dict[int, Any] = {}
    for seed in SEEDS:
        entry = keyed(master["runs"], seed)
        empirical_path = Path(entry["empirical_readout_path"])
        if base.sha256_file(empirical_path) != entry["empirical_readout_sha256"]:
            raise ValueError(f"K15 empirical readout hash mismatch seed {seed}")
        empirical = json.loads(empirical_path.read_text(encoding="utf-8"))
        if empirical["test_evaluated"] is not False:
            raise ValueError("K15 empirical readout touched test")
        seed_runs: dict[str, Any] = {}
        legacy = keyed(k3["seeds"], seed)["runs"]
        seed_runs["center_only"] = compact_run(legacy["center_only_correct"], None, None, None)
        seed_runs["unbounded"] = compact_run(legacy["unbounded_correct"], None, None, None)
        empirical_run = empirical["runs"]["isotropic"]
        seed_runs["empirical_rank"] = compact_run(
            empirical_run["semantic"], empirical_run["graph"], empirical_run["ccrl_validation"], None
        )
        new_inputs: dict[str, Any] = {}
        for mode in TRAIN_MODES:
            path = run_directory(root, seed, mode, True) / "validation_readout.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            checks = {
                "protocol": payload["protocol_version"] == PROTOCOL,
                "seed": int(payload["seed"]) == seed,
                "mode": payload["radius_mode"] == mode,
                "config": payload["master_config_sha256"] == base.sha256_file(args.config.resolve()),
                "source": payload["source_files_sha256"] == source_hashes(),
                "validation": payload["semantic_validation_evaluated"] is True,
                "test_unread": payload["test_evaluated"] is False,
            }
            if not all(checks.values()):
                raise ValueError(f"invalid K15 readout seed {seed} mode {mode}: {checks}")
            seed_runs[mode] = compact_run(
                payload["semantic"], payload["graph"], payload["ccrl_validation"], payload["representation_audit"]
            )
            new_inputs[mode] = {"path": str(path), "sha256": base.sha256_file(path)}
        results[seed] = seed_runs
        inputs[seed] = {
            "empirical": {"path": str(empirical_path), "sha256": entry["empirical_readout_sha256"]},
            "new": new_inputs,
        }
    comparisons: dict[str, Any] = {}
    alternatives = ("half_empirical", "fixed_unit", "double_empirical")
    for alternative in alternatives:
        deltas = {
            seed: float(results[seed]["empirical_rank"]["full_affine"]["mean_r2"] - results[seed][alternative]["full_affine"]["mean_r2"])
            for seed in SEEDS
        }
        comparisons[alternative] = {
            "empirical_minus_alternative_full_affine_r2": deltas,
            "positive_seeds": int(sum(value > 0.0 for value in deltas.values())),
            "mean_delta": float(np.mean(list(deltas.values()))),
        }
    summaries: dict[str, Any] = {}
    for mode in ALL_MODES:
        summaries[mode] = {
            "full_affine_r2_mean": float(np.mean([results[seed][mode]["full_affine"]["mean_r2"] for seed in SEEDS])),
            "raw_correlation_mean": float(np.mean([results[seed][mode]["raw"]["mean_direct_abs_correlation"] for seed in SEEDS])),
            "raw_r2_mean": float(np.mean([results[seed][mode]["raw"]["mean_r2"] for seed in SEEDS])),
        }
    preferred = int(master["decision"]["preferred_seed_count"])
    fixed_tolerance = float(master["evaluation"]["fixed_unit_equivalence_r2"])
    fixed_equivalent = all(
        abs(comparisons["fixed_unit"]["empirical_minus_alternative_full_affine_r2"][seed]) <= fixed_tolerance
        for seed in SEEDS
    )
    all_three = all(comparisons[name]["positive_seeds"] == 3 for name in alternatives)
    empirical_mean_best = all(
        summaries["empirical_rank"]["full_affine_r2_mean"] > summaries[name]["full_affine_r2_mean"]
        for name in alternatives
    )
    half_double_preferred = all(
        comparisons[name]["positive_seeds"] >= preferred
        and comparisons[name]["mean_delta"] > 0.0
        for name in ("half_empirical", "double_empirical")
    )
    majority_all = all(
        comparisons[name]["positive_seeds"] >= preferred and comparisons[name]["mean_delta"] > 0.0
        for name in alternatives
    )
    alternative_preferred = any(
        comparisons[name]["positive_seeds"] <= 1 and comparisons[name]["mean_delta"] < 0.0
        for name in alternatives
    )
    if all_three:
        verdict = "empirical_rank_radius_preferred_three_seed"
    elif half_double_preferred and fixed_equivalent:
        verdict = "empirical_scale_preferred_fixed_unit_equivalent"
    elif majority_all and empirical_mean_best:
        verdict = "empirical_rank_radius_preferred_majority"
    elif alternative_preferred:
        verdict = "alternative_radius_preferred"
    else:
        verdict = "radius_scale_sensitivity_mixed"
    output = {
        "protocol_version": PROTOCOL,
        "mode": "three_seed_k80_radius_scale_validation_aggregate",
        "master_config_sha256": base.sha256_file(args.config.resolve()),
        "source_files_sha256": source_hashes(),
        "inputs": inputs,
        "results": results,
        "comparisons": comparisons,
        "summaries": summaries,
        "decision": {
            "verdict": verdict,
            "fixed_unit_equivalent_within_r2_0p01_all_seeds": fixed_equivalent,
            "empirical_mean_best": empirical_mean_best,
            "test_evaluated": False,
        },
        "training_performed_for_new_modes": True,
        "semantic_validation_evaluated": True,
        "test_evaluated": False,
    }
    path = root / "radius_scale_aggregate.json"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite K15 aggregate: {path}")
    v11.atomic_json(path, output)
    print(json.dumps({"result_path": str(path), "sha256": base.sha256_file(path), "verdict": verdict}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("preflight", "smoke", "train", "readout", "aggregate", "validate"), required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--radius-mode", choices=TRAIN_MODES)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    master = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    validate_master(master)
    if args.mode == "validate":
        print(json.dumps({"protocol_version": PROTOCOL, "config_valid": True}, sort_keys=True))
    elif args.mode == "preflight":
        preflight_main(args, master)
    elif args.mode in ("smoke", "train"):
        train_main(args, master)
    elif args.mode == "readout":
        readout_main(args, master)
    else:
        aggregate_main(args, master)


if __name__ == "__main__":
    main()
