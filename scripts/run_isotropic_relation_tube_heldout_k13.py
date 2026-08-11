"""One-time held-out readout for the locked isotropic relation tube."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

import audit_calibrated_relation_tube_gauge_k1 as gauge
import audit_relation_tube_geometry_downstream_k6 as k6
import extract_heldout_baseline_comparator_k13 as baseline_extract
import run_conflict_projected_physics_k80_closed as v17
import run_diagnostic as base
import run_dynamic_residual_propagation_k0 as dynamic
import run_isotropic_relation_tube_budget_k11 as k11
import run_physics_functional_anchor_training as v11
import run_supervised_continuation_diagnostic as v6


PROTOCOL = "isotropic_relation_tube_heldout_k13_v1"
SOURCE_FILES = (
    "run_isotropic_relation_tube_heldout_k13.py",
    "extract_heldout_baseline_comparator_k13.py",
    "run_isotropic_relation_tube_budget_k11.py",
    "audit_relation_tube_geometry_downstream_k6.py",
    "audit_calibrated_relation_tube_gauge_k1.py",
    "run_relation_tube_geometry_ablation_k4.py",
    "run_calibrated_relation_tube_k0.py",
    "run_dynamic_residual_propagation_k0.py",
    "run_conflict_projected_physics_k80_closed.py",
    "run_supervised_continuation_diagnostic.py",
    "run_diagnostic.py",
)


def keyed(mapping: dict[Any, Any], key: int) -> Any:
    return mapping[key] if key in mapping else mapping[str(key)]


def source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    return {name: base.sha256_file(root / name) for name in SOURCE_FILES}


def validate_master(master: dict[str, Any]) -> None:
    if master["protocol_version"] != PROTOCOL:
        raise ValueError("unexpected K13 protocol")
    if tuple(master["evaluation"]["budgets"]) != k11.BUDGETS:
        raise ValueError("K13 budget registry changed")
    if tuple(master["evaluation"]["seeds"]) != k11.SEEDS:
        raise ValueError("K13 seed registry changed")
    if (
        int(master["evaluation"]["test_start"]),
        int(master["evaluation"]["test_end"]),
    ) != (9000, 10000):
        raise ValueError("K13 test rows changed")
    if master["evaluation"]["candidate_geometry"] != "isotropic":
        raise ValueError("K13 geometry changed")
    if master["evaluation"]["test_guided_selection_allowed"] is not False:
        raise ValueError("K13 cannot allow test-guided selection")
    if master["audit"]["test_evaluation_authorized"] is not True:
        raise ValueError("K13 test readout is not authorized")
    if master["audit"]["training_allowed"] is not False:
        raise ValueError("K13 must not train")
    for item in (master["k11"], master["baseline_comparator"]):
        path = Path(item.get("config_path", item.get("path", item.get("validation_aggregate_path"))))
        expected = item.get("config_sha256", item.get("sha256", item.get("validation_aggregate_sha256")))
        if base.sha256_file(path) != expected:
            raise ValueError(f"K13 frozen input hash mismatch: {path}")
    aggregate_path = Path(master["k11"]["validation_aggregate_path"])
    if base.sha256_file(aggregate_path) != master["k11"]["validation_aggregate_sha256"]:
        raise ValueError("K13 K11 aggregate hash mismatch")
    validation = json.loads(aggregate_path.read_text(encoding="utf-8"))
    if (
        validation["protocol_version"]
        != "isotropic_relation_tube_budget_k11_aggregate_v1"
        or validation["test_evaluated"] is not False
        or validation["decision"]["geometry_selection_reopened"] is not False
    ):
        raise ValueError("K13 invalid K11 validation aggregate")
    baseline_path = Path(master["baseline_comparator"]["path"])
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    if (
        baseline["protocol_version"] != baseline_extract.PROTOCOL
        or baseline["candidate_test_evaluated"] is not False
        or tuple(baseline["budgets"]) != k11.BUDGETS
        or tuple(baseline["seeds"]) != k11.SEEDS
    ):
        raise ValueError("K13 invalid historical baseline comparator")
    for seed in k11.SEEDS:
        item = keyed(master["k80_loaders"], seed)
        if base.sha256_file(Path(item["path"])) != item["sha256"]:
            raise ValueError(f"K13 K80 seed{seed} loader hash mismatch")


def load_k11_model(
    master: dict[str, Any], budget: int, seed: int, device: torch.device
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any], str]:
    k11_config_path = Path(master["k11"]["config_path"]).resolve()
    k11_master = yaml.safe_load(k11_config_path.read_text(encoding="utf-8"))
    config = k11.build_run_config(k11_master, budget, seed)
    payload, preflight_lock = k11.load_preflight(
        k11_master,
        k11_config_path,
        Path(master["k11"]["output_root"]),
        budget,
    )
    model, lock_hash = k11.load_model(
        k11_master,
        k11_config_path,
        Path(master["k11"]["output_root"]),
        config,
        payload,
        preflight_lock,
        budget,
        seed,
        device,
    )
    return model, config, payload, lock_hash


def load_k80_model(
    master: dict[str, Any], seed: int, device: torch.device
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any], str]:
    item = keyed(master["k80_loaders"], seed)
    k6_path = Path(item["path"]).resolve()
    if base.sha256_file(k6_path) != item["sha256"]:
        raise ValueError(f"K13 K80 seed{seed} K6 config hash mismatch")
    config = yaml.safe_load(k6_path.read_text(encoding="utf-8"))
    k6.validate_config(config)
    upstream = config["upstream"]
    geometry_path = k6.resolve_and_hash(
        upstream["geometry_config_path"],
        upstream["geometry_config_sha256"],
        "K13 K80 geometry config",
    )
    geometry = yaml.safe_load(geometry_path.read_text(encoding="utf-8"))
    engine = k6.configure_upstream_engine(upstream["geometry_protocol"])
    engine.validate_config(geometry)
    root = Path(geometry["runtime"]["output_root"]).resolve()
    payload, preflight_lock = engine.load_preflight(root, geometry_path, geometry)
    model, model_lock = engine.load_new_model(
        "isotropic",
        root,
        geometry_path,
        geometry,
        payload,
        preflight_lock,
        device,
    )
    if int(geometry["training"]["seed"]) != seed:
        raise ValueError("K13 K80 seed mismatch")
    return model, geometry, payload, str(model_lock)


def evaluate_one(
    master: dict[str, Any], budget: int, seed: int, device: torch.device
) -> dict[str, Any]:
    if budget == 80:
        model, config, payload, model_lock = load_k80_model(master, seed, device)
    else:
        model, config, payload, model_lock = load_k11_model(master, budget, seed, device)
    if tuple(
        int(config["split"][key])
        for key in ("train_end", "validation_end", "test_end")
    ) != (8000, 9000, 10000):
        raise ValueError("K13 source split changed")
    images = np.load(Path(config["dataset"]["cache_path"]), mmap_mode="r")
    raw_latents = base.load_latents(config)
    subset, subset_audit = dynamic.build_subset(config)
    latents, _, _ = v17.normalize_with_k80_rgb_statistics(
        raw_latents, int(config["split"]["train_end"]), subset
    )
    batch_size = int(config["training"]["batch_size"])
    calibration_prediction = gauge.predict_selected(
        model, images, latents, subset, batch_size, device
    )
    start = int(master["evaluation"]["test_start"])
    end = int(master["evaluation"]["test_end"])
    test_prediction = v6.predict_rgb(
        model, images, latents, start, end, batch_size, device
    )
    semantic = k6.semantic_bundle(
        calibration_prediction,
        test_prediction,
        latents[0, subset, :3],
        latents[0, start:end, :3],
        latents[0, start:end],
        list(config["model"]["learned_indices"]),
        list(config["model"]["anchor_indices"]),
        float(config["evaluation"]["full_affine_ridge"]),
    )
    geometry_sha256 = payload["geometry_sha256"]
    if isinstance(geometry_sha256, dict):
        geometry_sha256 = geometry_sha256["isotropic"]
    return {
        "protocol_version": PROTOCOL,
        "mode": "single_locked_checkpoint_heldout_readout",
        "budget": budget,
        "seed": seed,
        "geometry": "isotropic",
        "source_files_sha256": source_hashes(),
        "model_lock": model_lock,
        "subset": subset_audit,
        "geometry_sha256": geometry_sha256,
        "semantic": semantic,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "training_performed": False,
        "test_rows": [start, end],
        "test_evaluated": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    config_path = args.config.resolve()
    master = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_master(master)
    if args.budget not in k11.BUDGETS or args.seed not in k11.SEEDS:
        raise ValueError("unregistered K13 budget or seed")
    torch.set_num_threads(int(master["runtime"]["num_threads"]))
    torch.set_float32_matmul_precision("high")
    device = torch.device(
        master["runtime"]["device"] if torch.cuda.is_available() else "cpu"
    )
    output_root = Path(args.output_root or master["runtime"]["output_root"]).resolve()
    output = output_root / f"k{args.budget}" / f"seed{args.seed}" / "test_readout.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite K13 readout: {output}")
    result = evaluate_one(master, args.budget, args.seed, device)
    result["config_path"] = str(config_path)
    result["config_sha256"] = base.sha256_file(config_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    v11.atomic_json(output, result)


if __name__ == "__main__":
    main()
