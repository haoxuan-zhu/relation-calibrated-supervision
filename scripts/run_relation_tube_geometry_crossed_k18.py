"""Complete the K8 calibration-subset by model-seed geometry crossing."""

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

import run_diagnostic as base
import run_dynamic_residual_propagation_k0 as dynamic
import run_physics_functional_anchor_training as v11
import run_relation_tube_calibration_subset_k8 as k8


PROTOCOL = "relation_tube_geometry_crossed_k18_v1"
MODEL_SEEDS = (0, 42)
GEOMETRIES = ("diagonal", "isotropic")
INITIAL_STATE_SHA256 = {
    0: "6037afc85fa38492ba8b6854e7467d610016d6bfe4a06e96276c11a1c302a7b2",
    42: "9e21bbf9d3cd2c6abefe2029a6e845c2a12175b0a49ec445234fc1258dbddc56",
}
SEED3407_SHA256 = "4b599031478a138433ed0208cc4871be1895b15db07576b2e13b5b9f81572731"
SOURCE_FILES = ("run_relation_tube_geometry_crossed_k18.py", *k8.SOURCE_FILES)
ORIGINAL_VALIDATE_CONFIG = k8.validate_config


def source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    return {name: base.sha256_file(root / name) for name in SOURCE_FILES}


def validate_config(config: dict[str, Any]) -> None:
    seed = int(config["training"]["seed"])
    if config["protocol_version"] != PROTOCOL:
        raise ValueError("unexpected K18 protocol")
    if seed not in MODEL_SEEDS or int(config["initialization"]["seed"]) != seed:
        raise ValueError("K18 is restricted to model seeds 0 and 42")
    if config["initialization"]["expected_state_dict_sha256"] != INITIAL_STATE_SHA256[seed]:
        raise ValueError("K18 initial-state hash changed")
    if tuple(config["geometry_subset"]["modes"]) != k8.GEOMETRIES:
        raise ValueError("K18 geometry registry changed")
    if config["audit"]["test_evaluated"] is not False:
        raise ValueError("K18 test must remain closed")

    shadow = copy.deepcopy(config)
    shadow["training"]["seed"] = 3407
    shadow["initialization"]["seed"] = 3407
    shadow["initialization"]["expected_state_dict_sha256"] = SEED3407_SHA256
    previous_protocol = k8.PROTOCOL
    k8.PROTOCOL = PROTOCOL
    try:
        ORIGINAL_VALIDATE_CONFIG(shadow)
    finally:
        k8.PROTOCOL = previous_protocol


def configure_k8_engine() -> None:
    k8.PROTOCOL = PROTOCOL
    k8.source_hashes = source_hashes
    k8.validate_config = validate_config


def paired_readout(args: argparse.Namespace, config: dict[str, Any]) -> None:
    if args.geometry is not None or args.epochs is not None:
        raise ValueError("K18 readout accepts neither geometry nor epochs")
    root = Path(args.output_root or config["runtime"]["output_root"]).resolve()
    payload, preflight_lock_path = k8.load_preflight(root, args.config.resolve())
    torch.set_num_threads(int(config["runtime"]["num_threads"]))
    torch.set_float32_matmul_precision("high")
    device = torch.device(
        config["runtime"]["device"] if torch.cuda.is_available() else "cpu"
    )
    images = np.load(Path(config["dataset"]["cache_path"]), mmap_mode="r")
    raw_latents = base.load_latents(config)
    subset, subset_audit = dynamic.build_subset(config)
    latents, _, _ = k8.v17.normalize_with_k80_rgb_statistics(
        raw_latents, int(config["split"]["train_end"]), subset
    )
    anchor_maps, _ = k8.tube.v2.build_anchor_maps(config)
    start = int(config["split"]["train_end"])
    end = int(config["split"]["validation_end"])
    batch_size = int(config["training"]["batch_size"])
    runs: dict[str, Any] = {}
    locks: dict[str, str] = {}

    for name in GEOMETRIES:
        model, locks[name] = k8.load_model(
            name,
            root,
            args.config.resolve(),
            config,
            payload,
            preflight_lock_path,
            device,
        )
        calibration_prediction = k8.k6.gauge.predict_selected(
            model, images, latents, subset, batch_size, device
        )
        validation_prediction = k8.v6.predict_rgb(
            model, images, latents, start, end, batch_size, device
        )
        semantic = k8.k6.semantic_bundle(
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
            "ccrl_validation": k8.tube.v2.validate(
                model,
                images,
                latents,
                anchor_maps,
                k8.tube.v2.Condition("tube_correct", "oracle"),
                config,
                device,
            ),
            "representation_audit": k8.tube.representation_audit(
                model, images, latents, start, end, batch_size, device
            ),
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    comparison = k8.comparison(runs["diagonal"], runs["isotropic"])
    output = {
        "protocol_version": PROTOCOL,
        "mode": "paired_diagonal_isotropic_validation_only_readout",
        "subset_seed": int(config["calibration"]["subset_seed"]),
        "model_seed": int(config["training"]["seed"]),
        "config_path": str(args.config.resolve()),
        "config_sha256": base.sha256_file(args.config.resolve()),
        "source_files_sha256": source_hashes(),
        "preflight_lock_sha256": base.sha256_file(preflight_lock_path),
        "subset": subset_audit,
        "geometry_sha256": {
            name: payload["geometry_sha256"][name] for name in GEOMETRIES
        },
        "training_locks": locks,
        "runs": runs,
        "isotropic_minus_diagonal": comparison,
        "decision": {
            "direct_coordinate_metrics_positive": all(
                comparison["semantic_deltas"][metric] > 0.0
                for metric in ("raw_correlation", "rgb_mcc", "coordinatewise_r2")
            ),
            "test_evaluated": False,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": str(device),
        },
        "semantic_validation_evaluated": True,
        "test_evaluated": False,
    }
    output_path = root / "formal" / "paired_validation_readout.json"
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite K18 readout: {output_path}")
    v11.atomic_json(output_path, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("preflight", "smoke", "train", "readout"), required=True
    )
    parser.add_argument("--geometry", choices=GEOMETRIES)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    configure_k8_engine()
    validate_config(config)
    if args.mode == "preflight":
        k8.preflight_main(args, config)
    elif args.mode in ("smoke", "train"):
        if args.geometry not in GEOMETRIES:
            raise ValueError("K18 training requires diagonal or isotropic")
        k8.train_main(args, config)
    else:
        paired_readout(args, config)


if __name__ == "__main__":
    main()
