"""One-time K=80 held-out comparison of bounded and unbounded residuals."""

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
import audit_relation_tube_geometry_downstream_k6 as downstream
import run_calibrated_relation_tube_k0 as tube
import run_conflict_projected_physics_k80_closed as v17
import run_diagnostic as base
import run_dynamic_residual_propagation_k0 as dynamic
import run_physics_functional_anchor_training as v11
import run_supervised_continuation_diagnostic as v6


PROTOCOL = "bounded_unbounded_heldout_k45_v1"
SEEDS = (0, 42, 3407)
CONDITIONS = ("unbounded_correct", "tube_correct")
SOURCE_FILES = (
    "audit_bounded_unbounded_heldout_k45.py",
    "audit_calibrated_relation_tube_gauge_k1.py",
    "audit_relation_tube_geometry_downstream_k6.py",
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


def validate_contract(master: dict[str, Any]) -> None:
    if master["protocol_version"] != PROTOCOL:
        raise ValueError("unexpected K45 protocol")
    evaluation = master["evaluation"]
    if tuple(int(value) for value in evaluation["seeds"]) != SEEDS:
        raise ValueError("K45 seed registry changed")
    if tuple(evaluation["conditions"]) != CONDITIONS:
        raise ValueError("K45 condition registry changed")
    if (int(evaluation["test_start"]), int(evaluation["test_end"])) != (9000, 10000):
        raise ValueError("K45 held-out rows changed")
    if evaluation["training_allowed"] is not False:
        raise ValueError("K45 cannot train")
    if evaluation["selection_allowed"] is not False:
        raise ValueError("K45 cannot select a residual geometry")
    if evaluation["test_evaluation_authorized"] is not True:
        raise ValueError("K45 held-out read is not authorized")
    if master["implementation"]["entrypoint_sha256"] != base.sha256_file(
        Path(__file__).resolve()
    ):
        raise ValueError("K45 entrypoint hash mismatch")


def load_locked_model(
    entry: dict[str, Any], condition: str, seed: int, device: torch.device
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any], str]:
    config_path = Path(entry["config_path"])
    if base.sha256_file(config_path) != entry["config_sha256"]:
        raise ValueError(f"K45 seed {seed} config hash mismatch")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    tube.validate_config(config)
    if int(config["training"]["seed"]) != seed:
        raise ValueError("K45 seed/config mismatch")
    if tuple(int(config["split"][key]) for key in ("train_end", "validation_end", "test_end")) != (8000, 9000, 10000):
        raise ValueError("K45 source split changed")

    root = Path(entry["output_root"])
    preflight, _, preflight_lock_path = tube.load_preflight(root, config_path, config)
    lock_path = root / "formal" / condition / "training_lock.json"
    expected_lock = entry["training_lock_sha256"][condition]
    if base.sha256_file(lock_path) != expected_lock:
        raise ValueError(f"K45 seed {seed} {condition} lock hash mismatch")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    checks = {
        "formal": lock["mode"] == "formal",
        "status": lock["status"] == "locked_before_joint_semantic_validation_readout",
        "condition": lock["condition"] == condition,
        "config": lock["config_sha256"] == entry["config_sha256"],
        "source": lock["source_files_sha256"] == tube.source_hashes(),
        "preflight": lock["preflight_lock_sha256"] == base.sha256_file(preflight_lock_path),
        "validity": all(lock["validity"].values()),
        "test_unread_at_lock": lock["test_evaluated"] is False,
    }
    if not all(checks.values()):
        raise ValueError(f"invalid K45 lock seed={seed} condition={condition}: {checks}")
    model = tube.load_locked_model(condition, lock, config, preflight, device)
    return model, config, preflight, base.sha256_file(lock_path)


def evaluate_model(
    model: torch.nn.Module,
    config: dict[str, Any],
    master: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
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
    semantic = downstream.semantic_bundle(
        calibration_prediction,
        test_prediction,
        latents[0, subset, :3],
        latents[0, start:end, :3],
        latents[0, start:end],
        list(config["model"]["learned_indices"]),
        list(config["model"]["anchor_indices"]),
        float(master["evaluation"]["affine_ridge"]),
    )
    return {"subset": subset_audit, "semantic": semantic}


def summarize(runs: dict[str, Any]) -> dict[str, Any]:
    paired: dict[str, Any] = {}
    for seed in SEEDS:
        values = runs[str(seed)]
        unbounded = values["unbounded_correct"]["semantic"]
        bounded = values["tube_correct"]["semantic"]
        paired[str(seed)] = {}
        for readout in ("raw", "coordinatewise_affine", "full_affine"):
            paired[str(seed)][readout] = {
                "bounded_minus_unbounded_correlation": float(
                    bounded[readout]["mean_direct_abs_correlation"]
                    - unbounded[readout]["mean_direct_abs_correlation"]
                ),
                "bounded_minus_unbounded_r2": float(
                    bounded[readout]["mean_r2"] - unbounded[readout]["mean_r2"]
                ),
            }
    return {
        "paired": paired,
        "direction_counts": {
            readout: {
                metric: sum(
                    paired[str(seed)][readout][metric] > 0.0 for seed in SEEDS
                )
                for metric in (
                    "bounded_minus_unbounded_correlation",
                    "bounded_minus_unbounded_r2",
                )
            }
            for readout in ("raw", "coordinatewise_affine", "full_affine")
        },
        "performance_threshold_used": False,
        "geometry_selection_reopened": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config_path = args.config.resolve()
    master = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_contract(master)
    torch.set_num_threads(int(master["runtime"]["num_threads"]))
    torch.set_float32_matmul_precision("high")
    device = torch.device(
        master["runtime"]["device"] if torch.cuda.is_available() else "cpu"
    )
    output = Path(args.output or master["runtime"]["output_path"]).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite K45 result: {output}")

    runs: dict[str, Any] = {}
    for seed in SEEDS:
        entry = keyed(master["upstream"], seed)
        runs[str(seed)] = {}
        for condition in CONDITIONS:
            model, config, _, lock_sha = load_locked_model(
                entry, condition, seed, device
            )
            run = evaluate_model(model, config, master, device)
            run["training_lock_sha256"] = lock_sha
            runs[str(seed)][condition] = run
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    result = {
        "protocol_version": PROTOCOL,
        "mode": "post_lock_k80_bounded_unbounded_heldout_readout",
        "config_path": str(config_path),
        "config_sha256": base.sha256_file(config_path),
        "source_files_sha256": source_hashes(),
        "runs": runs,
        "comparison": summarize(runs),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "training_performed": False,
        "test_rows": [
            int(master["evaluation"]["test_start"]),
            int(master["evaluation"]["test_end"]),
        ],
        "test_evaluated": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    v11.atomic_json(output, result)
    print(json.dumps({"result_path": str(output), "sha256": base.sha256_file(output)}))


if __name__ == "__main__":
    main()
