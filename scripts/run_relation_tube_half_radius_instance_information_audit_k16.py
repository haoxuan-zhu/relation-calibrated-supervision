"""Audit whether the frozen K15 half-radius tube retains image-specific signal."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

import audit_relation_tube_geometry_downstream_k6 as k6
import run_conflict_projected_physics_k80_closed as v17
import run_diagnostic as base
import run_dynamic_residual_propagation_k0 as dynamic
import run_physics_functional_anchor_training as v11
import run_relation_tube_instance_information_audit_k14 as k14
import run_relation_tube_radius_scale_ablation_k15 as k15


PROTOCOL = "relation_tube_half_radius_instance_information_audit_k16_v1"
SEEDS = (3407, 0, 42)
RADIUS_MODE = "half_empirical"
CONDITIONS = k14.CONDITIONS
SOURCE_FILES = (
    "run_relation_tube_half_radius_instance_information_audit_k16.py",
    "run_relation_tube_instance_information_audit_k14.py",
    "run_relation_tube_radius_scale_ablation_k15.py",
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
        raise ValueError("unexpected K16 protocol")
    scope = config["scope"]
    if int(scope["budget"]) != 80 or tuple(int(value) for value in scope["seeds"]) != SEEDS:
        raise ValueError("K16 budget or seeds changed")
    if tuple(int(value) for value in scope["validation_rows"]) != (8000, 9000):
        raise ValueError("K16 validation split changed")
    if tuple(int(value) for value in scope["test_rows"]) != (9000, 10000):
        raise ValueError("K16 test registry changed")
    if scope["test_evaluated"] is not False:
        raise ValueError("K16 test must remain closed")
    if config["upstream"]["radius_mode"] != RADIUS_MODE:
        raise ValueError("K16 radius mode changed")
    if tuple(config["interventions"]["order"]) != CONDITIONS:
        raise ValueError("K16 intervention registry changed")
    if int(config["interventions"]["derangement_seed"]) != 20260803:
        raise ValueError("K16 derangement seed changed")
    evaluation = config["evaluation"]
    if not np.isclose(float(evaluation["full_affine_ridge"]), 0.001):
        raise ValueError("K16 readout ridge changed")
    if not np.isclose(float(evaluation["reproduction_atol"]), 5e-6):
        raise ValueError("K16 reproduction tolerance changed")
    if int(evaluation["bootstrap_replicates"]) != 2000:
        raise ValueError("K16 bootstrap count changed")
    if int(evaluation["bootstrap_seed"]) != 20260804:
        raise ValueError("K16 bootstrap seed changed")
    if int(config["decision"]["strong_directional_seeds"]) != 3:
        raise ValueError("K16 strong directional rule changed")
    if int(config["decision"]["strong_bootstrap_positive_seeds"]) != 2:
        raise ValueError("K16 bootstrap rule changed")
    if int(config["decision"]["mixed_directional_seeds"]) != 2:
        raise ValueError("K16 mixed directional rule changed")
    if {int(key) for key in config["runs"]} != set(SEEDS):
        raise ValueError("K16 run registry changed")


def load_seed_context(
    master: dict[str, Any], seed: int, device: torch.device
) -> tuple[dict[str, Any], torch.nn.Module, dict[str, Any]]:
    k15_config_path = Path(master["upstream"]["k15_config_path"])
    if base.sha256_file(k15_config_path) != master["upstream"]["k15_config_sha256"]:
        raise ValueError("K16 K15 config hash mismatch")
    k15_master = yaml.safe_load(k15_config_path.read_text(encoding="utf-8"))
    k15.validate_master(k15_master)
    model, config, payload, identity = k15.load_model(
        k15_master, k15_config_path.resolve(), None, seed, RADIUS_MODE, device
    )
    entry = keyed(master["runs"], seed)
    readout_path = Path(entry["validation_readout_path"])
    if base.sha256_file(readout_path) != entry["validation_readout_sha256"]:
        raise ValueError(f"K16 K15 validation readout hash mismatch for seed {seed}")
    readout = json.loads(readout_path.read_text(encoding="utf-8"))
    checks = {
        "protocol": readout["protocol_version"] == k15.PROTOCOL,
        "seed": int(readout["seed"]) == seed,
        "radius_mode": readout["radius_mode"] == RADIUS_MODE,
        "validation": readout["semantic_validation_evaluated"] is True,
        "test_unread": readout["test_evaluated"] is False,
    }
    if not all(checks.values()):
        raise ValueError(f"K16 upstream readout invalid for seed {seed}: {checks}")
    return config, model, {
        "semantic": readout["semantic"],
        "identity": {
            **identity,
            "validation_readout_path": str(readout_path),
            "validation_readout_sha256": entry["validation_readout_sha256"],
        },
    }


def seed_audit(master: dict[str, Any], seed: int, device: torch.device) -> dict[str, Any]:
    config, model, upstream = load_seed_context(master, seed, device)
    images = np.load(Path(config["dataset"]["cache_path"]), mmap_mode="r")
    raw_latents = base.load_latents(config)
    subset, subset_audit = dynamic.build_subset(config)
    latents, _, _ = v17.normalize_with_k80_rgb_statistics(
        raw_latents, int(config["split"]["train_end"]), subset
    )
    start, end = tuple(int(value) for value in master["scope"]["validation_rows"])
    if (start, end, int(config["split"]["test_end"])) != (8000, 9000, 10000):
        raise ValueError("K16 dataset split changed")
    rows = np.arange(start, end, dtype=np.int64)
    permutation = k14.make_derangement(
        len(rows), int(master["interventions"]["derangement_seed"])
    )
    visual_rows = rows[permutation]
    batch_size = int(config["training"]["batch_size"])
    calibration = k14.collect_correct_components(
        model, images, latents, subset, batch_size, device
    )
    correct = k14.collect_correct_components(
        model, images, latents, rows, batch_size, device
    )
    visual_swap = k14.collect_fixed_anchor_visual_swap(
        model, images, latents, rows, visual_rows, batch_size, device
    )
    center = correct["center"]
    projected_correct = correct["projected_residual"]
    projected_visual = visual_swap["projected_residual"]
    projected_reassigned = projected_correct[permutation]
    outputs = {
        "correct_image": center + projected_correct,
        "fixed_anchor_visual_derangement": center + projected_visual,
        "projected_residual_reassignment": center + projected_reassigned,
        "center_only": center,
    }
    residuals = {
        "correct_image": projected_correct,
        "fixed_anchor_visual_derangement": projected_visual,
        "projected_residual_reassignment": projected_reassigned,
        "center_only": np.zeros_like(projected_correct),
    }
    calibration_truth = latents[0, subset, :3]
    validation_truth = latents[0, start:end, :3]
    calibration_output = calibration["center"] + calibration["projected_residual"]
    calibration_residual_truth = calibration_truth - calibration["center"]
    validation_residual_truth = validation_truth - center
    ridge = float(master["evaluation"]["full_affine_ridge"])
    output_readout = k14.fit_frozen_readout(calibration_output, calibration_truth, ridge)
    residual_readout = k14.fit_frozen_readout(
        calibration["projected_residual"], calibration_residual_truth, ridge
    )
    output_metrics: dict[str, Any] = {}
    residual_metrics: dict[str, Any] = {}
    output_predictions: dict[str, dict[str, np.ndarray]] = {}
    residual_predictions: dict[str, dict[str, np.ndarray]] = {}
    for condition in CONDITIONS:
        output_metrics[condition], output_predictions[condition] = k14.apply_frozen_readout(
            outputs[condition], validation_truth, output_readout
        )
        residual_metrics[condition], residual_predictions[condition] = k14.apply_frozen_readout(
            residuals[condition], validation_residual_truth, residual_readout
        )
        residual_metrics[condition]["cosine"] = k14.cosine_summary(
            residuals[condition], validation_residual_truth
        )
    semantic = k6.semantic_bundle(
        calibration_output,
        outputs["correct_image"],
        calibration_truth,
        validation_truth,
        latents[0, start:end],
        list(config["model"]["learned_indices"]),
        list(config["model"]["anchor_indices"]),
        ridge,
    )
    reproduction = k14.reproduce_upstream(
        semantic, upstream["semantic"], float(master["evaluation"]["reproduction_atol"])
    )
    if not all(reproduction.values()):
        raise RuntimeError(f"K16 upstream reproduction failed for seed {seed}: {reproduction}")
    replicates = int(master["evaluation"]["bootstrap_replicates"])
    bootstrap_seed = int(master["evaluation"]["bootstrap_seed"]) + seed * 10
    controls = ("fixed_anchor_visual_derangement", "projected_residual_reassignment")
    bootstrap: dict[str, Any] = {}
    deltas: dict[str, Any] = {}
    for offset, control in enumerate(controls):
        bootstrap[control] = {
            "output_full_affine": k14.paired_bootstrap_mse_advantage(
                output_predictions["correct_image"]["full_affine"],
                output_predictions[control]["full_affine"],
                validation_truth,
                replicates,
                bootstrap_seed + offset * 2,
            ),
            "residual_full_affine": k14.paired_bootstrap_mse_advantage(
                residual_predictions["correct_image"]["full_affine"],
                residual_predictions[control]["full_affine"],
                validation_residual_truth,
                replicates,
                bootstrap_seed + offset * 2 + 1,
            ),
        }
        deltas[control] = {
            "correct_minus_control_output_full_affine_r2": float(
                output_metrics["correct_image"]["full_affine"]["mean_r2"]
                - output_metrics[control]["full_affine"]["mean_r2"]
            ),
            "correct_minus_control_output_raw_correlation": float(
                output_metrics["correct_image"]["raw"]["mean_direct_abs_correlation"]
                - output_metrics[control]["raw"]["mean_direct_abs_correlation"]
            ),
            "correct_minus_control_residual_full_affine_r2": float(
                residual_metrics["correct_image"]["full_affine"]["mean_r2"]
                - residual_metrics[control]["full_affine"]["mean_r2"]
            ),
        }
    distribution = {
        "correct_image": k14.distribution_summary(
            projected_correct, correct["projection_scale"], correct["raw_score"]
        ),
        "fixed_anchor_visual_derangement": k14.distribution_summary(
            projected_visual, visual_swap["projection_scale"], visual_swap["raw_score"]
        ),
        "projected_residual_reassignment": k14.distribution_summary(
            projected_reassigned,
            correct["projection_scale"][permutation],
            correct["raw_score"][permutation],
        ),
    }
    return {
        "seed": seed,
        "upstream": upstream["identity"],
        "subset": subset_audit,
        "permutation": {
            "seed": int(master["interventions"]["derangement_seed"]),
            "size": int(len(permutation)),
            "fixed_points": int(np.sum(permutation == np.arange(len(permutation)))),
            "sha256_int64": k14.sha256_array(permutation.astype("<i8")),
        },
        "output_metrics": output_metrics,
        "residual_metrics": residual_metrics,
        "distribution": distribution,
        "bootstrap": bootstrap,
        "deltas": deltas,
        "prediction_hashes": {
            "correct_output_float32": k14.sha256_array(outputs["correct_image"].astype("<f4")),
            "visual_swap_output_float32": k14.sha256_array(outputs["fixed_anchor_visual_derangement"].astype("<f4")),
            "reassigned_output_float32": k14.sha256_array(outputs["projected_residual_reassignment"].astype("<f4")),
        },
        "reproduction": reproduction,
        "test_evaluated": False,
    }


def decide(results: dict[int, dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    base_decision = k14.decide(results, config)
    verdict_map = {
        "invalid_instance_information_audit": "invalid_half_radius_audit",
        "instance_specific_residual_signal_supported": "image_specific_signal_retained_under_half_radius",
        "directional_instance_signal_supported_with_uncertainty": "directional_signal_retained_with_uncertainty",
        "mixed_instance_signal": "mixed_signal_retention",
        "instance_signal_not_supported": "instance_signal_not_retained_under_half_radius",
    }
    return {**base_decision, "k14_rule_verdict": base_decision["verdict"], "verdict": verdict_map[base_decision["verdict"]]}


def retention_summary(
    results: dict[int, dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    path = Path(config["upstream"]["k14_result_path"])
    if base.sha256_file(path) != config["upstream"]["k14_result_sha256"]:
        raise ValueError("K16 K14 result hash mismatch")
    empirical = json.loads(path.read_text(encoding="utf-8"))
    controls = ("fixed_anchor_visual_derangement", "projected_residual_reassignment")
    metrics = (
        "correct_minus_control_output_full_affine_r2",
        "correct_minus_control_output_raw_correlation",
        "correct_minus_control_residual_full_affine_r2",
    )
    summary: dict[str, Any] = {}
    for seed in SEEDS:
        seed_summary: dict[str, Any] = {}
        for control in controls:
            seed_summary[control] = {}
            for metric in metrics:
                current = float(results[seed]["deltas"][control][metric])
                baseline = float(empirical["results"][str(seed)]["deltas"][control][metric])
                seed_summary[control][metric] = {
                    "half_radius": current,
                    "empirical_radius": baseline,
                    "ratio": current / baseline if abs(baseline) > 1e-12 else None,
                }
        summary[str(seed)] = seed_summary
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_master(config)
    if args.validate_only:
        print(json.dumps({"protocol_version": PROTOCOL, "config_valid": True}, sort_keys=True))
        return
    output_root = Path(args.output_root or config["runtime"]["output_root"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / "half_radius_instance_information_audit.json"
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite K16 result: {output_path}")
    torch.set_num_threads(int(config["runtime"]["num_threads"]))
    torch.set_float32_matmul_precision("high")
    device = torch.device(config["runtime"]["device"] if torch.cuda.is_available() else "cpu")
    results: dict[int, dict[str, Any]] = {}
    for seed in SEEDS:
        results[seed] = seed_audit(config, seed, device)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    payload = {
        "protocol_version": PROTOCOL,
        "mode": "frozen_k80_half_radius_three_seed_validation_only_instance_information_audit",
        "config_path": str(config_path),
        "config_sha256": base.sha256_file(config_path),
        "source_files_sha256": source_hashes(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "results": results,
        "retention_vs_empirical_radius": retention_summary(results, config),
        "decision": decide(results, config),
        "training_performed": False,
        "semantic_validation_evaluated": True,
        "test_evaluated": False,
    }
    v11.atomic_json(output_path, payload)
    print(json.dumps({"result_path": str(output_path), "sha256": base.sha256_file(output_path), "verdict": payload["decision"]["verdict"]}, sort_keys=True))


if __name__ == "__main__":
    main()
