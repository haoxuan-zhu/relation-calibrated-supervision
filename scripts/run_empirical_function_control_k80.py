"""Matched K=80 empirical-function teacher control.

The control keeps the closed v17/v18 training protocol and replaces only the
full-Malus forward/pseudoinverse teacher by a frozen low-dimensional empirical
inverse, [1, m, m*tau] -> RGB, fitted on the same 80 observational labels.
"""

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
import torch.nn.functional as F
import yaml

import audit_physics_functional_anchor as v10
import run_clamped_anchor_diagnostic as v2
import run_conflict_projected_physics_from_scratch as v16
import run_conflict_projected_physics_k80_closed as v17
import run_conflict_projected_physics_training as v13
import run_diagnostic as base
import run_matched_point_control_k80 as point
import run_physics_functional_anchor_training as v11
import run_supervised_continuation_diagnostic as v6


PROTOCOL = "partial_anchor_crl_empirical_function_k80_v1"
CONDITION = "empirical_function_correct_floor"
RGB_STATISTICS_SOURCE = v17.RGB_STATISTICS_SOURCE


def fit_empirical_teacher(
    images: np.ndarray,
    raw_latents: np.ndarray,
    subset: np.ndarray,
    train_end: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit the frozen 7-by-3 interaction inverse on K observational rows."""
    subset = np.asarray(subset, dtype=np.int64)
    means = v10.image_channel_means(images, 0, 0, train_end)
    tau = v10.malus_tau(raw_latents)
    unit_rgb = raw_latents[..., :3].astype(np.float64) / 255.0
    coefficients = v10.fit_interaction_inverse(
        means[subset], tau[0, subset], unit_rgb[0, subset]
    )
    fitted = v10.predict_interaction_inverse(
        coefficients, means[subset], tau[0, subset], clip=False
    )
    design_shape = [int(subset.size), 7]
    coefficient_shape = list(coefficients.shape)
    if coefficient_shape != [7, 3]:
        raise ValueError(f"unexpected empirical coefficient shape: {coefficient_shape}")
    audit = {
        "family": "linear_interaction_inverse_[1,m,m*tau]_to_rgb",
        "fit_environment": "obs",
        "label_rows": subset.tolist(),
        "unique_label_rows": int(np.unique(subset).size),
        "design_shape": design_shape,
        "coefficient_shape": coefficient_shape,
        "coefficient_count": int(coefficients.size),
        "coefficients": coefficients.tolist(),
        "coefficients_sha256": hashlib.sha256(
            np.ascontiguousarray(coefficients, dtype=np.float64).tobytes()
        ).hexdigest(),
        "calibration_mse": float(np.mean((fitted - unit_rgb[0, subset]) ** 2)),
        "tau_source": "cos_squared_known_angle_difference",
        "semantic_rows_read": int(subset.size),
    }
    return coefficients.astype(np.float32), audit


def empirical_pseudo_target(
    x: torch.Tensor,
    raw_angles: torch.Tensor,
    coefficients: torch.Tensor,
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
    clip: bool,
) -> torch.Tensor:
    """Apply the frozen empirical inverse and return normalized RGB targets."""
    means = x.mean(dim=(2, 3))
    tau = torch.cos(torch.deg2rad(raw_angles[:, 0] - raw_angles[:, 1])).square()
    design = torch.cat(
        [torch.ones_like(tau[:, None]), means, means * tau[:, None]], dim=1
    )
    unit_rgb = design @ coefficients
    if clip:
        unit_rgb = unit_rgb.clamp(0.0, 1.0)
    return (unit_rgb * 255.0 - latent_mean[:3]) / latent_std[:3]


def empirical_loss_components(
    model: torch.nn.Module,
    condition: v2.Condition,
    images: np.ndarray,
    latents: np.ndarray,
    raw_latents: np.ndarray,
    anchor_maps: dict[str, np.ndarray],
    rows: np.ndarray,
    targets: np.ndarray,
    config: dict[str, Any],
    device: torch.device,
    coefficients: dict[str, torch.Tensor],
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Compute the unchanged CCRL loss plus empirical functional supervision."""
    if condition.name != CONDITION:
        raise ValueError(f"unexpected empirical condition: {condition.name}")
    ccrl, metrics = v2.objective_for_batch(
        model, images, latents, anchor_maps, condition, rows, targets, config, device
    )
    anchor_indices = list(config["model"]["anchor_indices"])
    losses = []
    for environments in (np.zeros_like(rows), targets + 1):
        x = base.image_batch(images, environments, rows, device)
        anchors = torch.from_numpy(
            np.array(latents[environments, rows][:, anchor_indices], copy=True)
        ).to(device)
        raw_angles = torch.from_numpy(
            np.array(raw_latents[environments, rows][:, anchor_indices], copy=True)
        ).to(device)
        pseudo = empirical_pseudo_target(
            x,
            raw_angles,
            coefficients["correct"],
            latent_mean,
            latent_std,
            bool(config["empirical_anchor"]["clip_predictions_to_unit_interval"]),
        )
        losses.append(F.mse_loss(model.embedding(x, anchors), pseudo))
    empirical = torch.stack(losses).mean()
    return ccrl, empirical, metrics


def teacher_validation_metrics(
    coefficients: np.ndarray,
    images: np.ndarray,
    raw_latents: np.ndarray,
    train_end: int,
    validation_end: int,
    clip: bool,
) -> dict[str, Any]:
    """Read validation RGB only after the checkpoint/training lock exists."""
    means = np.stack(
        [
            v10.image_channel_means(images, environment, train_end, validation_end)
            for environment in range(images.shape[0])
        ],
        axis=0,
    )
    tau = v10.malus_tau(raw_latents)[:, train_end:validation_end]
    truth = (
        raw_latents[:, train_end:validation_end, :3].astype(np.float64) / 255.0
    )
    prediction = v10.predict_interaction_inverse(
        coefficients,
        means.reshape(-1, 3),
        tau.reshape(-1),
        clip,
    )
    return v10.regression_metrics(prediction, truth.reshape(-1, 3))


def decide_validity(
    metrics: dict[str, Any],
    training: dict[str, Any],
    post_lock_validation: dict[str, Any],
    teacher_audit: dict[str, Any],
    subset_audit: dict[str, Any],
    generated_initial_state_hash: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Judge run integrity; effect interpretation is deferred to aggregation."""
    expected_initial = config["initialization"]["expected_state_dict_sha256"]
    reference = config["matched_references"]
    validation_value = float(post_lock_validation["mean_direct_abs_correlation"])
    test_value = float(np.mean(metrics["direct_abs_correlation"][:3]))
    validity = {
        "random_initial_state_exact": generated_initial_state_hash == expected_initial,
        "same_initial_state_loaded": training["initial_state_sha256"]
        == expected_initial,
        "no_warm_checkpoint_in_config": "warm_start" not in config,
        "k80_rgb_statistics_source_registered": config["normalization"][
            "rgb_statistics_source"
        ]
        == RGB_STATISTICS_SOURCE,
        "teacher_budget_exact": subset_audit["budget"]
        == int(config["empirical_anchor"]["budget"])
        == 80,
        "teacher_subset_matches_normalization": subset_audit["subset_sha256"]
        == config["normalization"]["subset_sha256"],
        "teacher_subset_matches_physical": subset_audit["subset_sha256"]
        == reference["physical_subset_sha256"],
        "teacher_subset_matches_point": subset_audit["subset_sha256"]
        == reference["point_subset_sha256"],
        "teacher_observational_rows_exact": teacher_audit["fit_environment"] == "obs"
        and teacher_audit["unique_label_rows"] == 80
        and teacher_audit["semantic_rows_read"] == 80,
        "teacher_dimensions_exact": teacher_audit["design_shape"] == [80, 7]
        and teacher_audit["coefficient_shape"] == [7, 3]
        and teacher_audit["coefficient_count"] == 21,
        "same_parameter_count_as_references": training["parameter_count"]
        == int(reference["expected_parameter_count"]),
        "floor_exact": bool(
            np.isclose(training["final_supervision_weight"], 0.1)
        ),
        "no_semantic_truth_before_lock": all(
            "semantic_validation" not in record for record in training["history"]
        )
        and not training["semantic_truth_read_during_training"],
        "anchors_exact": metrics["anchor_mean_direct_r2"]
        >= float(config["evaluation"]["minimum_oracle_anchor_r2"]),
        "training_values_finite": point.numeric_training_values_are_finite(training),
        "semantic_metrics_finite": bool(
            np.isfinite(validation_value) and np.isfinite(test_value)
        ),
    }
    seed = int(config["training"]["seed"])
    return {
        "verdict": (
            f"empirical_function_control_valid_seed{seed}"
            if all(validity.values())
            else f"empirical_function_control_invalid_seed{seed}"
        ),
        "validity": validity,
        "readout": {
            "validation_rgb_mean_direct_abs_correlation": validation_value,
            "test_rgb_mean_direct_abs_correlation": test_value,
        },
        "effect_judgment_deferred_to_three_seed_aggregate": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["protocol_version"] != PROTOCOL:
        raise ValueError("unexpected empirical-function protocol")
    if list(config["training"]["conditions"]) != [CONDITION]:
        raise ValueError("runner accepts exactly one empirical condition")
    if config["initialization"]["mode"] != "same_seed_random_no_warm":
        raise ValueError("empirical control must use the registered random state")
    actual_runner_sha = base.sha256_file(Path(__file__).resolve())
    if actual_runner_sha != config["implementation"]["runner_sha256"]:
        raise ValueError("empirical runner hash mismatch")

    epochs = int(args.epochs or config["training"]["epochs"])
    if not args.smoke and epochs != int(config["training"]["epochs"]):
        raise ValueError("formal empirical run must use frozen epochs")
    config["training"]["epochs"] = epochs
    output_dir = (args.output_dir or Path(config["runtime"]["output_dir"])).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(config["runtime"]["num_threads"]))
    torch.set_float32_matmul_precision("high")
    device = torch.device(
        config["runtime"]["device"] if torch.cuda.is_available() else "cpu"
    )

    images = np.load(Path(config["dataset"]["cache_path"]), mmap_mode="r")
    raw_latents = base.load_latents(config)
    spec, subset_audit = v11.build_calibration_spec(config)
    subset = np.asarray(spec["subset"], dtype=np.int64)
    if subset_audit["subset_sha256"] != config["normalization"]["subset_sha256"]:
        raise ValueError("teacher and normalization subsets differ")
    latents, latent_mean_np, latent_std_np = v17.normalize_with_k80_rgb_statistics(
        raw_latents, int(config["split"]["train_end"]), subset
    )
    coefficients_np, teacher_audit = fit_empirical_teacher(
        images, raw_latents, subset, int(config["split"]["train_end"])
    )
    coefficients = {"correct": torch.from_numpy(coefficients_np).to(device)}
    latent_mean = torch.from_numpy(latent_mean_np).to(device)
    latent_std = torch.from_numpy(latent_std_np).to(device)
    anchor_maps, anchor_map_audit = v2.build_anchor_maps(config)
    initial_state, initial_state_hash = v16.make_initial_state(config, device)
    expected_hash = config["initialization"]["expected_state_dict_sha256"]
    if initial_state_hash != expected_hash:
        raise ValueError(
            f"random initial state hash mismatch: {initial_state_hash} != {expected_hash}"
        )
    initial_path = output_dir / f"initial_state_seed{config['training']['seed']}.pt"
    torch.save(
        {
            "seed": int(config["training"]["seed"]),
            "state_dict_sha256": initial_state_hash,
            "state_dict": copy.deepcopy(initial_state),
        },
        initial_path,
    )

    condition = v2.Condition(CONDITION, "oracle")
    started = time.time()
    original_loss_components = v13.loss_components
    v13.loss_components = empirical_loss_components
    try:
        training = v16.train_condition(
            condition,
            initial_state,
            initial_state_hash,
            images,
            latents,
            raw_latents,
            anchor_maps,
            config,
            device,
            output_dir,
            coefficients,
            latent_mean,
            latent_std,
        )
    finally:
        v13.loss_components = original_loss_components

    source_names = [
        "run_empirical_function_control_k80.py",
        "audit_physics_functional_anchor.py",
        "run_conflict_projected_physics_k80_closed.py",
        "run_conflict_projected_physics_from_scratch.py",
        "run_conflict_projected_physics_training.py",
        "run_physics_functional_anchor_training.py",
        "run_conditioned_film_diagnostic.py",
        "run_clamped_anchor_diagnostic.py",
        "run_diagnostic.py",
    ]
    source_hashes = {
        name: base.sha256_file(Path(__file__).resolve().parent / name)
        for name in source_names
    }
    training_lock = {
        "status": "locked_before_semantic_validation_and_test_evaluation",
        "config_sha256": base.sha256_file(config_path),
        "source_files_sha256": source_hashes,
        "initial_state": {
            "path": str(initial_path),
            "file_sha256": base.sha256_file(initial_path),
            "state_dict_sha256": initial_state_hash,
        },
        "normalization": {
            "rgb_statistics_source": config["normalization"]["rgb_statistics_source"],
            "latent_mean": latent_mean_np,
            "latent_std": latent_std_np,
        },
        "calibration_subset": subset_audit,
        "empirical_teacher": teacher_audit,
        "anchor_map_audit": anchor_map_audit,
        "checkpoint": {
            "path": training["checkpoint"],
            "sha256": training["checkpoint_sha256"],
            "initial_state_sha256": training["initial_state_sha256"],
            "final_supervision_weight": training["final_supervision_weight"],
        },
        "semantic_validation_evaluated": False,
        "test_evaluated": False,
    }
    lock_path = output_dir / "training_lock.json"
    v11.atomic_json(lock_path, training_lock)
    lock_sha = base.sha256_file(lock_path)
    common = {
        "protocol_version": config["protocol_version"],
        "mode": "smoke" if args.smoke else "formal",
        "config": config,
        "config_sha256": base.sha256_file(config_path),
        "source_files_sha256": source_hashes,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "initial_state": training_lock["initial_state"],
        "normalization": training_lock["normalization"],
        "calibration_subset": subset_audit,
        "empirical_teacher": teacher_audit,
        "training": training,
        "training_lock": str(lock_path),
        "training_lock_sha256_before_semantic_and_test": lock_sha,
        "duration_seconds_before_semantic_and_test": time.time() - started,
    }
    if args.smoke:
        result = {**common, "verdict": "smoke_completed_no_semantic_or_test_read"}
        result_path = output_dir / "smoke_results.json"
    else:
        model = v13.load_model(Path(training["checkpoint"]), config, device)
        train_end = int(config["split"]["train_end"])
        validation_end = int(config["split"]["validation_end"])
        prediction = v6.predict_rgb(
            model,
            images,
            latents,
            train_end,
            validation_end,
            int(config["training"]["batch_size"]),
            device,
        )
        validation_semantics = v6.regression_metrics(
            prediction, latents[0, train_end:validation_end, :3]
        )
        metrics = v2.evaluate_model(
            model, images, latents, anchor_maps, condition, config, device
        )
        teacher_validation = teacher_validation_metrics(
            coefficients_np,
            images,
            raw_latents,
            train_end,
            validation_end,
            bool(config["empirical_anchor"]["clip_predictions_to_unit_interval"]),
        )
        decision = decide_validity(
            metrics,
            training,
            validation_semantics,
            teacher_audit,
            subset_audit,
            initial_state_hash,
            config,
        )
        result = {
            **common,
            "post_lock_teacher_validation": teacher_validation,
            "post_lock_validation_semantics": validation_semantics,
            "metrics": metrics,
            "decision": decision,
            "duration_seconds": time.time() - started,
        }
        result_path = output_dir / "diagnostic_results.json"
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    v11.atomic_json(result_path, result)
    print(
        json.dumps(
            {"result_path": str(result_path), "sha256": base.sha256_file(result_path)},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
