from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml

import audit_physics_ccrl_gradient_conflict as v12b
import run_clamped_anchor_diagnostic as v2
import run_conditioned_film_diagnostic as v3
import run_diagnostic as base
import run_physics_functional_anchor_training as v11
import run_supervised_continuation_diagnostic as v6


CONDITIONS = [
    "vanilla_correct_floor",
    "projected_correct_floor",
    "vanilla_permuted_floor",
    "projected_permuted_floor",
]


def supervision_weight(epoch_index: int, config: dict[str, Any]) -> float:
    schedule = config["floor_schedule"]
    hold = int(schedule["hold_epochs"])
    decay_end = int(schedule["decay_end_epoch"])
    floor = float(schedule["nonzero_floor"])
    if epoch_index < hold:
        return float(schedule["hold_weight"])
    if epoch_index < decay_end:
        fraction = (decay_end - 1 - epoch_index) / max(decay_end - hold - 1, 1)
        return floor + (float(schedule["hold_weight"]) - floor) * fraction
    return floor


def project_and_combine(
    ccrl_gradients: list[torch.Tensor],
    physics_gradients: list[torch.Tensor],
    weight: float,
    epsilon: float,
) -> tuple[list[torch.Tensor], dict[str, float]]:
    dot = sum(torch.sum(left * right) for left, right in zip(ccrl_gradients, physics_gradients))
    physics_norm_sq = sum(torch.sum(value * value) for value in physics_gradients)
    ccrl_norm_sq = sum(torch.sum(value * value) for value in ccrl_gradients)
    coefficient = dot / physics_norm_sq.clamp_min(epsilon)
    removal = torch.minimum(coefficient, torch.zeros_like(coefficient))
    combined = [
        ccrl - removal * physics + weight * physics
        for ccrl, physics in zip(ccrl_gradients, physics_gradients)
    ]
    cosine = dot / torch.sqrt(ccrl_norm_sq * physics_norm_sq).clamp_min(epsilon)
    return combined, {
        "ccrl_physics_cosine": float(cosine.detach().cpu()),
        "projection_active": float((coefficient < 0).detach().cpu()),
        "unweighted_physics_to_ccrl_norm_ratio": float(
            torch.sqrt(physics_norm_sq / ccrl_norm_sq.clamp_min(epsilon)).detach().cpu()
        ),
    }


def loss_components(
    model: v3.FilmConditionedModel,
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
    ccrl, metrics = v2.objective_for_batch(
        model, images, latents, anchor_maps, condition, rows, targets, config, device
    )
    mode = "permuted" if "permuted" in condition.name else "correct"
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
        pseudo = v11.physical_pseudo_target(
            x,
            raw_angles,
            coefficients[mode],
            latent_mean,
            latent_std,
            float(config["functional_anchor"]["pinv_rcond"]),
            bool(config["functional_anchor"]["clip_predictions_to_unit_interval"]),
        )
        losses.append(F.mse_loss(model.embedding(x, anchors), pseudo))
    physics = torch.stack(losses).mean()
    return ccrl, physics, metrics


def projected_backward(
    model: v3.FilmConditionedModel,
    ccrl: torch.Tensor,
    physics: torch.Tensor,
    weight: float,
    epsilon: float,
) -> dict[str, float]:
    embedding_parameters = [parameter for parameter in model.embedding.parameters() if parameter.requires_grad]
    causal_parameters = [parameter for parameter in model.parametric_part.parameters() if parameter.requires_grad]
    ccrl_embedding = list(
        torch.autograd.grad(ccrl, embedding_parameters, retain_graph=True)
    )
    physics_embedding = list(
        torch.autograd.grad(physics, embedding_parameters, retain_graph=True)
    )
    ccrl_causal = list(torch.autograd.grad(ccrl, causal_parameters, retain_graph=False))
    combined, audit = project_and_combine(
        ccrl_embedding,
        physics_embedding,
        weight,
        epsilon,
    )
    for parameter, gradient in zip(embedding_parameters, combined):
        parameter.grad = gradient
    for parameter, gradient in zip(causal_parameters, ccrl_causal):
        parameter.grad = gradient
    return audit


def train_condition(
    condition: v2.Condition,
    warm_state: dict[str, torch.Tensor],
    images: np.ndarray,
    latents: np.ndarray,
    raw_latents: np.ndarray,
    anchor_maps: dict[str, np.ndarray],
    config: dict[str, Any],
    device: torch.device,
    output_dir: Path,
    coefficients: dict[str, torch.Tensor],
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
) -> dict[str, Any]:
    model_cfg = config["model"]
    training_cfg = config["training"]
    seed = int(training_cfg["seed"])
    base.set_seed(seed)
    model = v3.FilmConditionedModel(
        int(model_cfg["latent_dim"]),
        list(model_cfg["learned_indices"]),
        list(model_cfg["anchor_indices"]),
        int(model_cfg["hidden_channels"]),
        int(model_cfg["conv_layers"]),
    ).to(device)
    model.load_state_dict(warm_state)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training_cfg["learning_rate"]))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=float(training_cfg["scheduler_factor"]),
        patience=int(training_cfg["scheduler_patience"]),
    )
    rows, targets = base.make_pairs(
        list(training_cfg["targets"]), 0, int(config["split"]["train_end"])
    )
    batch_size = int(training_cfg["batch_size"])
    interval = int(training_cfg["validation_interval"])
    epochs = int(training_cfg["epochs"])
    projected = condition.name.startswith("projected_")
    history = []
    started = time.time()
    for epoch in range(epochs):
        weight = supervision_weight(epoch, config)
        order = np.random.default_rng(seed + 1009 * epoch).permutation(len(rows))
        model.train()
        sums: dict[str, float] = {}
        count = 0
        for begin in range(0, len(rows), batch_size):
            indices = order[begin : begin + batch_size]
            batch_rows = rows[indices]
            batch_targets = targets[indices]
            ccrl, physics, values = loss_components(
                model,
                condition,
                images,
                latents,
                raw_latents,
                anchor_maps,
                batch_rows,
                batch_targets,
                config,
                device,
                coefficients,
                latent_mean,
                latent_std,
            )
            optimizer.zero_grad(set_to_none=True)
            if projected:
                projection = projected_backward(
                    model,
                    ccrl,
                    physics,
                    weight,
                    float(config["projection"]["epsilon"]),
                )
            else:
                (ccrl + weight * physics).backward()
                projection = {
                    "ccrl_physics_cosine": float("nan"),
                    "projection_active": 0.0,
                    "unweighted_physics_to_ccrl_norm_ratio": float("nan"),
                }
            optimizer.step()
            size = len(batch_rows)
            count += size
            batch_values = {
                **values,
                "physics_loss": float(physics.detach().cpu()),
                "supervision_weight": weight,
                "optimized_total": float((ccrl + weight * physics).detach().cpu()),
                **projection,
            }
            for name, value in batch_values.items():
                if np.isfinite(value):
                    sums[name] = sums.get(name, 0.0) + size * value
        record: dict[str, Any] = {
            "epoch": epoch + 1,
            "train": base.average_metrics(sums, count),
            "supervision_weight": weight,
        }
        if epoch % interval == 0 or epoch == epochs - 1:
            validation = v2.validate(
                model, images, latents, anchor_maps, condition, config, device
            )
            scheduler.step(validation["total"])
            prediction = v6.predict_rgb(
                model,
                images,
                latents,
                int(config["split"]["train_end"]),
                int(config["split"]["validation_end"]),
                batch_size,
                device,
            )
            semantic = v6.regression_metrics(
                prediction,
                latents[0, int(config["split"]["train_end"]):int(config["split"]["validation_end"]), :3],
            )
            record.update(validation=validation, semantic_validation=semantic)
            print(
                json.dumps(
                    {
                        "condition": condition.name,
                        "epoch": epoch + 1,
                        "lambda": weight,
                        "semantic_mean_r2": semantic["mean_r2"],
                        "semantic_mean_corr": semantic["mean_direct_abs_correlation"],
                        "projection_active_fraction": record["train"].get("projection_active"),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        history.append(record)
    checkpoint = output_dir / f"{condition.name}_seed{seed}.pt"
    torch.save(
        {"condition": condition.name, "seed": seed, "epoch": epochs, "state_dict": model.state_dict()},
        checkpoint,
    )
    result = {
        "condition": condition.name,
        "final_epoch": epochs,
        "final_supervision_weight": supervision_weight(epochs - 1, config),
        "duration_seconds": time.time() - started,
        "history": history,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": base.sha256_file(checkpoint),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def load_model(path: Path, config: dict[str, Any], device: torch.device) -> v3.FilmConditionedModel:
    model_cfg = config["model"]
    model = v3.FilmConditionedModel(
        int(model_cfg["latent_dim"]),
        list(model_cfg["learned_indices"]),
        list(model_cfg["anchor_indices"]),
        int(model_cfg["hidden_channels"]),
        int(model_cfg["conv_layers"]),
    ).to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=False)["state_dict"])
    return model


def decide(
    metrics: dict[str, dict[str, Any]],
    training: dict[str, dict[str, Any]],
    lock_before_test: bool,
    calibration_audit: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    evaluation = config["evaluation"]
    projected = metrics["projected_correct_floor"]["unanchored_mcc"]
    vanilla = metrics["vanilla_correct_floor"]["unanchored_mcc"]
    projected_permuted = metrics["projected_permuted_floor"]["unanchored_mcc"]
    denominator = float(evaluation["teacher_ceiling_rgb_correlation"]) - vanilla
    gap_closed = (projected - vanilla) / max(denominator, 1e-12)
    final_validation = training["projected_correct_floor"]["history"][-1]["semantic_validation"]
    final_projection = training["projected_correct_floor"]["history"][-1]["train"][
        "projection_active"
    ]
    validity = {
        "warm_checkpoint_exact": base.sha256_file(Path(config["warm_start"]["path"]))
        == config["warm_start"]["sha256"],
        "calibration_budget_exact": calibration_audit["budget"]
        == int(config["functional_anchor"]["budget"]),
        "permutation_deranged": calibration_audit["permutation_fixed_points"] == 0,
        "matched_parameter_count": len(
            {item["parameter_count"] for item in training.values()}
        )
        == 1,
        "floors_exact": all(
            item["final_supervision_weight"] == 0.1 for item in training.values()
        ),
        "vanilla_reproduced": abs(
            vanilla - float(evaluation["expected_vanilla_correct_rgb_mcc"])
        )
        <= float(evaluation["maximum_vanilla_reproduction_deviation"]),
        "locked_before_test": lock_before_test,
        "anchors_exact": min(item["anchor_mean_direct_r2"] for item in metrics.values())
        >= float(evaluation["minimum_oracle_anchor_r2"]),
    }
    gates = {
        "projected_absolute": projected >= float(evaluation["minimum_projected_rgb_mcc"]),
        "gain_over_vanilla": projected - vanilla
        >= float(evaluation["minimum_gain_over_vanilla_correct"]),
        "gain_over_projected_permuted": projected - projected_permuted
        >= float(evaluation["minimum_gain_over_projected_permuted"]),
        "teacher_gap_fraction_closed": gap_closed
        >= float(evaluation["minimum_teacher_gap_fraction_closed"]),
        "validation_semantics": final_validation["mean_direct_abs_correlation"]
        >= float(evaluation["minimum_final_validation_direct_correlation"]),
        "projection_active": final_projection
        >= float(evaluation["minimum_final_projection_active_fraction"]),
    }
    relative_keys = [
        "gain_over_vanilla",
        "gain_over_projected_permuted",
        "teacher_gap_fraction_closed",
        "projection_active",
    ]
    if not all(validity.values()):
        verdict = "conflict_projected_physics_training_invalid_seed3407"
    elif all(gates.values()):
        verdict = "conflict_projected_physics_training_supported_seed3407"
    elif all(gates[key] for key in relative_keys):
        verdict = "conflict_projected_physics_partial_recovery_seed3407"
    else:
        verdict = "conflict_projected_physics_not_supported_seed3407"
    return {
        "verdict": verdict,
        "validity": validity,
        "gates": gates,
        "rgb_mcc": {name: item["unanchored_mcc"] for name, item in metrics.items()},
        "deltas": {
            "projected_minus_vanilla_correct": projected - vanilla,
            "projected_correct_minus_projected_permuted": projected - projected_permuted,
            "teacher_gap_fraction_closed": gap_closed,
        },
        "final_projected_validation": final_validation,
        "final_projection_active_fraction": final_projection,
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
    if list(config["training"]["conditions"]) != CONDITIONS:
        raise ValueError("unexpected condition order")
    epochs = int(args.epochs or config["training"]["epochs"])
    if not args.smoke and epochs != int(config["training"]["epochs"]):
        raise ValueError("formal v13 must use frozen epochs")
    config["training"]["epochs"] = epochs
    output_dir = (args.output_dir or Path(config["runtime"]["output_dir"])).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(config["runtime"]["num_threads"]))
    torch.set_float32_matmul_precision("high")
    device = torch.device(config["runtime"]["device"] if torch.cuda.is_available() else "cpu")
    images = np.load(Path(config["dataset"]["cache_path"]), mmap_mode="r")
    raw_latents = base.load_latents(config)
    latents, latent_mean_np, latent_std_np = base.normalize_latents(
        raw_latents, int(config["split"]["train_end"])
    )
    latent_mean = torch.from_numpy(latent_mean_np).to(device)
    latent_std = torch.from_numpy(latent_std_np).to(device)
    anchor_maps, _ = v2.build_anchor_maps(config)
    spec, calibration_audit = v11.build_calibration_spec(config)
    coefficient_np, coefficient_audit = v11.calibrate_teachers(
        images, raw_latents, spec, config
    )
    coefficients = {name: torch.from_numpy(value).to(device) for name, value in coefficient_np.items()}
    warm_path = Path(config["warm_start"]["path"])
    if base.sha256_file(warm_path) != config["warm_start"]["sha256"]:
        raise ValueError("warm checkpoint hash mismatch")
    warm_state = torch.load(warm_path, map_location="cpu", weights_only=False)["state_dict"]
    started = time.time()
    training = {}
    for name in CONDITIONS:
        training[name] = train_condition(
            v2.Condition(name, "oracle"),
            warm_state,
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
    source_names = [
        "run_conflict_projected_physics_training.py",
        "run_physics_functional_anchor_training.py",
        "run_conditioned_film_diagnostic.py",
        "run_clamped_anchor_diagnostic.py",
        "run_diagnostic.py",
    ]
    source_hashes = {
        name: base.sha256_file(Path(__file__).resolve().parent / name) for name in source_names
    }
    training_lock = {
        "status": "locked_before_test_evaluation",
        "config_sha256": base.sha256_file(config_path),
        "source_files_sha256": source_hashes,
        "warm_checkpoint_sha256": base.sha256_file(warm_path),
        "calibration": calibration_audit,
        "teacher_coefficients": coefficient_audit,
        "checkpoints": {
            name: {
                "path": item["checkpoint"],
                "sha256": item["checkpoint_sha256"],
                "final_supervision_weight": item["final_supervision_weight"],
            }
            for name, item in training.items()
        },
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
        "data": {"latent_mean": latent_mean_np, "latent_std": latent_std_np},
        "calibration": calibration_audit,
        "teacher_coefficients": coefficient_audit,
        "training": training,
        "training_lock": str(lock_path),
        "training_lock_sha256_before_test": lock_sha,
        "duration_seconds_before_test": time.time() - started,
    }
    if args.smoke:
        result = {**common, "verdict": "smoke_completed_no_test_read"}
        path = output_dir / "smoke_results.json"
    else:
        metrics = {}
        for name, item in training.items():
            model = load_model(Path(item["checkpoint"]), config, device)
            metrics[name] = v2.evaluate_model(
                model, images, latents, anchor_maps, v2.Condition(name, "oracle"), config, device
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        result = {
            **common,
            "metrics": metrics,
            "decision": decide(metrics, training, True, calibration_audit, config),
            "duration_seconds": time.time() - started,
        }
        path = output_dir / "diagnostic_results.json"
    v11.atomic_json(path, result)
    print(json.dumps({"result_path": str(path), "sha256": base.sha256_file(path)}, sort_keys=True))


if __name__ == "__main__":
    main()
