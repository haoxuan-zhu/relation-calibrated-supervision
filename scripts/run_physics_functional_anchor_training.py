"""Physics-functional semantic-anchor training diagnostic for partial-anchor CRL v11."""

from __future__ import annotations

import argparse
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
import run_conditioned_film_diagnostic as v3
import run_diagnostic as base
import run_sparse_persistent_anchor_diagnostic as v9
import run_supervised_continuation_diagnostic as v6


CONDITIONS = [
    "physics_correct_floor",
    "physics_permuted_floor",
    "point_correct_floor",
    "zero_floor",
]


def supervision_weight(condition: str, epoch_index: int, config: dict[str, Any]) -> float:
    if condition not in CONDITIONS:
        raise ValueError(condition)
    schedule = config["floor_schedule"]
    hold = int(schedule["hold_epochs"])
    decay_end = int(schedule["decay_end_epoch"])
    floor = float(
        schedule["zero_floor"] if condition == "zero_floor" else schedule["nonzero_floor"]
    )
    if epoch_index < hold:
        return float(schedule["hold_weight"])
    if epoch_index < decay_end:
        fraction = (decay_end - 1 - epoch_index) / max(decay_end - hold - 1, 1)
        return floor + (float(schedule["hold_weight"]) - floor) * fraction
    return floor


def build_calibration_spec(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    item = config["functional_anchor"]
    train_end = int(config["split"]["train_end"])
    budget = int(item["budget"])
    master = np.random.default_rng(int(item["subset_seed"])).permutation(train_end)
    subset = np.array(master[:budget], dtype=np.int64, copy=True)
    derangement = v9.make_derangement(
        budget, int(item["permutation_seed"]) + 104729 * budget
    )
    permuted = subset[derangement]
    spec = {"budget": budget, "subset": subset, "permuted": permuted}
    audit = {
        "budget": budget,
        "subset_sha256": v9.sha256_int_array(subset),
        "permuted_sha256": v9.sha256_int_array(permuted),
        "permutation_fixed_points": int(np.sum(subset == permuted)),
        "subset_rows": subset.tolist(),
        "permuted_rows": permuted.tolist(),
    }
    return spec, audit


def calibrate_teachers(
    images: np.ndarray,
    raw_latents: np.ndarray,
    spec: dict[str, Any],
    config: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    train_end = int(config["split"]["train_end"])
    means = v10.image_channel_means(images, 0, 0, train_end)
    tau = v10.malus_tau(raw_latents)
    rgb = raw_latents[..., :3].astype(np.float64) / 255.0
    coefficients = {}
    audit = {}
    for mode, rows in [("correct", spec["subset"]), ("permuted", spec["permuted"])]:
        value = v10.fit_lstsq(
            v10.full_forward_design(rgb[0, rows], tau[0, spec["subset"]]),
            means[spec["subset"]],
        )
        coefficients[mode] = value.astype(np.float32)
        fit = v10.full_forward_design(rgb[0, rows], tau[0, spec["subset"]]) @ value
        audit[mode] = {
            "coefficients": value.tolist(),
            "coefficients_sha256": hashlib.sha256(
                np.ascontiguousarray(value, dtype=np.float64).tobytes()
            ).hexdigest(),
            "calibration_mse": float(np.mean((fit - means[spec["subset"]]) ** 2)),
        }
    return coefficients, audit


def physical_pseudo_target(
    x: torch.Tensor,
    raw_angles: torch.Tensor,
    coefficients: torch.Tensor,
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
    rcond: float,
    clip: bool,
) -> torch.Tensor:
    means = x.mean(dim=(2, 3))
    tau = torch.cos(torch.deg2rad(raw_angles[:, 0] - raw_angles[:, 1])).square()
    intercept = coefficients[0]
    matrices = coefficients[1:4].unsqueeze(0) + tau[:, None, None] * coefficients[
        4:7
    ].unsqueeze(0)
    inverse = torch.linalg.pinv(matrices, rtol=rcond)
    unit_rgb = torch.bmm((means - intercept).unsqueeze(1), inverse).squeeze(1)
    if clip:
        unit_rgb = unit_rgb.clamp(0.0, 1.0)
    return (unit_rgb * 255.0 - latent_mean[:3]) / latent_std[:3]


def objective(
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
    weight: float,
    calibration_spec: dict[str, Any],
    teacher_coefficients: dict[str, torch.Tensor],
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    ccrl, metrics = v2.objective_for_batch(
        model, images, latents, anchor_maps, condition, rows, targets, config, device
    )
    anchor_indices = list(config["model"]["anchor_indices"])
    if condition.name.startswith("physics_"):
        mode = "correct" if condition.name == "physics_correct_floor" else "permuted"
        losses = []
        for environments in (np.zeros_like(rows), targets + 1):
            x = base.image_batch(images, environments, rows, device)
            anchors = torch.from_numpy(
                np.array(latents[environments, rows][:, anchor_indices], copy=True)
            ).to(device)
            raw_angles = torch.from_numpy(
                np.array(raw_latents[environments, rows][:, anchor_indices], copy=True)
            ).to(device)
            pseudo = physical_pseudo_target(
                x,
                raw_angles,
                teacher_coefficients[mode],
                latent_mean,
                latent_std,
                float(config["functional_anchor"]["pinv_rcond"]),
                bool(config["functional_anchor"]["clip_predictions_to_unit_interval"]),
            )
            losses.append(F.mse_loss(model.embedding(x, anchors), pseudo))
        semantic_loss = torch.stack(losses).mean()
    else:
        subset = calibration_spec["subset"]
        anchor_rows = v9.anchor_rows_for_batch(rows, subset)
        x = base.image_batch(images, np.zeros_like(anchor_rows), anchor_rows, device)
        anchors = torch.from_numpy(
            np.array(latents[0, anchor_rows, 3:5], copy=True)
        ).to(device)
        truth = torch.from_numpy(
            np.array(latents[0, anchor_rows, :3], copy=True)
        ).to(device)
        semantic_loss = F.mse_loss(model.embedding(x, anchors), truth)
    total = ccrl + weight * semantic_loss
    return total, {
        **metrics,
        "semantic_loss": float(semantic_loss.detach().cpu()),
        "supervision_weight": weight,
        "optimized_total": float(total.detach().cpu()),
    }


def train_condition(
    condition: v2.Condition,
    warm_state: dict[str, torch.Tensor],
    images: np.ndarray,
    latents: np.ndarray,
    raw_latents: np.ndarray,
    anchor_maps: dict[str, np.ndarray],
    config: dict[str, Any],
    device: torch.device,
    epochs: int,
    output_dir: Path,
    calibration_spec: dict[str, Any],
    teacher_coefficients: dict[str, torch.Tensor],
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
    history = []
    started = time.time()
    for epoch in range(epochs):
        weight = supervision_weight(condition.name, epoch, config)
        order = np.random.default_rng(seed + 1009 * epoch).permutation(len(rows))
        model.train()
        sums: dict[str, float] = {}
        count = 0
        for begin in range(0, len(rows), batch_size):
            indices = order[begin : begin + batch_size]
            batch_rows = rows[indices]
            batch_targets = targets[indices]
            loss, values = objective(
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
                weight,
                calibration_spec,
                teacher_coefficients,
                latent_mean,
                latent_std,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            count += len(batch_rows)
            for name, value in values.items():
                sums[name] = sums.get(name, 0.0) + len(batch_rows) * value
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
            print(json.dumps({
                "condition": condition.name,
                "epoch": epoch + 1,
                "lambda": weight,
                "validation_total": validation["total"],
                "semantic_mean_r2": semantic["mean_r2"],
            }, sort_keys=True), flush=True)
        history.append(record)
    checkpoint = output_dir / f"{condition.name}_seed{seed}.pt"
    torch.save({
        "condition": condition.name,
        "seed": seed,
        "epoch": epochs,
        "state_dict": model.state_dict(),
    }, checkpoint)
    summary = {
        "condition": condition.name,
        "final_epoch": epochs,
        "final_supervision_weight": supervision_weight(condition.name, epochs - 1, config),
        "duration_seconds": time.time() - started,
        "history": history,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": base.sha256_file(checkpoint),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


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
    physics = metrics["physics_correct_floor"]["unanchored_mcc"]
    permuted = metrics["physics_permuted_floor"]["unanchored_mcc"]
    point = metrics["point_correct_floor"]["unanchored_mcc"]
    zero = metrics["zero_floor"]["unanchored_mcc"]
    final_validation = training["physics_correct_floor"]["history"][-1][
        "semantic_validation"
    ]
    validity = {
        "warm_checkpoint_exact": base.sha256_file(Path(config["warm_start"]["path"]))
        == config["warm_start"]["sha256"],
        "calibration_budget_exact": calibration_audit["budget"]
        == int(config["functional_anchor"]["budget"]),
        "permutation_deranged": calibration_audit["permutation_fixed_points"] == 0,
        "matched_parameter_count": len(
            {value["parameter_count"] for value in training.values()}
        ) == 1,
        "floors_exact": {
            name: value["final_supervision_weight"] for name, value in training.items()
        } == {
            "physics_correct_floor": 0.1,
            "physics_permuted_floor": 0.1,
            "point_correct_floor": 0.1,
            "zero_floor": 0.0,
        },
        "point_reproduced": abs(point - float(evaluation["expected_point_rgb_mcc"]))
        <= float(evaluation["maximum_reproduction_deviation"]),
        "zero_reproduced": abs(zero - float(evaluation["expected_zero_rgb_mcc"]))
        <= float(evaluation["maximum_reproduction_deviation"]),
        "locked_before_test": lock_before_test,
        "anchors_exact": min(value["anchor_mean_direct_r2"] for value in metrics.values())
        >= float(evaluation["minimum_oracle_anchor_r2"]),
    }
    deltas = {
        "physics_minus_point_rgb_mcc": physics - point,
        "physics_minus_permuted_rgb_mcc": physics - permuted,
        "physics_minus_zero_rgb_mcc": physics - zero,
    }
    gates = {
        "physics_absolute": physics >= float(evaluation["minimum_physics_rgb_mcc"]),
        "gain_over_point": deltas["physics_minus_point_rgb_mcc"]
        >= float(evaluation["minimum_gain_over_point"]),
        "gain_over_permuted": deltas["physics_minus_permuted_rgb_mcc"]
        >= float(evaluation["minimum_gain_over_permuted_physics"]),
        "gain_over_zero": deltas["physics_minus_zero_rgb_mcc"]
        >= float(evaluation["minimum_gain_over_zero"]),
        "validation_semantics": final_validation["mean_direct_abs_correlation"]
        >= float(evaluation["minimum_final_validation_direct_correlation"]),
    }
    if not all(validity.values()):
        verdict = "physics_functional_anchor_training_invalid_seed3407"
    elif all(gates.values()):
        verdict = "physics_functional_anchor_training_supported_seed3407"
    elif any(gates.values()):
        verdict = "physics_functional_anchor_training_partial_signal_seed3407"
    else:
        verdict = "physics_functional_anchor_training_not_supported_seed3407"
    return {
        "verdict": verdict,
        "validity": validity,
        "gates": gates,
        "rgb_mcc": {
            "physics_correct": physics,
            "physics_permuted": permuted,
            "point_correct": point,
            "zero": zero,
        },
        "deltas": deltas,
        "final_physics_validation": final_validation,
    }


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(base.json_ready(value), indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ccrl-epochs", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if list(config["training"]["conditions"]) != CONDITIONS:
        raise ValueError("unexpected condition order")
    epochs = int(args.ccrl_epochs or config["training"]["epochs"])
    if not args.smoke and epochs != int(config["training"]["epochs"]):
        raise ValueError("formal v11 must use frozen epochs")
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
    spec, calibration_audit = build_calibration_spec(config)
    coefficient_np, coefficient_audit = calibrate_teachers(images, raw_latents, spec, config)
    teacher_coefficients = {
        name: torch.from_numpy(value).to(device) for name, value in coefficient_np.items()
    }
    warm_path = Path(config["warm_start"]["path"])
    if base.sha256_file(warm_path) != config["warm_start"]["sha256"]:
        raise ValueError("warm checkpoint hash mismatch")
    warm_payload = torch.load(warm_path, map_location="cpu", weights_only=False)
    warm_state = warm_payload["state_dict"]
    started = time.time()
    training: dict[str, dict[str, Any]] = {}
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
            epochs,
            output_dir,
            spec,
            teacher_coefficients,
            latent_mean,
            latent_std,
        )
    source_names = [
        "run_physics_functional_anchor_training.py",
        "audit_physics_functional_anchor.py",
        "run_sparse_persistent_anchor_diagnostic.py",
        "run_supervised_continuation_diagnostic.py",
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
        "warm_checkpoint": str(warm_path),
        "warm_checkpoint_sha256": base.sha256_file(warm_path),
        "calibration": calibration_audit,
        "teacher_coefficients": coefficient_audit,
        "checkpoints": {
            name: {
                "path": value["checkpoint"],
                "sha256": value["checkpoint_sha256"],
                "final_supervision_weight": value["final_supervision_weight"],
            }
            for name, value in training.items()
        },
        "test_evaluated": False,
    }
    lock_path = output_dir / "training_lock.json"
    atomic_json(lock_path, training_lock)
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
        "warm_checkpoint": str(warm_path),
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
        for name, summary in training.items():
            model = load_model(Path(summary["checkpoint"]), config, device)
            metrics[name] = v2.evaluate_model(
                model, images, latents, anchor_maps, v2.Condition(name, "oracle"), config, device
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        decision = decide(metrics, training, True, calibration_audit, config)
        result = {
            **common,
            "metrics": metrics,
            "decision": decision,
            "duration_seconds": time.time() - started,
        }
        path = output_dir / "diagnostic_results.json"
    atomic_json(path, result)
    print(json.dumps({"result_path": str(path), "sha256": base.sha256_file(path)}, sort_keys=True))


if __name__ == "__main__":
    main()
