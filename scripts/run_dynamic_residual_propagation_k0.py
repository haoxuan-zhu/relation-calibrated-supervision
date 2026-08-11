"""Validation-only K0 for dynamic residual-gradient propagation.

The existing raw-ridge follow-up uses a fixed teacher target for all 100
epochs.  This isolated runner instead freezes the current student at the
start of each outer block, fits the same low-dimensional ridge family to the
    student's K-row squared-error residual gradients, and performs an
    approximate proximal update.

Formal training never reads semantic validation or test targets.  A separate
readout mode refuses to run until both correct and permuted condition locks
exist, and reads observational validation only.
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

import audit_physics_functional_anchor as physics_audit
import raw_ridge_propagation as raw
import run_clamped_anchor_diagnostic as v2
import run_conditioned_film_diagnostic as v3
import run_conflict_projected_physics_from_scratch as v16
import run_conflict_projected_physics_k80_closed as v17
import run_conflict_projected_physics_training as v13
import run_diagnostic as base
import run_physics_functional_anchor_training as v11
import run_supervised_continuation_diagnostic as v6


PROTOCOL = "dynamic_residual_propagation_k0_v1"
CONDITIONS = ("correct_residual", "permuted_residual")
SOURCE_FILES = (
    "run_dynamic_residual_propagation_k0.py",
    "raw_ridge_propagation.py",
    "run_conflict_projected_physics_from_scratch.py",
    "run_conflict_projected_physics_k80_closed.py",
    "run_conflict_projected_physics_training.py",
    "run_conditioned_film_diagnostic.py",
    "run_clamped_anchor_diagnostic.py",
    "run_diagnostic.py",
)


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        base.json_ready(value), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_int_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values, dtype=np.int64)
    return hashlib.sha256(array.tobytes()).hexdigest()


def validate_config(config: dict[str, Any]) -> None:
    if config["protocol_version"] != PROTOCOL:
        raise ValueError("unexpected dynamic-residual protocol")
    if list(config["training"]["conditions"]) != list(CONDITIONS):
        raise ValueError("dynamic-residual condition registry changed")
    if int(config["initialization"]["seed"]) != 3407:
        raise ValueError("K0 is locked to seed3407")
    if int(config["training"]["seed"]) != 3407:
        raise ValueError("training seed differs from registered initialization")
    if int(config["calibration"]["budget"]) != 80:
        raise ValueError("K0 is locked to K=80")
    epochs = int(config["training"]["epochs"])
    outer = int(config["training"]["outer_rounds"])
    inner = int(config["training"]["inner_epochs"])
    if (epochs, outer, inner, outer * inner) != (100, 10, 10, 100):
        raise ValueError("formal K0 must use 10 outer x 10 inner = 100 epochs")
    if not np.isclose(float(config["training"]["residual_gradient_factor"]), 2.0):
        raise ValueError("implemented mean-MSE gradient factor must remain 2")
    if not np.isclose(float(config["training"]["proximal_step_size"]), 0.5):
        raise ValueError("MSE proximal step is locked to 1/2")
    if not np.isclose(float(config["calibration"]["raw_ridge_alpha"]), 0.1):
        raise ValueError("K80 residual ridge alpha must reuse the locked 0.1")
    if config["evaluation"]["direct_teacher_comparator"] != (
        "locked_observational_validation_preflight"
    ):
        raise ValueError("direct teacher comparator must use the locked preflight")
    if int(config["implementation"]["expected_parameter_count"]) != 16902384:
        raise ValueError("unexpected model parameter count contract")


def build_subset(config: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    calibration = config["calibration"]
    budget = int(calibration["budget"])
    train_end = int(config["split"]["train_end"])
    master = np.random.default_rng(
        int(calibration["subset_seed"])
    ).permutation(train_end)
    subset = np.ascontiguousarray(master[:budget], dtype=np.int64)
    actual = sha256_int_array(subset)
    expected = str(calibration["subset_sha256"])
    if actual != expected:
        raise ValueError(f"registered K80 subset hash mismatch: {actual} != {expected}")
    return subset, {
        "budget": budget,
        "subset_seed": int(calibration["subset_seed"]),
        "subset_sha256": actual,
        "expected_subset_sha256": expected,
        "subset_rows": subset.tolist(),
        "unique_rows": int(np.unique(subset).size),
    }


def build_derangement(config: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    spec = config["permutation_control"]
    budget = int(config["calibration"]["budget"])
    identity = np.arange(budget, dtype=np.int64)
    generator = np.random.default_rng(int(spec["seed"]))
    attempts = 0
    while True:
        attempts += 1
        permutation = np.ascontiguousarray(
            generator.permutation(budget), dtype=np.int64
        )
        if np.all(permutation != identity):
            break
    digest = sha256_int_array(permutation)
    checks = {
        "attempts_exact": attempts == int(spec["expected_attempts"]),
        "sha256_exact": digest == str(spec["expected_sha256"]),
        "deranged": bool(np.all(permutation != identity)),
        "bijective": bool(np.array_equal(np.sort(permutation), identity)),
    }
    if not all(checks.values()):
        raise ValueError(f"registered derangement mismatch: {checks}")
    return permutation, {
        "seed": int(spec["seed"]),
        "attempts": attempts,
        "sha256": digest,
        "fixed_points": int(np.sum(permutation == identity)),
        "indices": permutation.tolist(),
        "checks": checks,
    }


def make_model(config: dict[str, Any], device: torch.device) -> v3.FilmConditionedModel:
    model_cfg = config["model"]
    return v3.FilmConditionedModel(
        int(model_cfg["latent_dim"]),
        list(model_cfg["learned_indices"]),
        list(model_cfg["anchor_indices"]),
        int(model_cfg["hidden_channels"]),
        int(model_cfg["conv_layers"]),
    ).to(device)


@torch.no_grad()
def predict_calibration_rows(
    model: torch.nn.Module,
    images: np.ndarray,
    latents: np.ndarray,
    subset: np.ndarray,
    config: dict[str, Any],
    device: torch.device,
) -> np.ndarray:
    model.eval()
    environments = np.zeros_like(subset)
    x = base.image_batch(images, environments, subset, device)
    anchors = torch.from_numpy(
        np.array(
            latents[0, subset][:, list(config["model"]["anchor_indices"])],
            dtype=np.float32,
            copy=True,
        )
    ).to(device)
    prediction = model.embedding(x, anchors)
    return prediction.detach().cpu().numpy().astype(np.float64)


def calibration_features(
    images: np.ndarray,
    raw_latents: np.ndarray,
    subset: np.ndarray,
    train_end: int,
) -> np.ndarray:
    means = physics_audit.image_channel_means(images, 0, 0, train_end)[subset]
    tau = physics_audit.malus_tau(raw_latents)[0, :train_end][subset]
    return raw.raw_features(means, tau)


def fit_residual_teacher(
    model: torch.nn.Module,
    condition: str,
    images: np.ndarray,
    latents: np.ndarray,
    raw_latents: np.ndarray,
    subset: np.ndarray,
    derangement: np.ndarray,
    config: dict[str, Any],
    device: torch.device,
    outer_round: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if condition not in CONDITIONS:
        raise ValueError(condition)
    prediction = predict_calibration_rows(
        model, images, latents, subset, config, device
    )
    targets = np.asarray(latents[0, subset, :3], dtype=np.float64)
    label_indices = np.arange(len(subset), dtype=np.int64)
    if condition == "permuted_residual":
        targets = targets[derangement]
        label_indices = derangement
    factor = float(config["training"]["residual_gradient_factor"])
    output_gradient = factor * (prediction - targets)
    features = calibration_features(
        images, raw_latents, subset, int(config["split"]["train_end"])
    )
    teacher = raw._fit_standardized_ridge(
        features,
        output_gradient,
        float(config["calibration"]["raw_ridge_alpha"]),
    )
    fitted = raw.predict_raw_ridge(teacher, features, clip=False)
    coefficient_bytes = np.ascontiguousarray(
        teacher["coefficients"], dtype=np.float64
    ).tobytes()
    audit = {
        "outer_round": int(outer_round),
        "condition": condition,
        "family": "fixed_alpha_raw_ridge_predicting_current_mse_gradient",
        "alpha": float(teacher["alpha"]),
        "gradient_factor": factor,
        "design_shape": [int(features.shape[0]), int(features.shape[1] + 1)],
        "target_shape": list(output_gradient.shape),
        "target_gradient_mean_l2": float(
            np.mean(np.linalg.norm(output_gradient, axis=1))
        ),
        "teacher_fitted_mean_l2": float(np.mean(np.linalg.norm(fitted, axis=1))),
        "calibration_gradient_mse": float(
            np.mean((fitted - output_gradient) ** 2)
        ),
        "student_calibration_mse": float(np.mean((prediction - targets) ** 2)),
        "coefficients_sha256": hashlib.sha256(coefficient_bytes).hexdigest(),
        "label_index_sha256": sha256_int_array(label_indices),
        "finite": bool(
            np.all(np.isfinite(prediction))
            and np.all(np.isfinite(output_gradient))
            and np.all(np.isfinite(fitted))
            and np.all(np.isfinite(np.asarray(teacher["coefficients"])))
        ),
    }
    if not audit["finite"]:
        raise FloatingPointError("residual teacher fit produced non-finite values")
    return teacher, audit


def torch_ridge_prediction(
    x: torch.Tensor,
    raw_angles: torch.Tensor,
    teacher: dict[str, torch.Tensor],
) -> torch.Tensor:
    means = x.mean(dim=(2, 3))
    tau = torch.cos(
        torch.deg2rad(raw_angles[:, 0] - raw_angles[:, 1])
    ).square()
    features = torch.cat([means, tau[:, None]], dim=1)
    standardized = (
        features - teacher["feature_mean"]
    ) / teacher["feature_scale"]
    design = torch.cat(
        [
            torch.ones(
                (features.shape[0], 1), dtype=x.dtype, device=x.device
            ),
            standardized,
        ],
        dim=1,
    )
    return design @ teacher["coefficients"]


def proximal_loss_for_batch(
    model: torch.nn.Module,
    snapshot: torch.nn.Module,
    images: np.ndarray,
    latents: np.ndarray,
    raw_latents: np.ndarray,
    rows: np.ndarray,
    targets: np.ndarray,
    config: dict[str, Any],
    device: torch.device,
    teacher: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    anchor_indices = list(config["model"]["anchor_indices"])
    step_size = float(config["training"]["proximal_step_size"])
    losses: list[torch.Tensor] = []
    target_shifts: list[torch.Tensor] = []
    for environments in (np.zeros_like(rows), targets + 1):
        x = base.image_batch(images, environments, rows, device)
        anchors = torch.from_numpy(
            np.array(
                latents[environments, rows][:, anchor_indices],
                dtype=np.float32,
                copy=True,
            )
        ).to(device)
        raw_angles = torch.from_numpy(
            np.array(
                raw_latents[environments, rows][:, anchor_indices],
                dtype=np.float32,
                copy=True,
            )
        ).to(device)
        with torch.no_grad():
            previous = snapshot.embedding(x, anchors)
            estimated_gradient = torch_ridge_prediction(
                x, raw_angles, teacher
            )
            proximal_target = previous - step_size * estimated_gradient
        current = model.embedding(x, anchors)
        losses.append(F.mse_loss(current, proximal_target))
        target_shifts.append(
            torch.linalg.vector_norm(
                proximal_target - previous, dim=1
            ).mean()
        )
    loss = torch.stack(losses).mean()
    return loss, {
        "proximal_target_shift_mean_l2": float(
            torch.stack(target_shifts).mean().detach().cpu()
        ),
        "proximal_loss": float(loss.detach().cpu()),
    }


def numeric_history_is_finite(history: list[dict[str, Any]]) -> bool:
    def walk(value: Any) -> bool:
        if isinstance(value, dict):
            return all(walk(item) for item in value.values())
        if isinstance(value, list):
            return all(walk(item) for item in value)
        if isinstance(value, (int, bool, str)) or value is None:
            return True
        if isinstance(value, float):
            return bool(np.isfinite(value))
        return True

    return walk(history)


def train_condition(
    condition: str,
    config: dict[str, Any],
    images: np.ndarray,
    latents: np.ndarray,
    raw_latents: np.ndarray,
    anchor_maps: dict[str, np.ndarray],
    subset: np.ndarray,
    derangement: np.ndarray,
    initial_state: dict[str, torch.Tensor],
    initial_state_hash: str,
    device: torch.device,
    output_dir: Path,
    epochs: int,
    smoke: bool,
) -> dict[str, Any]:
    if condition not in CONDITIONS:
        raise ValueError(condition)
    training = config["training"]
    seed = int(training["seed"])
    base.set_seed(seed)
    model = make_model(config, device)
    model.load_state_dict(initial_state)
    if v6.sha256_state_dict(model.state_dict()) != initial_state_hash:
        raise ValueError("condition did not load registered initial state")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(training["learning_rate"])
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=float(training["scheduler_factor"]),
        patience=int(training["scheduler_patience"]),
    )
    rows, intervention_targets = base.make_pairs(
        list(training["targets"]), 0, int(config["split"]["train_end"])
    )
    batch_size = int(training["batch_size"])
    interval = int(training["validation_interval"])
    if smoke:
        outer_rounds = 1
        inner_epochs = epochs
    else:
        outer_rounds = int(training["outer_rounds"])
        inner_epochs = int(training["inner_epochs"])
        if epochs != outer_rounds * inner_epochs:
            raise ValueError("formal outer/inner schedule does not equal epochs")

    history: list[dict[str, Any]] = []
    teacher_audits: list[dict[str, Any]] = []
    global_epoch = 0
    started = time.time()
    condition_token = v2.Condition(condition, "oracle")

    for outer_index in range(outer_rounds):
        snapshot = make_model(config, device)
        snapshot.load_state_dict(copy.deepcopy(model.state_dict()))
        snapshot.eval()
        for parameter in snapshot.parameters():
            parameter.requires_grad_(False)

        teacher_np, teacher_audit = fit_residual_teacher(
            snapshot,
            condition,
            images,
            latents,
            raw_latents,
            subset,
            derangement,
            config,
            device,
            outer_index + 1,
        )
        teacher_audits.append(teacher_audit)
        teacher = {
            key: torch.from_numpy(
                np.asarray(teacher_np[key], dtype=np.float32)
            ).to(device)
            for key in ("feature_mean", "feature_scale", "coefficients")
        }

        for _ in range(inner_epochs):
            weight = v13.supervision_weight(global_epoch, config)
            order = np.random.default_rng(
                seed + 1009 * global_epoch
            ).permutation(len(rows))
            model.train()
            sums: dict[str, float] = {}
            count = 0
            for begin in range(0, len(rows), batch_size):
                indices = order[begin : begin + batch_size]
                batch_rows = rows[indices]
                batch_targets = intervention_targets[indices]
                ccrl, values = v2.objective_for_batch(
                    model,
                    images,
                    latents,
                    anchor_maps,
                    condition_token,
                    batch_rows,
                    batch_targets,
                    config,
                    device,
                )
                proximal, proximal_values = proximal_loss_for_batch(
                    model,
                    snapshot,
                    images,
                    latents,
                    raw_latents,
                    batch_rows,
                    batch_targets,
                    config,
                    device,
                    teacher,
                )
                total = ccrl + weight * proximal
                optimizer.zero_grad(set_to_none=True)
                total.backward()
                optimizer.step()

                size = len(batch_rows)
                count += size
                batch_values = {
                    **values,
                    **proximal_values,
                    "supervision_weight": float(weight),
                    "optimized_total": float(total.detach().cpu()),
                }
                for name, value in batch_values.items():
                    if np.isfinite(value):
                        sums[name] = sums.get(name, 0.0) + size * float(value)

            global_epoch += 1
            record: dict[str, Any] = {
                "epoch": global_epoch,
                "outer_round": outer_index + 1,
                "inner_epoch": (global_epoch - 1) % inner_epochs + 1,
                "train": base.average_metrics(sums, count),
                "supervision_weight": float(weight),
                "residual_teacher_coefficients_sha256": teacher_audit[
                    "coefficients_sha256"
                ],
            }
            if (
                (global_epoch - 1) % interval == 0
                or global_epoch == epochs
            ):
                validation = v2.validate(
                    model,
                    images,
                    latents,
                    anchor_maps,
                    condition_token,
                    config,
                    device,
                )
                scheduler.step(validation["total"])
                record.update(
                    validation=validation,
                    learning_rate=float(optimizer.param_groups[0]["lr"]),
                )
                print(
                    json.dumps(
                        {
                            "condition": condition,
                            "epoch": global_epoch,
                            "outer_round": outer_index + 1,
                            "lambda": weight,
                            "validation_total": validation["total"],
                            "residual_gradient_l2": teacher_audit[
                                "target_gradient_mean_l2"
                            ],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            history.append(record)

        del snapshot
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if global_epoch != epochs:
        raise RuntimeError("completed epoch count differs from registered budget")
    checkpoint = output_dir / f"{condition}_seed{seed}.pt"
    torch.save(
        {
            "protocol_version": PROTOCOL,
            "condition": condition,
            "seed": seed,
            "epoch": epochs,
            "initial_state_sha256": initial_state_hash,
            "state_dict": model.state_dict(),
        },
        checkpoint,
    )
    result = {
        "condition": condition,
        "initial_state_sha256": initial_state_hash,
        "final_epoch": epochs,
        "outer_rounds": outer_rounds,
        "inner_epochs": inner_epochs,
        "final_supervision_weight": v13.supervision_weight(
            epochs - 1, config
        ),
        "duration_seconds": time.time() - started,
        "history": history,
        "residual_teacher_audits": teacher_audits,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": base.sha256_file(checkpoint),
        "parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "semantic_truth_read_during_training": False,
        "test_evaluated": False,
    }
    if not numeric_history_is_finite(history):
        raise FloatingPointError("training history contains non-finite values")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def source_hashes() -> dict[str, str]:
    script_dir = Path(__file__).resolve().parent
    return {
        name: base.sha256_file(script_dir / name) for name in SOURCE_FILES
    }


def baseline_main(args: argparse.Namespace, config: dict[str, Any]) -> None:
    """Lock the direct raw-ridge comparator on the same observational split.

    The historical teacher audit pools all six environments, whereas the
    representation readout is observational only.  This preflight is kept
    separate from candidate training so that the matched comparator is fixed
    before either candidate checkpoint is semantically evaluated.
    """
    if args.condition is not None or args.epochs is not None:
        raise ValueError("baseline accepts neither condition nor epochs")
    root = Path(args.output_root or config["runtime"]["output_root"]).resolve()
    output_dir = root / "preflight"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "direct_teacher_baseline.json"
    lock_path = output_dir / "baseline_lock.json"
    if output_path.exists() or lock_path.exists():
        raise FileExistsError("refusing to overwrite direct-teacher preflight")

    images = np.load(Path(config["dataset"]["cache_path"]), mmap_mode="r")
    raw_latents = base.load_latents(config).astype(np.float64)
    subset, subset_audit = build_subset(config)
    train_end = int(config["split"]["train_end"])
    validation_end = int(config["split"]["validation_end"])
    train_features = calibration_features(
        images, raw_latents, subset, train_end
    )
    train_targets = raw_latents[0, subset, :3] / 255.0
    teacher = raw._fit_standardized_ridge(
        train_features,
        train_targets,
        float(config["calibration"]["raw_ridge_alpha"]),
    )
    validation_rows = np.arange(train_end, validation_end, dtype=np.int64)
    validation_features = calibration_features(
        images, raw_latents, validation_rows, validation_end
    )
    prediction = raw.predict_raw_ridge(
        teacher,
        validation_features,
        clip=bool(config["calibration"]["clip_predictions_to_unit_interval"]),
    )
    truth = raw_latents[0, train_end:validation_end, :3] / 255.0
    metrics = v6.regression_metrics(prediction, truth)
    coefficient_sha = hashlib.sha256(
        np.ascontiguousarray(teacher["coefficients"], dtype=np.float64).tobytes()
    ).hexdigest()
    validity = {
        "subset_hash_exact": subset_audit["subset_sha256"]
        == config["calibration"]["subset_sha256"],
        "subset_unique_exact": subset_audit["unique_rows"] == 80,
        "validation_rows_exact": len(validation_rows) == 1000,
        "teacher_finite": bool(
            np.all(np.isfinite(prediction))
            and np.all(np.isfinite(np.asarray(teacher["coefficients"])))
        ),
        "metrics_finite": numeric_history_is_finite([metrics]),
        "test_not_read": True,
    }
    if not all(validity.values()):
        raise RuntimeError(f"direct-teacher preflight invalid: {validity}")
    hashes = source_hashes()
    lock = {
        "status": "locked_observational_validation_comparator_before_candidate_readout",
        "protocol_version": PROTOCOL,
        "config_sha256": base.sha256_file(args.config.resolve()),
        "source_files_sha256": hashes,
        "subset_sha256": subset_audit["subset_sha256"],
        "environment": "obs",
        "split": [train_end, validation_end],
        "teacher_alpha": float(teacher["alpha"]),
        "teacher_coefficients_sha256": coefficient_sha,
        "metrics": metrics,
        "validity": validity,
        "test_evaluated": False,
    }
    v11.atomic_json(lock_path, lock)
    payload = {
        "protocol_version": PROTOCOL,
        "mode": "matched_direct_teacher_baseline_preflight",
        "config_path": str(args.config.resolve()),
        "config_sha256": base.sha256_file(args.config.resolve()),
        "subset": subset_audit,
        "metrics": metrics,
        "teacher": {
            "alpha": float(teacher["alpha"]),
            "coefficients_sha256": coefficient_sha,
            "feature_mean": teacher["feature_mean"],
            "feature_scale": teacher["feature_scale"],
            "coefficients": teacher["coefficients"],
        },
        "validity": validity,
        "baseline_lock": str(lock_path),
        "baseline_lock_sha256": base.sha256_file(lock_path),
        "semantic_validation_evaluated": True,
        "test_evaluated": False,
    }
    v11.atomic_json(output_path, payload)
    print(
        json.dumps(
            {
                "result_path": str(output_path),
                "sha256": base.sha256_file(output_path),
                "correlation": metrics["mean_direct_abs_correlation"],
                "r2": metrics["mean_r2"],
            },
            sort_keys=True,
        )
    )


def train_main(args: argparse.Namespace, config: dict[str, Any]) -> None:
    if args.condition not in CONDITIONS:
        raise ValueError("train/smoke requires a registered condition")
    formal_epochs = int(config["training"]["epochs"])
    if args.mode == "train":
        epochs = int(args.epochs or formal_epochs)
        if epochs != formal_epochs:
            raise ValueError("formal train must use the frozen 100 epochs")
        smoke = False
    else:
        epochs = int(args.epochs or 2)
        if epochs != 2:
            raise ValueError("smoke is frozen to two epochs")
        smoke = True

    root = Path(args.output_root or config["runtime"]["output_root"]).resolve()
    output_dir = root / ("smoke" if smoke else "formal") / args.condition
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / (
        "smoke_results.json" if smoke else "training_results.json"
    )
    if result_path.exists():
        raise FileExistsError(f"refusing to overwrite existing result: {result_path}")

    torch.set_num_threads(int(config["runtime"]["num_threads"]))
    torch.set_float32_matmul_precision("high")
    device = torch.device(
        config["runtime"]["device"] if torch.cuda.is_available() else "cpu"
    )
    images = np.load(Path(config["dataset"]["cache_path"]), mmap_mode="r")
    raw_latents = base.load_latents(config)
    subset, subset_audit = build_subset(config)
    derangement, permutation_audit = build_derangement(config)
    latents, latent_mean_np, latent_std_np = v17.normalize_with_k80_rgb_statistics(
        raw_latents, int(config["split"]["train_end"]), subset
    )
    anchor_maps, anchor_map_audit = v2.build_anchor_maps(config)
    initial_state, initial_state_hash = v16.make_initial_state(config, device)
    expected_initial = str(
        config["initialization"]["expected_state_dict_sha256"]
    )
    if initial_state_hash != expected_initial:
        raise ValueError(
            f"initial state mismatch: {initial_state_hash} != {expected_initial}"
        )
    initial_path = output_dir / "initial_state_seed3407.pt"
    torch.save(
        {
            "seed": 3407,
            "state_dict_sha256": initial_state_hash,
            "state_dict": copy.deepcopy(initial_state),
        },
        initial_path,
    )

    started = time.time()
    training = train_condition(
        args.condition,
        config,
        images,
        latents,
        raw_latents,
        anchor_maps,
        subset,
        derangement,
        initial_state,
        initial_state_hash,
        device,
        output_dir,
        epochs,
        smoke,
    )
    hashes = source_hashes()
    checks = {
        "subset_hash_exact": subset_audit["subset_sha256"]
        == config["calibration"]["subset_sha256"],
        "subset_unique_exact": subset_audit["unique_rows"] == 80,
        "permutation_exact": all(permutation_audit["checks"].values()),
        "initial_state_exact": initial_state_hash == expected_initial,
        "parameter_count_exact": training["parameter_count"]
        == int(config["implementation"]["expected_parameter_count"]),
        "epoch_budget_exact": training["final_epoch"] == epochs,
        "floor_exact": bool(
            np.isclose(
                training["final_supervision_weight"],
                v13.supervision_weight(epochs - 1, config),
            )
        ),
        "residual_teachers_finite": all(
            item["finite"] for item in training["residual_teacher_audits"]
        ),
        "training_history_finite": numeric_history_is_finite(training["history"]),
        "semantic_truth_not_read": not training[
            "semantic_truth_read_during_training"
        ],
        "test_not_read": not training["test_evaluated"],
    }
    if not all(checks.values()):
        raise RuntimeError(f"training validity failed: {checks}")
    lock = {
        "status": "locked_before_semantic_validation_readout",
        "mode": "smoke" if smoke else "formal",
        "protocol_version": PROTOCOL,
        "condition": args.condition,
        "config_path": str(args.config.resolve()),
        "config_sha256": base.sha256_file(args.config.resolve()),
        "derived_config_sha256": canonical_sha256(config),
        "source_files_sha256": hashes,
        "subset": subset_audit,
        "permutation": permutation_audit,
        "normalization": {
            "latent_mean": latent_mean_np,
            "latent_std": latent_std_np,
        },
        "anchor_map_audit": anchor_map_audit,
        "initial_state": {
            "path": str(initial_path),
            "file_sha256": base.sha256_file(initial_path),
            "state_dict_sha256": initial_state_hash,
        },
        "checkpoint": {
            "path": training["checkpoint"],
            "sha256": training["checkpoint_sha256"],
            "epoch": training["final_epoch"],
        },
        "validity": checks,
        "semantic_validation_evaluated": False,
        "test_evaluated": False,
    }
    lock_path = output_dir / "training_lock.json"
    v11.atomic_json(lock_path, lock)
    payload = {
        "protocol_version": PROTOCOL,
        "mode": "smoke_train" if smoke else "formal_train",
        "condition": args.condition,
        "config": config,
        "config_sha256": base.sha256_file(args.config.resolve()),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None,
        },
        "subset": subset_audit,
        "permutation": permutation_audit,
        "normalization": lock["normalization"],
        "training": training,
        "training_lock": str(lock_path),
        "training_lock_sha256": base.sha256_file(lock_path),
        "duration_seconds": time.time() - started,
        "semantic_validation_evaluated": False,
        "test_evaluated": False,
        "verdict": "smoke_completed_no_semantic_or_test_read"
        if smoke
        else "formal_training_locked_awaiting_joint_validation_readout",
    }
    v11.atomic_json(result_path, payload)
    print(
        json.dumps(
            {
                "result_path": str(result_path),
                "sha256": base.sha256_file(result_path),
                "lock_path": str(lock_path),
                "lock_sha256": base.sha256_file(lock_path),
            },
            sort_keys=True,
        )
    )


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_locked_model(
    lock: dict[str, Any], config: dict[str, Any], device: torch.device
) -> torch.nn.Module:
    checkpoint_path = Path(lock["checkpoint"]["path"])
    if base.sha256_file(checkpoint_path) != lock["checkpoint"]["sha256"]:
        raise ValueError("checkpoint hash mismatch during readout")
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    if checkpoint["condition"] != lock["condition"]:
        raise ValueError("checkpoint condition mismatch")
    if int(checkpoint["epoch"]) != 100:
        raise ValueError("readout requires frozen epoch100 checkpoint")
    model = make_model(config, device)
    model.load_state_dict(checkpoint["state_dict"])
    return model


def gap_fraction(value: float, start: float, target: float) -> float:
    return (value - start) / max(target - start, 1e-12)


def decide_readout(
    metrics: dict[str, dict[str, Any]],
    direct_teacher: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    evaluation = config["evaluation"]
    correct = metrics["correct_residual"]
    permuted = metrics["permuted_residual"]
    static_corr = float(evaluation["static_raw_validation_correlation"])
    static_r2 = float(evaluation["static_raw_validation_r2"])
    teacher_corr = float(direct_teacher["mean_direct_abs_correlation"])
    teacher_r2 = float(direct_teacher["mean_r2"])
    correct_corr = float(correct["mean_direct_abs_correlation"])
    correct_r2 = float(correct["mean_r2"])
    permuted_corr = float(permuted["mean_direct_abs_correlation"])
    permuted_r2 = float(permuted["mean_r2"])
    direction = {
        "correlation_above_static": correct_corr > static_corr,
        "r2_above_static": correct_r2 > static_r2,
        "correlation_above_permuted": correct_corr > permuted_corr,
        "r2_above_permuted": correct_r2 > permuted_r2,
    }
    if all(direction.values()):
        verdict = "dynamic_transfer_supported"
    elif (
        not direction["correlation_above_static"]
        and not direction["r2_above_static"]
    ) or (
        not direction["correlation_above_permuted"]
        and not direction["r2_above_permuted"]
    ):
        verdict = "dynamic_transfer_not_supported"
    else:
        verdict = "dynamic_transfer_mixed"
    fractions = {
        "correlation": gap_fraction(correct_corr, static_corr, teacher_corr),
        "r2": gap_fraction(correct_r2, static_r2, teacher_r2),
    }
    soft_threshold = float(evaluation["strong_gap_reduction_fraction"])
    effect_level = (
        "solver_matched"
        if correct_corr >= teacher_corr and correct_r2 >= teacher_r2
        else "strong_gap_reduction"
        if min(fractions.values()) >= soft_threshold
        else "limited_gap_reduction"
    )
    return {
        "verdict": verdict,
        "direction_checks": direction,
        "effect_level": effect_level,
        "strong_gap_reduction_is_soft_threshold": True,
        "gap_closure_fraction": fractions,
        "deltas": {
            "correct_minus_static_correlation": correct_corr - static_corr,
            "correct_minus_static_r2": correct_r2 - static_r2,
            "correct_minus_permuted_correlation": correct_corr - permuted_corr,
            "correct_minus_permuted_r2": correct_r2 - permuted_r2,
            "direct_teacher_minus_correct_correlation": teacher_corr - correct_corr,
            "direct_teacher_minus_correct_r2": teacher_r2 - correct_r2,
        },
        "test_evaluated": False,
    }


def readout_main(args: argparse.Namespace, config: dict[str, Any]) -> None:
    if args.condition is not None or args.epochs is not None:
        raise ValueError("readout accepts neither condition nor epochs")
    root = Path(args.output_root or config["runtime"]["output_root"]).resolve()
    formal_root = root / "formal"
    baseline_lock_path = root / "preflight" / "baseline_lock.json"
    baseline_lock = read_json(baseline_lock_path)
    baseline_checks = {
        "status": baseline_lock["status"]
        == "locked_observational_validation_comparator_before_candidate_readout",
        "protocol": baseline_lock["protocol_version"] == PROTOCOL,
        "config": baseline_lock["config_sha256"]
        == base.sha256_file(args.config.resolve()),
        "environment": baseline_lock["environment"] == "obs",
        "split": baseline_lock["split"]
        == [int(config["split"]["train_end"]), int(config["split"]["validation_end"])],
        "subset": baseline_lock["subset_sha256"]
        == config["calibration"]["subset_sha256"],
        "all_validity": all(baseline_lock["validity"].values()),
        "test_unread": baseline_lock["test_evaluated"] is False,
    }
    if not all(baseline_checks.values()):
        raise ValueError(f"invalid direct-teacher baseline lock: {baseline_checks}")
    locks: dict[str, dict[str, Any]] = {}
    lock_hashes: dict[str, str] = {}
    for condition in CONDITIONS:
        lock_path = formal_root / condition / "training_lock.json"
        lock = read_json(lock_path)
        checks = {
            "formal": lock["mode"] == "formal",
            "status": lock["status"]
            == "locked_before_semantic_validation_readout",
            "protocol": lock["protocol_version"] == PROTOCOL,
            "condition": lock["condition"] == condition,
            "config": lock["config_sha256"]
            == base.sha256_file(args.config.resolve()),
            "all_training_validity": all(lock["validity"].values()),
            "semantic_unread": lock["semantic_validation_evaluated"] is False,
            "test_unread": lock["test_evaluated"] is False,
        }
        if not all(checks.values()):
            raise ValueError(f"invalid lock for {condition}: {checks}")
        locks[condition] = lock
        lock_hashes[condition] = base.sha256_file(lock_path)
    shared = {
        "initial_state": len(
            {
                lock["initial_state"]["state_dict_sha256"]
                for lock in locks.values()
            }
        )
        == 1,
        "subset": len(
            {lock["subset"]["subset_sha256"] for lock in locks.values()}
        )
        == 1,
        "permutation": len(
            {lock["permutation"]["sha256"] for lock in locks.values()}
        )
        == 1,
        "source": len(
            {canonical_sha256(lock["source_files_sha256"]) for lock in locks.values()}
        )
        == 1,
        "baseline_source": all(
            lock["source_files_sha256"] == baseline_lock["source_files_sha256"]
            for lock in locks.values()
        ),
    }
    if not all(shared.values()):
        raise ValueError(f"condition locks are not matched: {shared}")

    torch.set_num_threads(int(config["runtime"]["num_threads"]))
    torch.set_float32_matmul_precision("high")
    device = torch.device(
        config["runtime"]["device"] if torch.cuda.is_available() else "cpu"
    )
    images = np.load(Path(config["dataset"]["cache_path"]), mmap_mode="r")
    raw_latents = base.load_latents(config)
    subset, _ = build_subset(config)
    latents, _, _ = v17.normalize_with_k80_rgb_statistics(
        raw_latents, int(config["split"]["train_end"]), subset
    )
    train_end = int(config["split"]["train_end"])
    validation_end = int(config["split"]["validation_end"])
    batch_size = int(config["training"]["batch_size"])
    metrics: dict[str, dict[str, Any]] = {}
    for condition in CONDITIONS:
        model = load_locked_model(locks[condition], config, device)
        prediction = v6.predict_rgb(
            model,
            images,
            latents,
            train_end,
            validation_end,
            batch_size,
            device,
        )
        metrics[condition] = v6.regression_metrics(
            prediction, latents[0, train_end:validation_end, :3]
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    result = {
        "protocol_version": PROTOCOL,
        "mode": "joint_validation_only_readout",
        "config_path": str(args.config.resolve()),
        "config_sha256": base.sha256_file(args.config.resolve()),
        "training_locks": lock_hashes,
        "baseline_lock": {
            "path": str(baseline_lock_path),
            "sha256": base.sha256_file(baseline_lock_path),
            "checks": baseline_checks,
            "metrics": baseline_lock["metrics"],
        },
        "matched_lock_checks": shared,
        "metrics": metrics,
        "decision": decide_readout(metrics, baseline_lock["metrics"], config),
        "semantic_validation_evaluated": True,
        "test_evaluated": False,
    }
    output_path = formal_root / "validation_readout.json"
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite readout: {output_path}")
    v11.atomic_json(output_path, result)
    print(
        json.dumps(
            {
                "result_path": str(output_path),
                "sha256": base.sha256_file(output_path),
                "verdict": result["decision"]["verdict"],
            },
            sort_keys=True,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("baseline", "smoke", "train", "readout"), required=True
    )
    parser.add_argument("--condition", choices=CONDITIONS)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.resolve().read_text(encoding="utf-8"))
    validate_config(config)
    if args.mode == "baseline":
        baseline_main(args, config)
    elif args.mode == "readout":
        readout_main(args, config)
    else:
        train_main(args, config)


if __name__ == "__main__":
    main()
