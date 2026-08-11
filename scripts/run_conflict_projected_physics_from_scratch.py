"""Random-initialization audit for conflict-projected functional supervision (v16)."""

from __future__ import annotations

import argparse
import copy
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

import run_clamped_anchor_diagnostic as v2
import run_conditioned_film_diagnostic as v3
import run_conflict_projected_physics_training as v13
import run_diagnostic as base
import run_physics_functional_anchor_training as v11
import run_supervised_continuation_diagnostic as v6


CONDITIONS = list(v13.CONDITIONS)


def make_initial_state(
    config: dict[str, Any], device: torch.device
) -> tuple[dict[str, torch.Tensor], str]:
    model_cfg = config["model"]
    seed = int(config["training"]["seed"])
    base.set_seed(seed)
    model = v3.FilmConditionedModel(
        int(model_cfg["latent_dim"]),
        list(model_cfg["learned_indices"]),
        list(model_cfg["anchor_indices"]),
        int(model_cfg["hidden_channels"]),
        int(model_cfg["conv_layers"]),
    ).to(device)
    state = copy.deepcopy(model.state_dict())
    state_hash = v6.sha256_state_dict(state)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return state, state_hash


def train_condition(
    condition: v2.Condition,
    initial_state: dict[str, torch.Tensor],
    initial_state_hash: str,
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
    model.load_state_dict(initial_state)
    loaded_hash = v6.sha256_state_dict(model.state_dict())
    if loaded_hash != initial_state_hash:
        raise ValueError("condition did not load the exact registered initial state")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(training_cfg["learning_rate"])
    )
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
    history: list[dict[str, Any]] = []
    started = time.time()

    for epoch in range(epochs):
        weight = v13.supervision_weight(epoch, config)
        order = np.random.default_rng(seed + 1009 * epoch).permutation(len(rows))
        model.train()
        sums: dict[str, float] = {}
        count = 0
        for begin in range(0, len(rows), batch_size):
            indices = order[begin : begin + batch_size]
            batch_rows = rows[indices]
            batch_targets = targets[indices]
            ccrl, physics, values = v13.loss_components(
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
                projection = v13.projected_backward(
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
            # This objective uses images, environment IDs and the authorized angle anchors.
            # RGB semantic truth is deliberately not read before the training lock.
            validation = v2.validate(
                model, images, latents, anchor_maps, condition, config, device
            )
            scheduler.step(validation["total"])
            record.update(
                validation=validation,
                learning_rate=float(optimizer.param_groups[0]["lr"]),
            )
            print(
                json.dumps(
                    {
                        "condition": condition.name,
                        "epoch": epoch + 1,
                        "lambda": weight,
                        "validation_total": validation["total"],
                        "projection_active_fraction": record["train"].get(
                            "projection_active"
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        history.append(record)

    checkpoint = output_dir / f"{condition.name}_seed{seed}.pt"
    torch.save(
        {
            "condition": condition.name,
            "seed": seed,
            "epoch": epochs,
            "initial_state_sha256": initial_state_hash,
            "state_dict": model.state_dict(),
        },
        checkpoint,
    )
    result = {
        "condition": condition.name,
        "initial_state_sha256": initial_state_hash,
        "final_epoch": epochs,
        "final_supervision_weight": v13.supervision_weight(epochs - 1, config),
        "duration_seconds": time.time() - started,
        "history": history,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": base.sha256_file(checkpoint),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "semantic_truth_read_during_training": False,
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def decide(
    metrics: dict[str, dict[str, Any]],
    training: dict[str, dict[str, Any]],
    post_lock_validation: dict[str, dict[str, Any]],
    lock_before_semantic_and_test: bool,
    calibration_audit: dict[str, Any],
    generated_initial_state_hash: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    evaluation = config["evaluation"]
    seed = int(config["training"]["seed"])
    projected = metrics["projected_correct_floor"]["unanchored_mcc"]
    vanilla = metrics["vanilla_correct_floor"]["unanchored_mcc"]
    projected_permuted = metrics["projected_permuted_floor"]["unanchored_mcc"]
    denominator = float(evaluation["teacher_ceiling_rgb_correlation"]) - vanilla
    gap_closed = (projected - vanilla) / max(denominator, 1e-12)
    validation_semantics = post_lock_validation["projected_correct_floor"]
    final_projection = training["projected_correct_floor"]["history"][-1]["train"][
        "projection_active"
    ]
    expected_initial = config["initialization"]["expected_state_dict_sha256"]
    histories_have_no_semantic = all(
        all("semantic_validation" not in record for record in item["history"])
        and not item["semantic_truth_read_during_training"]
        for item in training.values()
    )
    validity = {
        "random_initial_state_exact": generated_initial_state_hash == expected_initial,
        "same_initial_state_all_conditions": {
            item["initial_state_sha256"] for item in training.values()
        }
        == {expected_initial},
        "no_warm_checkpoint_in_config": "warm_start" not in config,
        "full_rgb_statistics_explicit_for_v16": config["normalization"][
            "rgb_statistics_source"
        ]
        == "all_8000_observational_train_truth_v16_only",
        "calibration_budget_exact": calibration_audit["budget"]
        == int(config["functional_anchor"]["budget"]),
        "permutation_deranged": calibration_audit["permutation_fixed_points"] == 0,
        "matched_parameter_count": len(
            {item["parameter_count"] for item in training.values()}
        )
        == 1,
        "floors_exact": all(
            np.isclose(item["final_supervision_weight"], 0.1)
            for item in training.values()
        ),
        "no_semantic_truth_before_lock": histories_have_no_semantic,
        "locked_before_semantic_and_test": lock_before_semantic_and_test,
        "anchors_exact": min(item["anchor_mean_direct_r2"] for item in metrics.values())
        >= float(evaluation["minimum_oracle_anchor_r2"]),
    }
    gates = {
        "projected_absolute": projected
        >= float(evaluation["minimum_projected_rgb_mcc"]),
        "gain_over_vanilla": projected - vanilla
        >= float(evaluation["minimum_gain_over_vanilla_correct"]),
        "gain_over_projected_permuted": projected - projected_permuted
        >= float(evaluation["minimum_gain_over_projected_permuted"]),
        "teacher_gap_fraction_closed": gap_closed
        >= float(evaluation["minimum_teacher_gap_fraction_closed"]),
        "validation_semantics": validation_semantics[
            "mean_direct_abs_correlation"
        ]
        >= float(evaluation["minimum_post_lock_validation_direct_correlation"]),
        "projection_active": final_projection
        >= float(evaluation["minimum_final_projection_active_fraction"]),
    }
    all_valid = all(validity.values())
    all_gates = all(gates.values())
    relative_keys = [
        "gain_over_vanilla",
        "gain_over_projected_permuted",
        "teacher_gap_fraction_closed",
        "projection_active",
    ]
    if not all_valid:
        verdict = f"from_scratch_functional_anchor_invalid_seed{seed}"
    elif all_gates:
        verdict = f"from_scratch_functional_anchor_supported_seed{seed}"
    elif all(gates[key] for key in relative_keys):
        verdict = f"from_scratch_functional_signal_insufficient_seed{seed}"
    else:
        verdict = f"from_scratch_functional_anchor_not_supported_seed{seed}"
    return {
        "verdict": verdict,
        "validity": validity,
        "gates": gates,
        "rgb_mcc": {name: item["unanchored_mcc"] for name, item in metrics.items()},
        "deltas": {
            "projected_minus_vanilla_correct": projected - vanilla,
            "projected_correct_minus_projected_permuted": projected
            - projected_permuted,
            "teacher_gap_fraction_closed": gap_closed,
        },
        "post_lock_projected_validation": validation_semantics,
        "final_projection_active_fraction": final_projection,
        "next_stage": {
            "sparse_normalization_v17_unlocked": bool(all_valid and all_gates),
            "rule": "v17 unlocks only when every v16 validity and effect gate passes",
        },
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
    if config["initialization"]["mode"] != "same_seed_random_no_warm":
        raise ValueError("v16 must use random initialization without a warm checkpoint")
    epochs = int(args.epochs or config["training"]["epochs"])
    if not args.smoke and epochs != int(config["training"]["epochs"]):
        raise ValueError("formal v16 must use frozen epochs")
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
    coefficients = {
        name: torch.from_numpy(value).to(device)
        for name, value in coefficient_np.items()
    }
    initial_state, initial_state_hash = make_initial_state(config, device)
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
            "state_dict": initial_state,
        },
        initial_path,
    )

    started = time.time()
    training: dict[str, dict[str, Any]] = {}
    for name in CONDITIONS:
        training[name] = train_condition(
            v2.Condition(name, "oracle"),
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

    source_names = [
        "run_conflict_projected_physics_from_scratch.py",
        "run_conflict_projected_physics_training.py",
        "run_physics_functional_anchor_training.py",
        "run_supervised_continuation_diagnostic.py",
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
            "rgb_statistics_source": config["normalization"][
                "rgb_statistics_source"
            ],
            "latent_mean": latent_mean_np,
            "latent_std": latent_std_np,
        },
        "calibration": calibration_audit,
        "teacher_coefficients": coefficient_audit,
        "checkpoints": {
            name: {
                "path": item["checkpoint"],
                "sha256": item["checkpoint_sha256"],
                "initial_state_sha256": item["initial_state_sha256"],
                "final_supervision_weight": item["final_supervision_weight"],
            }
            for name, item in training.items()
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
        "calibration": calibration_audit,
        "teacher_coefficients": coefficient_audit,
        "training": training,
        "training_lock": str(lock_path),
        "training_lock_sha256_before_semantic_and_test": lock_sha,
        "duration_seconds_before_semantic_and_test": time.time() - started,
    }
    if args.smoke:
        result = {**common, "verdict": "smoke_completed_no_semantic_or_test_read"}
        path = output_dir / "smoke_results.json"
    else:
        metrics: dict[str, dict[str, Any]] = {}
        validation_semantics: dict[str, dict[str, Any]] = {}
        train_end = int(config["split"]["train_end"])
        validation_end = int(config["split"]["validation_end"])
        batch_size = int(config["training"]["batch_size"])
        for name, item in training.items():
            model = v13.load_model(Path(item["checkpoint"]), config, device)
            prediction = v6.predict_rgb(
                model,
                images,
                latents,
                train_end,
                validation_end,
                batch_size,
                device,
            )
            validation_semantics[name] = v6.regression_metrics(
                prediction, latents[0, train_end:validation_end, :3]
            )
            metrics[name] = v2.evaluate_model(
                model,
                images,
                latents,
                anchor_maps,
                v2.Condition(name, "oracle"),
                config,
                device,
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        decision = decide(
            metrics,
            training,
            validation_semantics,
            True,
            calibration_audit,
            initial_state_hash,
            config,
        )
        result = {
            **common,
            "post_lock_validation_semantics": validation_semantics,
            "metrics": metrics,
            "decision": decision,
            "duration_seconds": time.time() - started,
        }
        path = output_dir / "diagnostic_results.json"
    v11.atomic_json(path, result)
    print(
        json.dumps(
            {"result_path": str(path), "sha256": base.sha256_file(path)},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
