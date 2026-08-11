"""Matched K=80 isolated-point control for the closed functional-anchor runs.

This runner deliberately reuses the v17/v18 model, CCRL objective, optimizer,
normalization, initialization, schedule, evaluator, and lock boundary.  The
only training change is that RGB supervision is replayed on the registered 80
observational rows instead of being propagated by a calibrated function.
"""

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
import torch.nn.functional as F
import yaml

import run_clamped_anchor_diagnostic as v2
import run_conditioned_film_diagnostic as v3
import run_conflict_projected_physics_from_scratch as v16
import run_conflict_projected_physics_k80_closed as v17
import run_conflict_projected_physics_training as v13
import run_diagnostic as base
import run_physics_functional_anchor_training as v11
import run_sparse_persistent_anchor_diagnostic as v9
import run_supervised_continuation_diagnostic as v6


CONDITION = "matched_point_correct_floor"
RGB_STATISTICS_SOURCE = v17.RGB_STATISTICS_SOURCE


def build_point_spec(config: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Recreate the registered K-row subset without reading semantic holdouts."""
    item = config["point_anchor"]
    train_end = int(config["split"]["train_end"])
    budget = int(item["budget"])
    if budget <= 0 or budget > train_end:
        raise ValueError("point budget must lie inside observational train")
    master = np.random.default_rng(int(item["subset_seed"])).permutation(train_end)
    subset = np.array(master[:budget], dtype=np.int64, copy=True)
    audit = {
        "budget": budget,
        "subset_sha256": v9.sha256_int_array(subset),
        "subset_rows": subset.tolist(),
        "unique_rows": int(np.unique(subset).size),
        "supervised_environments": ["obs"],
        "sampling": item["sampling"],
    }
    return subset, audit


def point_rows_for_batch(batch_rows: np.ndarray, subset: np.ndarray) -> np.ndarray:
    """Replay exactly the K registered rows at the v9 row-modulo cadence."""
    subset = np.asarray(subset, dtype=np.int64)
    if subset.ndim != 1 or subset.size == 0:
        raise ValueError("point subset must be a non-empty 1D array")
    return subset[np.asarray(batch_rows, dtype=np.int64) % subset.size]


def point_supervision_arrays(
    latents: np.ndarray,
    point_rows: np.ndarray,
    learned_indices: list[int],
    anchor_indices: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Return known angle inputs and RGB targets from observational rows only."""
    rows = np.asarray(point_rows, dtype=np.int64)
    anchors = np.array(latents[0, rows][:, anchor_indices], copy=True)
    truth = np.array(latents[0, rows][:, learned_indices], copy=True)
    return anchors.astype(np.float32), truth.astype(np.float32)


def loss_components(
    model: v3.FilmConditionedModel,
    condition: v2.Condition,
    images: np.ndarray,
    latents: np.ndarray,
    anchor_maps: dict[str, np.ndarray],
    rows: np.ndarray,
    targets: np.ndarray,
    config: dict[str, Any],
    device: torch.device,
    subset: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Compute the shared CCRL loss and the isolated K-point RGB MSE."""
    ccrl, metrics = v2.objective_for_batch(
        model, images, latents, anchor_maps, condition, rows, targets, config, device
    )
    replay_rows = point_rows_for_batch(rows, subset)
    x_obs = base.image_batch(images, np.zeros_like(replay_rows), replay_rows, device)
    anchors_np, truth_np = point_supervision_arrays(
        latents,
        replay_rows,
        list(config["model"]["learned_indices"]),
        list(config["model"]["anchor_indices"]),
    )
    anchors = torch.from_numpy(anchors_np).to(device)
    truth = torch.from_numpy(truth_np).to(device)
    point_mse = F.mse_loss(model.embedding(x_obs, anchors), truth)
    return ccrl, point_mse, metrics


def train_condition(
    initial_state: dict[str, torch.Tensor],
    initial_state_hash: str,
    images: np.ndarray,
    latents: np.ndarray,
    anchor_maps: dict[str, np.ndarray],
    subset: np.ndarray,
    config: dict[str, Any],
    device: torch.device,
    output_dir: Path,
) -> dict[str, Any]:
    """Train the one registered vanilla point condition without semantic peeking."""
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
        raise ValueError("point condition did not load the registered initial state")

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
    condition = v2.Condition(CONDITION, "oracle")
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
            ccrl, point_mse, values = loss_components(
                model,
                condition,
                images,
                latents,
                anchor_maps,
                batch_rows,
                batch_targets,
                config,
                device,
                subset,
            )
            optimizer.zero_grad(set_to_none=True)
            total = ccrl + weight * point_mse
            total.backward()
            optimizer.step()
            size = len(batch_rows)
            count += size
            batch_values = {
                **values,
                "point_mse": float(point_mse.detach().cpu()),
                "supervision_weight": weight,
                "optimized_total": float(total.detach().cpu()),
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
            record.update(
                validation=validation,
                learning_rate=float(optimizer.param_groups[0]["lr"]),
            )
            print(
                json.dumps(
                    {
                        "condition": CONDITION,
                        "epoch": epoch + 1,
                        "lambda": weight,
                        "validation_total": validation["total"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        history.append(record)

    checkpoint = output_dir / f"{CONDITION}_seed{seed}.pt"
    torch.save(
        {
            "condition": CONDITION,
            "seed": seed,
            "epoch": epochs,
            "initial_state_sha256": initial_state_hash,
            "point_subset_sha256": v9.sha256_int_array(subset),
            "state_dict": model.state_dict(),
        },
        checkpoint,
    )
    result = {
        "condition": CONDITION,
        "initial_state_sha256": initial_state_hash,
        "final_epoch": epochs,
        "final_supervision_weight": v13.supervision_weight(epochs - 1, config),
        "duration_seconds": time.time() - started,
        "history": history,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": base.sha256_file(checkpoint),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "semantic_truth_read_during_training": False,
        "point_unique_supervised_rows": int(np.unique(subset).size),
        "point_supervised_environments": ["obs"],
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def numeric_training_values_are_finite(training: dict[str, Any]) -> bool:
    """Check logged optimization scalars while ignoring identifiers and paths."""
    for record in training["history"]:
        for block in (record.get("train", {}), record.get("validation", {})):
            for value in block.values():
                if isinstance(value, (int, float)) and not np.isfinite(value):
                    return False
        for key in ("supervision_weight", "learning_rate"):
            if key in record and not np.isfinite(record[key]):
                return False
    return True


def decide_validity(
    metrics: dict[str, Any],
    training: dict[str, Any],
    post_lock_validation: dict[str, Any],
    lock_before_semantic_and_test: bool,
    point_audit: dict[str, Any],
    generated_initial_state_hash: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Judge only whether the baseline is matched and interpretable."""
    expected_initial = config["initialization"]["expected_state_dict_sha256"]
    registered_subset = config["normalization"]["subset_sha256"]
    expected_budget = int(config["point_anchor"]["budget"])
    expected_parameters = int(config["matched_reference"]["expected_parameter_count"])
    validation_value = float(post_lock_validation["mean_direct_abs_correlation"])
    test_direct = float(np.mean(metrics["direct_abs_correlation"][:3]))
    validity = {
        "random_initial_state_exact": generated_initial_state_hash == expected_initial,
        "same_initial_state_loaded": training["initial_state_sha256"] == expected_initial,
        "no_warm_checkpoint_in_config": "warm_start" not in config,
        "k80_rgb_statistics_source_registered": config["normalization"][
            "rgb_statistics_source"
        ]
        == RGB_STATISTICS_SOURCE,
        "point_budget_exact": point_audit["budget"] == expected_budget == 80,
        "point_subset_matches_normalization": point_audit["subset_sha256"]
        == registered_subset,
        "point_subset_matches_functional_reference": point_audit["subset_sha256"]
        == config["matched_reference"]["functional_subset_sha256"],
        "point_unique_rows_exact": point_audit["unique_rows"] == expected_budget,
        "point_observational_only": point_audit["supervised_environments"] == ["obs"],
        "point_sampling_registered": point_audit["sampling"]
        == "ccrl_row_modulo_registered_subset",
        "same_parameter_count_as_functional": training["parameter_count"]
        == expected_parameters,
        "floor_exact": bool(
            np.isclose(training["final_supervision_weight"], 0.1)
        ),
        "no_semantic_truth_before_lock": all(
            "semantic_validation" not in record for record in training["history"]
        )
        and not training["semantic_truth_read_during_training"],
        "locked_before_semantic_and_test": lock_before_semantic_and_test,
        "anchors_exact": metrics["anchor_mean_direct_r2"]
        >= float(config["evaluation"]["minimum_oracle_anchor_r2"]),
        "training_values_finite": numeric_training_values_are_finite(training),
        "semantic_metrics_finite": bool(
            np.isfinite(validation_value) and np.isfinite(test_direct)
        ),
    }
    seed = int(config["training"]["seed"])
    all_valid = all(validity.values())
    return {
        "verdict": (
            f"matched_point_control_valid_seed{seed}"
            if all_valid
            else f"matched_point_control_invalid_seed{seed}"
        ),
        "validity": validity,
        "readout": {
            "validation_rgb_mean_direct_abs_correlation": validation_value,
            "test_rgb_mean_direct_abs_correlation": test_direct,
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
    if config["protocol_version"] != "partial_anchor_crl_matched_point_k80_v1":
        raise ValueError("unexpected matched-point protocol")
    if list(config["training"]["conditions"]) != [CONDITION]:
        raise ValueError("matched-point runner accepts exactly one registered condition")
    if config["initialization"]["mode"] != "same_seed_random_no_warm":
        raise ValueError("matched point must start from the registered random state")
    actual_runner_sha = base.sha256_file(Path(__file__).resolve())
    expected_runner_sha = config["implementation"]["runner_sha256"]
    if actual_runner_sha != expected_runner_sha:
        raise ValueError(
            f"matched-point runner hash mismatch: {actual_runner_sha} != {expected_runner_sha}"
        )

    epochs = int(args.epochs or config["training"]["epochs"])
    if not args.smoke and epochs != int(config["training"]["epochs"]):
        raise ValueError("formal matched point run must use frozen epochs")
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
    subset, point_audit = build_point_spec(config)
    if point_audit["subset_sha256"] != config["normalization"]["subset_sha256"]:
        raise ValueError("point and normalization subsets differ")
    latents, latent_mean_np, latent_std_np = v17.normalize_with_k80_rgb_statistics(
        raw_latents, int(config["split"]["train_end"]), subset
    )
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

    started = time.time()
    training = train_condition(
        initial_state,
        initial_state_hash,
        images,
        latents,
        anchor_maps,
        subset,
        config,
        device,
        output_dir,
    )
    source_names = [
        "run_matched_point_control_k80.py",
        "run_conflict_projected_physics_k80_closed.py",
        "run_conflict_projected_physics_from_scratch.py",
        "run_conflict_projected_physics_training.py",
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
        "point_anchor": point_audit,
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
        "point_anchor": point_audit,
        "training": training,
        "training_lock": str(lock_path),
        "training_lock_sha256_before_semantic_and_test": lock_sha,
        "duration_seconds_before_semantic_and_test": time.time() - started,
    }
    if args.smoke:
        result = {**common, "verdict": "smoke_completed_no_semantic_or_test_read"}
        result_path = output_dir / "smoke_results.json"
    else:
        condition = v2.Condition(CONDITION, "oracle")
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
        decision = decide_validity(
            metrics,
            training,
            validation_semantics,
            True,
            point_audit,
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
