"""Sparse persistent semantic-anchor budget diagnostic for partial-anchor CRL v9."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml

import run_clamped_anchor_diagnostic as v2
import run_conditioned_film_diagnostic as v3
import run_diagnostic as base
import run_supervised_continuation_diagnostic as v6


CONDITION_RE = re.compile(r"^k(?P<budget>[0-9]+)_(?P<mode>correct|permuted|zero)_floor$")


def parse_condition(name: str) -> tuple[int, str]:
    match = CONDITION_RE.fullmatch(name)
    if match is None:
        raise ValueError(f"invalid v9 condition name: {name}")
    return int(match.group("budget")), match.group("mode")


def floor_weight(condition: str, epoch_index: int, config: dict[str, Any]) -> float:
    _, mode = parse_condition(condition)
    schedule = config["floor_schedule"]
    hold = int(schedule["hold_epochs"])
    decay_end = int(schedule["decay_end_epoch"])
    floor = float(schedule["zero_floor"] if mode == "zero" else schedule["nonzero_floor"])
    if epoch_index < hold:
        return float(schedule["hold_weight"])
    if epoch_index < decay_end:
        fraction = (decay_end - 1 - epoch_index) / max(decay_end - hold - 1, 1)
        return floor + (float(schedule["hold_weight"]) - floor) * fraction
    return floor


def sha256_int_array(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values, dtype=np.int64)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def make_derangement(size: int, seed: int) -> np.ndarray:
    if size < 2:
        raise ValueError("a permuted-label control needs at least two anchors")
    rng = np.random.default_rng(seed)
    identity = np.arange(size, dtype=np.int64)
    for _ in range(10000):
        candidate = rng.permutation(size).astype(np.int64, copy=False)
        if not np.any(candidate == identity):
            return candidate
    raise RuntimeError(f"could not construct derangement for size={size}")


def build_budget_specs(config: dict[str, Any]) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    train_end = int(config["split"]["train_end"])
    budget_cfg = config["persistent_anchor_budget"]
    budgets = [int(value) for value in budget_cfg["budgets"]]
    if len(budgets) != len(set(budgets)) or budgets != sorted(budgets, reverse=True):
        raise ValueError("budgets must be unique and listed from largest to smallest")
    if budgets[0] != train_end or any(value < 2 or value > train_end for value in budgets):
        raise ValueError("budgets must start at train_end and remain in [2, train_end]")
    subset_seed = int(budget_cfg["subset_seed"])
    permutation_seed = int(budget_cfg["permutation_seed"])
    master = np.random.default_rng(subset_seed).permutation(train_end).astype(np.int64)
    specs: dict[int, dict[str, Any]] = {}
    for budget in budgets:
        subset = (
            np.arange(train_end, dtype=np.int64)
            if budget == train_end
            else np.array(master[:budget], dtype=np.int64, copy=True)
        )
        derangement = make_derangement(budget, permutation_seed + 104729 * budget)
        permuted_targets = subset[derangement]
        target_lookup = np.full(train_end, -1, dtype=np.int64)
        target_lookup[subset] = permuted_targets
        specs[budget] = {
            "budget": budget,
            "subset": subset,
            "permuted_targets": permuted_targets,
            "permuted_target_lookup": target_lookup,
            "subset_sha256": sha256_int_array(subset),
            "permuted_targets_sha256": sha256_int_array(permuted_targets),
            "permutation_fixed_points": int(np.sum(permuted_targets == subset)),
        }
    nested = all(
        set(specs[smaller]["subset"].tolist()).issubset(
            set(specs[larger]["subset"].tolist())
        )
        for larger, smaller in zip(budgets, budgets[1:])
    )
    audit = {
        "budgets": budgets,
        "subset_seed": subset_seed,
        "permutation_seed": permutation_seed,
        "nested": nested,
        "budget_records": {
            str(budget): {
                "count": int(len(specs[budget]["subset"])),
                "subset_sha256": specs[budget]["subset_sha256"],
                "permuted_targets_sha256": specs[budget]["permuted_targets_sha256"],
                "permutation_fixed_points": specs[budget]["permutation_fixed_points"],
                "subset_rows": specs[budget]["subset"].tolist(),
                "permuted_target_rows": specs[budget]["permuted_targets"].tolist(),
            }
            for budget in budgets
        },
    }
    return specs, audit


def anchor_rows_for_batch(batch_rows: np.ndarray, subset: np.ndarray) -> np.ndarray:
    return subset[np.asarray(batch_rows, dtype=np.int64) % len(subset)]


def objective_with_sparse_supervision(
    model: v3.FilmConditionedModel,
    images: np.ndarray,
    latents: np.ndarray,
    anchor_maps: dict[str, np.ndarray],
    condition: v2.Condition,
    rows: np.ndarray,
    targets: np.ndarray,
    config: dict[str, Any],
    device: torch.device,
    weight: float,
    budget_spec: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    ccrl_loss, metrics = v2.objective_for_batch(
        model, images, latents, anchor_maps, condition, rows, targets, config, device
    )
    _, mode = parse_condition(condition.name)
    anchor_rows = anchor_rows_for_batch(rows, budget_spec["subset"])
    if mode == "permuted":
        target_rows = budget_spec["permuted_target_lookup"][anchor_rows]
        if np.any(target_rows < 0):
            raise RuntimeError("permuted target lookup escaped the registered subset")
    else:
        target_rows = anchor_rows
    obs_envs = np.zeros_like(anchor_rows)
    x_obs = base.image_batch(images, obs_envs, anchor_rows, device)
    anchors = torch.from_numpy(
        np.array(latents[0, anchor_rows, 3:5], copy=True)
    ).to(device)
    truth = torch.from_numpy(
        np.array(latents[0, target_rows, :3], copy=True)
    ).to(device)
    rgb_mse = F.mse_loss(model.embedding(x_obs, anchors), truth)
    total = ccrl_loss + weight * rgb_mse
    return total, {
        **metrics,
        "rgb_mse": float(rgb_mse.detach().cpu()),
        "supervision_weight": weight,
        "optimized_total": float(total.detach().cpu()),
    }


def train_path(
    condition: v2.Condition,
    warm_state: dict[str, torch.Tensor],
    images: np.ndarray,
    latents: np.ndarray,
    anchor_maps: dict[str, np.ndarray],
    config: dict[str, Any],
    device: torch.device,
    epochs: int,
    output_dir: Path,
    budget_spec: dict[str, Any],
) -> dict[str, Any]:
    model_cfg = config["model"]
    training = config["training"]
    seed = int(training["seed"])
    base.set_seed(seed)
    model = v3.FilmConditionedModel(
        int(model_cfg["latent_dim"]),
        list(model_cfg["learned_indices"]),
        list(model_cfg["anchor_indices"]),
        int(model_cfg["hidden_channels"]),
        int(model_cfg["conv_layers"]),
    ).to(device)
    model.load_state_dict(warm_state)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training["learning_rate"]))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=float(training["scheduler_factor"]),
        patience=int(training["scheduler_patience"]),
    )
    rows, targets = base.make_pairs(
        list(training["targets"]), 0, int(config["split"]["train_end"])
    )
    batch_size = int(training["batch_size"])
    interval = int(training["validation_interval"])
    history: list[dict[str, Any]] = []
    started = time.time()
    for epoch in range(epochs):
        weight = floor_weight(condition.name, epoch, config)
        order = np.random.default_rng(seed + 1009 * epoch).permutation(len(rows))
        model.train()
        sums: dict[str, float] = {}
        count = 0
        for begin in range(0, len(rows), batch_size):
            indices = order[begin : begin + batch_size]
            batch_rows = rows[indices]
            batch_targets = targets[indices]
            loss, metrics = objective_with_sparse_supervision(
                model,
                images,
                latents,
                anchor_maps,
                condition,
                batch_rows,
                batch_targets,
                config,
                device,
                weight,
                budget_spec,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            count += len(batch_rows)
            for name, value in metrics.items():
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
                latents[
                    0,
                    int(config["split"]["train_end"]) : int(config["split"]["validation_end"]),
                    :3,
                ],
            )
            record.update(validation=validation, semantic_validation=semantic)
            print(
                json.dumps(
                    {
                        "condition": condition.name,
                        "epoch": epoch + 1,
                        "lambda": weight,
                        "validation_total": validation["total"],
                        "semantic_mean_r2": semantic["mean_r2"],
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
            "budget": int(budget_spec["budget"]),
            "state_dict": model.state_dict(),
        },
        checkpoint,
    )
    summary = {
        "condition": condition.name,
        "budget": int(budget_spec["budget"]),
        "final_epoch": epochs,
        "final_supervision_weight": floor_weight(condition.name, epochs - 1, config),
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


def load_checkpoint_model(
    checkpoint: Path, config: dict[str, Any], device: torch.device
) -> v3.FilmConditionedModel:
    model_cfg = config["model"]
    model = v3.FilmConditionedModel(
        int(model_cfg["latent_dim"]),
        list(model_cfg["learned_indices"]),
        list(model_cfg["anchor_indices"]),
        int(model_cfg["hidden_channels"]),
        int(model_cfg["conv_layers"]),
    ).to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["state_dict"])
    return model


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(base.json_ready(payload), indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)


def decide(
    metrics: dict[str, dict[str, Any]],
    training: dict[str, dict[str, Any]],
    pretrain: dict[str, Any],
    budget_audit: dict[str, Any],
    locked_before_test: bool,
    config: dict[str, Any],
) -> dict[str, Any]:
    evaluation = config["evaluation"]
    budgets = [int(value) for value in config["persistent_anchor_budget"]["budgets"]]
    all_conditions = [
        f"k{budget}_{mode}_floor"
        for budget in budgets
        for mode in ("correct", "permuted", "zero")
    ]
    parameter_counts = {training[name]["parameter_count"] for name in all_conditions}
    floors_exact = all(
        training[f"k{budget}_correct_floor"]["final_supervision_weight"] == 0.1
        and training[f"k{budget}_permuted_floor"]["final_supervision_weight"] == 0.1
        and training[f"k{budget}_zero_floor"]["final_supervision_weight"] == 0.0
        for budget in budgets
    )
    all_anchor_r2 = min(metrics[name]["anchor_mean_direct_r2"] for name in all_conditions)
    validity = {
        "pretrain_learned_rgb": pretrain["best_validation_metrics"]["mean_r2"]
        >= float(config["pretraining"]["minimum_validation_mean_direct_r2"]),
        "full_budget_positive_control": metrics[f"k{budgets[0]}_correct_floor"]["unanchored_mcc"]
        >= float(evaluation["minimum_full_budget_rgb_mcc"]),
        "nested_subsets": bool(budget_audit["nested"]),
        "subset_counts_exact": all(
            budget_audit["budget_records"][str(budget)]["count"] == budget
            for budget in budgets
        ),
        "permutations_are_derangements": all(
            budget_audit["budget_records"][str(budget)]["permutation_fixed_points"] == 0
            for budget in budgets
        ),
        "floors_exact": floors_exact,
        "matched_parameter_count": len(parameter_counts) == 1,
        "anchors_exact": all_anchor_r2 >= float(evaluation["minimum_oracle_anchor_r2"]),
        "locked_before_test": bool(locked_before_test),
    }
    budget_results: dict[str, Any] = {}
    supported: list[int] = []
    for budget in budgets:
        correct_name = f"k{budget}_correct_floor"
        permuted_name = f"k{budget}_permuted_floor"
        zero_name = f"k{budget}_zero_floor"
        correct = metrics[correct_name]
        permuted = metrics[permuted_name]
        zero = metrics[zero_name]
        final_semantic = training[correct_name]["history"][-1]["semantic_validation"]
        deltas = {
            "correct_minus_zero_rgb_mcc": correct["unanchored_mcc"] - zero["unanchored_mcc"],
            "correct_minus_permuted_rgb_mcc": correct["unanchored_mcc"]
            - permuted["unanchored_mcc"],
        }
        gates = {
            "rgb_absolute": correct["unanchored_mcc"]
            >= float(evaluation["minimum_sparse_rgb_mcc"]),
            "gain_over_zero": deltas["correct_minus_zero_rgb_mcc"]
            >= float(evaluation["minimum_gain_over_zero_floor"]),
            "gain_over_permuted": deltas["correct_minus_permuted_rgb_mcc"]
            >= float(evaluation["minimum_gain_over_permuted_floor"]),
            "validation_semantics": final_semantic["mean_direct_abs_correlation"]
            >= float(evaluation["minimum_final_validation_direct_correlation"]),
        }
        passed = all(gates.values())
        if passed:
            supported.append(budget)
        budget_results[str(budget)] = {
            "passed": passed,
            "gates": gates,
            "deltas": deltas,
            "correct_rgb_mcc": correct["unanchored_mcc"],
            "permuted_rgb_mcc": permuted["unanchored_mcc"],
            "zero_rgb_mcc": zero["unanchored_mcc"],
            "final_correct_validation": final_semantic,
        }
    sparse_threshold = int(evaluation["sparse_budget_threshold"])
    if not all(validity.values()):
        verdict = "sparse_persistent_anchor_invalid_seed3407"
    elif any(budget <= sparse_threshold for budget in supported):
        verdict = "sparse_persistent_anchor_supported_seed3407"
    elif budgets[0] in supported:
        verdict = "full_persistent_anchor_only_seed3407"
    else:
        verdict = "persistent_anchor_not_supported_seed3407"
    return {
        "verdict": verdict,
        "validity": validity,
        "budget_results": budget_results,
        "supported_budgets": supported,
        "smallest_supported_budget": min(supported) if supported else None,
        "all_path_minimum_anchor_r2": all_anchor_r2,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pretrain-epochs", type=int)
    parser.add_argument("--ccrl-epochs", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    pretrain_epochs = int(args.pretrain_epochs or config["pretraining"]["epochs"])
    ccrl_epochs = int(args.ccrl_epochs or config["training"]["epochs"])
    if not args.smoke and (
        pretrain_epochs != int(config["pretraining"]["epochs"])
        or ccrl_epochs != int(config["training"]["epochs"])
    ):
        raise ValueError("formal v9 must use the frozen epoch counts")
    output_dir = (args.output_dir or Path(config["runtime"]["output_dir"])).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(config["runtime"]["num_threads"]))
    torch.set_float32_matmul_precision("high")
    device = torch.device(config["runtime"]["device"] if torch.cuda.is_available() else "cpu")
    images = np.load(Path(config["dataset"]["cache_path"]), mmap_mode="r")
    raw_latents = base.load_latents(config)
    latents, latent_mean, latent_std = base.normalize_latents(
        raw_latents, int(config["split"]["train_end"])
    )
    anchor_maps, _ = v2.build_anchor_maps(config)
    budget_specs, budget_audit = build_budget_specs(config)
    warm_state, pretrain = v6.pretrain_embedding(
        images, latents, config, device, pretrain_epochs
    )
    warm_path = output_dir / f"warm_state_seed{int(config['training']['seed'])}.pt"
    torch.save({"state_dict": warm_state, "pretraining": pretrain}, warm_path)
    warm_sha256 = base.sha256_file(warm_path)
    training: dict[str, dict[str, Any]] = {}
    started = time.time()
    budgets = [int(value) for value in config["persistent_anchor_budget"]["budgets"]]
    for budget in budgets:
        for mode in ("correct", "permuted", "zero"):
            name = f"k{budget}_{mode}_floor"
            condition = v2.Condition(name, "oracle")
            training[name] = train_path(
                condition,
                warm_state,
                images,
                latents,
                anchor_maps,
                config,
                device,
                ccrl_epochs,
                output_dir,
                budget_specs[budget],
            )
    source_names = [
        "run_sparse_persistent_anchor_diagnostic.py",
        "run_homotopy_continuation_diagnostic.py",
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
        "warm_checkpoint_sha256": warm_sha256,
        "budget_audit": budget_audit,
        "checkpoints": {
            name: {
                "path": summary["checkpoint"],
                "sha256": summary["checkpoint_sha256"],
                "final_supervision_weight": summary["final_supervision_weight"],
            }
            for name, summary in training.items()
        },
        "test_evaluated": False,
    }
    lock_path = output_dir / "training_lock.json"
    atomic_write_json(lock_path, training_lock)
    lock_sha256_before_test = base.sha256_file(lock_path)
    result_base = {
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
        "data": {"latent_mean": latent_mean, "latent_std": latent_std},
        "pretraining": pretrain,
        "warm_checkpoint": str(warm_path),
        "warm_checkpoint_sha256": warm_sha256,
        "budget_audit": budget_audit,
        "training": training,
        "training_lock": str(lock_path),
        "training_lock_sha256_before_test": lock_sha256_before_test,
        "duration_seconds_before_test": time.time() - started + pretrain["duration_seconds"],
    }
    if args.smoke:
        result = {**result_base, "verdict": "smoke_completed_no_test_read"}
        result_path = output_dir / "smoke_results.json"
    else:
        metrics: dict[str, dict[str, Any]] = {}
        for name, summary in training.items():
            condition = v2.Condition(name, "oracle")
            model = load_checkpoint_model(Path(summary["checkpoint"]), config, device)
            metrics[name] = v2.evaluate_model(
                model, images, latents, anchor_maps, condition, config, device
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        decision = decide(metrics, training, pretrain, budget_audit, True, config)
        result = {
            **result_base,
            "metrics": metrics,
            "decision": decision,
            "duration_seconds": time.time() - started + pretrain["duration_seconds"],
        }
        result_path = output_dir / "diagnostic_results.json"
    atomic_write_json(result_path, result)
    print(
        json.dumps(
            {"result_path": str(result_path), "sha256": base.sha256_file(result_path)},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
