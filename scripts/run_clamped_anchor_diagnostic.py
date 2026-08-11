"""Observed-anchor clamping pilot for partial-anchor CRL.

This v2 imports the frozen v1 data, encoder, density-ratio head, and metric
utilities but does not modify v1.  The learned encoder emits only unobserved
coordinates; registered observed coordinates are inserted directly.  A
within-environment, within-split anchor permutation is the matched control.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

import run_diagnostic as base


@dataclass(frozen=True)
class Condition:
    name: str
    anchor_mode: str


class ClampedContrastiveModel(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        learned_indices: list[int],
        anchor_indices: list[int],
        hidden_channels: int,
        conv_layers: int,
    ):
        super().__init__()
        if sorted(learned_indices + anchor_indices) != list(range(latent_dim)):
            raise ValueError("learned and anchor indices must partition the latent coordinates")
        self.latent_dim = latent_dim
        self.learned_indices = list(learned_indices)
        self.anchor_indices = list(anchor_indices)
        self.embedding = base.ImageEncoderChambers(
            len(learned_indices), hidden_channels, conv_layers
        )
        self.parametric_part = base.ParametricPart(latent_dim)

    def compose_z(self, learned: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        if learned.shape[1] != len(self.learned_indices):
            raise ValueError("unexpected learned coordinate count")
        if anchors.shape[1] != len(self.anchor_indices):
            raise ValueError("unexpected anchor coordinate count")
        columns: list[torch.Tensor | None] = [None] * self.latent_dim
        for source, target in enumerate(self.learned_indices):
            columns[target] = learned[:, source]
        for source, target in enumerate(self.anchor_indices):
            columns[target] = anchors[:, source]
        if any(column is None for column in columns):
            raise RuntimeError("incomplete latent composition")
        return torch.stack([column for column in columns if column is not None], dim=1)

    def get_z(self, x: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        return self.compose_z(self.embedding(x), anchors)

    def forward(
        self, x: torch.Tensor, anchors: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.get_z(x, anchors)
        return self.parametric_part(z, targets), z


def validate_config(config: dict[str, Any]) -> None:
    model = config["model"]
    training = config["training"]
    learned = list(model["learned_indices"])
    anchors = list(model["anchor_indices"])
    latent_dim = int(model["latent_dim"])
    if sorted(learned + anchors) != list(range(latent_dim)):
        raise ValueError("coordinate blocks do not partition latent space")
    if list(training["targets"]) != learned:
        raise ValueError("v2 targets must equal learned coordinates")
    if list(training["conditions"]) != ["oracle_clamped", "shuffled_clamped"]:
        raise ValueError("unexpected registered condition order")


def conditions_from_config(config: dict[str, Any]) -> list[Condition]:
    mapping = {
        "oracle_clamped": Condition("oracle_clamped", "oracle"),
        "shuffled_clamped": Condition("shuffled_clamped", "shuffled"),
    }
    return [mapping[name] for name in config["training"]["conditions"]]


def split_ranges(config: dict[str, Any]) -> list[tuple[str, int, int]]:
    split = config["split"]
    return [
        ("train", 0, int(split["train_end"])),
        ("validation", int(split["train_end"]), int(split["validation_end"])),
        ("test", int(split["validation_end"]), int(split["test_end"])),
    ]


def build_anchor_maps(config: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    n_env = len(config["dataset"]["environments"])
    n_rows = int(config["dataset"]["samples_per_environment"])
    identity = np.tile(np.arange(n_rows, dtype=np.int64), (n_env, 1))
    shuffled = identity.copy()
    shuffle_seed = int(config["training"]["shuffle_seed"])
    checks: dict[str, Any] = {"all_bijective": True, "splits": {}}
    for split_index, (name, start, end) in enumerate(split_ranges(config)):
        split_checks = {}
        target_set = np.arange(start, end, dtype=np.int64)
        for environment in range(n_env):
            rng = np.random.default_rng(
                shuffle_seed + 1009 * environment + 104729 * split_index
            )
            permutation = rng.permutation(target_set)
            shuffled[environment, start:end] = permutation
            bijective = bool(np.array_equal(np.sort(permutation), target_set))
            checks["all_bijective"] = bool(checks["all_bijective"] and bijective)
            split_checks[str(environment)] = {
                "bijective": bijective,
                "fixed_point_fraction": float(np.mean(permutation == target_set)),
            }
        checks["splits"][name] = split_checks
    return {"oracle": identity, "shuffled": shuffled}, checks


def anchor_values(
    latents: np.ndarray,
    maps: dict[str, np.ndarray],
    mode: str,
    environments: np.ndarray,
    rows: np.ndarray,
    anchor_indices: list[int],
) -> np.ndarray:
    mapped_rows = maps[mode][environments, rows]
    return np.array(
        latents[environments, mapped_rows][:, anchor_indices], dtype=np.float32, copy=True
    )


def objective_for_batch(
    model: ClampedContrastiveModel,
    images: np.ndarray,
    latents: np.ndarray,
    anchor_maps: dict[str, np.ndarray],
    condition: Condition,
    rows: np.ndarray,
    targets: np.ndarray,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    training = config["training"]
    anchor_indices = list(config["model"]["anchor_indices"])
    obs_envs = np.zeros_like(rows)
    int_envs = targets + 1
    obs_anchors = torch.from_numpy(
        anchor_values(
            latents, anchor_maps, condition.anchor_mode, obs_envs, rows, anchor_indices
        )
    ).to(device)
    int_anchors = torch.from_numpy(
        anchor_values(
            latents, anchor_maps, condition.anchor_mode, int_envs, rows, anchor_indices
        )
    ).to(device)
    x_obs = base.image_batch(images, obs_envs, rows, device)
    x_int = base.image_batch(images, int_envs, rows, device)
    target_tensor = torch.from_numpy(targets).to(device=device, dtype=torch.long)

    logits_int, z_int = model(x_int, int_anchors, target_tensor)
    logits_obs, z_obs = model(x_obs, obs_anchors, target_tensor)
    classifier = F.cross_entropy(
        logits_obs, torch.zeros(len(rows), device=device, dtype=torch.long)
    ) + F.cross_entropy(logits_int, torch.ones(len(rows), device=device, dtype=torch.long))
    mean_regularizer = float(training["kappa"]) * torch.sum(torch.mean(z_obs, dim=0) ** 2)
    graph_regularizer = float(training["eta"]) * torch.sum(
        torch.abs(model.parametric_part.A)
    ) + float(training["mu"]) * base.notears_loss(model.parametric_part.A)
    total = classifier + mean_regularizer + graph_regularizer
    metrics = {
        "total": float(total.detach().cpu()),
        "classifier": float(classifier.detach().cpu()),
        "mean_regularizer": float(mean_regularizer.detach().cpu()),
        "graph_regularizer": float(graph_regularizer.detach().cpu()),
    }
    return total, metrics


@torch.no_grad()
def validate(
    model: ClampedContrastiveModel,
    images: np.ndarray,
    latents: np.ndarray,
    anchor_maps: dict[str, np.ndarray],
    condition: Condition,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, float]:
    split = config["split"]
    rows, targets = base.make_pairs(
        list(config["training"]["targets"]),
        int(split["train_end"]),
        int(split["validation_end"]),
    )
    batch_size = int(config["training"]["batch_size"])
    sums: dict[str, float] = {}
    total_count = 0
    model.eval()
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        batch_targets = targets[start : start + batch_size]
        _, metrics = objective_for_batch(
            model,
            images,
            latents,
            anchor_maps,
            condition,
            batch_rows,
            batch_targets,
            config,
            device,
        )
        size = len(batch_rows)
        total_count += size
        for name, value in metrics.items():
            sums[name] = sums.get(name, 0.0) + size * value
    return base.average_metrics(sums, total_count)


def train_condition(
    condition: Condition,
    images: np.ndarray,
    latents: np.ndarray,
    anchor_maps: dict[str, np.ndarray],
    config: dict[str, Any],
    device: torch.device,
    epochs: int,
    output_dir: Path,
) -> tuple[ClampedContrastiveModel, dict[str, Any]]:
    training = config["training"]
    model_cfg = config["model"]
    seed = int(training["seed"])
    base.set_seed(seed)
    model = ClampedContrastiveModel(
        int(model_cfg["latent_dim"]),
        list(model_cfg["learned_indices"]),
        list(model_cfg["anchor_indices"]),
        int(model_cfg["hidden_channels"]),
        int(model_cfg["conv_layers"]),
    ).to(device)
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
    validation_interval = int(training["validation_interval"])
    best_objective = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    history = []
    started = time.time()

    for epoch in range(epochs):
        generator = np.random.default_rng(seed + 1009 * epoch)
        order = generator.permutation(len(rows))
        sums: dict[str, float] = {}
        total_count = 0
        model.train()
        for start in range(0, len(rows), batch_size):
            indices = order[start : start + batch_size]
            batch_rows = rows[indices]
            batch_targets = targets[indices]
            loss, metrics = objective_for_batch(
                model,
                images,
                latents,
                anchor_maps,
                condition,
                batch_rows,
                batch_targets,
                config,
                device,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            size = len(batch_rows)
            total_count += size
            for name, value in metrics.items():
                sums[name] = sums.get(name, 0.0) + size * value

        train_metrics = base.average_metrics(sums, total_count)
        should_validate = epoch % validation_interval == 0 or epoch == epochs - 1
        record: dict[str, Any] = {"epoch": epoch + 1, "train": train_metrics}
        if should_validate:
            validation_metrics = validate(
                model, images, latents, anchor_maps, condition, config, device
            )
            selection_objective = validation_metrics["total"]
            scheduler.step(selection_objective)
            record["validation"] = validation_metrics
            record["learning_rate"] = optimizer.param_groups[0]["lr"]
            if selection_objective < best_objective:
                best_objective = selection_objective
                best_epoch = epoch + 1
                best_state = copy.deepcopy(model.state_dict())
            print(
                json.dumps(
                    {
                        "condition": condition.name,
                        "epoch": epoch + 1,
                        "train_total": train_metrics["total"],
                        "validation_total": validation_metrics["total"],
                        "validation_classifier": validation_metrics["classifier"],
                        "lr": optimizer.param_groups[0]["lr"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        history.append(record)

    if best_state is None:
        raise RuntimeError(f"no validation checkpoint for {condition.name}")
    model.load_state_dict(best_state)
    checkpoint = output_dir / f"{condition.name}_seed{seed}.pt"
    torch.save(
        {
            "condition": condition.name,
            "anchor_mode": condition.anchor_mode,
            "seed": seed,
            "epoch": best_epoch,
            "state_dict": best_state,
        },
        checkpoint,
    )
    summary = {
        "condition": condition.name,
        "anchor_mode": condition.anchor_mode,
        "best_epoch": best_epoch,
        "best_validation_objective": best_objective,
        "duration_seconds": time.time() - started,
        "history": history,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": base.sha256_file(checkpoint),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    return model, summary


@torch.no_grad()
def encode_rows(
    model: ClampedContrastiveModel,
    images: np.ndarray,
    latents: np.ndarray,
    anchor_maps: dict[str, np.ndarray],
    condition: Condition,
    environment: int,
    start: int,
    end: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    outputs = []
    anchor_indices = list(model.anchor_indices)
    model.eval()
    for batch_start in range(start, end, batch_size):
        rows = np.arange(batch_start, min(batch_start + batch_size, end), dtype=np.int64)
        envs = np.full(len(rows), environment, dtype=np.int64)
        x = base.image_batch(images, envs, rows, device)
        anchors = torch.from_numpy(
            anchor_values(
                latents,
                anchor_maps,
                condition.anchor_mode,
                envs,
                rows,
                anchor_indices,
            )
        ).to(device)
        outputs.append(model.get_z(x, anchors).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def evaluate_model(
    model: ClampedContrastiveModel,
    images: np.ndarray,
    latents: np.ndarray,
    anchor_maps: dict[str, np.ndarray],
    condition: Condition,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    split = config["split"]
    batch_size = int(config["training"]["batch_size"])
    validation_embedding = encode_rows(
        model,
        images,
        latents,
        anchor_maps,
        condition,
        0,
        int(split["train_end"]),
        int(split["validation_end"]),
        batch_size,
        device,
    )
    test_embedding = encode_rows(
        model,
        images,
        latents,
        anchor_maps,
        condition,
        0,
        int(split["validation_end"]),
        int(split["test_end"]),
        batch_size,
        device,
    )
    validation_truth = latents[0, int(split["train_end"]) : int(split["validation_end"])]
    test_truth = latents[0, int(split["validation_end"]) : int(split["test_end"])]
    full_corr = base.absolute_correlation(test_embedding, test_truth)
    full_mcc, full_assignment = base.hungarian_mcc(full_corr)
    learned = list(config["model"]["learned_indices"])
    anchors = list(config["model"]["anchor_indices"])
    learned_corr = base.absolute_correlation(
        test_embedding[:, learned], test_truth[:, learned]
    )
    learned_mcc, learned_assignment = base.hungarian_mcc(learned_corr)
    anchor_r2 = base.direct_r2(test_embedding[:, anchors], test_truth[:, anchors])
    readout = base.linear_readout_r2(
        validation_embedding, validation_truth, test_embedding, test_truth
    )
    estimated_A = model.parametric_part.A.detach().cpu().numpy()
    return {
        "full_mcc": full_mcc,
        "full_assignment": full_assignment,
        "unanchored_mcc": learned_mcc,
        "unanchored_assignment": learned_assignment,
        "direct_abs_correlation": np.diag(full_corr),
        "anchor_direct_r2": anchor_r2,
        "anchor_mean_direct_r2": float(anchor_r2.mean()),
        "linear_readout_r2": readout,
        "linear_readout_mean_r2": float(readout.mean()),
        "absolute_correlation_matrix": full_corr,
        "estimated_A": estimated_A,
        "graph": base.graph_metrics(
            estimated_A, float(config["evaluation"]["graph_threshold"])
        ),
    }


def decide(
    metrics: dict[str, dict[str, Any]],
    training: dict[str, dict[str, Any]],
    shuffle_checks: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    evaluation = config["evaluation"]
    oracle = metrics["oracle_clamped"]
    shuffled = metrics["shuffled_clamped"]
    baseline = float(evaluation["anchor_mse_baseline_rgb_mcc"])
    deltas = {
        "oracle_minus_anchor_mse_rgb_mcc": oracle["unanchored_mcc"] - baseline,
        "oracle_minus_shuffled_rgb_mcc": oracle["unanchored_mcc"]
        - shuffled["unanchored_mcc"],
    }
    validity = {
        "oracle_anchor_exact": oracle["anchor_mean_direct_r2"]
        >= float(evaluation["minimum_oracle_anchor_r2"]),
        "shuffle_bijective": bool(shuffle_checks["all_bijective"]),
        "matched_parameter_count": training["oracle_clamped"]["parameter_count"]
        == training["shuffled_clamped"]["parameter_count"],
    }
    success = {
        "gain_over_anchor_mse": deltas["oracle_minus_anchor_mse_rgb_mcc"]
        >= float(evaluation["minimum_gain_over_anchor_mse"]),
        "gain_over_shuffle": deltas["oracle_minus_shuffled_rgb_mcc"]
        >= float(evaluation["minimum_gain_over_shuffle"]),
        "rgb_absolute": oracle["unanchored_mcc"]
        >= float(evaluation["minimum_rgb_absolute"]),
    }
    if not all(validity.values()):
        verdict = "clamping_branch_invalid"
    elif all(success.values()):
        verdict = "clamping_branch_supported"
    elif any(success.values()):
        verdict = "clamping_branch_partial_signal"
    else:
        verdict = "clamping_branch_not_supported"
    return {
        "verdict": verdict,
        "validity": validity,
        "success": success,
        "deltas": deltas,
        "next_action": (
            "independent_confirmation"
            if verdict == "clamping_branch_supported"
            else "failure_attribution_and_next_structural_operation"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    validate_config(config)
    epochs = int(args.epochs or config["training"]["epochs"])
    output_dir = (args.output_dir or Path(config["runtime"]["output_dir"])).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(config["runtime"]["num_threads"]))
    torch.set_float32_matmul_precision("high")
    device = torch.device(config["runtime"]["device"] if torch.cuda.is_available() else "cpu")

    cache_path = Path(config["dataset"]["cache_path"])
    images = np.load(cache_path, mmap_mode="r")
    expected_shape = (
        len(config["dataset"]["environments"]),
        int(config["dataset"]["samples_per_environment"]),
        64,
        64,
        3,
    )
    if images.shape != expected_shape or images.dtype != np.uint8:
        raise ValueError(f"unexpected image cache: {images.shape}, {images.dtype}")
    raw_latents = base.load_latents(config)
    latents, latent_mean, latent_std = base.normalize_latents(
        raw_latents, int(config["split"]["train_end"])
    )
    anchor_maps, shuffle_checks = build_anchor_maps(config)

    models: dict[str, ClampedContrastiveModel] = {}
    training_summaries: dict[str, Any] = {}
    run_started = time.time()
    for condition in conditions_from_config(config):
        model, summary = train_condition(
            condition,
            images,
            latents,
            anchor_maps,
            config,
            device,
            epochs,
            output_dir,
        )
        models[condition.name] = model
        training_summaries[condition.name] = summary

    source_path = Path(__file__).resolve()
    base_source_path = Path(base.__file__).resolve()
    base_result = {
        "protocol_version": config["protocol_version"],
        "mode": "smoke" if args.smoke else "formal",
        "config": config,
        "config_path": str(config_path),
        "config_sha256": base.sha256_file(config_path),
        "source_path": str(source_path),
        "source_sha256": base.sha256_file(source_path),
        "base_source_path": str(base_source_path),
        "base_source_sha256": base.sha256_file(base_source_path),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "data": {
            "cache_path": str(cache_path),
            "cache_shape": images.shape,
            "cache_dtype": str(images.dtype),
            "latent_mean": latent_mean,
            "latent_std": latent_std,
            "shuffle_checks": shuffle_checks,
        },
        "epochs": epochs,
        "training": training_summaries,
        "duration_seconds": time.time() - run_started,
    }
    if args.smoke:
        result = {**base_result, "verdict": "smoke_completed_no_test_read"}
        result_path = output_dir / "smoke_results.json"
    else:
        metrics = {
            condition.name: evaluate_model(
                models[condition.name],
                images,
                latents,
                anchor_maps,
                condition,
                config,
                device,
            )
            for condition in conditions_from_config(config)
        }
        decision = decide(metrics, training_summaries, shuffle_checks, config)
        result = {**base_result, "metrics": metrics, "decision": decision}
        result_path = output_dir / "diagnostic_results.json"
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(base.json_ready(result), handle, indent=2, sort_keys=True)
    print(
        json.dumps({"result_path": str(result_path), "sha256": base.sha256_file(result_path)}),
        flush=True,
    )


if __name__ == "__main__":
    main()
