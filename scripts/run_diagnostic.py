"""Bounded partial-anchor CRL diagnostic on the Light Tunnel benchmark.

The encoder, parametric head, and contrastive objective are adapted from
simonbing/CRLSanityCheck at commit
2532cd4998695cccf566803b2061cb13be72496e (MIT license).  Data access is
implemented directly against the frozen CSV/uint8 cache so this diagnostic can
reuse the existing lightweight PyTorch environment without installing the
official repository's full dependency stack.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import platform
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata


TRUE_W = np.array(
    [
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.71943922, 0.0, 0.0, 0.0, 0.67726298],
        [0.0, 0.89303215, 0.0, 0.0, 0.98534901],
        [0.84868401, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ],
    dtype=np.float32,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def load_latents(config: dict[str, Any]) -> np.ndarray:
    data_cfg = config["dataset"]
    root = Path(data_cfg["root"])
    prefix = data_cfg["experiment_prefix"]
    suffix = data_cfg["domain_suffix"]
    columns = list(data_cfg["latent_columns"])
    arrays = []
    expected = int(data_cfg["samples_per_environment"])
    for environment in data_cfg["environments"]:
        name = f"{prefix}_{environment}{suffix}"
        csv_path = root / name / f"{name}.csv"
        rows = []
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                rows.append([float(row[column]) for column in columns])
        array = np.asarray(rows, dtype=np.float32)
        if array.shape != (expected, len(columns)):
            raise ValueError(f"unexpected latent shape for {name}: {array.shape}")
        arrays.append(array)
    return np.stack(arrays, axis=0)


def normalize_latents(latents: np.ndarray, train_end: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    obs_train = latents[0, :train_end]
    mean = obs_train.mean(axis=0, keepdims=True)
    std = obs_train.std(axis=0, keepdims=True)
    if np.any(std <= 0):
        raise ValueError("non-positive observational latent standard deviation")
    normalized = (latents - mean[None, :, :]) / std[None, :, :]
    return normalized.astype(np.float32), mean.squeeze(0), std.squeeze(0)


def make_pairs(targets: list[int], start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
    rows = np.tile(np.arange(start, end, dtype=np.int64), len(targets))
    target_array = np.repeat(np.asarray(targets, dtype=np.int64), end - start)
    return rows, target_array


def image_batch(
    images: np.ndarray,
    environment_indices: np.ndarray,
    row_indices: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    array = np.array(images[environment_indices, row_indices], dtype=np.uint8, copy=True)
    tensor = torch.from_numpy(array).permute(0, 3, 1, 2).contiguous().float().div_(255.0)
    return tensor.to(device=device, non_blocking=True)


class ImageEncoderChambers(nn.Module):
    """Official two-block Light Tunnel image encoder."""

    def __init__(self, latent_dim: int = 5, hidden_channels: int = 64, conv_layers: int = 2):
        super().__init__()
        h_dim = hidden_channels
        blocks = []
        for index in range(conv_layers):
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(3 if index == 0 else h_dim, h_dim, 3, 2, 1, bias=False),
                    nn.GroupNorm(num_groups=8, num_channels=h_dim),
                    nn.SiLU(),
                    nn.Conv2d(h_dim, h_dim, 3, 1, 1, bias=False),
                    nn.GroupNorm(num_groups=8, num_channels=h_dim),
                    nn.SiLU(),
                )
            )
        spatial = 64 // (2**conv_layers)
        self.network = nn.Sequential(
            *blocks,
            nn.Flatten(),
            nn.Linear(spatial * spatial * h_dim, 16 * h_dim),
            nn.LayerNorm(16 * h_dim),
            nn.SiLU(),
            nn.Linear(16 * h_dim, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ParametricPart(nn.Module):
    """Official CCRL Gaussian density-ratio head."""

    def __init__(self, latent_dim: int):
        super().__init__()
        self.d = latent_dim
        self.intercepts = nn.Parameter(torch.ones(latent_dim))
        self.shifts = nn.Parameter(torch.zeros(latent_dim))
        self.lambdas = nn.Parameter(torch.ones(latent_dim))
        self.scales = nn.Parameter(torch.ones(1, latent_dim))
        self.A = nn.Parameter(torch.eye(latent_dim))
        self.register_buffer("exclusion_matrix", torch.ones(latent_dim, latent_dim) - torch.eye(latent_dim))

    def get_B(self) -> torch.Tensor:
        identity = torch.eye(self.d, device=self.A.device)
        return self.scales.view(-1, 1) * (identity - self.A * self.exclusion_matrix)

    def forward(self, z: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        row = torch.arange(z.size(0), device=z.device)
        z_selected = z[row, targets]
        transformed = self.get_B()[targets]
        logit = (
            self.shifts[targets]
            + z_selected * self.intercepts[targets]
            + (z_selected * self.lambdas[targets]) ** 2
            - torch.sum(z * transformed, dim=1) ** 2
        )
        return torch.stack((logit, torch.zeros_like(logit)), dim=1)


class ContrastiveModel(nn.Module):
    def __init__(self, latent_dim: int, hidden_channels: int, conv_layers: int):
        super().__init__()
        self.embedding = ImageEncoderChambers(latent_dim, hidden_channels, conv_layers)
        self.parametric_part = ParametricPart(latent_dim)

    def get_z(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x)

    def forward(self, x: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.embedding(x)
        return self.parametric_part(z, targets), z


def notears_loss(A: torch.Tensor) -> torch.Tensor:
    return torch.trace(torch.matrix_exp(A * A)) - A.size(0)


@dataclass(frozen=True)
class Condition:
    name: str
    targets: list[int]
    use_anchors: bool


def conditions_from_config(config: dict[str, Any]) -> list[Condition]:
    training = config["training"]
    mapping = {
        "full_crl": Condition("full_crl", list(training["full_targets"]), False),
        "reduced_crl": Condition("reduced_crl", list(training["reduced_targets"]), False),
        "anchored_reduced_crl": Condition(
            "anchored_reduced_crl", list(training["reduced_targets"]), True
        ),
    }
    return [mapping[name] for name in training["conditions"]]


def objective_for_batch(
    model: ContrastiveModel,
    images: np.ndarray,
    latents: np.ndarray,
    rows: np.ndarray,
    targets: np.ndarray,
    anchor_indices: list[int],
    use_anchors: bool,
    anchor_weight: float,
    kappa: float,
    eta: float,
    mu: float,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    obs_envs = np.zeros_like(rows)
    int_envs = targets + 1
    x_obs = image_batch(images, obs_envs, rows, device)
    x_int = image_batch(images, int_envs, rows, device)
    target_tensor = torch.from_numpy(targets).to(device=device, dtype=torch.long)

    logits_int, z_int = model(x_int, target_tensor)
    logits_obs, z_obs = model(x_obs, target_tensor)
    classifier = F.cross_entropy(
        logits_obs, torch.zeros(len(rows), device=device, dtype=torch.long)
    ) + F.cross_entropy(logits_int, torch.ones(len(rows), device=device, dtype=torch.long))
    mean_regularizer = kappa * torch.sum(torch.mean(z_obs, dim=0) ** 2)
    graph_regularizer = eta * torch.sum(torch.abs(model.parametric_part.A)) + mu * notears_loss(
        model.parametric_part.A
    )

    anchor = torch.zeros((), device=device)
    if use_anchors:
        anchor_tensor = torch.as_tensor(anchor_indices, device=device, dtype=torch.long)
        y_obs = torch.from_numpy(np.array(latents[0, rows][:, anchor_indices], copy=True)).to(device)
        y_int = torch.from_numpy(
            np.array(latents[int_envs, rows][:, anchor_indices], copy=True)
        ).to(device)
        anchor = F.mse_loss(z_obs.index_select(1, anchor_tensor), y_obs) + F.mse_loss(
            z_int.index_select(1, anchor_tensor), y_int
        )

    total = classifier + mean_regularizer + graph_regularizer + anchor_weight * anchor
    metrics = {
        "total": float(total.detach().cpu()),
        "classifier": float(classifier.detach().cpu()),
        "mean_regularizer": float(mean_regularizer.detach().cpu()),
        "graph_regularizer": float(graph_regularizer.detach().cpu()),
        "anchor": float(anchor.detach().cpu()),
    }
    return total, metrics


def average_metrics(weighted: dict[str, float], count: int) -> dict[str, float]:
    return {name: value / max(count, 1) for name, value in weighted.items()}


@torch.no_grad()
def validate(
    model: ContrastiveModel,
    images: np.ndarray,
    latents: np.ndarray,
    condition: Condition,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, float]:
    training = config["training"]
    split = config["split"]
    rows, targets = make_pairs(condition.targets, split["train_end"], split["validation_end"])
    batch_size = int(training["batch_size"])
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
            batch_rows,
            batch_targets,
            list(training["anchor_indices"]),
            condition.use_anchors,
            float(training["anchor_weight"]),
            float(training["kappa"]),
            float(training["eta"]),
            float(training["mu"]),
            device,
        )
        size = len(batch_rows)
        total_count += size
        for name, value in metrics.items():
            sums[name] = sums.get(name, 0.0) + size * value
    return average_metrics(sums, total_count)


def train_condition(
    condition: Condition,
    images: np.ndarray,
    latents: np.ndarray,
    config: dict[str, Any],
    device: torch.device,
    epochs: int,
    output_dir: Path,
) -> tuple[ContrastiveModel, dict[str, Any]]:
    training = config["training"]
    model_cfg = config["model"]
    seed = int(training["seed"])
    set_seed(seed)
    model = ContrastiveModel(
        int(model_cfg["latent_dim"]),
        int(model_cfg["hidden_channels"]),
        int(model_cfg["conv_layers"]),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training["learning_rate"]), weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=float(training["scheduler_factor"]),
        patience=int(training["scheduler_patience"]),
    )

    rows, targets = make_pairs(condition.targets, 0, int(config["split"]["train_end"]))
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
                batch_rows,
                batch_targets,
                list(training["anchor_indices"]),
                condition.use_anchors,
                float(training["anchor_weight"]),
                float(training["kappa"]),
                float(training["eta"]),
                float(training["mu"]),
                device,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            size = len(batch_rows)
            total_count += size
            for name, value in metrics.items():
                sums[name] = sums.get(name, 0.0) + size * value

        train_metrics = average_metrics(sums, total_count)
        should_validate = epoch % validation_interval == 0 or epoch == epochs - 1
        record: dict[str, Any] = {"epoch": epoch + 1, "train": train_metrics}
        if should_validate:
            validation_metrics = validate(model, images, latents, condition, config, device)
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
                        "validation_anchor": validation_metrics["anchor"],
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
            "seed": seed,
            "epoch": best_epoch,
            "state_dict": best_state,
        },
        checkpoint,
    )
    summary = {
        "condition": condition.name,
        "targets": condition.targets,
        "use_anchors": condition.use_anchors,
        "best_epoch": best_epoch,
        "best_validation_objective": best_objective,
        "duration_seconds": time.time() - started,
        "history": history,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
    }
    return model, summary


@torch.no_grad()
def encode_rows(
    model: ContrastiveModel,
    images: np.ndarray,
    environment: int,
    start: int,
    end: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    outputs = []
    model.eval()
    for batch_start in range(start, end, batch_size):
        rows = np.arange(batch_start, min(batch_start + batch_size, end), dtype=np.int64)
        envs = np.full(len(rows), environment, dtype=np.int64)
        x = image_batch(images, envs, rows, device)
        outputs.append(model.get_z(x).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def absolute_correlation(predicted: np.ndarray, truth: np.ndarray) -> np.ndarray:
    x = predicted - predicted.mean(axis=0, keepdims=True)
    y = truth - truth.mean(axis=0, keepdims=True)
    denom = np.sqrt(np.sum(x * x, axis=0)[:, None] * np.sum(y * y, axis=0)[None, :])
    return np.abs((x.T @ y) / np.maximum(denom, 1e-12))


def hungarian_mcc(correlation: np.ndarray) -> tuple[float, list[list[int]]]:
    rows, columns = linear_sum_assignment(-correlation)
    score = float(correlation[rows, columns].mean())
    return score, [[int(row), int(column)] for row, column in zip(rows, columns)]


def direct_r2(predicted: np.ndarray, truth: np.ndarray) -> np.ndarray:
    residual = np.sum((truth - predicted) ** 2, axis=0)
    total = np.sum((truth - truth.mean(axis=0, keepdims=True)) ** 2, axis=0)
    return 1.0 - residual / np.maximum(total, 1e-12)


def linear_readout_r2(
    fit_embedding: np.ndarray,
    fit_truth: np.ndarray,
    test_embedding: np.ndarray,
    test_truth: np.ndarray,
) -> np.ndarray:
    fit_design = np.concatenate((fit_embedding, np.ones((len(fit_embedding), 1))), axis=1)
    test_design = np.concatenate((test_embedding, np.ones((len(test_embedding), 1))), axis=1)
    weights = np.linalg.lstsq(fit_design, fit_truth, rcond=None)[0]
    prediction = test_design @ weights
    return direct_r2(prediction, test_truth)


def edge_auroc(true_w: np.ndarray, estimated: np.ndarray) -> float:
    mask = ~np.eye(true_w.shape[0], dtype=bool)
    labels = (np.abs(true_w[mask]) > 0.01).astype(np.int64)
    scores = np.abs(estimated[mask])
    ranks = rankdata(scores, method="average")
    positives = labels == 1
    n_pos = int(positives.sum())
    n_neg = int((~positives).sum())
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def graph_metrics(estimated: np.ndarray, threshold: float) -> dict[str, float]:
    truth = (np.abs(TRUE_W) > 0.01).astype(np.int64)
    prediction = (np.abs(estimated) > threshold).astype(np.int64)
    np.fill_diagonal(prediction, 0)
    return {
        "fixed_threshold_shd": int(np.abs(truth - prediction).sum()),
        "edge_auroc": edge_auroc(TRUE_W, estimated),
    }


def evaluate_model(
    model: ContrastiveModel,
    images: np.ndarray,
    latents: np.ndarray,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    split = config["split"]
    training = config["training"]
    batch_size = int(training["batch_size"])
    validation_embedding = encode_rows(
        model,
        images,
        0,
        int(split["train_end"]),
        int(split["validation_end"]),
        batch_size,
        device,
    )
    test_embedding = encode_rows(
        model,
        images,
        0,
        int(split["validation_end"]),
        int(split["test_end"]),
        batch_size,
        device,
    )
    validation_truth = latents[0, int(split["train_end"]) : int(split["validation_end"])]
    test_truth = latents[0, int(split["validation_end"]) : int(split["test_end"])]
    full_corr = absolute_correlation(test_embedding, test_truth)
    full_mcc, full_assignment = hungarian_mcc(full_corr)
    unanchored = list(training["reduced_targets"])
    anchors = list(training["anchor_indices"])
    unanchored_corr = absolute_correlation(test_embedding[:, unanchored], test_truth[:, unanchored])
    unanchored_mcc, unanchored_assignment = hungarian_mcc(unanchored_corr)
    direct_corr = np.diag(full_corr)
    anchor_r2 = direct_r2(test_embedding[:, anchors], test_truth[:, anchors])
    readout = linear_readout_r2(validation_embedding, validation_truth, test_embedding, test_truth)
    estimated_A = model.parametric_part.A.detach().cpu().numpy()
    return {
        "full_mcc": full_mcc,
        "full_assignment": full_assignment,
        "unanchored_mcc": unanchored_mcc,
        "unanchored_assignment": unanchored_assignment,
        "direct_abs_correlation": direct_corr,
        "anchor_direct_r2": anchor_r2,
        "anchor_mean_direct_r2": float(anchor_r2.mean()),
        "linear_readout_r2": readout,
        "linear_readout_mean_r2": float(readout.mean()),
        "absolute_correlation_matrix": full_corr,
        "estimated_A": estimated_A,
        "graph": graph_metrics(estimated_A, float(config["evaluation"]["graph_threshold"])),
    }


def decide(metrics: dict[str, dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    evaluation = config["evaluation"]
    full = metrics["full_crl"]
    reduced = metrics["reduced_crl"]
    anchored = metrics["anchored_reduced_crl"]
    deltas = {
        "anchored_minus_reduced_full_mcc": anchored["full_mcc"] - reduced["full_mcc"],
        "anchored_minus_full_full_mcc": anchored["full_mcc"] - full["full_mcc"],
        "anchored_minus_full_unanchored_mcc": anchored["unanchored_mcc"]
        - full["unanchored_mcc"],
    }
    validity = {
        "full_mcc": full["full_mcc"] >= float(evaluation["minimum_full_mcc"]),
        "anchor_path": anchored["anchor_mean_direct_r2"]
        >= float(evaluation["minimum_anchor_valid_r2"]),
    }
    success = {
        "all_mcc_gain": deltas["anchored_minus_reduced_full_mcc"]
        >= float(evaluation["minimum_all_mcc_gain"]),
        "near_full_all": deltas["anchored_minus_full_full_mcc"]
        >= -float(evaluation["maximum_full_mcc_gap"]),
        "preserves_unanchored": deltas["anchored_minus_full_unanchored_mcc"]
        >= -float(evaluation["maximum_unanchored_mcc_gap"]),
        "strong_anchor": anchored["anchor_mean_direct_r2"]
        >= float(evaluation["minimum_anchor_success_r2"]),
    }
    if not all(validity.values()):
        verdict = "diagnostic_invalid"
    elif all(success.values()):
        verdict = "preliminary_signal"
    elif deltas["anchored_minus_reduced_full_mcc"] < 0.05:
        verdict = "mechanism_not_supported_in_this_implementation"
    else:
        verdict = "mixed_signal"
    return {"verdict": verdict, "validity": validity, "success": success, "deltas": deltas}


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
    raw_latents = load_latents(config)
    latents, latent_mean, latent_std = normalize_latents(
        raw_latents, int(config["split"]["train_end"])
    )

    models: dict[str, ContrastiveModel] = {}
    training_summaries: dict[str, Any] = {}
    run_started = time.time()
    for condition in conditions_from_config(config):
        model, summary = train_condition(
            condition, images, latents, config, device, epochs, output_dir
        )
        models[condition.name] = model
        training_summaries[condition.name] = summary

    base_result = {
        "protocol_version": config["protocol_version"],
        "mode": "smoke" if args.smoke else "formal",
        "config": config,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "source_path": str(Path(__file__).resolve()),
        "source_sha256": sha256_file(Path(__file__).resolve()),
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
            name: evaluate_model(model, images, latents, config, device)
            for name, model in models.items()
        }
        result = {**base_result, "metrics": metrics, "decision": decide(metrics, config)}
        result_path = output_dir / "diagnostic_results.json"
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(result), handle, indent=2, sort_keys=True)
    print(json.dumps({"result_path": str(result_path), "sha256": sha256_file(result_path)}), flush=True)


if __name__ == "__main__":
    main()
