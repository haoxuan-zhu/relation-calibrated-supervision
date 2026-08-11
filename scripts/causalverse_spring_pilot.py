from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.nn import functional as F


LATENT_COLUMNS = ("h", "r", "m", "k", "l")
RELATION_INDICES = (2, 3, 4)
FREE_INDICES = (0, 1)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def make_id_splits(
    ids: Sequence[int],
    seed: int,
    train_fraction: float,
    validation_fraction: float,
) -> dict[str, np.ndarray]:
    unique_ids = np.asarray(sorted(set(int(value) for value in ids)), dtype=np.int64)
    if unique_ids.size < 3:
        raise ValueError("at least three unique ids are required")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("train and validation fractions leave no test ids")

    permutation = np.random.default_rng(seed).permutation(unique_ids)
    train_count = int(math.floor(unique_ids.size * train_fraction))
    validation_count = int(math.floor(unique_ids.size * validation_fraction))
    if min(train_count, validation_count, unique_ids.size - train_count - validation_count) < 1:
        raise ValueError("each split must contain at least one id")
    return {
        "train": permutation[:train_count],
        "validation": permutation[train_count : train_count + validation_count],
        "test": permutation[train_count + validation_count :],
    }


@dataclass(frozen=True)
class GroupedFeatures:
    ids: np.ndarray
    features: np.ndarray
    latents: np.ndarray
    views: np.ndarray


def group_features_by_id(
    ids: np.ndarray,
    views: np.ndarray,
    features: np.ndarray,
    latents: np.ndarray,
    required_views: Iterable[int] = (0, 1, 2, 3),
) -> GroupedFeatures:
    ids = np.asarray(ids, dtype=np.int64)
    views = np.asarray(views, dtype=np.int64)
    features = np.asarray(features)
    # Keep released physical metadata in float64. The Hooke residual is exact to
    # roughly 1e-14 in the source parquet and must not be invalidated by a cache cast.
    latents = np.asarray(latents, dtype=np.float64)
    required = np.asarray(tuple(required_views), dtype=np.int64)
    if not (len(ids) == len(views) == len(features) == len(latents)):
        raise ValueError("ids, views, features, and latents must have equal length")
    if features.ndim != 2 or latents.shape != (len(ids), len(LATENT_COLUMNS)):
        raise ValueError("unexpected feature or latent shape")
    if not np.isfinite(features).all() or not np.isfinite(latents).all():
        raise ValueError("feature cache contains non-finite values")

    grouped_ids: list[int] = []
    grouped_features: list[np.ndarray] = []
    grouped_latents: list[np.ndarray] = []
    for sample_id in np.unique(ids):
        indices = np.flatnonzero(ids == sample_id)
        order = np.argsort(views[indices])
        indices = indices[order]
        if not np.array_equal(views[indices], required):
            raise ValueError(f"id {sample_id} does not have exactly the required views")
        if not np.allclose(latents[indices], latents[indices[0]], rtol=0.0, atol=0.0):
            raise ValueError(f"id {sample_id} has inconsistent latents across views")
        grouped_ids.append(int(sample_id))
        grouped_features.append(features[indices])
        grouped_latents.append(latents[indices[0]])

    return GroupedFeatures(
        ids=np.asarray(grouped_ids, dtype=np.int64),
        features=np.stack(grouped_features).astype(np.float32),
        latents=np.stack(grouped_latents).astype(np.float64),
        views=required.copy(),
    )


def select_ids(grouped: GroupedFeatures, selected_ids: Sequence[int]) -> GroupedFeatures:
    index_by_id = {int(sample_id): index for index, sample_id in enumerate(grouped.ids)}
    try:
        indices = np.asarray([index_by_id[int(sample_id)] for sample_id in selected_ids], dtype=np.int64)
    except KeyError as error:
        raise ValueError(f"selected id is absent from feature cache: {error.args[0]}") from error
    return GroupedFeatures(
        ids=grouped.ids[indices],
        features=grouped.features[indices],
        latents=grouped.latents[indices],
        views=grouped.views,
    )


def fit_relation_coefficient(latents: np.ndarray, mass_permutation: np.ndarray | None = None) -> float:
    values = np.asarray(latents, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(LATENT_COLUMNS):
        raise ValueError("latents must have shape [N, 5]")
    mass = values[:, 2]
    if mass_permutation is not None:
        permutation = np.asarray(mass_permutation, dtype=np.int64)
        if not np.array_equal(np.sort(permutation), np.arange(len(values))):
            raise ValueError("mass_permutation must be a permutation of row indices")
        mass = mass[permutation]
    response = values[:, 4] * values[:, 3]
    denominator = float(np.dot(mass, mass))
    if denominator <= 0.0 or not np.isfinite(denominator):
        raise ValueError("relation calibration has an invalid denominator")
    coefficient = float(np.dot(mass, response) / denominator)
    if not np.isfinite(coefficient):
        raise ValueError("relation coefficient is non-finite")
    return coefficient


def relation_residual_numpy(latents: np.ndarray, coefficient: float) -> np.ndarray:
    values = np.asarray(latents, dtype=np.float64)
    return values[:, 4] * values[:, 3] - float(coefficient) * values[:, 2]


def relation_scale(latents: np.ndarray) -> float:
    values = np.asarray(latents, dtype=np.float64)
    response = values[:, 4] * values[:, 3]
    scale = float(np.std(response))
    if not np.isfinite(scale) or scale <= 1e-8:
        scale = float(np.mean(np.abs(response)))
    if not np.isfinite(scale) or scale <= 1e-8:
        raise ValueError("relation scale is degenerate")
    return scale


class SpringSemanticHead(nn.Module):
    def __init__(self, feature_dim: int, hidden_dims: Sequence[int], output_dim: int = 5) -> None:
        super().__init__()
        dimensions = [int(feature_dim), *(int(value) for value in hidden_dims), int(output_dim)]
        layers: list[nn.Module] = []
        for index, (input_dim, output) in enumerate(zip(dimensions[:-1], dimensions[1:])):
            layers.append(nn.Linear(input_dim, output))
            if index < len(dimensions) - 2:
                layers.extend((nn.LayerNorm(output), nn.GELU()))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def build_head(feature_dim: int, hidden_dims: Sequence[int], seed: int, device: torch.device) -> SpringSemanticHead:
    set_global_seed(seed)
    return SpringSemanticHead(feature_dim, hidden_dims).to(device)


def _off_diagonal_covariance_loss(values: torch.Tensor) -> torch.Tensor:
    centered = values - values.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / max(values.shape[0] - 1, 1)
    off_diagonal = covariance - torch.diag(torch.diagonal(covariance))
    return off_diagonal.square().sum() / values.shape[1]


def multiview_losses(predictions: torch.Tensor, range_limit: float) -> dict[str, torch.Tensor]:
    if predictions.ndim != 3 or predictions.shape[1:] != (4, len(LATENT_COLUMNS)):
        raise ValueError("predictions must have shape [N_ids, 4, 5]")
    id_mean = predictions.mean(dim=1)
    view = (predictions - id_mean[:, None, :]).square().mean()
    std = torch.sqrt(id_mean.var(dim=0, unbiased=False) + 1e-4)
    variance = F.relu(1.0 - std).mean()
    covariance = _off_diagonal_covariance_loss(id_mean)
    range_penalty = F.relu(id_mean.abs() - float(range_limit)).square().mean()
    return {
        "view": view,
        "variance": variance,
        "covariance": covariance,
        "range": range_penalty,
    }


def physical_relation_loss(
    normalized_predictions: torch.Tensor,
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
    coefficient: float,
    scale: float,
) -> torch.Tensor:
    physical = normalized_predictions * latent_std + latent_mean
    residual = (physical[:, 4] * physical[:, 3] - float(coefficient) * physical[:, 2]) / float(scale)
    return F.smooth_l1_loss(residual, torch.zeros_like(residual), beta=1.0)


@dataclass
class TrainingResult:
    model: SpringSemanticHead
    initial_state_sha256: str
    final_state_sha256: str
    history: list[dict[str, float]]
    normalization_mean: np.ndarray
    normalization_std: np.ndarray
    relation_coefficient: float | None
    relation_scale: float


def train_condition(
    train: GroupedFeatures,
    label_ids: Sequence[int],
    condition: str,
    feature_dim: int,
    hidden_dims: Sequence[int],
    seed: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    loss_weights: Mapping[str, float],
    range_limit: float,
    permutation_seed: int,
    device: torch.device,
) -> TrainingResult:
    if condition not in {"point", "correct_relation", "permuted_relation"}:
        raise ValueError(f"unknown condition: {condition}")
    label_set = {int(sample_id) for sample_id in label_ids}
    label_indices = np.asarray(
        [index for index, sample_id in enumerate(train.ids) if int(sample_id) in label_set], dtype=np.int64
    )
    if len(label_indices) != len(label_set):
        raise ValueError("some label ids are absent from the train split")
    label_latents = train.latents[label_indices].astype(np.float64)
    latent_mean = label_latents.mean(axis=0)
    latent_std = label_latents.std(axis=0)
    if np.any(~np.isfinite(latent_std)) or np.any(latent_std <= 1e-8):
        raise ValueError("K-label normalization is degenerate")

    coefficient: float | None = None
    if condition == "correct_relation":
        coefficient = fit_relation_coefficient(label_latents)
    elif condition == "permuted_relation":
        permutation = np.random.default_rng(permutation_seed).permutation(len(label_latents))
        coefficient = fit_relation_coefficient(label_latents, permutation)
    rel_scale = relation_scale(label_latents)

    features = torch.as_tensor(train.features, dtype=torch.float32, device=device)
    targets = torch.as_tensor(
        (train.latents - latent_mean) / latent_std, dtype=torch.float32, device=device
    )
    label_tensor = torch.as_tensor(label_indices, dtype=torch.long, device=device)
    mean_tensor = torch.as_tensor(latent_mean, dtype=torch.float32, device=device)
    std_tensor = torch.as_tensor(latent_std, dtype=torch.float32, device=device)

    model = build_head(feature_dim, hidden_dims, seed, device)
    initial_sha = state_dict_sha256(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    history: list[dict[str, float]] = []
    set_global_seed(seed)

    for epoch in range(1, epochs + 1):
        model.train()
        predictions = model(features.reshape(-1, feature_dim)).reshape(len(train.ids), 4, -1)
        losses = multiview_losses(predictions, range_limit)
        point = F.mse_loss(predictions[label_tensor], targets[label_tensor, None, :].expand(-1, 4, -1))
        relation = torch.zeros((), dtype=predictions.dtype, device=device)
        if coefficient is not None:
            relation = physical_relation_loss(
                predictions.mean(dim=1), mean_tensor, std_tensor, coefficient, rel_scale
            )
        total = (
            float(loss_weights["view"]) * losses["view"]
            + float(loss_weights["variance"]) * losses["variance"]
            + float(loss_weights["covariance"]) * losses["covariance"]
            + float(loss_weights["point"]) * point
            + float(loss_weights["relation"]) * relation
            + float(loss_weights["range"]) * losses["range"]
        )
        if not torch.isfinite(total):
            raise FloatingPointError(f"non-finite loss at epoch {epoch}")
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        optimizer.step()

        if epoch == 1 or epoch == epochs or epoch % 50 == 0:
            history.append(
                {
                    "epoch": epoch,
                    "total": float(total.detach().cpu()),
                    "view": float(losses["view"].detach().cpu()),
                    "variance": float(losses["variance"].detach().cpu()),
                    "covariance": float(losses["covariance"].detach().cpu()),
                    "point": float(point.detach().cpu()),
                    "relation": float(relation.detach().cpu()),
                    "range": float(losses["range"].detach().cpu()),
                }
            )

    return TrainingResult(
        model=model,
        initial_state_sha256=initial_sha,
        final_state_sha256=state_dict_sha256(model.state_dict()),
        history=history,
        normalization_mean=latent_mean.astype(np.float32),
        normalization_std=latent_std.astype(np.float32),
        relation_coefficient=coefficient,
        relation_scale=rel_scale,
    )


def train_supervised_upper_bound(
    train: GroupedFeatures,
    feature_dim: int,
    hidden_dims: Sequence[int],
    seed: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
) -> TrainingResult:
    latent_mean = train.latents.mean(axis=0)
    latent_std = train.latents.std(axis=0)
    if np.any(latent_std <= 1e-8):
        raise ValueError("full-train normalization is degenerate")
    features = torch.as_tensor(train.features, dtype=torch.float32, device=device)
    targets = torch.as_tensor((train.latents - latent_mean) / latent_std, dtype=torch.float32, device=device)
    model = build_head(feature_dim, hidden_dims, seed, device)
    initial_sha = state_dict_sha256(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    history: list[dict[str, float]] = []
    set_global_seed(seed)
    for epoch in range(1, epochs + 1):
        model.train()
        predictions = model(features.reshape(-1, feature_dim)).reshape(len(train.ids), 4, -1)
        loss = F.mse_loss(predictions, targets[:, None, :].expand(-1, 4, -1))
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite upper-bound loss at epoch {epoch}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch == epochs or epoch % 50 == 0:
            history.append({"epoch": epoch, "supervised_mse": float(loss.detach().cpu())})
    return TrainingResult(
        model=model,
        initial_state_sha256=initial_sha,
        final_state_sha256=state_dict_sha256(model.state_dict()),
        history=history,
        normalization_mean=latent_mean.astype(np.float32),
        normalization_std=latent_std.astype(np.float32),
        relation_coefficient=None,
        relation_scale=relation_scale(train.latents),
    )


def predict_physical(
    model: nn.Module,
    grouped: GroupedFeatures,
    feature_dim: int,
    normalization_mean: np.ndarray,
    normalization_std: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        features = torch.as_tensor(grouped.features, dtype=torch.float32, device=device)
        normalized = model(features.reshape(-1, feature_dim)).reshape(len(grouped.ids), 4, -1).mean(dim=1)
    return (
        normalized.detach().cpu().numpy() * np.asarray(normalization_std)[None, :]
        + np.asarray(normalization_mean)[None, :]
    )


def _absolute_pearson(true: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    correlations = []
    for index in range(true.shape[1]):
        if np.std(true[:, index]) <= 1e-12 or np.std(predicted[:, index]) <= 1e-12:
            correlations.append(0.0)
        else:
            correlations.append(abs(float(np.corrcoef(true[:, index], predicted[:, index])[0, 1])))
    return np.asarray(correlations, dtype=np.float64)


def evaluate_predictions(true: np.ndarray, predicted: np.ndarray) -> dict[str, object]:
    true = np.asarray(true, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if true.shape != predicted.shape or true.ndim != 2 or true.shape[1] != len(LATENT_COLUMNS):
        raise ValueError("true and predicted must both have shape [N, 5]")
    direct = _absolute_pearson(true, predicted)
    denominator = np.sum((true - true.mean(axis=0, keepdims=True)) ** 2, axis=0)
    direct_r2 = 1.0 - np.sum((true - predicted) ** 2, axis=0) / np.maximum(denominator, 1e-12)

    correlation_matrix = np.empty((true.shape[1], predicted.shape[1]), dtype=np.float64)
    for row in range(true.shape[1]):
        for column in range(predicted.shape[1]):
            if np.std(true[:, row]) <= 1e-12 or np.std(predicted[:, column]) <= 1e-12:
                correlation_matrix[row, column] = 0.0
            else:
                correlation_matrix[row, column] = abs(
                    float(np.corrcoef(true[:, row], predicted[:, column])[0, 1])
                )
    rows, columns = linear_sum_assignment(-correlation_matrix)
    hungarian_mcc = float(correlation_matrix[rows, columns].mean())

    design = np.concatenate((predicted, np.ones((len(predicted), 1))), axis=1)
    coefficients = np.linalg.lstsq(design, true, rcond=None)[0]
    fitted = design @ coefficients
    block_denominator = float(np.sum((true - true.mean(axis=0, keepdims=True)) ** 2))
    linear_block_r2 = 1.0 - float(np.sum((true - fitted) ** 2)) / max(block_denominator, 1e-12)

    return {
        "direct_abs_correlation": direct.tolist(),
        "mean_direct_abs_correlation_all5": float(direct.mean()),
        "mean_direct_abs_correlation_relation3": float(direct[list(RELATION_INDICES)].mean()),
        "mean_direct_abs_correlation_free2": float(direct[list(FREE_INDICES)].mean()),
        "direct_r2": direct_r2.tolist(),
        "mean_direct_r2": float(direct_r2.mean()),
        "hungarian_mcc": hungarian_mcc,
        "linear_block_r2": float(linear_block_r2),
        "prediction_std": np.std(predicted, axis=0).tolist(),
    }
