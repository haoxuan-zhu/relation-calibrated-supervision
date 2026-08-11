"""Core model and physical relations for the CausalVerse Slope K22 preflight."""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.nn import functional as F


LATENT_COLUMNS = ("roughness", "theta", "v0", "mu_1", "mu_2", "v1", "l")
FREE_INDICES = (0, 1, 2)
RELATION_INDICES = (3, 4, 5, 6)
CONDITIONS = ("point", "correct_relation", "coefficient_permuted_relation")


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
    ids: Sequence[int], seed: int, train_fraction: float, validation_fraction: float
) -> dict[str, np.ndarray]:
    unique = np.asarray(sorted(set(int(value) for value in ids)), dtype=np.int64)
    if len(unique) < 3 or train_fraction <= 0 or validation_fraction <= 0:
        raise ValueError("invalid split inputs")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("split fractions leave no test IDs")
    ordered = np.random.default_rng(seed).permutation(unique)
    train_count = int(math.floor(len(unique) * train_fraction))
    validation_count = int(math.floor(len(unique) * validation_fraction))
    if min(train_count, validation_count, len(unique) - train_count - validation_count) < 1:
        raise ValueError("each split must contain at least one ID")
    return {
        "train": ordered[:train_count],
        "validation": ordered[train_count : train_count + validation_count],
        "test": ordered[train_count + validation_count :],
    }


@dataclass(frozen=True)
class GroupedFeatures:
    ids: np.ndarray
    features: np.ndarray
    latents: np.ndarray
    views: np.ndarray


def incomplete_product_ids(
    ids: np.ndarray,
    views: np.ndarray,
    required_views: Sequence[int] = (0, 1, 2, 3),
) -> np.ndarray:
    """Return IDs whose selected parquet rows do not form one complete view set."""
    ids = np.asarray(ids, dtype=np.int64)
    views = np.asarray(views, dtype=np.int64)
    required = np.asarray(required_views, dtype=np.int64)
    if len(ids) != len(views):
        raise ValueError("ids and views have inconsistent lengths")
    incomplete: list[int] = []
    for sample_id in np.unique(ids):
        observed = np.sort(views[ids == sample_id])
        if not np.array_equal(observed, required):
            incomplete.append(int(sample_id))
    return np.asarray(incomplete, dtype=np.int64)


def group_features_by_id(
    ids: np.ndarray,
    views: np.ndarray,
    features: np.ndarray,
    latents: np.ndarray,
    required_views: Sequence[int] = (0, 1, 2, 3),
) -> GroupedFeatures:
    ids = np.asarray(ids, dtype=np.int64)
    views = np.asarray(views, dtype=np.int64)
    features = np.asarray(features)
    latents = np.asarray(latents, dtype=np.float64)
    required = np.asarray(required_views, dtype=np.int64)
    if not (len(ids) == len(views) == len(features) == len(latents)):
        raise ValueError("cache arrays have inconsistent lengths")
    if features.ndim != 2 or latents.shape != (len(ids), len(LATENT_COLUMNS)):
        raise ValueError("unexpected feature or latent shape")
    if not np.isfinite(features).all() or not np.isfinite(latents).all():
        raise ValueError("cache contains non-finite values")

    grouped_ids: list[int] = []
    grouped_features: list[np.ndarray] = []
    grouped_latents: list[np.ndarray] = []
    for sample_id in np.unique(ids):
        indices = np.flatnonzero(ids == sample_id)
        indices = indices[np.argsort(views[indices])]
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
    index = {int(sample_id): row for row, sample_id in enumerate(grouped.ids)}
    try:
        rows = np.asarray([index[int(sample_id)] for sample_id in selected_ids], dtype=np.int64)
    except KeyError as error:
        raise ValueError(f"selected id is absent: {error.args[0]}") from error
    return GroupedFeatures(
        ids=grouped.ids[rows],
        features=grouped.features[rows],
        latents=grouped.latents[rows],
        views=grouped.views,
    )


def ordered_label_indices(train_ids: np.ndarray, label_ids: Sequence[int]) -> np.ndarray:
    requested = np.asarray(label_ids, dtype=np.int64)
    if len(np.unique(requested)) != len(requested):
        raise ValueError("label IDs must be unique")
    index = {int(sample_id): row for row, sample_id in enumerate(train_ids)}
    try:
        return np.asarray([index[int(sample_id)] for sample_id in requested], dtype=np.int64)
    except KeyError as error:
        raise ValueError(f"label id is absent from train split: {error.args[0]}") from error


def explicit_derangement(label_ids: Sequence[int], seed: int) -> np.ndarray:
    ids = np.asarray(label_ids, dtype=np.int64)
    if len(ids) < 2 or len(np.unique(ids)) != len(ids):
        raise ValueError("at least two unique IDs are required")
    randomized = np.random.default_rng(seed).permutation(ids)
    mapping = {
        int(randomized[index]): int(randomized[(index + 1) % len(randomized)])
        for index in range(len(randomized))
    }
    mapped = np.asarray([mapping[int(sample_id)] for sample_id in ids], dtype=np.int64)
    if np.any(mapped == ids) or set(mapped.tolist()) != set(ids.tolist()):
        raise AssertionError("mapping is not a complete derangement")
    return mapped


def relation_mapping_sha256(label_ids: Sequence[int], mapped_ids: Sequence[int]) -> str:
    pairs = np.stack(
        (np.asarray(label_ids, dtype="<i8"), np.asarray(mapped_ids, dtype="<i8")), axis=1
    )
    return hashlib.sha256(pairs.tobytes()).hexdigest()


@dataclass(frozen=True)
class RelationParameters:
    mu1_slope: float
    mu1_intercept: float
    mu2_slope: float
    mu2_intercept: float
    deceleration: float
    incline: float

    def as_array(self) -> np.ndarray:
        return np.asarray(list(asdict(self).values()), dtype=np.float64)


def _fit_affine(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    design = np.column_stack((x, np.ones(len(x), dtype=np.float64)))
    slope, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
    return float(slope), float(intercept)


def _fit_through_origin(x: np.ndarray, y: np.ndarray) -> float:
    denominator = float(np.dot(x, x))
    if not np.isfinite(denominator) or denominator <= 1e-12:
        raise ValueError("relation calibration denominator is degenerate")
    value = float(np.dot(x, y) / denominator)
    if not np.isfinite(value):
        raise ValueError("relation coefficient is non-finite")
    return value


def fit_relation_parameters(
    source_latents: np.ndarray, target_latents: np.ndarray | None = None
) -> RelationParameters:
    source = np.asarray(source_latents, dtype=np.float64)
    target = source if target_latents is None else np.asarray(target_latents, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != len(LATENT_COLUMNS):
        raise ValueError("relation arrays must have matching [N,7] shapes")
    a1, b1 = _fit_affine(source[:, 0], target[:, 3])
    a2, b2 = _fit_affine(source[:, 0], target[:, 4])
    deceleration = _fit_through_origin(source[:, 3], target[:, 2] - target[:, 5])
    theta = np.deg2rad(source[:, 1])
    incline_term = source[:, 6] * (np.sin(theta) + source[:, 4] * np.cos(theta))
    incline = _fit_through_origin(incline_term, target[:, 5] ** 2)
    return RelationParameters(a1, b1, a2, b2, deceleration, incline)


def relation_residuals_numpy(latents: np.ndarray, parameters: RelationParameters) -> np.ndarray:
    values = np.asarray(latents, dtype=np.float64)
    theta = np.deg2rad(values[:, 1])
    return np.column_stack(
        (
            values[:, 3] - (parameters.mu1_slope * values[:, 0] + parameters.mu1_intercept),
            values[:, 4] - (parameters.mu2_slope * values[:, 0] + parameters.mu2_intercept),
            (values[:, 2] - values[:, 5]) - parameters.deceleration * values[:, 3],
            values[:, 5] ** 2
            - parameters.incline
            * values[:, 6]
            * (np.sin(theta) + values[:, 4] * np.cos(theta)),
        )
    )


def relation_scales(latents: np.ndarray) -> np.ndarray:
    values = np.asarray(latents, dtype=np.float64)
    components = (
        values[:, 3],
        values[:, 4],
        values[:, 2] - values[:, 5],
        values[:, 5] ** 2,
    )
    scales = []
    for component in components:
        scale = float(np.std(component))
        if not np.isfinite(scale) or scale <= 1e-8:
            scale = float(np.mean(np.abs(component)))
        if not np.isfinite(scale) or scale <= 1e-8:
            raise ValueError("relation scale is degenerate")
        scales.append(scale)
    return np.asarray(scales, dtype=np.float64)


def physical_relation_loss(
    normalized_predictions: torch.Tensor,
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
    parameters: RelationParameters,
    scales: torch.Tensor,
) -> torch.Tensor:
    values = normalized_predictions * latent_std + latent_mean
    theta = torch.deg2rad(values[:, 1])
    residuals = torch.stack(
        (
            values[:, 3] - (parameters.mu1_slope * values[:, 0] + parameters.mu1_intercept),
            values[:, 4] - (parameters.mu2_slope * values[:, 0] + parameters.mu2_intercept),
            (values[:, 2] - values[:, 5]) - parameters.deceleration * values[:, 3],
            values[:, 5].square()
            - parameters.incline
            * values[:, 6]
            * (torch.sin(theta) + values[:, 4] * torch.cos(theta)),
        ),
        dim=1,
    )
    scaled = residuals / scales[None, :]
    return F.smooth_l1_loss(scaled, torch.zeros_like(scaled), beta=1.0)


class SlopeSemanticHead(nn.Module):
    def __init__(self, feature_dim: int, hidden_dims: Sequence[int]) -> None:
        super().__init__()
        dimensions = [int(feature_dim), *(int(value) for value in hidden_dims), len(LATENT_COLUMNS)]
        layers: list[nn.Module] = []
        for index, (input_dim, output_dim) in enumerate(zip(dimensions[:-1], dimensions[1:])):
            layers.append(nn.Linear(input_dim, output_dim))
            if index < len(dimensions) - 2:
                layers.extend((nn.LayerNorm(output_dim), nn.GELU()))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def build_head(
    feature_dim: int, hidden_dims: Sequence[int], seed: int, device: torch.device
) -> SlopeSemanticHead:
    set_global_seed(seed)
    return SlopeSemanticHead(feature_dim, hidden_dims).to(device)


def _off_diagonal_covariance_loss(values: torch.Tensor) -> torch.Tensor:
    centered = values - values.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / max(values.shape[0] - 1, 1)
    off_diagonal = covariance - torch.diag(torch.diagonal(covariance))
    return off_diagonal.square().sum() / values.shape[1]


def multiview_losses(predictions: torch.Tensor, range_limit: float) -> dict[str, torch.Tensor]:
    if predictions.ndim != 3 or predictions.shape[1:] != (4, len(LATENT_COLUMNS)):
        raise ValueError("predictions must have shape [N,4,7]")
    id_mean = predictions.mean(dim=1)
    std = torch.sqrt(id_mean.var(dim=0, unbiased=False) + 1e-4)
    return {
        "view": (predictions - id_mean[:, None, :]).square().mean(),
        "variance": F.relu(1.0 - std).mean(),
        "covariance": _off_diagonal_covariance_loss(id_mean),
        "range": F.relu(id_mean.abs() - float(range_limit)).square().mean(),
    }


@dataclass
class TrainingResult:
    model: SlopeSemanticHead
    initial_state_sha256: str
    final_state_sha256: str
    history: list[dict[str, float]]
    normalization_mean: np.ndarray
    normalization_std: np.ndarray
    relation_parameters: RelationParameters | None
    relation_scales: np.ndarray
    mapped_label_ids: np.ndarray | None
    mapping_sha256: str | None


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
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition: {condition}")
    label_ids = np.asarray(label_ids, dtype=np.int64)
    label_indices = ordered_label_indices(train.ids, label_ids)
    labels = train.latents[label_indices]
    latent_mean = labels.mean(axis=0)
    latent_std = labels.std(axis=0)
    if np.any(~np.isfinite(latent_std)) or np.any(latent_std <= 1e-8):
        raise ValueError("K40 normalization is degenerate")

    parameters: RelationParameters | None = None
    mapped_ids: np.ndarray | None = None
    mapping_sha: str | None = None
    if condition == "correct_relation":
        parameters = fit_relation_parameters(labels)
    elif condition == "coefficient_permuted_relation":
        mapped_ids = explicit_derangement(label_ids, permutation_seed)
        mapped_indices = ordered_label_indices(train.ids, mapped_ids)
        parameters = fit_relation_parameters(labels, train.latents[mapped_indices])
        mapping_sha = relation_mapping_sha256(label_ids, mapped_ids)
    scales = relation_scales(labels)

    features = torch.as_tensor(np.ascontiguousarray(train.features), dtype=torch.float32, device=device)
    targets = torch.as_tensor((train.latents - latent_mean) / latent_std, dtype=torch.float32, device=device)
    label_tensor = torch.as_tensor(label_indices, dtype=torch.long, device=device)
    mean_tensor = torch.as_tensor(latent_mean, dtype=torch.float32, device=device)
    std_tensor = torch.as_tensor(latent_std, dtype=torch.float32, device=device)
    scale_tensor = torch.as_tensor(scales, dtype=torch.float32, device=device)

    model = build_head(feature_dim, hidden_dims, seed, device)
    initial_sha = state_dict_sha256(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    history: list[dict[str, float]] = []
    set_global_seed(seed)
    for epoch in range(1, epochs + 1):
        model.train()
        predictions = model(features.reshape(-1, feature_dim)).reshape(len(train.ids), 4, -1)
        common = multiview_losses(predictions, range_limit)
        point = F.mse_loss(predictions[label_tensor], targets[label_tensor, None, :].expand(-1, 4, -1))
        relation = torch.zeros((), dtype=predictions.dtype, device=device)
        if parameters is not None:
            relation = physical_relation_loss(
                predictions.mean(dim=1), mean_tensor, std_tensor, parameters, scale_tensor
            )
        total = (
            float(loss_weights["view"]) * common["view"]
            + float(loss_weights["variance"]) * common["variance"]
            + float(loss_weights["covariance"]) * common["covariance"]
            + float(loss_weights["point"]) * point
            + float(loss_weights["relation"]) * relation
            + float(loss_weights["range"]) * common["range"]
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
                    "point": float(point.detach().cpu()),
                    "relation": float(relation.detach().cpu()),
                    **{name: float(value.detach().cpu()) for name, value in common.items()},
                }
            )
    return TrainingResult(
        model=model,
        initial_state_sha256=initial_sha,
        final_state_sha256=state_dict_sha256(model.state_dict()),
        history=history,
        normalization_mean=latent_mean.astype(np.float32),
        normalization_std=latent_std.astype(np.float32),
        relation_parameters=parameters,
        relation_scales=scales.astype(np.float32),
        mapped_label_ids=mapped_ids,
        mapping_sha256=mapping_sha,
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
        normalized = model(features.reshape(-1, feature_dim)).reshape(len(grouped.ids), 4, -1).mean(1)
    return normalized.cpu().numpy() * normalization_std[None, :] + normalization_mean[None, :]


def evaluate_predictions(true: np.ndarray, predicted: np.ndarray) -> dict[str, object]:
    true = np.asarray(true, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if true.shape != predicted.shape or true.ndim != 2 or true.shape[1] != len(LATENT_COLUMNS):
        raise ValueError("true and predicted must have matching [N,7] shapes")
    direct = []
    for index in range(true.shape[1]):
        if np.std(true[:, index]) <= 1e-12 or np.std(predicted[:, index]) <= 1e-12:
            direct.append(0.0)
        else:
            direct.append(abs(float(np.corrcoef(true[:, index], predicted[:, index])[0, 1])))
    direct = np.asarray(direct, dtype=np.float64)
    denominator = np.sum((true - true.mean(axis=0, keepdims=True)) ** 2, axis=0)
    direct_r2 = 1.0 - np.sum((true - predicted) ** 2, axis=0) / np.maximum(denominator, 1e-12)
    matrix = np.zeros((true.shape[1], predicted.shape[1]), dtype=np.float64)
    for row in range(true.shape[1]):
        for column in range(predicted.shape[1]):
            if np.std(true[:, row]) > 1e-12 and np.std(predicted[:, column]) > 1e-12:
                matrix[row, column] = abs(float(np.corrcoef(true[:, row], predicted[:, column])[0, 1]))
    rows, columns = linear_sum_assignment(-matrix)
    design = np.column_stack((predicted, np.ones(len(predicted))))
    fitted = design @ np.linalg.lstsq(design, true, rcond=None)[0]
    block_denom = float(np.sum((true - true.mean(axis=0, keepdims=True)) ** 2))
    return {
        "direct_abs_correlation": direct.tolist(),
        "direct_r2": direct_r2.tolist(),
        "mean_direct_abs_correlation_all7": float(direct.mean()),
        "mean_direct_abs_correlation_relation4": float(direct[list(RELATION_INDICES)].mean()),
        "mean_direct_abs_correlation_free3": float(direct[list(FREE_INDICES)].mean()),
        "mean_direct_r2_all7": float(direct_r2.mean()),
        "mean_direct_r2_relation4": float(direct_r2[list(RELATION_INDICES)].mean()),
        "mean_direct_r2_free3": float(direct_r2[list(FREE_INDICES)].mean()),
        "hungarian_mcc": float(matrix[rows, columns].mean()),
        "linear_block_r2": 1.0 - float(np.sum((true - fitted) ** 2)) / max(block_denom, 1e-12),
        "prediction_std": np.std(predicted, axis=0).tolist(),
    }
