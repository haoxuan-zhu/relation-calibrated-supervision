"""Correctness-controlled Spring relation training for the v3 audit.

This module intentionally leaves the historical v2 implementation untouched.
It fixes the label-order ambiguity by materializing an ID-to-ID relation map and
adds two matched one-coefficient relations with incorrect variable roles.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F

import causalverse_spring_pilot as base


@dataclass(frozen=True)
class RelationSpec:
    lhs_left: int
    lhs_right: int
    rhs: int
    equation: str


RELATION_SPECS: dict[str, RelationSpec] = {
    "correct_relation": RelationSpec(4, 3, 2, "l*k=c*m"),
    "coefficient_permuted_relation": RelationSpec(4, 3, 2, "l*k=c*m_permuted"),
    "wrong_lm_to_k_relation": RelationSpec(4, 2, 3, "l*m=c*k"),
    "wrong_km_to_l_relation": RelationSpec(3, 2, 4, "k*m=c*l"),
}
CONDITIONS = ("point", *RELATION_SPECS)


def ordered_label_indices(train_ids: np.ndarray, label_ids: Sequence[int]) -> np.ndarray:
    train_ids = np.asarray(train_ids, dtype=np.int64)
    requested = np.asarray(label_ids, dtype=np.int64)
    if len(np.unique(requested)) != len(requested):
        raise ValueError("label_ids must be unique")
    index_by_id = {int(sample_id): index for index, sample_id in enumerate(train_ids)}
    try:
        return np.asarray([index_by_id[int(sample_id)] for sample_id in requested], dtype=np.int64)
    except KeyError as error:
        raise ValueError(f"label id is absent from train split: {error.args[0]}") from error


def explicit_derangement(label_ids: Sequence[int], seed: int) -> np.ndarray:
    """Return mapped IDs in the original label-ID order with no fixed points."""
    ids = np.asarray(label_ids, dtype=np.int64)
    if len(ids) < 2 or len(np.unique(ids)) != len(ids):
        raise ValueError("at least two unique label ids are required")
    randomized = np.random.default_rng(seed).permutation(ids)
    mapping = {
        int(randomized[index]): int(randomized[(index + 1) % len(randomized)])
        for index in range(len(randomized))
    }
    mapped = np.asarray([mapping[int(sample_id)] for sample_id in ids], dtype=np.int64)
    if np.any(mapped == ids) or set(mapped.tolist()) != set(ids.tolist()):
        raise AssertionError("explicit relation mapping is not a derangement")
    return mapped


def relation_mapping_sha256(label_ids: Sequence[int], mapped_ids: Sequence[int]) -> str:
    pairs = np.stack(
        (np.asarray(label_ids, dtype="<i8"), np.asarray(mapped_ids, dtype="<i8")), axis=1
    )
    return hashlib.sha256(pairs.tobytes()).hexdigest()


def fit_relation_coefficient(
    latents: np.ndarray,
    spec: RelationSpec,
    rhs_latents: np.ndarray | None = None,
) -> float:
    values = np.asarray(latents, dtype=np.float64)
    rhs_values = values if rhs_latents is None else np.asarray(rhs_latents, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(base.LATENT_COLUMNS):
        raise ValueError("latents must have shape [N, 5]")
    if rhs_values.shape != values.shape:
        raise ValueError("rhs_latents must match latents")
    lhs = values[:, spec.lhs_left] * values[:, spec.lhs_right]
    rhs = rhs_values[:, spec.rhs]
    denominator = float(np.dot(rhs, rhs))
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("relation calibration has an invalid denominator")
    coefficient = float(np.dot(rhs, lhs) / denominator)
    if not np.isfinite(coefficient):
        raise ValueError("relation coefficient is non-finite")
    return coefficient


def relation_residual_numpy(latents: np.ndarray, coefficient: float, spec: RelationSpec) -> np.ndarray:
    values = np.asarray(latents, dtype=np.float64)
    return (
        values[:, spec.lhs_left] * values[:, spec.lhs_right]
        - float(coefficient) * values[:, spec.rhs]
    )


def relation_scale(latents: np.ndarray, spec: RelationSpec) -> float:
    values = np.asarray(latents, dtype=np.float64)
    lhs = values[:, spec.lhs_left] * values[:, spec.lhs_right]
    scale = float(np.std(lhs))
    if not np.isfinite(scale) or scale <= 1e-8:
        scale = float(np.mean(np.abs(lhs)))
    if not np.isfinite(scale) or scale <= 1e-8:
        raise ValueError("relation scale is degenerate")
    return scale


def physical_relation_loss(
    normalized_predictions: torch.Tensor,
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
    coefficient: float,
    scale: float,
    spec: RelationSpec,
) -> torch.Tensor:
    physical = normalized_predictions * latent_std + latent_mean
    residual = (
        physical[:, spec.lhs_left] * physical[:, spec.lhs_right]
        - float(coefficient) * physical[:, spec.rhs]
    ) / float(scale)
    return F.smooth_l1_loss(residual, torch.zeros_like(residual), beta=1.0)


@dataclass
class TrainingResult:
    model: base.SpringSemanticHead
    initial_state_sha256: str
    final_state_sha256: str
    history: list[dict[str, float]]
    normalization_mean: np.ndarray
    normalization_std: np.ndarray
    relation_coefficient: float | None
    relation_scale: float
    relation_spec: str | None
    mapped_label_ids: np.ndarray | None
    mapping_sha256: str | None


def train_condition(
    train: base.GroupedFeatures,
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
    label_ids_array = np.asarray(label_ids, dtype=np.int64)
    label_indices = ordered_label_indices(train.ids, label_ids_array)
    label_latents = train.latents[label_indices].astype(np.float64)
    latent_mean = label_latents.mean(axis=0)
    latent_std = label_latents.std(axis=0)
    if np.any(~np.isfinite(latent_std)) or np.any(latent_std <= 1e-8):
        raise ValueError("K-label normalization is degenerate")

    coefficient: float | None = None
    spec: RelationSpec | None = None
    mapped_ids: np.ndarray | None = None
    mapping_sha: str | None = None
    if condition != "point":
        spec = RELATION_SPECS[condition]
        rhs_latents = None
        if condition == "coefficient_permuted_relation":
            mapped_ids = explicit_derangement(label_ids_array, permutation_seed)
            mapped_indices = ordered_label_indices(train.ids, mapped_ids)
            rhs_latents = train.latents[mapped_indices].astype(np.float64)
            mapping_sha = relation_mapping_sha256(label_ids_array, mapped_ids)
        coefficient = fit_relation_coefficient(label_latents, spec, rhs_latents)
        rel_scale = relation_scale(label_latents, spec)
    else:
        rel_scale = base.relation_scale(label_latents)

    features = torch.as_tensor(np.ascontiguousarray(train.features), dtype=torch.float32, device=device)
    targets = torch.as_tensor((train.latents - latent_mean) / latent_std, dtype=torch.float32, device=device)
    label_tensor = torch.as_tensor(label_indices, dtype=torch.long, device=device)
    mean_tensor = torch.as_tensor(latent_mean, dtype=torch.float32, device=device)
    std_tensor = torch.as_tensor(latent_std, dtype=torch.float32, device=device)

    model = base.build_head(feature_dim, hidden_dims, seed, device)
    initial_sha = base.state_dict_sha256(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    history: list[dict[str, float]] = []
    base.set_global_seed(seed)

    for epoch in range(1, epochs + 1):
        model.train()
        predictions = model(features.reshape(-1, feature_dim)).reshape(len(train.ids), 4, -1)
        losses = base.multiview_losses(predictions, range_limit)
        point = F.mse_loss(predictions[label_tensor], targets[label_tensor, None, :].expand(-1, 4, -1))
        relation = torch.zeros((), dtype=predictions.dtype, device=device)
        if coefficient is not None and spec is not None:
            relation = physical_relation_loss(
                predictions.mean(dim=1), mean_tensor, std_tensor, coefficient, rel_scale, spec
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
            history.append({
                "epoch": epoch,
                "total": float(total.detach().cpu()),
                "view": float(losses["view"].detach().cpu()),
                "variance": float(losses["variance"].detach().cpu()),
                "covariance": float(losses["covariance"].detach().cpu()),
                "point": float(point.detach().cpu()),
                "relation": float(relation.detach().cpu()),
                "range": float(losses["range"].detach().cpu()),
            })

    return TrainingResult(
        model=model,
        initial_state_sha256=initial_sha,
        final_state_sha256=base.state_dict_sha256(model.state_dict()),
        history=history,
        normalization_mean=latent_mean.astype(np.float32),
        normalization_std=latent_std.astype(np.float32),
        relation_coefficient=coefficient,
        relation_scale=rel_scale,
        relation_spec=None if spec is None else spec.equation,
        mapped_label_ids=mapped_ids,
        mapping_sha256=mapping_sha,
    )


def json_mapping(label_ids: Sequence[int], mapped_ids: Sequence[int]) -> list[dict[str, int]]:
    return [
        {"source_id": int(source), "mapped_id": int(mapped)}
        for source, mapped in zip(label_ids, mapped_ids)
    ]


def contract_json(result: TrainingResult, label_ids: Sequence[int]) -> dict[str, object]:
    payload: dict[str, object] = {
        "relation_spec": result.relation_spec,
        "relation_coefficient": result.relation_coefficient,
        "relation_scale": result.relation_scale,
        "mapping_sha256": result.mapping_sha256,
    }
    if result.mapped_label_ids is not None:
        payload["explicit_id_mapping"] = json_mapping(label_ids, result.mapped_label_ids)
    # Round-trip through JSON here catches NumPy scalar leakage before a lock is written.
    json.dumps(payload, sort_keys=True)
    return payload
