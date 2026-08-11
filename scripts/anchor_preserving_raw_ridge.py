"""Raw-ridge propagation that retains the measured observational targets."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

import raw_ridge_propagation as raw
import run_clamped_anchor_diagnostic as v2
import run_diagnostic as base


CONDITION = "anchor_preserving_raw_ridge_floor"


def measured_row_mask(rows: np.ndarray, subset: np.ndarray) -> np.ndarray:
    """Mark batch rows whose observational RGB values were measured."""
    rows = np.asarray(rows, dtype=np.int64)
    subset = np.asarray(subset, dtype=np.int64)
    if rows.ndim != 1 or subset.ndim != 1 or subset.size == 0:
        raise ValueError("rows and the non-empty measured subset must be 1D")
    if np.unique(subset).size != subset.size:
        raise ValueError("measured subset contains duplicate rows")
    return np.isin(rows, subset, assume_unique=False)


def retain_measured_observations(
    pseudo_target: torch.Tensor,
    observational_truth: torch.Tensor,
    rows: np.ndarray,
    subset: np.ndarray,
) -> tuple[torch.Tensor, int]:
    """Replace relation targets only where observational measurements exist."""
    if pseudo_target.shape != observational_truth.shape:
        raise ValueError("pseudo targets and measured truth must have equal shape")
    mask_np = measured_row_mask(rows, subset)
    if pseudo_target.shape[0] != mask_np.size:
        raise ValueError("target batch and row batch differ")
    mask = torch.from_numpy(mask_np).to(device=pseudo_target.device)
    mixed = torch.where(mask[:, None], observational_truth, pseudo_target)
    return mixed, int(mask_np.sum())


def loss_components(
    model: torch.nn.Module,
    condition: v2.Condition,
    images: np.ndarray,
    latents: np.ndarray,
    raw_latents: np.ndarray,
    anchor_maps: dict[str, np.ndarray],
    rows: np.ndarray,
    targets: np.ndarray,
    config: dict[str, Any],
    device: torch.device,
    coefficients: dict[str, torch.Tensor],
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
    subset: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Compute CCRL plus dense propagation with measured rows retained."""
    if condition.name != CONDITION:
        raise ValueError(f"unexpected anchor-preserving condition: {condition.name}")
    ccrl, metrics = v2.objective_for_batch(
        model, images, latents, anchor_maps, condition, rows, targets, config, device
    )
    anchor_indices = list(config["model"]["anchor_indices"])
    learned_indices = list(config["model"]["learned_indices"])
    losses: list[torch.Tensor] = []
    retained = 0

    for position, environments in enumerate((np.zeros_like(rows), targets + 1)):
        x = base.image_batch(images, environments, rows, device)
        anchors = torch.from_numpy(
            np.array(latents[environments, rows][:, anchor_indices], copy=True)
        ).to(device)
        raw_angles = torch.from_numpy(
            np.array(raw_latents[environments, rows][:, anchor_indices], copy=True)
        ).to(device)
        pseudo = raw.torch_pseudo_target(
            x,
            raw_angles,
            coefficients,
            latent_mean,
            latent_std,
            bool(config["raw_ridge_anchor"]["clip_predictions_to_unit_interval"]),
        )
        target = pseudo
        if position == 0:
            observational_truth = torch.from_numpy(
                np.array(latents[0, rows][:, learned_indices], copy=True)
            ).to(device)
            target, retained = retain_measured_observations(
                pseudo, observational_truth, rows, subset
            )
        losses.append(F.mse_loss(model.embedding(x, anchors), target))

    metrics = {
        **metrics,
        "measured_observations_in_batch": float(retained),
    }
    return ccrl, torch.stack(losses).mean(), metrics
