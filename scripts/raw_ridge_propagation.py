"""Frozen raw-feature ridge teacher for the raw-ridge propagation study.

The teacher deliberately receives only image-channel means and the observed
angle statistic tau.  It does not receive the handcrafted m*tau interaction
used by the existing empirical control.
"""

from __future__ import annotations

import hashlib
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F

import audit_physics_functional_anchor as v10
import run_clamped_anchor_diagnostic as v2
import run_conflict_projected_physics_training as v13
import run_diagnostic as base


CONDITION = "raw_ridge_correct_floor"
DEFAULT_ALPHAS = (0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)


def raw_features(means: np.ndarray, tau: np.ndarray) -> np.ndarray:
    means = np.asarray(means, dtype=np.float64)
    tau = np.asarray(tau, dtype=np.float64).reshape(-1, 1)
    if means.ndim != 2 or means.shape[1] != 3 or means.shape[0] != tau.shape[0]:
        raise ValueError("raw ridge expects matched N-by-3 means and N tau values")
    return np.concatenate([means, tau], axis=1)


def _fit_standardized_ridge(
    features: np.ndarray, targets: np.ndarray, alpha: float
) -> dict[str, np.ndarray | float]:
    features = np.asarray(features, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if features.ndim != 2 or features.shape[1] != 4:
        raise ValueError("raw ridge features must be N-by-4")
    if targets.shape != (features.shape[0], 3):
        raise ValueError("raw ridge targets must be N-by-3")
    if features.shape[0] < 2:
        raise ValueError("raw ridge needs at least two calibration rows")
    if alpha < 0:
        raise ValueError("ridge alpha must be nonnegative")

    center = features.mean(axis=0)
    scale = features.std(axis=0)
    if not np.all(np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("raw ridge calibration features have degenerate scale")
    standardized = (features - center) / scale
    design = np.concatenate(
        [np.ones((features.shape[0], 1), dtype=np.float64), standardized], axis=1
    )
    penalty = np.diag([0.0, alpha, alpha, alpha, alpha])
    coefficients = np.linalg.pinv(design.T @ design + penalty) @ design.T @ targets
    return {
        "alpha": float(alpha),
        "feature_mean": center,
        "feature_scale": scale,
        "coefficients": coefficients,
    }


def predict_raw_ridge(
    teacher: dict[str, Any], features: np.ndarray, clip: bool
) -> np.ndarray:
    features = np.asarray(features, dtype=np.float64)
    center = np.asarray(teacher["feature_mean"], dtype=np.float64)
    scale = np.asarray(teacher["feature_scale"], dtype=np.float64)
    coefficients = np.asarray(teacher["coefficients"], dtype=np.float64)
    if center.shape != (4,) or scale.shape != (4,) or coefficients.shape != (5, 3):
        raise ValueError("invalid raw ridge teacher dimensions")
    design = np.concatenate(
        [
            np.ones((features.shape[0], 1), dtype=np.float64),
            (features - center) / scale,
        ],
        axis=1,
    )
    prediction = design @ coefficients
    return np.clip(prediction, 0.0, 1.0) if clip else prediction


def fit_raw_ridge(
    features: np.ndarray,
    targets: np.ndarray,
    alphas: Iterable[float] = DEFAULT_ALPHAS,
) -> tuple[dict[str, Any], dict[str, Any]]:
    features = np.asarray(features, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    candidates = tuple(float(value) for value in alphas)
    if not candidates or tuple(sorted(set(candidates))) != candidates:
        raise ValueError("ridge alpha candidates must be unique and increasing")

    cv_curve: list[dict[str, float]] = []
    best_alpha: float | None = None
    best_mse = np.inf
    for alpha in candidates:
        predictions = np.empty_like(targets)
        for holdout in range(features.shape[0]):
            keep = np.arange(features.shape[0]) != holdout
            fold_teacher = _fit_standardized_ridge(
                features[keep], targets[keep], alpha
            )
            predictions[holdout] = predict_raw_ridge(
                fold_teacher, features[holdout : holdout + 1], clip=False
            )[0]
        mse = float(np.mean((predictions - targets) ** 2))
        cv_curve.append({"alpha": alpha, "leave_one_out_mse": mse})
        if mse < best_mse - 1e-12 or (
            abs(mse - best_mse) <= 1e-12
            and (best_alpha is None or alpha > best_alpha)
        ):
            best_alpha = alpha
            best_mse = mse

    assert best_alpha is not None
    teacher = _fit_standardized_ridge(features, targets, best_alpha)
    fitted = predict_raw_ridge(teacher, features, clip=False)
    coefficient_bytes = np.ascontiguousarray(
        teacher["coefficients"], dtype=np.float64
    ).tobytes()
    audit = {
        "family": "raw_ridge_[m,tau]_to_rgb",
        "alpha_candidates": list(candidates),
        "selected_alpha": best_alpha,
        "selection_rule": "deterministic_leave_one_out_mse_ties_choose_larger_alpha",
        "leave_one_out_curve": cv_curve,
        "leave_one_out_mse": best_mse,
        "design_shape": [int(features.shape[0]), 5],
        "coefficient_shape": [5, 3],
        "coefficient_count": 15,
        "coefficients_sha256": hashlib.sha256(coefficient_bytes).hexdigest(),
        "calibration_mse": float(np.mean((fitted - targets) ** 2)),
    }
    return teacher, audit


def fit_teacher_from_calibration_rows(
    images: np.ndarray,
    raw_latents: np.ndarray,
    subset: np.ndarray,
    train_end: int,
    alphas: Iterable[float] = DEFAULT_ALPHAS,
) -> tuple[dict[str, Any], dict[str, Any]]:
    subset = np.asarray(subset, dtype=np.int64)
    means = v10.image_channel_means(images, 0, 0, train_end)
    tau = v10.malus_tau(raw_latents)[0, :train_end]
    unit_rgb = raw_latents[0, :train_end, :3].astype(np.float64) / 255.0
    teacher, audit = fit_raw_ridge(
        raw_features(means[subset], tau[subset]), unit_rgb[subset], alphas
    )
    audit.update(
        {
            "fit_environment": "obs",
            "label_rows": subset.tolist(),
            "unique_label_rows": int(np.unique(subset).size),
            "semantic_rows_read": int(subset.size),
            "tau_source": "cos_squared_known_angle_difference",
        }
    )
    return teacher, audit


def torch_pseudo_target(
    x: torch.Tensor,
    raw_angles: torch.Tensor,
    teacher: dict[str, torch.Tensor],
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
    clip: bool,
) -> torch.Tensor:
    means = x.mean(dim=(2, 3))
    tau = torch.cos(torch.deg2rad(raw_angles[:, 0] - raw_angles[:, 1])).square()
    features = torch.cat([means, tau[:, None]], dim=1)
    standardized = (features - teacher["feature_mean"]) / teacher["feature_scale"]
    design = torch.cat(
        [torch.ones((features.shape[0], 1), dtype=x.dtype, device=x.device), standardized],
        dim=1,
    )
    unit_rgb = design @ teacher["coefficients"]
    if clip:
        unit_rgb = unit_rgb.clamp(0.0, 1.0)
    return (unit_rgb * 255.0 - latent_mean[:3]) / latent_std[:3]


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
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    if condition.name != CONDITION:
        raise ValueError(f"unexpected raw-ridge condition: {condition.name}")
    ccrl, metrics = v2.objective_for_batch(
        model, images, latents, anchor_maps, condition, rows, targets, config, device
    )
    anchor_indices = list(config["model"]["anchor_indices"])
    losses = []
    for environments in (np.zeros_like(rows), targets + 1):
        x = base.image_batch(images, environments, rows, device)
        anchors = torch.from_numpy(
            np.array(latents[environments, rows][:, anchor_indices], copy=True)
        ).to(device)
        raw_angles = torch.from_numpy(
            np.array(raw_latents[environments, rows][:, anchor_indices], copy=True)
        ).to(device)
        pseudo = torch_pseudo_target(
            x,
            raw_angles,
            coefficients,
            latent_mean,
            latent_std,
            bool(config["raw_ridge_anchor"]["clip_predictions_to_unit_interval"]),
        )
        losses.append(F.mse_loss(model.embedding(x, anchors), pseudo))
    return ccrl, torch.stack(losses).mean(), metrics


def validation_metrics(
    teacher: dict[str, Any],
    images: np.ndarray,
    raw_latents: np.ndarray,
    train_end: int,
    validation_end: int,
    clip: bool,
) -> dict[str, Any]:
    means = np.stack(
        [
            v10.image_channel_means(images, environment, train_end, validation_end)
            for environment in range(images.shape[0])
        ],
        axis=0,
    )
    tau = v10.malus_tau(raw_latents)[:, train_end:validation_end]
    truth = raw_latents[:, train_end:validation_end, :3].astype(np.float64) / 255.0
    prediction = predict_raw_ridge(
        teacher,
        raw_features(means.reshape(-1, 3), tau.reshape(-1)),
        clip,
    )
    return v10.regression_metrics(prediction, truth.reshape(-1, 3))
