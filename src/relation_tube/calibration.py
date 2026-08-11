from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class RidgeCenter:
    alpha: float
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    coefficients: np.ndarray

    def predict(self, features: np.ndarray, *, clip: bool = False) -> np.ndarray:
        x = _matrix("features", features)
        if x.shape[1] != self.feature_mean.size:
            raise ValueError("feature dimension does not match the fitted center")
        design = np.column_stack(
            (np.ones(x.shape[0], dtype=np.float64), (x - self.feature_mean) / self.feature_scale)
        )
        prediction = design @ self.coefficients
        return np.clip(prediction, 0.0, 1.0) if clip else prediction


@dataclass(frozen=True)
class TubeGeometry:
    kind: str
    second_moment: np.ndarray
    precision: np.ndarray
    radius_squared: float
    coverage: float
    order_index: int
    empirical_coverage: float


def _matrix(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite matrix")
    return array


def fit_ridge_center(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    alpha: float,
) -> RidgeCenter:
    """Fit the affine ridge center used by the relation-tube experiments."""

    x = _matrix("features", features)
    y = _matrix("targets", targets)
    if x.shape[0] != y.shape[0] or x.shape[0] < 2:
        raise ValueError("features and targets need the same number of calibration rows")
    if alpha < 0:
        raise ValueError("alpha must be nonnegative")

    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    if np.any(scale <= 0):
        raise ValueError("calibration features contain a constant column")

    design = np.column_stack((np.ones(x.shape[0]), (x - mean) / scale))
    penalty = np.diag(np.r_[0.0, np.full(x.shape[1], float(alpha))])
    coefficients = np.linalg.pinv(design.T @ design + penalty) @ design.T @ y
    return RidgeCenter(float(alpha), mean, scale, coefficients)


def fit_ridge_center_with_loo(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    alpha: float,
) -> tuple[RidgeCenter, np.ndarray]:
    """Fit the center and return one held-out prediction per calibration row."""

    x = _matrix("features", features)
    y = _matrix("targets", targets)
    if x.shape[0] != y.shape[0] or x.shape[0] < 3:
        raise ValueError("leave-one-out calibration needs at least three paired rows")

    predictions = np.empty_like(y)
    for held_out in range(x.shape[0]):
        keep = np.arange(x.shape[0]) != held_out
        fold = fit_ridge_center(x[keep], y[keep], alpha=alpha)
        predictions[held_out] = fold.predict(x[held_out : held_out + 1])[0]
    return fit_ridge_center(x, y, alpha=alpha), predictions


def calibrate_tube(
    residuals: np.ndarray,
    *,
    coverage: float = 0.95,
    ridge: float = 1e-6,
    geometry: Literal["isotropic", "diagonal", "full"] = "isotropic",
) -> TubeGeometry:
    """Calibrate an empirical Mahalanobis radius from held-out residuals."""

    values = _matrix("residuals", residuals)
    if values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError("residual calibration needs at least two rows")
    if not 0.0 < coverage < 1.0 or ridge <= 0.0:
        raise ValueError("coverage and ridge are outside their valid ranges")

    second_moment = values.T @ values / values.shape[0]
    second_moment += ridge * np.eye(values.shape[1], dtype=np.float64)
    if geometry == "isotropic":
        second_moment = np.eye(values.shape[1]) * (
            np.trace(second_moment) / values.shape[1]
        )
    elif geometry == "diagonal":
        second_moment = np.diag(np.diag(second_moment))
    elif geometry != "full":
        raise ValueError(f"unknown geometry: {geometry}")

    eigenvalues = np.linalg.eigvalsh(second_moment)
    if eigenvalues.min() <= 0.0:
        raise ValueError("tube geometry is not positive definite")
    precision = np.linalg.inv(second_moment)
    scores = np.einsum("ni,ij,nj->n", values, precision, values)
    order_index = min(len(scores), math.ceil((len(scores) + 1) * coverage))
    radius_squared = float(np.sort(scores)[order_index - 1])
    empirical_coverage = float(np.mean(scores <= radius_squared + 1e-12))
    return TubeGeometry(
        kind=geometry,
        second_moment=second_moment,
        precision=precision,
        radius_squared=radius_squared,
        coverage=float(coverage),
        order_index=order_index,
        empirical_coverage=empirical_coverage,
    )
