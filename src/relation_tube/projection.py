from __future__ import annotations

import torch
from torch import nn


def project_residual(
    residual: torch.Tensor,
    precision: torch.Tensor,
    radius_squared: torch.Tensor | float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project residuals onto a calibrated ellipsoid without changing direction."""

    if residual.ndim < 2:
        raise ValueError("residual must have a batch dimension")
    dimension = residual.shape[-1]
    if precision.shape != (dimension, dimension):
        raise ValueError("precision has the wrong shape")
    score = torch.einsum("...i,ij,...j->...", residual, precision, residual)
    radius = torch.as_tensor(radius_squared, dtype=residual.dtype, device=residual.device)
    if radius.ndim != 0 or bool(radius <= 0):
        raise ValueError("radius_squared must be positive")
    scale = torch.sqrt(radius / score.clamp_min(1e-12)).clamp(max=1.0)
    return residual * scale.unsqueeze(-1), scale


class TubeProjector(nn.Module):
    """Torch layer for the bounded residual branch."""

    def __init__(self, precision: torch.Tensor, radius_squared: float):
        super().__init__()
        matrix = torch.as_tensor(precision, dtype=torch.float32)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("precision must be square")
        if radius_squared <= 0.0:
            raise ValueError("radius_squared must be positive")
        self.register_buffer("precision", matrix)
        self.register_buffer("radius_squared", torch.tensor(float(radius_squared)))

    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        projected, _ = project_residual(residual, self.precision, self.radius_squared)
        return projected
