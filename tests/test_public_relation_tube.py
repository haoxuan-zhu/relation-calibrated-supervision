from __future__ import annotations

import numpy as np
import torch

import raw_ridge_propagation as legacy_center
import run_isotropic_relation_tube_budget_k11 as legacy_isotropic
from relation_tube import (
    TubeProjector,
    calibrate_tube,
    fit_ridge_center,
    fit_ridge_center_with_loo,
    project_residual,
)


def test_reference_center_matches_experiment_implementation() -> None:
    rng = np.random.default_rng(17)
    features = rng.normal(size=(24, 4))
    targets = rng.normal(size=(24, 3))

    public = fit_ridge_center(features, targets, alpha=0.1)
    legacy = legacy_center._fit_standardized_ridge(features, targets, 0.1)

    np.testing.assert_allclose(public.feature_mean, legacy["feature_mean"])
    np.testing.assert_allclose(public.feature_scale, legacy["feature_scale"])
    np.testing.assert_allclose(public.coefficients, legacy["coefficients"])
    np.testing.assert_allclose(
        public.predict(features),
        legacy_center.predict_raw_ridge(legacy, features, clip=False),
    )


def test_reference_loo_predictions_match_fold_fits() -> None:
    rng = np.random.default_rng(23)
    features = rng.normal(size=(12, 4))
    targets = rng.normal(size=(12, 3))
    center, predictions = fit_ridge_center_with_loo(features, targets, alpha=0.1)

    expected = np.empty_like(targets)
    for held_out in range(len(features)):
        keep = np.arange(len(features)) != held_out
        fold = legacy_center._fit_standardized_ridge(features[keep], targets[keep], 0.1)
        expected[held_out] = legacy_center.predict_raw_ridge(
            fold, features[held_out : held_out + 1], clip=False
        )[0]
    np.testing.assert_allclose(predictions, expected)
    np.testing.assert_allclose(center.predict(features).shape, targets.shape)
def test_isotropic_geometry_matches_locked_experiment_code() -> None:
    residuals = np.random.default_rng(31).normal(size=(80, 3))
    public = calibrate_tube(
        residuals, coverage=0.95, ridge=1e-6, geometry="isotropic"
    )
    legacy, audit = legacy_isotropic.build_isotropic_geometry(residuals, 0.95, 1e-6)

    np.testing.assert_allclose(public.second_moment, legacy["covariance"])
    np.testing.assert_allclose(public.precision, legacy["precision"])
    assert public.radius_squared == legacy["radius_squared"]
    assert public.order_index == audit["score_order_index_one_based"]
    assert public.empirical_coverage == audit["empirical_coverage"]


def test_projection_preserves_interior_and_clips_exterior() -> None:
    precision = torch.eye(3)
    residual = torch.tensor([[0.2, 0.0, 0.0], [3.0, 4.0, 0.0]])
    projected, scale = project_residual(residual, precision, 1.0)

    torch.testing.assert_close(projected[0], residual[0])
    torch.testing.assert_close(scale, torch.tensor([1.0, 0.2]))
    torch.testing.assert_close((projected[1] @ projected[1]), torch.tensor(1.0))

    layer = TubeProjector(precision, 1.0)
    torch.testing.assert_close(layer(residual), projected)
