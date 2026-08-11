import numpy as np
import pytest
import torch

import causalverse_spring_pilot as base
import causalverse_spring_correctness_v3 as module
import evaluate_causalverse_spring_correctness_v3 as evaluator


def _grouped(n_ids: int = 40, feature_dim: int = 8) -> base.GroupedFeatures:
    rng = np.random.default_rng(19)
    ids = np.arange(1000, 1000 + n_ids, dtype=np.int64)
    latents = rng.uniform(0.2, 1.2, size=(n_ids, 5)).astype(np.float64)
    latents[:, 4] = 9.81 * latents[:, 2] / latents[:, 3]
    projection = rng.normal(size=(5, feature_dim))
    base_feature = latents @ projection
    features = np.stack([base_feature + 0.001 * view for view in range(4)], axis=1).astype(np.float32)
    return base.GroupedFeatures(ids=ids[::-1], features=features[::-1], latents=latents[::-1], views=np.arange(4))


def test_ordered_label_indices_follow_label_id_order_not_train_row_order():
    train_ids = np.asarray([30, 10, 40, 20], dtype=np.int64)
    indices = module.ordered_label_indices(train_ids, [10, 20, 30])
    np.testing.assert_array_equal(indices, [1, 3, 0])


def test_explicit_derangement_is_stable_complete_and_hashed_with_id_pairs():
    ids = np.asarray([101, 305, 207, 999], dtype=np.int64)
    first = module.explicit_derangement(ids, 20260733)
    second = module.explicit_derangement(ids, 20260733)
    np.testing.assert_array_equal(first, second)
    assert set(first.tolist()) == set(ids.tolist())
    assert np.all(first != ids)
    assert module.relation_mapping_sha256(ids, first) == module.relation_mapping_sha256(ids, second)
    assert module.relation_mapping_sha256(ids[::-1], first[::-1]) != module.relation_mapping_sha256(ids, first)


def test_correct_and_wrong_relation_specs_have_expected_calibration_behavior():
    grouped = _grouped()
    values = grouped.latents
    correct_spec = module.RELATION_SPECS["correct_relation"]
    correct = module.fit_relation_coefficient(values, correct_spec)
    assert correct == pytest.approx(9.81, abs=1e-10)
    assert np.max(np.abs(module.relation_residual_numpy(values, correct, correct_spec))) < 1e-10

    for name in ("wrong_lm_to_k_relation", "wrong_km_to_l_relation"):
        spec = module.RELATION_SPECS[name]
        coefficient = module.fit_relation_coefficient(values, spec)
        residual = module.relation_residual_numpy(values, coefficient, spec)
        assert np.sqrt(np.mean(residual**2)) > 1e-3


def test_all_v3_conditions_share_initial_state_and_run_without_readout():
    grouped = _grouped(24, 6)
    label_ids = grouped.ids[[3, 7, 1, 11, 5, 13, 17, 19]]
    weights = {"view": 1.0, "variance": 1.0, "covariance": 0.04, "point": 1.0, "relation": 1.0, "range": 0.001}
    results = {}
    for condition in module.CONDITIONS:
        results[condition] = module.train_condition(
            train=grouped,
            label_ids=label_ids,
            condition=condition,
            feature_dim=6,
            hidden_dims=(12, 8),
            seed=42,
            epochs=2,
            learning_rate=1e-3,
            weight_decay=1e-4,
            loss_weights=weights,
            range_limit=6.0,
            permutation_seed=20260733,
            device=torch.device("cpu"),
        )
    assert len({result.initial_state_sha256 for result in results.values()}) == 1
    assert results["point"].relation_coefficient is None
    assert results["correct_relation"].relation_coefficient == pytest.approx(9.81, abs=1e-5)
    permuted = results["coefficient_permuted_relation"]
    assert permuted.mapping_sha256 is not None
    assert permuted.mapped_label_ids is not None
    assert np.all(permuted.mapped_label_ids != label_ids)
    assert all(np.isfinite(entry["total"]) for result in results.values() for entry in result.history)


def test_scale_sensitive_metrics_separate_pearson_from_direct_r2():
    rng = np.random.default_rng(11)
    truth = rng.normal(size=(200, 5))
    predicted = truth.copy()
    predicted[:, 2:5] *= 1.5
    metrics = evaluator.enhanced_metrics(truth, predicted)
    assert metrics["mean_direct_abs_correlation_relation3"] == pytest.approx(1.0)
    assert metrics["relation3_mean_direct_r2"] < 1.0
    assert metrics["relation3_mean_normalized_rmse"] > 0.0


def test_paired_bootstrap_detects_uniformly_better_direct_prediction():
    rng = np.random.default_rng(29)
    truth = rng.normal(size=(120, 5))
    correct = truth + rng.normal(scale=0.05, size=truth.shape)
    control = truth + rng.normal(scale=0.40, size=truth.shape)
    result = evaluator.paired_bootstrap_r2_difference(truth, correct, control, 500, 123)
    assert result["point"] > 0.0
    assert result["ci95_percentile"][0] > 0.0
    assert result["positive_fraction"] > 0.99
