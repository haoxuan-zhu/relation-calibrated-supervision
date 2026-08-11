from __future__ import annotations

import evaluate_causalverse_spring_correctness_v4 as evaluator


THRESHOLDS = {
    "required_correct_minus_point_positive_seed_splits": 6,
    "required_correct_minus_coefficient_positive_seed_splits": 5,
    "required_positive_coefficient_validation_bootstrap_seeds": 2,
}


def test_primary_replication_is_not_controlled_by_topology_or_soft_effect_size() -> None:
    decision, supported = evaluator.make_decision(
        valid=True,
        point_positive_count=6,
        coefficient_positive_count=5,
        coefficient_split_means_positive=True,
        coefficient_ci_positive_count=2,
        free2_consistent_degradation=False,
        relation_residual_means_better=True,
        topology_all_positive=False,
        thresholds=THRESHOLDS,
    )
    assert supported
    assert decision == "unread_shard_primary_replication_supported_pending_state_disjoint"


def test_invalid_contract_cannot_become_method_failure_or_success() -> None:
    decision, supported = evaluator.make_decision(
        valid=False,
        point_positive_count=6,
        coefficient_positive_count=6,
        coefficient_split_means_positive=True,
        coefficient_ci_positive_count=3,
        free2_consistent_degradation=False,
        relation_residual_means_better=True,
        topology_all_positive=True,
        thresholds=THRESHOLDS,
    )
    assert not supported
    assert decision == "causalverse_spring_correctness_v4_invalid"


def test_point_only_result_is_classified_as_relation_regularization_only() -> None:
    decision, supported = evaluator.make_decision(
        valid=True,
        point_positive_count=6,
        coefficient_positive_count=4,
        coefficient_split_means_positive=True,
        coefficient_ci_positive_count=1,
        free2_consistent_degradation=False,
        relation_residual_means_better=True,
        topology_all_positive=True,
        thresholds=THRESHOLDS,
    )
    assert not supported
    assert decision == "unread_shard_relation_regularization_only"
