from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "evaluate_movi_collision_formal_relation_v1.py"
SPEC = importlib.util.spec_from_file_location("movi_formal_relation_v1", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def test_equal_weight_multiview_preserves_ten_unit_blocks() -> None:
    rng = np.random.default_rng(7)
    train_names = np.asarray(["a", "b", "c"])
    evaluation_names = np.asarray(["d", "e"])
    train_archive = {
        "video_name": train_names,
        "pair_cls": rng.normal(size=(3, 3, 4)),
        "object_cls": rng.normal(size=(3, 2, 3, 4)),
        "mask_motion": rng.normal(size=(3, 3, 2, 8)),
    }
    evaluation_archive = {
        "video_name": evaluation_names,
        "pair_cls": rng.normal(size=(2, 3, 4)),
        "object_cls": rng.normal(size=(2, 2, 3, 4)),
        "mask_motion": rng.normal(size=(2, 3, 2, 8)),
    }
    train, evaluation, record = module.visual_features(
        train_names,
        evaluation_names,
        train_archive,
        evaluation_archive,
        {"visual_representation": "equal_weight_multiview_v1"},
    )
    np.testing.assert_allclose(np.linalg.norm(train, axis=1), 1.0, atol=1.0e-7)
    np.testing.assert_allclose(np.linalg.norm(evaluation, axis=1), 1.0, atol=1.0e-7)
    assert train.shape == (3, 84)
    assert evaluation.shape == (2, 84)
    assert record["dino_view_blocks"] == 9


def test_isotropic_projection_respects_radius() -> None:
    center = np.zeros((2, 3))
    point = np.asarray([[3.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    projected, rate = module.project_isotropic(point, center, 1.0)
    np.testing.assert_allclose(np.linalg.norm(projected - center, axis=1), [1.0, 0.5])
    assert rate == 0.5


def test_formal_evaluator_uses_materialized_label_order_hash() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'train["label_order_hash"]' in source
    assert 'train["label_hash"]' not in source


def test_residual_input_branch_returns_matched_outputs() -> None:
    rng = np.random.default_rng(19)
    train_x = rng.normal(size=(8, 4))
    evaluation_x = rng.normal(size=(3, 4))
    train_base = rng.normal(size=(8, 3))
    train_target = 1.4 * train_base + 0.05 * rng.normal(size=(8, 3))
    center = 1.4 * rng.normal(size=(3, 3))
    result = module.cross_fitted_conic_branch(
        train_x,
        evaluation_x,
        train_base,
        train_target,
        center,
        np.arange(8),
        alpha=0.1,
        bounds=(1.0, 2.0),
        coverage=0.95,
    )
    assert result["unbounded"].shape == (3, 3)
    assert result["gated"].shape == (3, 3)
    assert result["conic"].shape == (3, 3)
    assert 0.0 <= result["beta"] <= 1.0
