from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "anchor_preserving_raw_ridge",
    ROOT / "scripts/anchor_preserving_raw_ridge.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_measured_row_mask_tracks_only_registered_rows() -> None:
    rows = np.array([8, 2, 5, 2, 9], dtype=np.int64)
    mask = MODULE.measured_row_mask(rows, np.array([2, 9], dtype=np.int64))
    np.testing.assert_array_equal(mask, [False, True, False, True, True])


def test_measured_observations_replace_relation_targets_only_on_mask() -> None:
    pseudo = torch.tensor([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]])
    truth = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    mixed, retained = MODULE.retain_measured_observations(
        pseudo,
        truth,
        np.array([4, 5, 6]),
        np.array([5]),
    )
    torch.testing.assert_close(
        mixed, torch.tensor([[10.0, 10.0], [2.0, 2.0], [30.0, 30.0]])
    )
    assert retained == 1
    torch.testing.assert_close(pseudo, torch.tensor([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]]))


def test_duplicate_measurements_are_rejected() -> None:
    with np.testing.assert_raises(ValueError):
        MODULE.measured_row_mask(np.array([0]), np.array([0, 0]))
