from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "bounded_unbounded_heldout_k45",
    ROOT / "scripts/audit_bounded_unbounded_heldout_k45.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_contract_is_read_only_and_fixed_to_k80_controls() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/config_bounded_unbounded_heldout_k45.yaml").read_text(
            encoding="utf-8"
        )
    )
    MODULE.validate_contract(config)
    assert config["evaluation"]["training_allowed"] is False
    assert config["evaluation"]["selection_allowed"] is False
    assert config["evaluation"]["conditions"] == [
        "unbounded_correct",
        "tube_correct",
    ]
    assert set(config["upstream"]) == {0, 42, 3407}


def test_summary_uses_paired_bounded_minus_unbounded_differences() -> None:
    runs = {}
    for seed in MODULE.SEEDS:
        blocks = {}
        for condition, shift in (("unbounded_correct", 0.0), ("tube_correct", 0.2)):
            semantic = {
                name: {
                    "mean_direct_abs_correlation": 0.4 + shift,
                    "mean_r2": 0.1 + shift,
                }
                for name in ("raw", "coordinatewise_affine", "full_affine")
            }
            blocks[condition] = {"semantic": semantic}
        runs[str(seed)] = blocks
    result = MODULE.summarize(runs)
    assert result["performance_threshold_used"] is False
    assert result["geometry_selection_reopened"] is False
    for readout in ("raw", "coordinatewise_affine", "full_affine"):
        assert result["direction_counts"][readout] == {
            "bounded_minus_unbounded_correlation": 3,
            "bounded_minus_unbounded_r2": 3,
        }
