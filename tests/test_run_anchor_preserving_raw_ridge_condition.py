from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_anchor_preserving_raw_ridge_condition",
    ROOT / "scripts/run_anchor_preserving_raw_ridge_condition.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def registry() -> dict:
    return yaml.safe_load(
        (ROOT / "configs/config_anchor_preserving_raw_ridge_v1.yaml").read_text(
            encoding="utf-8"
        )
    )


def test_registry_and_subsets_are_frozen() -> None:
    config = registry()
    MODULE.validate_registry(config)
    for budget in config["budget_curve"]["budgets"]:
        subset, audit = MODULE.budget.build_subset(config, budget)
        assert subset.shape == (budget,)
        assert audit["subset_sha256"] == audit["expected_subset_sha256"]


def test_derived_condition_and_preservation_scope_are_exact() -> None:
    config = MODULE.budget.build_run_config(
        registry(), MODULE.INTERNAL_CONDITION_KIND, 80, 42
    )
    assert config["selected_run"]["condition_name"] == MODULE.anchor.CONDITION
    assert config["anchor_preservation"] == {
        "measured_environments": ["obs"],
        "unmeasured_target_source": "raw_ridge_relation",
        "intervention_target_source": "raw_ridge_relation",
    }
    assert "warm_start" not in config
