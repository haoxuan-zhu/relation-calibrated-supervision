"""Raw-ridge condition for the frozen calibration-subset robustness blocks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import raw_ridge_propagation as raw
import run_raw_ridge_propagation_condition as raw_condition


budget = raw_condition.budget
PROTOCOL = "raw_ridge_propagation_subset_robustness_v1"
ALLOWED_SUBSET_SEEDS = (20260811, 20260821, 20260831)
INTERNAL_CONDITION_KIND = raw_condition.INTERNAL_CONDITION_KIND


def validate_registry(registry: dict[str, Any]) -> None:
    if registry["protocol_version"] != PROTOCOL:
        raise ValueError("unexpected raw-ridge subset-robustness protocol")
    if list(registry["budget_curve"]["budgets"]) != [20, 40, 80, 160]:
        raise ValueError("raw-ridge subset budget order changed")
    if list(registry["budget_curve"]["conditions"]) != [
        INTERNAL_CONDITION_KIND
    ]:
        raise ValueError("raw-ridge subset protocol accepts one isolated condition")
    if list(registry["budget_curve"]["seeds"]) != [3407]:
        raise ValueError("raw-ridge subset robustness fixes model initialization")
    subset_seed = int(registry["budget_curve"]["subset_seed"])
    if subset_seed not in ALLOWED_SUBSET_SEEDS:
        raise ValueError("unregistered raw-ridge calibration subset seed")
    if int(registry["subset_replication"]["seed"]) != subset_seed:
        raise ValueError("raw-ridge subset seed metadata mismatch")
    if registry["initialization"]["mode"] != "same_seed_random_no_warm":
        raise ValueError("raw-ridge subset runs must start from the registered state")
    if tuple(float(value) for value in registry["raw_ridge_anchor"]["alphas"]) != raw.DEFAULT_ALPHAS:
        raise ValueError("raw-ridge alpha registry changed")

    implementation = registry["implementation"]
    if budget.base.sha256_file(Path(__file__).resolve()) != implementation[
        "entrypoint_sha256"
    ]:
        raise ValueError("raw-ridge subset entrypoint hash mismatch")
    if budget.base.sha256_file(Path(raw_condition.__file__).resolve()) != implementation[
        "raw_condition_entrypoint_sha256"
    ]:
        raise ValueError("raw-ridge base condition hash mismatch")
    if budget.base.sha256_file(Path(raw.__file__).resolve()) != implementation[
        "teacher_module_sha256"
    ]:
        raise ValueError("raw-ridge teacher-module hash mismatch")


def install_protocol_hooks() -> None:
    # The imported teacher-audit writer reads the base module constant directly,
    # while the formal result writer reads ``budget.PROTOCOL``.  Keep both
    # identities synchronized before either path can run.
    raw_condition.PROTOCOL = PROTOCOL
    budget.PROTOCOL = PROTOCOL
    budget.SOURCE_FILES = list(
        dict.fromkeys(
            [
                *budget.SOURCE_FILES,
                "run_raw_ridge_propagation_subset_robustness.py",
            ]
        )
    )
    budget.validate_registry = validate_registry


install_protocol_hooks()


if __name__ == "__main__":
    budget.main()
