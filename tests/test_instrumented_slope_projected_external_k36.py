from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate_instrumented_slope_projected_external_k36 as k36


def test_external_protocol_matches_replication_seeds() -> None:
    config = k36.load_protocol(
        ROOT / "configs" / "config_instrumented_slope_projected_external_k36.yaml"
    )
    assert tuple(config["evaluation"]["seeds"]) == k36.SEEDS
    assert config["evaluation"]["bootstrap_repetitions"] == 2000
    assert config["evaluation"]["candidates"] == [
        "raw_baseline_curve",
        "projected_only_curve",
    ]
