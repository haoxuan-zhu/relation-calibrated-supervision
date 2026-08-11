import hashlib
import subprocess
from pathlib import Path

import numpy as np
import pytest

import audit_bounded_unbounded_heldout_k45_v2 as audit


ROOT = Path(__file__).resolve().parents[1]


def test_historical_seed3407_entrypoint_is_recoverable() -> None:
    if not (ROOT / ".git").exists():
        pytest.skip("the copied remote project uses the registered source snapshot")
    commit = "097a1319a65f488eaa94eaf75781429852e63afa"
    payload = subprocess.run(
        [
            "git",
            "-C",
            str(ROOT),
            "show",
            f"{commit}:scripts/run_calibrated_relation_tube_k0.py",
        ],
        check=True,
        capture_output=True,
    ).stdout
    assert hashlib.sha256(payload).hexdigest() == (
        "1229789aa71789a7fac640b5843b123e136c6a19281392d53950661fabde8d28"
    )


def test_semantic_metric_comparison_covers_scalars_and_vectors() -> None:
    metrics = {
        "mean_direct_abs_correlation": 0.8,
        "mean_r2": 0.6,
        "mse": 0.4,
        "direct_abs_correlation": [0.7, 0.8, 0.9],
        "r2": [0.5, 0.6, 0.7],
    }
    close = {key: value[:] if isinstance(value, list) else value for key, value in metrics.items()}
    close["r2"][1] += 1e-7
    far = {key: value[:] if isinstance(value, list) else value for key, value in metrics.items()}
    far["mean_r2"] += 1e-3
    assert audit.semantic_metrics_close(close, metrics, 5e-6)
    assert not audit.semantic_metrics_close(far, metrics, 5e-6)


def test_summary_keeps_the_original_three_readouts() -> None:
    runs = {}
    for seed in audit.SEEDS:
        runs[str(seed)] = {}
        for condition, offset in (("unbounded_correct", 0.0), ("tube_correct", 0.1)):
            runs[str(seed)][condition] = {
                "semantic": {
                    name: {
                        "mean_direct_abs_correlation": 0.5 + offset,
                        "mean_r2": 0.2 + offset,
                    }
                    for name in ("raw", "coordinatewise_affine", "full_affine")
                }
            }
    summary = audit.k45.summarize(runs)
    assert all(
        summary["direction_counts"][name]["bounded_minus_unbounded_r2"] == 3
        for name in ("raw", "coordinatewise_affine", "full_affine")
    )
