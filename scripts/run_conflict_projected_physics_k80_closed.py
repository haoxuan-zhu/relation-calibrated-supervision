"""Close the RGB supervision ledger of v16 to the registered K=80 subset."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

import run_conflict_projected_physics_from_scratch as v16


RGB_STATISTICS_SOURCE = "k80_calibration_subset_rgb_full_train_known_angles_v17"
V16_RGB_STATISTICS_SOURCE = "all_8000_observational_train_truth_v16_only"
_V16_DECIDE = v16.decide


def normalize_with_k80_rgb_statistics(
    raw_latents: np.ndarray,
    train_end: int,
    subset: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use K RGB rows and all known angle anchors to define target units."""
    subset = np.asarray(subset, dtype=np.int64)
    if subset.ndim != 1 or subset.size == 0:
        raise ValueError("normalization subset must be a non-empty 1D array")
    if np.any(subset < 0) or np.any(subset >= train_end):
        raise ValueError("normalization subset leaves observational train split")
    rgb = raw_latents[0, subset, :3]
    known_angles = raw_latents[0, :train_end, 3:]
    mean = np.concatenate([rgb.mean(axis=0), known_angles.mean(axis=0)])
    std = np.concatenate([rgb.std(axis=0), known_angles.std(axis=0)])
    if np.any(std <= 0):
        raise ValueError("non-positive registered latent standard deviation")
    normalized = (raw_latents - mean[None, None, :]) / std[None, None, :]
    return normalized.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def decide(
    metrics: dict[str, dict[str, Any]],
    training: dict[str, dict[str, Any]],
    post_lock_validation: dict[str, dict[str, Any]],
    lock_before_semantic_and_test: bool,
    calibration_audit: dict[str, Any],
    generated_initial_state_hash: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Reuse frozen v16 effect gates while replacing its normalization validity gate."""
    shadow = copy.deepcopy(config)
    shadow["normalization"]["rgb_statistics_source"] = V16_RGB_STATISTICS_SOURCE
    result = _V16_DECIDE(
        metrics,
        training,
        post_lock_validation,
        lock_before_semantic_and_test,
        calibration_audit,
        generated_initial_state_hash,
        shadow,
    )
    validity = dict(result["validity"])
    validity.pop("full_rgb_statistics_explicit_for_v16")
    registered = config["normalization"]
    validity.update(
        {
            "k80_rgb_statistics_source_registered": registered[
                "rgb_statistics_source"
            ]
            == RGB_STATISTICS_SOURCE,
            "rgb_statistics_budget_exact": int(registered["rgb_statistics_budget"])
            == int(calibration_audit["budget"]),
            "rgb_statistics_subset_matches_calibration": registered[
                "subset_sha256"
            ]
            == calibration_audit["subset_sha256"],
        }
    )
    all_valid = all(validity.values())
    all_gates = all(result["gates"].values())
    relative_keys = [
        "gain_over_vanilla",
        "gain_over_projected_permuted",
        "teacher_gap_fraction_closed",
        "projection_active",
    ]
    seed = int(config["training"]["seed"])
    if not all_valid:
        verdict = f"k80_closed_training_invalid_seed{seed}"
    elif all_gates:
        verdict = f"k80_closed_training_supported_seed{seed}"
    elif all(result["gates"][key] for key in relative_keys):
        verdict = f"k80_closed_functional_signal_insufficient_seed{seed}"
    else:
        verdict = f"k80_closed_training_not_supported_seed{seed}"
    result["verdict"] = verdict
    result["validity"] = validity
    result["next_stage"] = {
        "k80_closed_multiseed_unlocked": bool(all_valid and all_gates),
        "rule": "independent seeds unlock only when every v17 validity and effect gate passes",
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path, required=True)
    known, _ = parser.parse_known_args()
    config_path = known.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    actual_wrapper_sha = v16.base.sha256_file(Path(__file__).resolve())
    expected_wrapper_sha = config["implementation"]["wrapper_sha256"]
    if actual_wrapper_sha != expected_wrapper_sha:
        raise ValueError(
            f"v17 wrapper hash mismatch: {actual_wrapper_sha} != {expected_wrapper_sha}"
        )
    if config["normalization"]["rgb_statistics_source"] != RGB_STATISTICS_SOURCE:
        raise ValueError("v17 must use the registered K=80 RGB statistics source")
    spec, audit = v16.v11.build_calibration_spec(config)
    if audit["subset_sha256"] != config["normalization"]["subset_sha256"]:
        raise ValueError("normalization and teacher calibration subsets differ")
    subset = np.array(spec["subset"], dtype=np.int64, copy=True)

    original_normalize = v16.base.normalize_latents
    original_decide = v16.decide

    def registered_normalize(
        raw_latents: np.ndarray, train_end: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return normalize_with_k80_rgb_statistics(raw_latents, train_end, subset)

    v16.base.normalize_latents = registered_normalize
    v16.decide = decide
    try:
        v16.main()
    finally:
        v16.base.normalize_latents = original_normalize
        v16.decide = original_decide


if __name__ == "__main__":
    main()
