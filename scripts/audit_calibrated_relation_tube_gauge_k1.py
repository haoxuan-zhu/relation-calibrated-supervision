"""Validation-only sparse gauge audit for the calibrated relation tube."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

import run_calibrated_relation_tube_k0 as tube
import run_clamped_anchor_diagnostic as v2
import run_conditioned_film_diagnostic as v3
import run_conflict_projected_physics_k80_closed as v17
import run_diagnostic as base
import run_physics_functional_anchor_training as v11
import run_sparse_alignment_audit as sparse
import run_supervised_continuation_diagnostic as v6


PROTOCOL = "calibrated_relation_tube_gauge_k1_v1"
RUNS = (
    "center_only_correct",
    "unbounded_correct",
    "tube_correct",
    "tube_permuted_matched",
    "projected_physical_correct",
)
SOURCE_FILES = (
    "audit_calibrated_relation_tube_gauge_k1.py",
    "run_calibrated_relation_tube_k0.py",
    "run_sparse_alignment_audit.py",
    "run_conflict_projected_physics_k80_closed.py",
    "run_conditioned_film_diagnostic.py",
    "run_clamped_anchor_diagnostic.py",
    "run_supervised_continuation_diagnostic.py",
    "run_diagnostic.py",
)


def source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    return {name: base.sha256_file(root / name) for name in SOURCE_FILES}


def validate_config(config: dict[str, Any]) -> None:
    if config["protocol_version"] != PROTOCOL:
        raise ValueError("unexpected gauge-audit protocol")
    if int(config["calibration"]["budget"]) != 80:
        raise ValueError("gauge K1 is locked to K80")
    if not np.isclose(float(config["calibration"]["ridge"]), 0.001):
        raise ValueError("gauge ridge changed")
    if list(config["evaluation"]["tube_conditions"]) != list(tube.CONDITIONS):
        raise ValueError("tube condition registry changed")
    if not np.isclose(
        float(
            config["evaluation"][
                "maximum_full_minus_diagonal_r2_for_scale_bias_explanation"
            ]
        ),
        0.02,
    ):
        raise ValueError("scale/bias diagnostic tolerance changed")


def fit_coordinatewise_affine(
    prediction: np.ndarray, truth: np.ndarray, ridge: float
) -> dict[str, np.ndarray | float]:
    prediction = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if prediction.shape != truth.shape or prediction.ndim != 2:
        raise ValueError("coordinatewise affine inputs must have matched matrices")
    mean = prediction.mean(axis=0)
    scale = np.maximum(prediction.std(axis=0), 1e-6)
    standardized = (prediction - mean) / scale
    intercept = np.empty(prediction.shape[1], dtype=np.float64)
    slope = np.empty(prediction.shape[1], dtype=np.float64)
    for index in range(prediction.shape[1]):
        design = np.stack(
            [np.ones(len(prediction)), standardized[:, index]], axis=1
        )
        penalty = np.diag([0.0, ridge])
        weights = np.linalg.solve(
            design.T @ design + penalty, design.T @ truth[:, index]
        )
        intercept[index], slope[index] = weights
    return {
        "prediction_mean": mean,
        "prediction_scale": scale,
        "intercept": intercept,
        "slope": slope,
        "ridge": float(ridge),
    }


def apply_coordinatewise_affine(
    prediction: np.ndarray, model: dict[str, np.ndarray | float]
) -> np.ndarray:
    standardized = (
        np.asarray(prediction, dtype=np.float64) - model["prediction_mean"]
    ) / model["prediction_scale"]
    return model["intercept"] + standardized * model["slope"]


@torch.no_grad()
def predict_selected(
    model: torch.nn.Module,
    images: np.ndarray,
    latents: np.ndarray,
    rows: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    model.eval()
    for begin in range(0, len(rows), batch_size):
        selected = rows[begin : begin + batch_size]
        envs = np.zeros(len(selected), dtype=np.int64)
        x = base.image_batch(images, envs, selected, device)
        anchors = torch.from_numpy(
            np.array(latents[0, selected, 3:5], dtype=np.float32, copy=True)
        ).to(device)
        chunks.append(model.get_z(x, anchors)[:, :3].cpu().numpy())
    return np.concatenate(chunks, axis=0)


def load_tube_context(
    config: dict[str, Any], device: torch.device
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    tube_config_path = Path(config["upstream"]["tube_config_path"])
    tube_config = yaml.safe_load(tube_config_path.read_text(encoding="utf-8"))
    tube.validate_config(tube_config)
    tube_root = Path(config["upstream"]["tube_output_root"])
    preflight, _, _ = tube.load_preflight(tube_root, tube_config_path, tube_config)
    readout_path = tube_root / "formal" / "validation_readout.json"
    if base.sha256_file(readout_path) != config["upstream"][
        "tube_validation_readout_sha256"
    ]:
        raise ValueError("tube validation readout hash mismatch")
    return tube_config, preflight, readout_path


def load_tube_model(
    condition: str,
    config: dict[str, Any],
    tube_config: dict[str, Any],
    preflight: dict[str, Any],
    device: torch.device,
) -> torch.nn.Module:
    lock_path = (
        Path(config["upstream"]["tube_output_root"])
        / "formal"
        / condition
        / "training_lock.json"
    )
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if not all(lock["validity"].values()) or lock["test_evaluated"] is not False:
        raise ValueError(f"invalid upstream tube lock: {condition}")
    return tube.load_locked_model(condition, lock, tube_config, preflight, device)


def load_projected_model(
    config: dict[str, Any], device: torch.device
) -> tuple[torch.nn.Module, dict[str, Any]]:
    projected_config_path = Path(config["upstream"]["projected_config_path"])
    projected_config = yaml.safe_load(
        projected_config_path.read_text(encoding="utf-8")
    )
    checkpoint_path = Path(config["upstream"]["projected_checkpoint_path"])
    if base.sha256_file(checkpoint_path) != config["upstream"][
        "projected_checkpoint_sha256"
    ]:
        raise ValueError("projected physical checkpoint hash mismatch")
    model_cfg = projected_config["model"]
    model = v3.FilmConditionedModel(
        int(model_cfg["latent_dim"]),
        list(model_cfg["learned_indices"]),
        list(model_cfg["anchor_indices"]),
        int(model_cfg["hidden_channels"]),
        int(model_cfg["conv_layers"]),
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    return model, projected_config


def metric_bundle(
    calibration_prediction: np.ndarray,
    validation_prediction: np.ndarray,
    calibration_truth: np.ndarray,
    validation_truth: np.ndarray,
    ridge: float,
) -> dict[str, Any]:
    diagonal = fit_coordinatewise_affine(
        calibration_prediction, calibration_truth, ridge
    )
    diagonal_validation = apply_coordinatewise_affine(
        validation_prediction, diagonal
    )
    full_weights, full_stats = sparse.fit_affine_ridge(
        calibration_prediction, calibration_truth, ridge
    )
    full_validation = sparse.apply_affine_ridge(
        validation_prediction, full_weights, full_stats
    )
    return {
        "raw": v6.regression_metrics(validation_prediction, validation_truth),
        "coordinatewise_affine": v6.regression_metrics(
            diagonal_validation, validation_truth
        ),
        "full_affine": v6.regression_metrics(full_validation, validation_truth),
        "coordinatewise_parameters": diagonal,
        "full_weights": full_weights,
        "full_input_stats": full_stats,
        "calibration_coordinatewise_mse": float(
            np.mean(
                (
                    apply_coordinatewise_affine(calibration_prediction, diagonal)
                    - calibration_truth
                )
                ** 2
            )
        ),
    }


def decide(results: dict[str, dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    tube_result = results["tube_correct"]
    diagonal = tube_result["coordinatewise_affine"]
    full = tube_result["full_affine"]
    slopes = np.asarray(
        tube_result["coordinatewise_parameters"]["slope"], dtype=np.float64
    )
    raw_corr = np.asarray(tube_result["raw"]["direct_abs_correlation"])
    aligned_corr = np.asarray(diagonal["direct_abs_correlation"])
    tolerance = float(
        config["evaluation"][
            "maximum_full_minus_diagonal_r2_for_scale_bias_explanation"
        ]
    )
    mechanism = {
        "diagonal_r2_above_raw": diagonal["mean_r2"]
        > tube_result["raw"]["mean_r2"],
        "diagonal_r2_above_unbounded": diagonal["mean_r2"]
        > results["unbounded_correct"]["coordinatewise_affine"]["mean_r2"],
        "diagonal_r2_above_permuted": diagonal["mean_r2"]
        > results["tube_permuted_matched"]["coordinatewise_affine"]["mean_r2"],
        "full_minus_diagonal_within_tolerance": full["mean_r2"]
        - diagonal["mean_r2"]
        <= tolerance,
        "slopes_finite_nonzero": bool(
            np.all(np.isfinite(slopes)) and np.all(np.abs(slopes) > 1e-8)
        ),
        "absolute_correlation_preserved": bool(
            np.allclose(raw_corr, aligned_corr, atol=1e-6, rtol=0.0)
        ),
    }
    projected = results["projected_physical_correct"]["coordinatewise_affine"]
    competitive = {
        "r2_above_matched_projected": diagonal["mean_r2"]
        > projected["mean_r2"],
        "correlation_above_matched_projected": diagonal[
            "mean_direct_abs_correlation"
        ]
        > projected["mean_direct_abs_correlation"],
    }
    mechanism_supported = all(mechanism.values())
    paper_competitive = all(competitive.values())
    if mechanism_supported and paper_competitive:
        verdict = "scale_bias_gauge_supported_and_matched_competitive"
    elif mechanism_supported:
        verdict = "scale_bias_gauge_supported_not_matched_competitive"
    else:
        verdict = "scale_bias_gauge_not_supported"
    return {
        "verdict": verdict,
        "mechanism_checks": mechanism,
        "matched_competitive_checks": competitive,
        "deltas": {
            "tube_diagonal_minus_raw_r2": diagonal["mean_r2"]
            - tube_result["raw"]["mean_r2"],
            "tube_full_minus_diagonal_r2": full["mean_r2"]
            - diagonal["mean_r2"],
            "tube_diagonal_minus_projected_diagonal_r2": diagonal["mean_r2"]
            - projected["mean_r2"],
            "tube_diagonal_minus_projected_diagonal_correlation": diagonal[
                "mean_direct_abs_correlation"
            ]
            - projected["mean_direct_abs_correlation"],
        },
        "test_evaluated": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    output_root = Path(config["runtime"]["output_root"])
    output_path = output_root / "gauge_audit.json"
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite gauge audit: {output_path}")
    torch.set_num_threads(int(config["runtime"]["num_threads"]))
    torch.set_float32_matmul_precision("high")
    device = torch.device(
        config["runtime"]["device"] if torch.cuda.is_available() else "cpu"
    )
    images = np.load(Path(config["dataset"]["cache_path"]), mmap_mode="r")
    raw_latents = base.load_latents(config)
    subset, subset_audit = tube.dynamic.build_subset(config)
    latents, latent_mean, latent_std = v17.normalize_with_k80_rgb_statistics(
        raw_latents, int(config["split"]["train_end"]), subset
    )
    calibration_truth = latents[0, subset, :3]
    start = int(config["split"]["train_end"])
    end = int(config["split"]["validation_end"])
    validation_truth = latents[0, start:end, :3]
    batch_size = 512
    ridge = float(config["calibration"]["ridge"])
    tube_config, preflight, tube_readout_path = load_tube_context(config, device)
    upstream_readout = json.loads(tube_readout_path.read_text(encoding="utf-8"))
    results: dict[str, dict[str, Any]] = {}
    for condition in tube.CONDITIONS:
        model = load_tube_model(
            condition, config, tube_config, preflight, device
        )
        calibration_prediction = predict_selected(
            model, images, latents, subset, batch_size, device
        )
        validation_prediction = v6.predict_rgb(
            model, images, latents, start, end, batch_size, device
        )
        results[condition] = metric_bundle(
            calibration_prediction,
            validation_prediction,
            calibration_truth,
            validation_truth,
            ridge,
        )
        expected = upstream_readout["semantic_validation"][condition]
        if not (
            np.isclose(
                results[condition]["raw"]["mean_direct_abs_correlation"],
                expected["mean_direct_abs_correlation"],
                atol=5e-6,
            )
            and np.isclose(
                results[condition]["raw"]["mean_r2"],
                expected["mean_r2"],
                atol=5e-6,
            )
        ):
            raise RuntimeError(f"raw tube metric drift: {condition}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    projected, _ = load_projected_model(config, device)
    projected_calibration = predict_selected(
        projected, images, latents, subset, batch_size, device
    )
    projected_validation = v6.predict_rgb(
        projected, images, latents, start, end, batch_size, device
    )
    results["projected_physical_correct"] = metric_bundle(
        projected_calibration,
        projected_validation,
        calibration_truth,
        validation_truth,
        ridge,
    )
    projected_raw = results["projected_physical_correct"]["raw"]
    projected_checks = {
        "correlation_exact": bool(
            np.isclose(
                projected_raw["mean_direct_abs_correlation"],
                float(config["upstream"]["projected_raw_validation_correlation"]),
                atol=5e-6,
            )
        ),
        "r2_exact": bool(
            np.isclose(
                projected_raw["mean_r2"],
                float(config["upstream"]["projected_raw_validation_r2"]),
                atol=5e-6,
            )
        ),
    }
    if not all(projected_checks.values()):
        raise RuntimeError(f"projected comparator drift: {projected_checks}")
    result = {
        "protocol_version": PROTOCOL,
        "mode": "frozen_k80_train_fit_validation_only_gauge_audit",
        "config_sha256": base.sha256_file(config_path),
        "source_files_sha256": source_hashes(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "subset": subset_audit,
        "normalization": {"latent_mean": latent_mean, "latent_std": latent_std},
        "upstream": {
            "tube_validation_readout": str(tube_readout_path),
            "tube_validation_readout_sha256": base.sha256_file(tube_readout_path),
            "projected_checkpoint": config["upstream"][
                "projected_checkpoint_path"
            ],
            "projected_checkpoint_sha256": config["upstream"][
                "projected_checkpoint_sha256"
            ],
        },
        "raw_projected_checks": projected_checks,
        "runs": results,
        "decision": decide(results, config),
        "fit_rows": "registered_k80_observational_train_only",
        "evaluation_rows": "observational_validation_8000_9000",
        "test_evaluated": False,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    v11.atomic_json(output_path, result)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "sha256": base.sha256_file(output_path),
                "verdict": result["decision"]["verdict"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
