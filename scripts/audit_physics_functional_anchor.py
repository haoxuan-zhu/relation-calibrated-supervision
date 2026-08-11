"""Train/validation-only physical functional-anchor audit for partial-anchor CRL v10."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

import run_diagnostic as base
import run_sparse_persistent_anchor_diagnostic as v9


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def malus_tau(latents: np.ndarray) -> np.ndarray:
    return np.cos(np.deg2rad(latents[..., 3] - latents[..., 4])) ** 2


def image_channel_means(
    images: np.ndarray, environment: int, start: int, end: int, chunk: int = 512
) -> np.ndarray:
    output = np.empty((end - start, 3), dtype=np.float64)
    cursor = 0
    for left in range(start, end, chunk):
        right = min(left + chunk, end)
        batch = np.asarray(images[environment, left:right], dtype=np.float64)
        output[cursor : cursor + len(batch)] = batch.mean(axis=(1, 2)) / 255.0
        cursor += len(batch)
    return output


def full_forward_design(rgb: np.ndarray, tau: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [np.ones((len(rgb), 1)), rgb, rgb * tau[:, None]], axis=1
    )


def affine_forward_design(rgb: np.ndarray) -> np.ndarray:
    return np.concatenate([np.ones((len(rgb), 1)), rgb], axis=1)


def fit_lstsq(design: np.ndarray, target: np.ndarray) -> np.ndarray:
    coefficients, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    return coefficients


def forward_predict(
    coefficients: np.ndarray, rgb: np.ndarray, tau: np.ndarray, kind: str
) -> np.ndarray:
    if kind == "full_malus_forward":
        return full_forward_design(rgb, tau) @ coefficients
    if kind == "affine_forward":
        return affine_forward_design(rgb) @ coefficients
    raise ValueError(kind)


def invert_forward(
    coefficients: np.ndarray,
    channel_means: np.ndarray,
    tau: np.ndarray,
    kind: str,
    rcond: float,
    clip: bool,
) -> tuple[np.ndarray, np.ndarray]:
    intercept = coefficients[0]
    if kind == "affine_forward":
        matrices = np.repeat(coefficients[1:4][None, :, :], len(tau), axis=0)
    elif kind == "full_malus_forward":
        matrices = (
            coefficients[1:4][None, :, :]
            + tau[:, None, None] * coefficients[4:7][None, :, :]
        )
    else:
        raise ValueError(kind)
    prediction = np.empty((len(tau), 3), dtype=np.float64)
    condition_numbers = np.empty(len(tau), dtype=np.float64)
    for index, matrix in enumerate(matrices):
        condition_numbers[index] = np.linalg.cond(matrix)
        prediction[index] = (channel_means[index] - intercept) @ np.linalg.pinv(
            matrix, rcond=rcond
        )
    if clip:
        prediction = np.clip(prediction, 0.0, 1.0)
    return prediction, condition_numbers


def fit_diagonal_forward(
    rgb: np.ndarray, tau: np.ndarray, channel_means: np.ndarray
) -> np.ndarray:
    coefficients = np.empty((3, 3), dtype=np.float64)
    for channel in range(3):
        design = np.stack(
            [np.ones(len(rgb)), rgb[:, channel], rgb[:, channel] * tau], axis=1
        )
        coefficients[:, channel] = fit_lstsq(design, channel_means[:, channel])
    return coefficients


def invert_diagonal_forward(
    coefficients: np.ndarray,
    channel_means: np.ndarray,
    tau: np.ndarray,
    clip: bool,
) -> tuple[np.ndarray, np.ndarray]:
    denominator = coefficients[1][None, :] + tau[:, None] * coefficients[2][None, :]
    prediction = (channel_means - coefficients[0][None, :]) / np.where(
        np.abs(denominator) < 1e-8, np.nan, denominator
    )
    if clip:
        prediction = np.clip(prediction, 0.0, 1.0)
    return prediction, np.abs(denominator)


def fit_interaction_inverse(
    channel_means: np.ndarray, tau: np.ndarray, rgb: np.ndarray
) -> np.ndarray:
    design = np.concatenate(
        [
            np.ones((len(channel_means), 1)),
            channel_means,
            channel_means * tau[:, None],
        ],
        axis=1,
    )
    return fit_lstsq(design, rgb)


def predict_interaction_inverse(
    coefficients: np.ndarray,
    channel_means: np.ndarray,
    tau: np.ndarray,
    clip: bool,
) -> np.ndarray:
    design = np.concatenate(
        [
            np.ones((len(channel_means), 1)),
            channel_means,
            channel_means * tau[:, None],
        ],
        axis=1,
    )
    prediction = design @ coefficients
    return np.clip(prediction, 0.0, 1.0) if clip else prediction


def regression_metrics(prediction: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    if not np.all(np.isfinite(prediction)):
        return {"finite": False}
    direct = []
    r2 = []
    for channel in range(3):
        direct.append(
            float(np.corrcoef(prediction[:, channel], truth[:, channel])[0, 1])
        )
        residual = np.sum((truth[:, channel] - prediction[:, channel]) ** 2)
        total = np.sum((truth[:, channel] - truth[:, channel].mean()) ** 2)
        r2.append(float(1.0 - residual / max(total, 1e-12)))
    correlation = base.absolute_correlation(prediction, truth)
    mcc, assignment = base.hungarian_mcc(correlation)
    return {
        "finite": True,
        "direct_correlation": direct,
        "mean_direct_correlation": float(np.mean(direct)),
        "direct_r2": r2,
        "mean_direct_r2": float(np.mean(r2)),
        "hungarian_mcc": mcc,
        "assignment": assignment,
        "mae": float(np.mean(np.abs(prediction - truth))),
    }


def forward_metrics(prediction: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    residual = np.sum((truth - prediction) ** 2, axis=0)
    total = np.sum((truth - truth.mean(axis=0)) ** 2, axis=0)
    r2 = 1.0 - residual / np.maximum(total, 1e-12)
    return {
        "r2": r2.tolist(),
        "mean_r2": float(np.mean(r2)),
        "mse": np.mean((truth - prediction) ** 2, axis=0).tolist(),
    }


def summarize_distribution(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    return {
        "minimum": float(np.min(finite)),
        "median": float(np.median(finite)),
        "p95": float(np.percentile(finite, 95)),
        "maximum": float(np.max(finite)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    split = config["split"]
    train_end = int(split["train_end"])
    validation_end = int(split["validation_end"])
    if not 0 < train_end < validation_end <= int(split["test_end"]):
        raise ValueError("invalid split")
    calibration = config["calibration"]
    if calibration["environment"] != "obs":
        raise ValueError("v10 is frozen to observational calibration")

    raw_latents = base.load_latents(config).astype(np.float64)
    rgb = raw_latents[..., :3] / float(calibration["rgb_scale"])
    tau = malus_tau(raw_latents)
    images = np.load(Path(config["dataset"]["cache_path"]), mmap_mode="r")
    environment_names = list(config["dataset"]["environments"])
    train_means = image_channel_means(images, 0, 0, train_end)
    validation_means = np.stack(
        [
            image_channel_means(images, env, train_end, validation_end)
            for env in range(len(environment_names))
        ],
        axis=0,
    )

    budget_config = {
        "split": config["split"],
        "persistent_anchor_budget": {
            "budgets": calibration["budgets"],
            "subset_seed": calibration["subset_seed"],
            "permutation_seed": calibration["permutation_seed"],
        },
    }
    budget_specs, budget_audit = v9.build_budget_specs(budget_config)
    pooled_truth = rgb[:, train_end:validation_end].reshape(-1, 3)
    pooled_means = validation_means.reshape(-1, 3)
    pooled_tau = tau[:, train_end:validation_end].reshape(-1)
    rcond = float(calibration["pinv_rcond"])
    clip = bool(calibration["clip_predictions_to_unit_interval"])

    results: dict[str, Any] = {}
    for budget in [int(value) for value in calibration["budgets"]]:
        spec = budget_specs[budget]
        subset = spec["subset"]
        permuted_targets = spec["permuted_targets"]
        calibration_tau = tau[0, subset]
        calibration_means = train_means[subset]
        condition_results: dict[str, Any] = {}
        for control, label_rows in [
            ("correct_labels", subset),
            ("permuted_labels", permuted_targets),
        ]:
            calibration_rgb = rgb[0, label_rows]
            physical_coefficients = fit_lstsq(
                full_forward_design(calibration_rgb, calibration_tau),
                calibration_means,
            )
            physical_prediction, condition_numbers = invert_forward(
                physical_coefficients,
                pooled_means,
                pooled_tau,
                "full_malus_forward",
                rcond,
                clip,
            )
            per_environment = {}
            for env, name in enumerate(environment_names):
                start = env * (validation_end - train_end)
                end = start + (validation_end - train_end)
                per_environment[name] = regression_metrics(
                    physical_prediction[start:end], pooled_truth[start:end]
                )
            item: dict[str, Any] = {
                "full_malus_forward": {
                    "coefficients": physical_coefficients.tolist(),
                    "pooled_inverse": regression_metrics(
                        physical_prediction, pooled_truth
                    ),
                    "per_environment_inverse": per_environment,
                    "condition_number": summarize_distribution(condition_numbers),
                }
            }
            if control == "correct_labels":
                forward_prediction = forward_predict(
                    physical_coefficients,
                    pooled_truth,
                    pooled_tau,
                    "full_malus_forward",
                )
                item["full_malus_forward"]["pooled_forward"] = forward_metrics(
                    forward_prediction, pooled_means
                )
                affine_coefficients = fit_lstsq(
                    affine_forward_design(calibration_rgb), calibration_means
                )
                affine_prediction, affine_condition = invert_forward(
                    affine_coefficients,
                    pooled_means,
                    pooled_tau,
                    "affine_forward",
                    rcond,
                    clip,
                )
                item["affine_forward"] = {
                    "pooled_inverse": regression_metrics(
                        affine_prediction, pooled_truth
                    ),
                    "pooled_forward": forward_metrics(
                        forward_predict(
                            affine_coefficients,
                            pooled_truth,
                            pooled_tau,
                            "affine_forward",
                        ),
                        pooled_means,
                    ),
                    "condition_number": summarize_distribution(affine_condition),
                }
                diagonal_coefficients = fit_diagonal_forward(
                    calibration_rgb, calibration_tau, calibration_means
                )
                diagonal_prediction, diagonal_denominator = invert_diagonal_forward(
                    diagonal_coefficients, pooled_means, pooled_tau, clip
                )
                item["diagonal_malus_forward"] = {
                    "pooled_inverse": regression_metrics(
                        diagonal_prediction, pooled_truth
                    ),
                    "absolute_denominator": summarize_distribution(
                        diagonal_denominator
                    ),
                }
                inverse_coefficients = fit_interaction_inverse(
                    calibration_means, calibration_tau, calibration_rgb
                )
                inverse_prediction = predict_interaction_inverse(
                    inverse_coefficients, pooled_means, pooled_tau, clip
                )
                item["matched_interaction_inverse"] = {
                    "pooled_inverse": regression_metrics(
                        inverse_prediction, pooled_truth
                    )
                }
            condition_results[control] = item
        results[str(budget)] = condition_results

    evaluation = config["evaluation"]
    full_budget = str(int(calibration["budgets"][0]))
    sparse_budget = str(int(evaluation["target_sparse_budget"]))
    full_forward_r2 = results[full_budget]["correct_labels"][
        "full_malus_forward"
    ]["pooled_forward"]["mean_r2"]
    sparse_correct = results[sparse_budget]["correct_labels"][
        "full_malus_forward"
    ]
    sparse_permuted = results[sparse_budget]["permuted_labels"][
        "full_malus_forward"
    ]
    sparse_corr = sparse_correct["pooled_inverse"]["mean_direct_correlation"]
    sparse_r2 = sparse_correct["pooled_inverse"]["mean_direct_r2"]
    permuted_corr = sparse_permuted["pooled_inverse"]["mean_direct_correlation"]
    worst_environment = min(
        value["mean_direct_correlation"]
        for value in sparse_correct["per_environment_inverse"].values()
    )
    gates = {
        "full_budget_forward_fidelity": full_forward_r2
        >= float(evaluation["minimum_full_budget_forward_mean_r2"]),
        "sparse_pooled_correlation": sparse_corr
        >= float(evaluation["minimum_sparse_pooled_direct_correlation"]),
        "sparse_pooled_r2": sparse_r2
        >= float(evaluation["minimum_sparse_pooled_direct_r2"]),
        "semantic_mapping_gain": sparse_corr - permuted_corr
        >= float(evaluation["minimum_gain_over_permuted_correlation"]),
        "worst_environment_correlation": worst_environment
        >= float(evaluation["minimum_worst_environment_direct_correlation"]),
    }
    verdict = (
        "physics_functional_anchor_calibration_supported_v10"
        if all(gates.values())
        else "physics_functional_anchor_calibration_not_supported_v10"
    )
    result = {
        "protocol_version": config["protocol_version"],
        "scope": "train_calibration_validation_readout_no_test",
        "official_equation": "tau=cos^2(deg2rad(pol_1-pol_2))",
        "forward_constraint": "m=b+z_rgb^T(A0+tau*A1)",
        "budget_audit": budget_audit,
        "results": results,
        "decision": {
            "verdict": verdict,
            "gates": gates,
            "full_budget_forward_mean_r2": full_forward_r2,
            "target_sparse_budget": int(sparse_budget),
            "sparse_pooled_direct_correlation": sparse_corr,
            "sparse_pooled_direct_r2": sparse_r2,
            "sparse_permuted_direct_correlation": permuted_corr,
            "sparse_gain_over_permuted_correlation": sparse_corr - permuted_corr,
            "sparse_worst_environment_direct_correlation": worst_environment,
        },
        "config": config,
        "config_sha256": sha256_file(config_path),
        "source_sha256": sha256_file(Path(__file__).resolve()),
    }
    output = (args.output or Path(config["runtime"]["output"])).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(base.json_ready(result), indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
