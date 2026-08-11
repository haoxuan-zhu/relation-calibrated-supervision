"""Cross-validated state and metadata headroom audit for the MOVi collision track."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml


PROTOCOL = "movi_collision_state_headroom_v1"


def validate_config(config: dict[str, Any]) -> None:
    if config.get("protocol_version") != PROTOCOL:
        raise ValueError("unexpected MOVi state-headroom protocol")
    if config["data"].get("development_only") is not True:
        raise ValueError("this audit is development-only")
    if config["data"].get("test_evaluated") is not False:
        raise ValueError("test must remain closed")
    cross_validation = config["cross_validation"]
    if int(cross_validation["outer_folds"]) != 5 or int(cross_validation["inner_folds"]) != 4:
        raise ValueError("registered folds changed")
    if [float(value) for value in cross_validation["ridge_alphas"]] != [
        0.01,
        0.1,
        1.0,
        10.0,
        100.0,
    ]:
        raise ValueError("registered alpha grid changed")


def fold_id(name: str, salt: str, folds: int) -> int:
    digest = hashlib.sha256(f"{salt}:{name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big") % folds


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> dict[str, np.ndarray]:
    features = np.asarray(x, dtype=np.float64)
    targets = np.asarray(y, dtype=np.float64)
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale = np.where(scale > 1.0e-12, scale, 1.0)
    design = np.column_stack([np.ones(features.shape[0]), (features - mean) / scale])
    penalty = np.eye(design.shape[1], dtype=np.float64) * alpha
    penalty[0, 0] = 0.0
    coefficients = np.linalg.pinv(design.T @ design + penalty) @ design.T @ targets
    return {"mean": mean, "scale": scale, "coefficients": coefficients}


def predict_ridge(model: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    features = np.asarray(x, dtype=np.float64)
    design = np.column_stack(
        [np.ones(features.shape[0]), (features - model["mean"]) / model["scale"]]
    )
    return design @ model["coefficients"]


def select_alpha(
    x: np.ndarray,
    y: np.ndarray,
    names: list[str],
    alphas: list[float],
    folds: int,
    salt: str,
) -> tuple[float, dict[str, float]]:
    fold_ids = np.asarray([fold_id(name, salt, folds) for name in names], dtype=np.int64)
    scores: dict[str, float] = {}
    target_scale = np.std(y, axis=0)
    target_scale = np.where(target_scale > 1.0e-12, target_scale, 1.0)
    for alpha in alphas:
        errors: list[np.ndarray] = []
        for fold in range(folds):
            validation = fold_ids == fold
            training = ~validation
            if np.sum(validation) == 0 or np.sum(training) < 3:
                continue
            model = fit_ridge(x[training], y[training], alpha)
            errors.append((y[validation] - predict_ridge(model, x[validation])) / target_scale)
        if not errors:
            raise ValueError("inner folds contain no valid split")
        scores[str(alpha)] = float(np.mean(np.concatenate(errors, axis=0) ** 2))
    selected = min(alphas, key=lambda value: (scores[str(value)], value))
    return selected, scores


def metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    truth = np.asarray(y, dtype=np.float64)
    estimate = np.asarray(prediction, dtype=np.float64)
    if truth.shape != estimate.shape or truth.ndim != 2 or truth.shape[1] != 3:
        raise ValueError("collision metrics need matched N-by-3 arrays")
    centered = truth - truth.mean(axis=0)
    total = np.sum(centered**2, axis=0)
    squared = np.sum((truth - estimate) ** 2, axis=0)
    r2 = 1.0 - squared / np.maximum(total, 1.0e-15)
    scale = np.std(truth, axis=0)
    nrmse = np.sqrt(np.mean((truth - estimate) ** 2, axis=0)) / np.maximum(scale, 1.0e-15)
    truth_norm = np.linalg.norm(truth, axis=1)
    estimate_norm = np.linalg.norm(estimate, axis=1)
    valid = (truth_norm > 1.0e-12) & (estimate_norm > 1.0e-12)
    cosine = np.sum(truth[valid] * estimate[valid], axis=1) / (
        truth_norm[valid] * estimate_norm[valid]
    )
    return {
        "coordinate_r2": r2.tolist(),
        "mean_coordinate_r2": float(np.mean(r2)),
        "coordinate_normalized_rmse": nrmse.tolist(),
        "mean_normalized_rmse": float(np.mean(nrmse)),
        "mean_vector_cosine": float(np.mean(cosine)) if cosine.size else None,
        "impulse_magnitude_mae": float(np.mean(np.abs(truth_norm - estimate_norm))),
        "mse": float(np.mean((truth - estimate) ** 2)),
    }


def categorical_features(materials: np.ndarray, shapes: np.ndarray) -> np.ndarray:
    material_pairs = [(0, 0), (0, 1), (1, 1)]
    shape_pairs = [(first, second) for first in range(11) for second in range(first, 11)]
    material_lookup = {pair: index for index, pair in enumerate(material_pairs)}
    shape_lookup = {pair: index for index, pair in enumerate(shape_pairs)}
    output = np.zeros((materials.shape[0], len(material_pairs) + len(shape_pairs)))
    for row, (material, shape) in enumerate(zip(materials, shapes, strict=True)):
        material_pair = tuple(sorted(int(value) for value in material))
        shape_pair = tuple(sorted(int(value) for value in shape))
        output[row, material_lookup[material_pair]] = 1.0
        output[row, len(material_pairs) + shape_lookup[shape_pair]] = 1.0
    return output


def oof_ridge(
    x: np.ndarray,
    y: np.ndarray,
    names: list[str],
    config: dict[str, Any],
    condition: str,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    cross_validation = config["cross_validation"]
    outer_folds = int(cross_validation["outer_folds"])
    inner_folds = int(cross_validation["inner_folds"])
    alphas = [float(value) for value in cross_validation["ridge_alphas"]]
    salt = str(cross_validation["split_salt"])
    outer = np.asarray([fold_id(name, salt, outer_folds) for name in names], dtype=np.int64)
    prediction = np.empty_like(y)
    fold_records: list[dict[str, Any]] = []
    for fold in range(outer_folds):
        validation = outer == fold
        training = ~validation
        training_names = [name for name, keep in zip(names, training, strict=True) if keep]
        alpha, inner_scores = select_alpha(
            x[training],
            y[training],
            training_names,
            alphas,
            inner_folds,
            f"{salt}:{condition}:outer{fold}",
        )
        model = fit_ridge(x[training], y[training], alpha)
        prediction[validation] = predict_ridge(model, x[validation])
        fold_records.append(
            {
                "fold": fold,
                "training_events": int(np.sum(training)),
                "validation_events": int(np.sum(validation)),
                "selected_alpha": alpha,
                "inner_normalized_mse": inner_scores,
            }
        )
    return prediction, fold_records


def bootstrap_r2_delta(
    truth: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    repetitions: int,
    seed: int,
    confidence_level: float,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    deltas = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        sample = rng.integers(0, truth.shape[0], size=truth.shape[0])
        deltas[index] = metrics(truth[sample], first[sample])["mean_coordinate_r2"] - metrics(
            truth[sample], second[sample]
        )["mean_coordinate_r2"]
    tail = (1.0 - confidence_level) / 2.0
    return {
        "metric": "mean_coordinate_r2",
        "unit": "selected_collision_event_and_video",
        "repetitions": repetitions,
        "seed": seed,
        "observed_delta": metrics(truth, first)["mean_coordinate_r2"]
        - metrics(truth, second)["mean_coordinate_r2"],
        "ci_lower": float(np.quantile(deltas, tail)),
        "ci_upper": float(np.quantile(deltas, 1.0 - tail)),
    }


def run(config: dict[str, Any], metadata_path: Path) -> dict[str, Any]:
    validate_config(config)
    source = json.loads(metadata_path.read_text(encoding="utf-8"))
    selected = [row for row in source["audit"]["examples"] if row["selected"]]
    if len(selected) < 50:
        raise ValueError("too few selected collision events for headroom audit")
    names = [str(row["video_name"]) for row in selected]
    state = np.asarray([row["event"]["instrument_state"] for row in selected], dtype=np.float64)
    target = np.asarray([row["event"]["impulse"] for row in selected], dtype=np.float64)
    materials = np.asarray([row["event"]["material_pair"] for row in selected], dtype=np.int64)
    shapes = np.asarray([row["event"]["shape_pair"] for row in selected], dtype=np.int64)
    categories = categorical_features(materials, shapes)

    state_prediction, state_folds = oof_ridge(state, target, names, config, "state")
    augmented = np.column_stack([state, categories])
    augmented_prediction, augmented_folds = oof_ridge(
        augmented, target, names, config, "state_plus_categories"
    )
    outer_folds = int(config["cross_validation"]["outer_folds"])
    salt = str(config["cross_validation"]["split_salt"])
    outer = np.asarray([fold_id(name, salt, outer_folds) for name in names], dtype=np.int64)
    mean_prediction = np.empty_like(target)
    for fold in range(outer_folds):
        validation = outer == fold
        mean_prediction[validation] = target[~validation].mean(axis=0)

    evaluation = config["evaluation"]
    return {
        "protocol_version": PROTOCOL,
        "status": "completed_protocol_development_headroom_audit",
        "benchmark_metrics": False,
        "test_evaluated": False,
        "input": {
            "path": str(metadata_path),
            "sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
            "selected_events": len(selected),
            "state_dimension": int(state.shape[1]),
            "category_dimension": int(categories.shape[1]),
        },
        "conditions": {
            "fold_train_mean": metrics(target, mean_prediction),
            "instrument_state_ridge": metrics(target, state_prediction),
            "instrument_state_plus_audit_categories_ridge": metrics(
                target, augmented_prediction
            ),
        },
        "folds": {
            "instrument_state_ridge": state_folds,
            "instrument_state_plus_audit_categories_ridge": augmented_folds,
        },
        "paired_bootstrap": {
            "state_minus_mean": bootstrap_r2_delta(
                target,
                state_prediction,
                mean_prediction,
                int(evaluation["bootstrap_replicates"]),
                int(evaluation["bootstrap_seed"]),
                float(evaluation["confidence_level"]),
            ),
            "categories_minus_state": bootstrap_r2_delta(
                target,
                augmented_prediction,
                state_prediction,
                int(evaluation["bootstrap_replicates"]),
                int(evaluation["bootstrap_seed"]) + 1,
                float(evaluation["confidence_level"]),
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    validate_config(config)
    metadata_path = args.metadata or Path(config["data"]["metadata_json"])
    output_path = args.output or Path(config["output"]["json_path"])
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite result: {output_path}")
    result = run(config, metadata_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path), "benchmark_metrics": False}, sort_keys=True))


if __name__ == "__main__":
    main()
