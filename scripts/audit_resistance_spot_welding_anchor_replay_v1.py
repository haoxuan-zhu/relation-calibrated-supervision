"""Replay the anchor-preserving experiment with an independent ridge solver."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from relation_tube import welding


class DirectDualRidge:
    """Small independent implementation used only for result auditing."""

    def __init__(self, regularization: float):
        self.regularization = float(regularization)

    def fit(self, features: np.ndarray, target: np.ndarray) -> "DirectDualRidge":
        features = np.asarray(features, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        self.feature_mean = features.mean(axis=0)
        self.feature_scale = features.std(axis=0)
        self.feature_scale[self.feature_scale <= 1.0e-8] = 1.0
        self.standardized = (features - self.feature_mean) / self.feature_scale
        self.target_mean = target.mean(axis=0)
        centered_target = target - self.target_mean
        gram = self.standardized @ self.standardized.T
        penalty = self.regularization * max(1, self.standardized.shape[1])
        self.coefficients = np.linalg.solve(
            gram + penalty * np.eye(len(gram)), centered_target
        )
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        standardized = (
            np.asarray(features, dtype=np.float64) - self.feature_mean
        ) / self.feature_scale
        return (
            standardized @ self.standardized.T @ self.coefficients + self.target_mean
        )


def derangement(size: int, seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    identity = np.arange(size)
    for _ in range(10_000):
        permutation = generator.permutation(size)
        if np.all(permutation != identity):
            return permutation
    raise RuntimeError("could not reconstruct derangement")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    result = json.loads(args.result.read_text(encoding="utf-8"))
    units = welding.load_units(args.csv)
    split_by_id, _ = welding.load_split(args.split, set(units))
    with np.load(args.features, allow_pickle=False) as archive:
        feature_ids = np.asarray(archive["sample_ids"])
        views = np.asarray(archive["view_features"])
    train_ids = sorted(sample_id for sample_id, split in split_by_id.items() if split == "train")
    train_index = {sample_id: index for index, sample_id in enumerate(train_ids)}
    train_side = np.stack([units[sample_id]["side"] for sample_id in train_ids])
    train_target = np.asarray([units[sample_id]["pull_force"] for sample_id in train_ids])
    train_images = welding.reorder_features(train_ids, feature_ids, views)[:, 2]

    maximum_error = 0.0
    checked_predictions = 0
    split_overlap = {}
    for evaluation_split, evaluation in result["evaluations"].items():
        evaluation_ids = sorted(
            sample_id for sample_id, split in split_by_id.items() if split == evaluation_split
        )
        evaluation_images = welding.reorder_features(evaluation_ids, feature_ids, views)[:, 2]
        split_overlap[evaluation_split] = len(set(train_ids) & set(evaluation_ids))
        for run in evaluation["runs"]:
            subset_ids = [int(value) for value in run["subset_ids"]]
            if not set(subset_ids) <= set(train_ids):
                raise ValueError("calibration subset contains a non-training unit")
            subset = np.asarray([train_index[sample_id] for sample_id in subset_ids])
            mean = train_target[subset].mean()
            scale = train_target[subset].std()
            normalized = (train_target[subset] - mean) / scale

            relation = DirectDualRidge(0.1).fit(train_side[subset], normalized)
            propagated = relation.predict(train_side)
            permutation = derangement(len(subset), 20260805 + int(run["seed"]))
            permuted_relation = DirectDualRidge(0.1).fit(
                train_side[subset], normalized[permutation]
            )
            permuted = permuted_relation.predict(train_side)
            anchored = propagated.copy()
            anchored[subset] = normalized
            anchored_permuted = permuted.copy()
            anchored_permuted[subset] = normalized
            conditions = {
                "point": (train_images[subset], normalized),
                "relation_propagation": (train_images, propagated),
                "anchor_preserving": (train_images, anchored),
                "anchor_preserving_permuted": (train_images, anchored_permuted),
            }
            stored = run["evaluation_predictions"]
            if [int(value) for value in stored["sample_ids"]] != evaluation_ids:
                raise ValueError("evaluation row order changed")
            for name, (features, target) in conditions.items():
                prediction = DirectDualRidge(0.1).fit(features, target).predict(
                    evaluation_images
                )
                prediction = prediction * scale + mean
                reference = np.asarray(stored[name], dtype=np.float64)
                maximum_error = max(
                    maximum_error, float(np.max(np.abs(prediction - reference)))
                )
                checked_predictions += len(reference)

    audit = {
        "protocol": "resistance_spot_welding_anchor_replay_audit_v1",
        "independent_solver": "numpy.linalg.solve_on_regularized_dual_system",
        "checked_prediction_values": checked_predictions,
        "train_evaluation_unit_overlap": split_overlap,
        "maximum_prediction_absolute_error": maximum_error,
        "passed": maximum_error < 1.0e-8 and all(value == 0 for value in split_overlap.values()),
    }
    args.output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit))


if __name__ == "__main__":
    main()
