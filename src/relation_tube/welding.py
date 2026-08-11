"""Utilities for the resistance spot-welding relation benchmark."""

from __future__ import annotations

import csv
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


SIDE_FEATURE_NAMES = (
    "Pressure (PSI)",
    "Welding Time (ms)",
    "Angle (Deg)",
    "Thickness A median (mm)",
    "Thickness B median (mm)",
    "Force mean (N)",
    "Force min (N)",
    "Force max (N)",
    "Current mean (A)",
    "Current min (A)",
    "Current max (A)",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _constant(rows: list[dict[str, str]], name: str) -> float:
    values = {float(row[name]) for row in rows}
    if len(values) != 1:
        raise ValueError(f"{name} changes within sample {rows[0]['Sample ID']}")
    return values.pop()


def load_units(csv_path: Path) -> dict[int, dict[str, Any]]:
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            grouped[int(row["Sample ID"])].append(row)
    if not grouped:
        raise ValueError("spot-welding CSV contains no samples")

    units: dict[int, dict[str, Any]] = {}
    constant_fields = (
        "Pressure (PSI)",
        "Welding Time (ms)",
        "Angle (Deg)",
        "NuggetDiameter (mm)",
    )
    for sample_id, rows in grouped.items():
        constants = {name: _constant(rows, name) for name in constant_fields}
        force = np.asarray([float(row["Force (N)"]) for row in rows], dtype=np.float64)
        current = np.asarray([float(row["Current (A)"]) for row in rows], dtype=np.float64)
        thickness_a = np.asarray(
            [float(row["Thickness A (mm)"]) for row in rows], dtype=np.float64
        )
        thickness_b = np.asarray(
            [float(row["Thickness B (mm)"]) for row in rows], dtype=np.float64
        )
        pull_force = np.asarray(
            [float(row["PullTest (N)"]) for row in rows], dtype=np.float64
        )
        units[sample_id] = {
            "group": (
                constants["Pressure (PSI)"],
                constants["Welding Time (ms)"],
                constants["Angle (Deg)"],
            ),
            "side": np.asarray(
                [
                    constants["Pressure (PSI)"],
                    constants["Welding Time (ms)"],
                    constants["Angle (Deg)"],
                    np.median(thickness_a),
                    np.median(thickness_b),
                    force.mean(),
                    force.min(),
                    force.max(),
                    current.mean(),
                    current.min(),
                    current.max(),
                ],
                dtype=np.float64,
            ),
            "pull_force": float(np.median(pull_force)),
            "nugget_diameter": constants["NuggetDiameter (mm)"],
            "time_rows": len(rows),
        }
    return units


def load_split(path: Path, sample_ids: set[int]) -> tuple[dict[int, str], dict[str, Any]]:
    import json

    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("protocol") != "weld_product_split_v1":
        raise ValueError("unexpected weld split protocol")
    rows = record.get("rows", [])
    split_by_id = {int(row["sample_id"]): str(row["split"]) for row in rows}
    if len(split_by_id) != len(rows):
        raise ValueError("duplicate sample ID in weld split")
    if set(split_by_id) != sample_ids:
        raise ValueError("weld split IDs do not match CSV IDs")
    allowed = {"train", "validation", "heldout"}
    if set(split_by_id.values()) - allowed:
        raise ValueError("unknown weld split name")
    observed = {
        name: sum(value == name for value in split_by_id.values()) for name in sorted(allowed)
    }
    if observed != {key: int(value) for key, value in record["counts"].items()}:
        raise ValueError("weld split counts changed")
    return split_by_id, record


def reorder_features(
    requested_ids: list[int], feature_ids: np.ndarray, view_features: np.ndarray
) -> np.ndarray:
    source_ids = [int(value) for value in feature_ids.tolist()]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("duplicate sample ID in feature archive")
    lookup = {sample_id: index for index, sample_id in enumerate(source_ids)}
    missing = [sample_id for sample_id in requested_ids if sample_id not in lookup]
    if missing:
        raise ValueError(f"missing weld features for sample IDs: {missing[:5]}")
    return np.asarray(view_features)[[lookup[sample_id] for sample_id in requested_ids]]


def calibration_subset(
    train_ids: list[int], units: dict[int, dict[str, Any]], budget: int, seed: int
) -> list[int]:
    by_group: dict[tuple[float, float, float], list[int]] = defaultdict(list)
    for sample_id in train_ids:
        by_group[units[sample_id]["group"]].append(sample_id)
    groups = sorted(
        by_group,
        key=lambda group: stable_hash(f"{seed}|group|{'|'.join(map(str, group))}"),
    )
    for group in groups:
        by_group[group].sort(key=lambda sample_id: stable_hash(f"{seed}|sample|{sample_id}"))

    selected: list[int] = []
    layer = 0
    while len(selected) < budget:
        added = False
        for group in groups:
            if layer < len(by_group[group]):
                selected.append(by_group[group][layer])
                added = True
                if len(selected) == budget:
                    return selected
        if not added:
            break
        layer += 1
    raise ValueError("label budget exceeds the training split")


def derangement(size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    identity = np.arange(size)
    for _ in range(10_000):
        permutation = rng.permutation(size)
        if np.all(permutation != identity):
            return permutation
    raise RuntimeError("could not construct a derangement")


class StandardizedDualRidge:
    def __init__(self, regularization: float):
        self.regularization = float(regularization)

    def fit(self, features: np.ndarray, target: np.ndarray) -> "StandardizedDualRidge":
        features = np.asarray(features, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        self.mean = features.mean(axis=0)
        self.scale = features.std(axis=0)
        self.scale = np.where(self.scale > 1.0e-8, self.scale, 1.0)
        self.train = (features - self.mean) / self.scale
        self.target_mean = target.mean(axis=0)
        centered = target - self.target_mean
        gram = self.train @ self.train.T
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        alpha = self.regularization * max(1, self.train.shape[1])
        denominator = np.maximum(eigenvalues + alpha, 1.0e-12)
        if centered.ndim > 1:
            denominator = denominator[:, None]
        self.dual = eigenvectors @ ((eigenvectors.T @ centered) / denominator)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        standardized = (np.asarray(features, dtype=np.float64) - self.mean) / self.scale
        return standardized @ self.train.T @ self.dual + self.target_mean


def regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    denominator = float(np.sum((target - target.mean()) ** 2))
    if denominator <= 0.0:
        raise ValueError("R2 is undefined for a constant target")
    error = target - prediction
    correlation = float(np.corrcoef(target, prediction)[0, 1])
    return {
        "r2": float(1.0 - np.sum(error**2) / denominator),
        "correlation": correlation,
        "rmse": float(np.sqrt(np.mean(error**2))),
        "normalized_rmse": float(np.sqrt(np.mean(error**2)) / target.std()),
        "mae": float(np.mean(np.abs(error))),
    }


def bootstrap_r2_delta(
    target: np.ndarray,
    candidate: np.ndarray,
    baseline: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    candidate = np.asarray(candidate, dtype=np.float64).reshape(-1)
    baseline = np.asarray(baseline, dtype=np.float64).reshape(-1)
    observed = regression_metrics(target, candidate)["r2"] - regression_metrics(
        target, baseline
    )["r2"]
    rng = np.random.default_rng(seed)
    values = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        rows = rng.integers(0, len(target), len(target))
        values[index] = regression_metrics(target[rows], candidate[rows])["r2"] - regression_metrics(
            target[rows], baseline[rows]
        )["r2"]
    low, high = np.quantile(values, [0.025, 0.975])
    return {
        "observed": float(observed),
        "bootstrap_mean": float(values.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
    }
