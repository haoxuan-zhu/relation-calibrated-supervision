from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from relation_tube import welding

import importlib.util


SPLIT_SCRIPT = (
    Path(__file__).parents[1] / "scripts" / "prepare_resistance_spot_welding_split_v1.py"
)
SPLIT_SPEC = importlib.util.spec_from_file_location("weld_split_v1", SPLIT_SCRIPT)
assert SPLIT_SPEC is not None and SPLIT_SPEC.loader is not None
split_module = importlib.util.module_from_spec(SPLIT_SPEC)
SPLIT_SPEC.loader.exec_module(split_module)


def write_csv(path: Path) -> None:
    fields = [
        "Sample ID",
        "Pressure (PSI)",
        "Welding Time (ms)",
        "Angle (Deg)",
        "Thickness A (mm)",
        "Thickness B (mm)",
        "Force (N)",
        "Current (A)",
        "PullTest (N)",
        "NuggetDiameter (mm)",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sample_id in (1, 2):
            for step in (0, 1):
                writer.writerow(
                    {
                        "Sample ID": sample_id,
                        "Pressure (PSI)": 40 + sample_id,
                        "Welding Time (ms)": 200,
                        "Angle (Deg)": 0,
                        "Thickness A (mm)": 1,
                        "Thickness B (mm)": 1,
                        "Force (N)": 100 + sample_id + step,
                        "Current (A)": 5 + sample_id + step,
                        "PullTest (N)": 1000 + 10 * sample_id,
                        "NuggetDiameter (mm)": 3 + 0.1 * sample_id,
                    }
                )


def test_loader_aggregates_instrument_traces_by_weld(tmp_path: Path) -> None:
    path = tmp_path / "weld.csv"
    write_csv(path)
    units = welding.load_units(path)
    assert sorted(units) == [1, 2]
    assert units[1]["time_rows"] == 2
    np.testing.assert_allclose(units[1]["side"][-6:], [101.5, 101, 102, 6.5, 6, 7])
    assert units[2]["pull_force"] == 1020


def test_loader_uses_product_median_for_repeated_measurement_outliers(tmp_path: Path) -> None:
    path = tmp_path / "weld.csv"
    fields = [
        "Sample ID",
        "Pressure (PSI)",
        "Welding Time (ms)",
        "Angle (Deg)",
        "Thickness A (mm)",
        "Thickness B (mm)",
        "Force (N)",
        "Current (A)",
        "PullTest (N)",
        "NuggetDiameter (mm)",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for thickness, pull in ((0.61, 900), (0.64, 1000), (0.64, 1000)):
            writer.writerow(
                {
                    "Sample ID": 7,
                    "Pressure (PSI)": 40,
                    "Welding Time (ms)": 200,
                    "Angle (Deg)": 0,
                    "Thickness A (mm)": thickness,
                    "Thickness B (mm)": 0.65,
                    "Force (N)": 100,
                    "Current (A)": 6,
                    "PullTest (N)": pull,
                    "NuggetDiameter (mm)": 3.1,
                }
            )
    unit = welding.load_units(path)[7]
    assert unit["side"][3] == 0.64
    assert unit["pull_force"] == 1000


def test_product_split_is_deterministic_and_product_level(tmp_path: Path) -> None:
    path = tmp_path / "weld.csv"
    fields = [
        "Sample ID",
        "Pressure (PSI)",
        "Welding Time (ms)",
        "Angle (Deg)",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sample_id in range(1, 21):
            writer.writerow(
                {
                    "Sample ID": sample_id,
                    "Pressure (PSI)": 40,
                    "Welding Time (ms)": 200,
                    "Angle (Deg)": 0,
                }
            )
    first = split_module.build_split(path)
    second = split_module.build_split(path)
    assert first == second
    assert first["counts"] == {"train": 14, "validation": 3, "heldout": 3}
    assert len({row["sample_id"] for row in first["rows"]}) == 20


def test_calibration_subset_covers_groups_before_repeating() -> None:
    units = {
        sample_id: {"group": (float(sample_id % 3), 200.0, 0.0)}
        for sample_id in range(12)
    }
    selected = welding.calibration_subset(list(units), units, budget=3, seed=42)
    assert len({units[sample_id]["group"] for sample_id in selected}) == 3
    assert selected == welding.calibration_subset(list(units), units, budget=3, seed=42)


def test_derangement_has_no_fixed_points() -> None:
    permutation = welding.derangement(20, 20260847)
    assert sorted(permutation.tolist()) == list(range(20))
    assert np.all(permutation != np.arange(20))


def test_dual_ridge_recovers_a_linear_signal() -> None:
    rng = np.random.default_rng(7)
    features = rng.normal(size=(30, 5))
    target = features @ np.asarray([1.0, -2.0, 0.5, 0.0, 1.5])
    prediction = welding.StandardizedDualRidge(1.0e-8).fit(features, target).predict(features)
    assert welding.regression_metrics(target, prediction)["r2"] > 0.999


def test_paired_bootstrap_reports_candidate_advantage() -> None:
    target = np.linspace(-2.0, 2.0, 60)
    candidate = target + 0.05 * np.sin(np.arange(60))
    baseline = target + 0.8 * np.cos(np.arange(60))
    result = welding.bootstrap_r2_delta(
        target, candidate, baseline, replicates=500, seed=9
    )
    assert result["observed"] > 0.0
    assert result["ci95_low"] > 0.0
