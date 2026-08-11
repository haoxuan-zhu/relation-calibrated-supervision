"""Validation-only audit of image-specific information in the frozen K80 tube residual."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

import audit_calibrated_relation_tube_gauge_k1 as gauge
import audit_relation_tube_geometry_downstream_k6 as k6
import run_conflict_projected_physics_k80_closed as v17
import run_diagnostic as base
import run_dynamic_residual_propagation_k0 as dynamic
import run_physics_functional_anchor_training as v11
import run_relation_tube_geometry_ablation_k4 as k4
import run_relation_tube_geometry_tradeoff_k5 as k5
import run_sparse_alignment_audit as sparse
import run_supervised_continuation_diagnostic as v6


PROTOCOL = "relation_tube_instance_information_audit_k14_v1"
SEEDS = (3407, 0, 42)
CONDITIONS = (
    "correct_image",
    "fixed_anchor_visual_derangement",
    "projected_residual_reassignment",
    "center_only",
)
SOURCE_FILES = ("run_relation_tube_instance_information_audit_k14.py",)


def source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    return {name: base.sha256_file(root / name) for name in SOURCE_FILES}


def keyed(mapping: dict[Any, Any], key: int) -> Any:
    if key in mapping:
        return mapping[key]
    return mapping[str(key)]


def validate_master(config: dict[str, Any]) -> None:
    if config["protocol_version"] != PROTOCOL:
        raise ValueError("unexpected K14 protocol")
    scope = config["scope"]
    if int(scope["budget"]) != 80 or tuple(int(v) for v in scope["seeds"]) != SEEDS:
        raise ValueError("K14 budget or seed registry changed")
    if tuple(int(v) for v in scope["validation_rows"]) != (8000, 9000):
        raise ValueError("K14 validation split changed")
    if tuple(int(v) for v in scope["test_rows"]) != (9000, 10000):
        raise ValueError("K14 test registry changed")
    if scope["test_evaluated"] is not False:
        raise ValueError("K14 test must remain closed")
    if tuple(config["interventions"]["order"]) != CONDITIONS:
        raise ValueError("K14 intervention registry changed")
    if int(config["interventions"]["derangement_seed"]) != 20260803:
        raise ValueError("K14 derangement seed changed")
    evaluation = config["evaluation"]
    if not np.isclose(float(evaluation["full_affine_ridge"]), 0.001):
        raise ValueError("K14 affine ridge changed")
    if not np.isclose(float(evaluation["reproduction_atol"]), 5e-6):
        raise ValueError("K14 reproduction tolerance changed")
    if int(evaluation["bootstrap_replicates"]) != 2000:
        raise ValueError("K14 bootstrap count changed")
    if int(evaluation["bootstrap_seed"]) != 20260804:
        raise ValueError("K14 bootstrap seed changed")
    for seed in SEEDS:
        entry = keyed(config["runs"], seed)
        expected = (
            "relation_tube_geometry_ablation_k4_v1"
            if seed == 3407
            else "relation_tube_geometry_tradeoff_k5_v1"
        )
        if entry["geometry_protocol"] != expected:
            raise ValueError(f"K14 geometry protocol changed for seed {seed}")


def make_derangement(size: int, seed: int) -> np.ndarray:
    if size < 2:
        raise ValueError("derangement requires at least two rows")
    rng = np.random.default_rng(seed)
    identity = np.arange(size, dtype=np.int64)
    for _ in range(10000):
        permutation = rng.permutation(size).astype(np.int64, copy=False)
        if not np.any(permutation == identity):
            return permutation
    raise RuntimeError("failed to construct deterministic derangement")


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def project_residual(
    embedding: torch.nn.Module, raw_residual: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    score = torch.einsum(
        "ni,ij,nj->n", raw_residual, embedding.tube_precision, raw_residual
    )
    scale = torch.sqrt(
        embedding.tube_radius_squared / score.clamp_min(1e-12)
    ).clamp(max=1.0)
    return raw_residual * scale[:, None], scale, score


@torch.no_grad()
def collect_correct_components(
    model: torch.nn.Module,
    images: np.ndarray,
    latents: np.ndarray,
    rows: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    chunks: dict[str, list[np.ndarray]] = {
        "center": [],
        "raw_residual": [],
        "projected_residual": [],
        "projection_scale": [],
        "raw_score": [],
    }
    model.eval()
    for begin in range(0, len(rows), batch_size):
        selected = rows[begin : begin + batch_size]
        envs = np.zeros(len(selected), dtype=np.int64)
        x = base.image_batch(images, envs, selected, device)
        anchors = torch.from_numpy(
            np.array(latents[0, selected, 3:5], dtype=np.float32, copy=True)
        ).to(device)
        _, values = model.embedding.components(x, anchors)
        for name, source in (
            ("center", values["center"]),
            ("raw_residual", values["raw_residual"]),
            ("projected_residual", values["projected_residual"]),
            ("projection_scale", values["projection_scale"]),
            ("raw_score", values["raw_mahalanobis_score"]),
        ):
            chunks[name].append(source.detach().cpu().numpy())
    return {name: np.concatenate(parts, axis=0) for name, parts in chunks.items()}


@torch.no_grad()
def collect_fixed_anchor_visual_swap(
    model: torch.nn.Module,
    images: np.ndarray,
    latents: np.ndarray,
    rows: np.ndarray,
    visual_rows: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    if rows.shape != visual_rows.shape or np.any(rows == visual_rows):
        raise ValueError("K14 visual rows must form a no-self matched intervention")
    chunks: dict[str, list[np.ndarray]] = {
        "projected_residual": [],
        "projection_scale": [],
        "raw_score": [],
    }
    model.eval()
    for begin in range(0, len(rows), batch_size):
        selected = rows[begin : begin + batch_size]
        selected_visual = visual_rows[begin : begin + batch_size]
        envs = np.zeros(len(selected), dtype=np.int64)
        x_visual = base.image_batch(images, envs, selected_visual, device)
        anchors = torch.from_numpy(
            np.array(latents[0, selected, 3:5], dtype=np.float32, copy=True)
        ).to(device)
        raw = model.embedding.raw_residual(x_visual, anchors)
        projected, scale, score = project_residual(model.embedding, raw)
        for name, source in (
            ("projected_residual", projected),
            ("projection_scale", scale),
            ("raw_score", score),
        ):
            chunks[name].append(source.detach().cpu().numpy())
    return {name: np.concatenate(parts, axis=0) for name, parts in chunks.items()}


def fit_frozen_readout(
    prediction: np.ndarray, truth: np.ndarray, ridge: float
) -> dict[str, Any]:
    coordinatewise = gauge.fit_coordinatewise_affine(prediction, truth, ridge)
    full_weights, full_stats = sparse.fit_affine_ridge(prediction, truth, ridge)
    return {
        "coordinatewise": coordinatewise,
        "full_weights": full_weights,
        "full_stats": full_stats,
    }


def apply_frozen_readout(
    prediction: np.ndarray, truth: np.ndarray, readout: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    coordinatewise = gauge.apply_coordinatewise_affine(
        prediction, readout["coordinatewise"]
    )
    full = sparse.apply_affine_ridge(
        prediction, readout["full_weights"], readout["full_stats"]
    )
    matrix = base.absolute_correlation(prediction, truth) if all_nonconstant(prediction, truth) else None
    if matrix is None or not np.all(np.isfinite(matrix)):
        mcc, assignment = None, None
    else:
        mcc, assignment = base.hungarian_mcc(matrix)
    metrics = {
        "raw": safe_regression_metrics(prediction, truth),
        "coordinatewise_affine": safe_regression_metrics(coordinatewise, truth),
        "full_affine": safe_regression_metrics(full, truth),
        "rgb_hungarian_mcc": mcc,
        "rgb_hungarian_assignment": assignment,
    }
    return metrics, {"raw": prediction, "coordinatewise_affine": coordinatewise, "full_affine": full}


def all_nonconstant(prediction: np.ndarray, truth: np.ndarray) -> bool:
    return bool(
        np.all(np.std(prediction, axis=0) > 1e-12)
        and np.all(np.std(truth, axis=0) > 1e-12)
    )


def safe_regression_metrics(prediction: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    if all_nonconstant(prediction, truth):
        return v6.regression_metrics(prediction, truth)
    prediction = np.asarray(prediction)
    truth = np.asarray(truth)
    residual = np.sum((truth - prediction) ** 2, axis=0)
    total = np.sum((truth - truth.mean(axis=0)) ** 2, axis=0)
    r2 = 1.0 - residual / np.maximum(total, 1e-12)
    correlations: list[float | None] = []
    for index in range(truth.shape[1]):
        if np.std(prediction[:, index]) <= 1e-12 or np.std(truth[:, index]) <= 1e-12:
            correlations.append(None)
        else:
            correlations.append(
                float(abs(np.corrcoef(prediction[:, index], truth[:, index])[0, 1]))
            )
    finite_correlations = [value for value in correlations if value is not None]
    return {
        "r2": r2,
        "mean_r2": float(r2.mean()),
        "direct_abs_correlation": correlations,
        "mean_direct_abs_correlation": (
            float(np.mean(finite_correlations)) if finite_correlations else None
        ),
        "mse": float(np.mean((truth - prediction) ** 2)),
    }


def cosine_summary(prediction: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    numerator = np.sum(prediction * truth, axis=1)
    denominator = np.linalg.norm(prediction, axis=1) * np.linalg.norm(truth, axis=1)
    valid = denominator > 1e-12
    values = numerator[valid] / denominator[valid]
    return {
        "valid_fraction": float(np.mean(valid)),
        "mean": float(np.mean(values)) if len(values) else None,
        "median": float(np.median(values)) if len(values) else None,
    }


def distribution_summary(
    projected: np.ndarray, scale: np.ndarray, score: np.ndarray
) -> dict[str, float]:
    norms = np.linalg.norm(projected, axis=1)
    return {
        "projected_l2_mean": float(np.mean(norms)),
        "projected_l2_p95": float(np.percentile(norms, 95)),
        "boundary_active_fraction": float(np.mean(scale < 1.0 - 1e-12)),
        "projection_scale_mean": float(np.mean(scale)),
        "raw_score_mean": float(np.mean(score)),
        "raw_score_p95": float(np.percentile(score, 95)),
    }


def paired_bootstrap_mse_advantage(
    correct: np.ndarray,
    control: np.ndarray,
    truth: np.ndarray,
    replicates: int,
    seed: int,
) -> dict[str, float | int | bool]:
    if correct.shape != control.shape or correct.shape != truth.shape:
        raise ValueError("paired bootstrap arrays must match")
    correct_error = np.mean((truth - correct) ** 2, axis=1)
    control_error = np.mean((truth - control) ** 2, axis=1)
    paired = control_error - correct_error
    rng = np.random.default_rng(seed)
    values = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        rows = rng.integers(0, len(paired), size=len(paired))
        values[index] = float(np.mean(paired[rows]))
    low, high = np.percentile(values, [2.5, 97.5])
    return {
        "observed_control_minus_correct_mse": float(np.mean(paired)),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "replicates": int(replicates),
        "seed": int(seed),
        "ci_strictly_positive": bool(low > 0.0),
    }


def reproduce_upstream(
    observed: dict[str, Any], expected: dict[str, Any], atol: float
) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    for block in ("raw", "coordinatewise_affine", "full_affine"):
        for metric in ("mean_r2", "mean_direct_abs_correlation", "mse"):
            checks[f"{block}.{metric}"] = bool(
                np.isclose(
                    float(observed[block][metric]),
                    float(expected[block][metric]),
                    atol=atol,
                    rtol=0.0,
                )
            )
    checks["rgb_hungarian_mcc"] = bool(
        np.isclose(
            float(observed["rgb_hungarian_mcc"]),
            float(expected["rgb_hungarian_mcc"]),
            atol=atol,
            rtol=0.0,
        )
    )
    return checks


def configure_engine(protocol: str) -> Any:
    if protocol == k4.PROTOCOL:
        return k4
    if protocol == k5.PROTOCOL:
        k5.configure_k4_engine()
        return k4
    raise ValueError(f"unsupported geometry protocol: {protocol}")


def load_seed_context(
    master: dict[str, Any], seed: int, device: torch.device
) -> tuple[dict[str, Any], dict[str, Any], torch.nn.Module, dict[str, Any]]:
    entry = keyed(master["runs"], seed)
    config_path = Path(entry["geometry_config_path"])
    if base.sha256_file(config_path) != entry["geometry_config_sha256"]:
        raise ValueError(f"K14 geometry config hash mismatch for seed {seed}")
    geometry_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    engine = configure_engine(entry["geometry_protocol"])
    engine.validate_config(geometry_config)
    root = Path(geometry_config["runtime"]["output_root"]).resolve()
    payload, preflight_lock_path = engine.load_preflight(
        root, config_path.resolve(), geometry_config
    )
    model, training_lock_sha256 = engine.load_new_model(
        "isotropic",
        root,
        config_path.resolve(),
        geometry_config,
        payload,
        preflight_lock_path,
        device,
    )
    readout_path = Path(entry["downstream_readout_path"])
    if base.sha256_file(readout_path) != entry["downstream_readout_sha256"]:
        raise ValueError(f"K14 downstream readout hash mismatch for seed {seed}")
    readout = json.loads(readout_path.read_text(encoding="utf-8"))
    if readout["test_evaluated"] is not False:
        raise ValueError("K14 upstream readout touched test")
    lock_path = root / "formal" / "isotropic" / "training_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    identity = {
        "geometry_config_path": str(config_path),
        "geometry_config_sha256": entry["geometry_config_sha256"],
        "preflight_lock_sha256": base.sha256_file(preflight_lock_path),
        "training_lock_path": str(lock_path),
        "training_lock_sha256": training_lock_sha256,
        "checkpoint_path": lock["checkpoint"]["path"],
        "checkpoint_sha256": lock["checkpoint"]["sha256"],
        "downstream_readout_path": str(readout_path),
        "downstream_readout_sha256": entry["downstream_readout_sha256"],
    }
    return geometry_config, payload, model, {"readout": readout, "identity": identity}


def seed_audit(
    master: dict[str, Any], seed: int, device: torch.device
) -> dict[str, Any]:
    config, payload, model, upstream = load_seed_context(master, seed, device)
    images = np.load(Path(config["dataset"]["cache_path"]), mmap_mode="r")
    raw_latents = base.load_latents(config)
    subset, subset_audit = dynamic.build_subset(config)
    latents, _, _ = v17.normalize_with_k80_rgb_statistics(
        raw_latents, int(config["split"]["train_end"]), subset
    )
    start, end = tuple(int(v) for v in master["scope"]["validation_rows"])
    if (start, end, int(config["split"]["test_end"])) != (8000, 9000, 10000):
        raise ValueError("K14 dataset split changed")
    validation_rows = np.arange(start, end, dtype=np.int64)
    permutation = make_derangement(
        len(validation_rows), int(master["interventions"]["derangement_seed"])
    )
    visual_rows = validation_rows[permutation]
    batch_size = int(config["training"]["batch_size"])
    calibration = collect_correct_components(
        model, images, latents, subset, batch_size, device
    )
    correct = collect_correct_components(
        model, images, latents, validation_rows, batch_size, device
    )
    visual_swap = collect_fixed_anchor_visual_swap(
        model, images, latents, validation_rows, visual_rows, batch_size, device
    )
    center = correct["center"]
    projected_correct = correct["projected_residual"]
    projected_visual = visual_swap["projected_residual"]
    projected_reassigned = projected_correct[permutation]
    projected_zero = np.zeros_like(projected_correct)
    outputs = {
        "correct_image": center + projected_correct,
        "fixed_anchor_visual_derangement": center + projected_visual,
        "projected_residual_reassignment": center + projected_reassigned,
        "center_only": center,
    }
    residuals = {
        "correct_image": projected_correct,
        "fixed_anchor_visual_derangement": projected_visual,
        "projected_residual_reassignment": projected_reassigned,
        "center_only": projected_zero,
    }
    calibration_truth = latents[0, subset, :3]
    validation_truth = latents[0, start:end, :3]
    calibration_output = calibration["center"] + calibration["projected_residual"]
    calibration_residual_truth = calibration_truth - calibration["center"]
    validation_residual_truth = validation_truth - center
    ridge = float(master["evaluation"]["full_affine_ridge"])
    output_readout = fit_frozen_readout(calibration_output, calibration_truth, ridge)
    residual_readout = fit_frozen_readout(
        calibration["projected_residual"], calibration_residual_truth, ridge
    )
    output_metrics: dict[str, Any] = {}
    residual_metrics: dict[str, Any] = {}
    output_predictions: dict[str, dict[str, np.ndarray]] = {}
    residual_predictions: dict[str, dict[str, np.ndarray]] = {}
    for condition in CONDITIONS:
        output_metrics[condition], output_predictions[condition] = apply_frozen_readout(
            outputs[condition], validation_truth, output_readout
        )
        residual_metrics[condition], residual_predictions[condition] = apply_frozen_readout(
            residuals[condition], validation_residual_truth, residual_readout
        )
        residual_metrics[condition]["cosine"] = cosine_summary(
            residuals[condition], validation_residual_truth
        )
    semantic = k6.semantic_bundle(
        calibration_output,
        outputs["correct_image"],
        calibration_truth,
        validation_truth,
        latents[0, start:end],
        list(config["model"]["learned_indices"]),
        list(config["model"]["anchor_indices"]),
        ridge,
    )
    reproduction = reproduce_upstream(
        semantic,
        upstream["readout"]["runs"]["isotropic"]["semantic"],
        float(master["evaluation"]["reproduction_atol"]),
    )
    if not all(reproduction.values()):
        raise RuntimeError(f"K14 upstream reproduction failed for seed {seed}: {reproduction}")
    replicates = int(master["evaluation"]["bootstrap_replicates"])
    bootstrap_seed = int(master["evaluation"]["bootstrap_seed"]) + seed * 10
    controls = (
        "fixed_anchor_visual_derangement",
        "projected_residual_reassignment",
    )
    bootstrap: dict[str, Any] = {}
    for offset, control in enumerate(controls):
        bootstrap[control] = {
            "output_full_affine": paired_bootstrap_mse_advantage(
                output_predictions["correct_image"]["full_affine"],
                output_predictions[control]["full_affine"],
                validation_truth,
                replicates,
                bootstrap_seed + offset * 2,
            ),
            "residual_full_affine": paired_bootstrap_mse_advantage(
                residual_predictions["correct_image"]["full_affine"],
                residual_predictions[control]["full_affine"],
                validation_residual_truth,
                replicates,
                bootstrap_seed + offset * 2 + 1,
            ),
        }
    deltas: dict[str, Any] = {}
    for control in controls:
        deltas[control] = {
            "correct_minus_control_output_full_affine_r2": float(
                output_metrics["correct_image"]["full_affine"]["mean_r2"]
                - output_metrics[control]["full_affine"]["mean_r2"]
            ),
            "correct_minus_control_output_raw_correlation": float(
                output_metrics["correct_image"]["raw"]["mean_direct_abs_correlation"]
                - output_metrics[control]["raw"]["mean_direct_abs_correlation"]
            ),
            "correct_minus_control_residual_full_affine_r2": float(
                residual_metrics["correct_image"]["full_affine"]["mean_r2"]
                - residual_metrics[control]["full_affine"]["mean_r2"]
            ),
        }
    distribution = {
        "correct_image": distribution_summary(
            projected_correct, correct["projection_scale"], correct["raw_score"]
        ),
        "fixed_anchor_visual_derangement": distribution_summary(
            projected_visual,
            visual_swap["projection_scale"],
            visual_swap["raw_score"],
        ),
        "projected_residual_reassignment": distribution_summary(
            projected_reassigned,
            correct["projection_scale"][permutation],
            correct["raw_score"][permutation],
        ),
    }
    hashes = {
        "validation_derangement": sha256_array(permutation.astype("<i8")),
        "correct_output_float32": sha256_array(outputs["correct_image"].astype("<f4")),
        "visual_swap_output_float32": sha256_array(
            outputs["fixed_anchor_visual_derangement"].astype("<f4")
        ),
        "reassigned_output_float32": sha256_array(
            outputs["projected_residual_reassignment"].astype("<f4")
        ),
    }
    return {
        "seed": seed,
        "upstream": upstream["identity"],
        "subset": subset_audit,
        "permutation": {
            "seed": int(master["interventions"]["derangement_seed"]),
            "size": int(len(permutation)),
            "fixed_points": int(np.sum(permutation == np.arange(len(permutation)))),
            "sha256_int64": hashes["validation_derangement"],
        },
        "output_metrics": output_metrics,
        "residual_metrics": residual_metrics,
        "distribution": distribution,
        "bootstrap": bootstrap,
        "deltas": deltas,
        "prediction_hashes": hashes,
        "reproduction": reproduction,
        "test_evaluated": False,
    }


def decide(results: dict[int, dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    controls = (
        "fixed_anchor_visual_derangement",
        "projected_residual_reassignment",
    )
    directional: dict[str, Any] = {}
    all_four_counts: list[int] = []
    for control in controls:
        output_positive = sum(
            results[seed]["deltas"][control][
                "correct_minus_control_output_full_affine_r2"
            ]
            > 0.0
            for seed in SEEDS
        )
        residual_positive = sum(
            results[seed]["deltas"][control][
                "correct_minus_control_residual_full_affine_r2"
            ]
            > 0.0
            for seed in SEEDS
        )
        directional[control] = {
            "output_full_affine_r2_positive_seeds": int(output_positive),
            "residual_full_affine_r2_positive_seeds": int(residual_positive),
        }
        all_four_counts.extend([output_positive, residual_positive])
    visual = "fixed_anchor_visual_derangement"
    output_ci = sum(
        results[seed]["bootstrap"][visual]["output_full_affine"][
            "ci_strictly_positive"
        ]
        for seed in SEEDS
    )
    residual_ci = sum(
        results[seed]["bootstrap"][visual]["residual_full_affine"][
            "ci_strictly_positive"
        ]
        for seed in SEEDS
    )
    strong_directional = int(config["decision"]["strong_directional_seeds"])
    strong_ci = int(config["decision"]["strong_bootstrap_positive_seeds"])
    mixed_directional = int(config["decision"]["mixed_directional_seeds"])
    validity = {
        "all_upstream_reproduced": all(
            all(results[seed]["reproduction"].values()) for seed in SEEDS
        ),
        "all_derangements_no_self": all(
            results[seed]["permutation"]["fixed_points"] == 0 for seed in SEEDS
        ),
        "all_test_unread": all(results[seed]["test_evaluated"] is False for seed in SEEDS),
    }
    if not all(validity.values()):
        verdict = "invalid_instance_information_audit"
    elif all(value == strong_directional for value in all_four_counts):
        if output_ci >= strong_ci and residual_ci >= strong_ci:
            verdict = "instance_specific_residual_signal_supported"
        else:
            verdict = "directional_instance_signal_supported_with_uncertainty"
    elif all(value >= mixed_directional for value in all_four_counts):
        verdict = "mixed_instance_signal"
    else:
        verdict = "instance_signal_not_supported"
    return {
        "verdict": verdict,
        "directional_counts": directional,
        "visual_derangement_bootstrap_positive_counts": {
            "output_full_affine": int(output_ci),
            "residual_full_affine": int(residual_ci),
        },
        "validity": validity,
        "test_evaluated": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_master(config)
    if args.validate_only:
        print(json.dumps({"protocol_version": PROTOCOL, "config_valid": True}, sort_keys=True))
        return
    output_root = Path(args.output_root or config["runtime"]["output_root"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / "instance_information_audit.json"
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite K14 result: {output_path}")
    torch.set_num_threads(int(config["runtime"]["num_threads"]))
    torch.set_float32_matmul_precision("high")
    device = torch.device(
        config["runtime"]["device"] if torch.cuda.is_available() else "cpu"
    )
    results: dict[int, dict[str, Any]] = {}
    for seed in SEEDS:
        results[seed] = seed_audit(config, seed, device)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    payload = {
        "protocol_version": PROTOCOL,
        "mode": "frozen_k80_three_seed_validation_only_instance_information_audit",
        "config_path": str(config_path),
        "config_sha256": base.sha256_file(config_path),
        "source_files_sha256": source_hashes(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "results": results,
        "decision": decide(results, config),
        "training_performed": False,
        "semantic_validation_evaluated": True,
        "test_evaluated": False,
    }
    v11.atomic_json(output_path, payload)
    print(
        json.dumps(
            {
                "result_path": str(output_path),
                "sha256": base.sha256_file(output_path),
                "verdict": payload["decision"]["verdict"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
