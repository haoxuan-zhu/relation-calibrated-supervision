"""Frozen validation-only downstream audit for K4/K5 geometry checkpoints."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

import audit_calibrated_relation_tube_gauge_k1 as gauge
import run_conflict_projected_physics_k80_closed as v17
import run_diagnostic as base
import run_dynamic_residual_propagation_k0 as dynamic
import run_physics_functional_anchor_training as v11
import run_relation_tube_geometry_ablation_k4 as k4
import run_relation_tube_geometry_tradeoff_k5 as k5
import run_supervised_continuation_diagnostic as v6


PROTOCOL = "relation_tube_geometry_downstream_audit_k6_v1"
GEOMETRIES = ("full_anisotropic", "diagonal", "isotropic", "rotated")
SUPPORTED_UPSTREAM_PROTOCOLS = {
    "relation_tube_geometry_ablation_k4_v1",
    "relation_tube_geometry_tradeoff_k5_v1",
}
REGISTERED_METRICS = (
    "signed_direct_rgb_correlation",
    "rgb_hungarian_mcc",
    "full_latent_hungarian_mcc",
    "coordinatewise_affine_rgb",
    "raw_rgb",
    "full_affine_rgb",
    "edge_auroc",
    "fixed_threshold_shd",
    "ccrl_validation",
)
SOURCE_FILES = (
    "audit_relation_tube_geometry_downstream_k6.py",
    "run_relation_tube_geometry_tradeoff_k5.py",
    *k4.SOURCE_FILES,
)


def source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    return {name: base.sha256_file(root / name) for name in SOURCE_FILES}


def configure_upstream_engine(protocol: str) -> Any:
    if protocol == "relation_tube_geometry_tradeoff_k5_v1":
        k5.configure_k4_engine()
    elif protocol != "relation_tube_geometry_ablation_k4_v1":
        raise ValueError(f"unsupported upstream protocol: {protocol}")
    return k4


def validate_config(config: dict[str, Any]) -> None:
    if config["protocol_version"] != PROTOCOL:
        raise ValueError("unexpected K6 protocol")
    if int(config["seed"]) not in (0, 42, 3407):
        raise ValueError("K6 is restricted to the three K4/K5 seeds")
    if config["upstream"]["geometry_protocol"] not in SUPPORTED_UPSTREAM_PROTOCOLS:
        raise ValueError("unregistered upstream geometry protocol")
    if tuple(config["audit"]["metrics"]) != REGISTERED_METRICS:
        raise ValueError("K6 metric registry changed")
    if config["audit"]["split"] != "validation_only_rows_8000_8999":
        raise ValueError("K6 validation split changed")
    if config["audit"]["test_evaluated"] is not False:
        raise ValueError("K6 test must remain closed")
    if not np.isclose(float(config["audit"]["reproduction_atol"]), 5e-6):
        raise ValueError("K6 reproduction tolerance changed")


def signed_direct_correlation(prediction: np.ndarray, truth: np.ndarray) -> list[float]:
    if prediction.shape != truth.shape or prediction.ndim != 2:
        raise ValueError("signed correlation expects matched two-dimensional arrays")
    values: list[float] = []
    for index in range(truth.shape[1]):
        value = float(np.corrcoef(prediction[:, index], truth[:, index])[0, 1])
        if not np.isfinite(value):
            raise ValueError("non-finite signed correlation")
        values.append(value)
    return values


def compose_full_latent(
    rgb_prediction: np.ndarray,
    latent_truth: np.ndarray,
    learned_indices: list[int],
    anchor_indices: list[int],
) -> np.ndarray:
    output = np.empty_like(latent_truth, dtype=np.float64)
    output[:, learned_indices] = rgb_prediction
    output[:, anchor_indices] = latent_truth[:, anchor_indices]
    return output


def semantic_bundle(
    calibration_prediction: np.ndarray,
    validation_prediction: np.ndarray,
    calibration_truth: np.ndarray,
    validation_truth: np.ndarray,
    validation_full_truth: np.ndarray,
    learned_indices: list[int],
    anchor_indices: list[int],
    ridge: float,
) -> dict[str, Any]:
    affine = gauge.metric_bundle(
        calibration_prediction,
        validation_prediction,
        calibration_truth,
        validation_truth,
        ridge,
    )
    signed = signed_direct_correlation(validation_prediction, validation_truth)
    rgb_matrix = base.absolute_correlation(validation_prediction, validation_truth)
    rgb_mcc, rgb_assignment = base.hungarian_mcc(rgb_matrix)
    full_prediction = compose_full_latent(
        validation_prediction,
        validation_full_truth,
        learned_indices,
        anchor_indices,
    )
    full_matrix = base.absolute_correlation(full_prediction, validation_full_truth)
    full_mcc, full_assignment = base.hungarian_mcc(full_matrix)
    return {
        "raw": affine["raw"],
        "coordinatewise_affine": affine["coordinatewise_affine"],
        "full_affine": affine["full_affine"],
        "signed_direct_rgb_correlation": signed,
        "mean_signed_direct_rgb_correlation": float(np.mean(signed)),
        "rgb_absolute_correlation_matrix": rgb_matrix,
        "rgb_hungarian_mcc": rgb_mcc,
        "rgb_hungarian_assignment": rgb_assignment,
        "full_latent_absolute_correlation_matrix": full_matrix,
        "full_latent_hungarian_mcc": full_mcc,
        "full_latent_hungarian_assignment": full_assignment,
    }


def reproduce_geometry_readout(
    observed: dict[str, Any], expected: dict[str, Any], atol: float
) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    for geometry in GEOMETRIES:
        for block in ("raw", "full_affine"):
            for metric in ("mean_r2", "mean_direct_abs_correlation", "mse"):
                key = f"{geometry}.{block}.{metric}"
                checks[key] = bool(
                    np.isclose(
                        float(observed[geometry][block][metric]),
                        float(expected[geometry][block][metric]),
                        atol=atol,
                        rtol=0.0,
                    )
                )
    return checks


def graph_comparison(full: dict[str, Any], comparator: dict[str, Any]) -> dict[str, Any]:
    edge_delta = float(full["edge_auroc"] - comparator["edge_auroc"])
    shd_delta = int(comparator["fixed_threshold_shd"] - full["fixed_threshold_shd"])
    if edge_delta > 0.0 and shd_delta >= 0:
        outcome = "full_graph_advantage"
    elif edge_delta < 0.0 and shd_delta <= 0:
        outcome = "comparator_graph_advantage"
    else:
        outcome = "mixed_graph_metrics"
    return {
        "full_minus_comparator_edge_auroc": edge_delta,
        "comparator_minus_full_shd": shd_delta,
        "outcome": outcome,
    }


def decide(runs: dict[str, Any]) -> dict[str, Any]:
    full = runs["full_anisotropic"]
    comparisons: dict[str, Any] = {}
    for name in GEOMETRIES[1:]:
        comparison = graph_comparison(full["graph"], runs[name]["graph"])
        comparison.update(
            full_minus_comparator_rgb_mcc=float(
                full["semantic"]["rgb_hungarian_mcc"]
                - runs[name]["semantic"]["rgb_hungarian_mcc"]
            ),
            full_minus_comparator_coordinatewise_r2=float(
                full["semantic"]["coordinatewise_affine"]["mean_r2"]
                - runs[name]["semantic"]["coordinatewise_affine"]["mean_r2"]
            ),
            full_minus_comparator_full_affine_r2=float(
                full["semantic"]["full_affine"]["mean_r2"]
                - runs[name]["semantic"]["full_affine"]["mean_r2"]
            ),
        )
        comparisons[name] = comparison
    generic_outcomes = {
        comparisons["diagonal"]["outcome"], comparisons["isotropic"]["outcome"]
    }
    if generic_outcomes == {"full_graph_advantage"}:
        verdict = "full_graph_advantage_over_generic_geometries_seed"
    elif generic_outcomes == {"comparator_graph_advantage"}:
        verdict = "generic_graph_advantage_over_full_seed"
    else:
        verdict = "generic_geometry_graph_outcomes_mixed_seed"
    return {
        "verdict": verdict,
        "comparisons": comparisons,
        "test_evaluated": False,
    }


def resolve_and_hash(path_text: str, expected_sha256: str, label: str) -> Path:
    path = Path(path_text).resolve()
    if base.sha256_file(path) != expected_sha256:
        raise ValueError(f"{label} hash mismatch")
    return path


def run(config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    validate_config(config)
    upstream = config["upstream"]
    geometry_config_path = resolve_and_hash(
        upstream["geometry_config_path"],
        upstream["geometry_config_sha256"],
        "geometry config",
    )
    geometry_readout_path = resolve_and_hash(
        upstream["geometry_readout_path"],
        upstream["geometry_readout_sha256"],
        "geometry readout",
    )
    full_readout_path = resolve_and_hash(
        upstream["full_validation_readout_path"],
        upstream["full_validation_readout_sha256"],
        "full validation readout",
    )
    geometry_config = yaml.safe_load(geometry_config_path.read_text(encoding="utf-8"))
    if geometry_config["protocol_version"] != upstream["geometry_protocol"]:
        raise ValueError("upstream geometry protocol mismatch")
    if int(geometry_config["training"]["seed"]) != int(config["seed"]):
        raise ValueError("upstream geometry seed mismatch")
    engine = configure_upstream_engine(upstream["geometry_protocol"])
    engine.validate_config(geometry_config)
    geometry_readout = json.loads(geometry_readout_path.read_text(encoding="utf-8"))
    full_readout = json.loads(full_readout_path.read_text(encoding="utf-8"))
    if geometry_readout["test_evaluated"] is not False:
        raise ValueError("upstream geometry readout touched test")
    if full_readout["test_evaluated"] is not False:
        raise ValueError("upstream full readout touched test")

    root = Path(geometry_config["runtime"]["output_root"]).resolve()
    payload, preflight_lock_path = engine.load_preflight(
        root, geometry_config_path, geometry_config
    )
    torch.set_num_threads(int(geometry_config["runtime"]["num_threads"]))
    torch.set_float32_matmul_precision("high")
    device = torch.device(
        geometry_config["runtime"]["device"] if torch.cuda.is_available() else "cpu"
    )
    images = np.load(Path(geometry_config["dataset"]["cache_path"]), mmap_mode="r")
    raw_latents = base.load_latents(geometry_config)
    subset, subset_audit = dynamic.build_subset(geometry_config)
    latents, _, _ = v17.normalize_with_k80_rgb_statistics(
        raw_latents, int(geometry_config["split"]["train_end"]), subset
    )
    start = int(geometry_config["split"]["train_end"])
    end = int(geometry_config["split"]["validation_end"])
    if (start, end, int(geometry_config["split"]["test_end"])) != (8000, 9000, 10000):
        raise ValueError("registered dataset split changed")
    batch_size = int(geometry_config["training"]["batch_size"])
    ridge = float(geometry_config["evaluation"]["full_affine_ridge"])
    threshold = float(geometry_config["evaluation"]["graph_threshold"])
    learned_indices = list(geometry_config["model"]["learned_indices"])
    anchor_indices = list(geometry_config["model"]["anchor_indices"])

    models: dict[str, torch.nn.Module] = {}
    model_locks: dict[str, Any] = {}
    models["full_anisotropic"], model_locks["full_anisotropic"] = (
        engine.load_upstream_full_model(geometry_config, payload, device)
    )
    for name in engine.TRAIN_GEOMETRIES:
        models[name], model_locks[name] = engine.load_new_model(
            name,
            root,
            geometry_config_path,
            geometry_config,
            payload,
            preflight_lock_path,
            device,
        )

    runs: dict[str, Any] = {}
    for name in GEOMETRIES:
        model = models[name]
        calibration_prediction = gauge.predict_selected(
            model, images, latents, subset, batch_size, device
        )
        validation_prediction = v6.predict_rgb(
            model, images, latents, start, end, batch_size, device
        )
        semantic = semantic_bundle(
            calibration_prediction,
            validation_prediction,
            latents[0, subset, :3],
            latents[0, start:end, :3],
            latents[0, start:end],
            learned_indices,
            anchor_indices,
            ridge,
        )
        estimated_a = model.parametric_part.A.detach().cpu().numpy()
        runs[name] = {
            "semantic": semantic,
            "graph": base.graph_metrics(estimated_a, threshold),
            "estimated_A": estimated_a,
            "ccrl_validation": geometry_readout["runs"][name]["ccrl_validation"],
        }
        del models[name]
        if device.type == "cuda":
            torch.cuda.empty_cache()

    reproduction = reproduce_geometry_readout(
        {name: runs[name]["semantic"] for name in GEOMETRIES},
        geometry_readout["runs"],
        float(config["audit"]["reproduction_atol"]),
    )
    expected_graph = full_readout["graph_parameter_metrics"]["tube_correct"]
    observed_graph = runs["full_anisotropic"]["graph"]
    full_graph_reproduction = {
        "edge_auroc": bool(
            np.isclose(
                observed_graph["edge_auroc"], expected_graph["edge_auroc"], atol=1e-12
            )
        ),
        "fixed_threshold_shd": bool(
            observed_graph["fixed_threshold_shd"]
            == expected_graph["fixed_threshold_shd"]
        ),
    }
    if not all(reproduction.values()) or not all(full_graph_reproduction.values()):
        raise RuntimeError(
            f"frozen checkpoint reproduction failed: semantic={reproduction}, "
            f"graph={full_graph_reproduction}"
        )

    return {
        "protocol_version": PROTOCOL,
        "mode": "frozen_k4_k5_validation_only_downstream_audit",
        "seed": int(config["seed"]),
        "config_path": str(config_path.resolve()),
        "config_sha256": base.sha256_file(config_path.resolve()),
        "source_files_sha256": source_hashes(),
        "upstream": {
            "geometry_config_path": str(geometry_config_path),
            "geometry_config_sha256": base.sha256_file(geometry_config_path),
            "geometry_readout_path": str(geometry_readout_path),
            "geometry_readout_sha256": base.sha256_file(geometry_readout_path),
            "full_validation_readout_path": str(full_readout_path),
            "full_validation_readout_sha256": base.sha256_file(full_readout_path),
            "preflight_lock_sha256": base.sha256_file(preflight_lock_path),
            "model_locks": model_locks,
        },
        "subset": subset_audit,
        "reproduction": reproduction,
        "full_graph_reproduction": full_graph_reproduction,
        "runs": runs,
        "decision": decide(runs),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "semantic_validation_evaluated": True,
        "test_evaluated": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    result = run(config_path, config)
    output = Path(args.output or config["runtime"]["output_path"]).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite K6 readout: {output}")
    v11.atomic_json(output, result)


if __name__ == "__main__":
    main()
