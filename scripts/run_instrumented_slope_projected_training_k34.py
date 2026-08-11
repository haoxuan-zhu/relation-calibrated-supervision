"""Train and evaluate the Instrumented Slope K34 projected objective."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml
from torch.nn import functional as F

import audit_instrumented_slope_geometry_k28 as k28
import audit_instrumented_slope_interface_k27 as k27
import causalverse_slope_preflight as slope
import evaluate_instrumented_slope_validation_k29 as k29
import run_causalverse_slope_preflight as preflight
import run_instrumented_slope_neural_tube_k30 as k30


CONDITIONS = ("joint_projected", "projected_only")


def source_hashes() -> dict[str, str]:
    paths = {
        "k34": Path(__file__),
        "k30": Path(k30.__file__),
        "k29": Path(k29.__file__),
        "k28": Path(k28.__file__),
        "k27": Path(k27.__file__),
        "slope": Path(slope.__file__),
        "preflight": Path(preflight.__file__),
    }
    return {name: preflight.sha256_file(path) for name, path in paths.items()}


def load_protocol(config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["protocol_version"] != "instrumented_slope_projected_training_k34":
        raise ValueError("unexpected K34 protocol")
    if tuple(config["model"]["conditions"]) != CONDITIONS:
        raise ValueError("K34 condition registry changed")
    if [int(value) for value in config["training"]["seeds"]] != [3407, 42, 0]:
        raise ValueError("K34 seed registry changed")
    if int(config["training"]["epochs"]) != 500:
        raise ValueError("K34 epoch registry changed")
    if int(config["projection"]["iterations"]) != 8:
        raise ValueError("K34 projection iteration registry changed")
    expected = ("k30_posthoc_curve", "joint_projected_curve", "projected_only_curve")
    if tuple(config["evaluation"]["candidates"]) != expected:
        raise ValueError("K34 evaluation registry changed")
    return config


def load_context(config: Mapping[str, Any]):
    context = k30.load_context_chain(config)
    source = config["source_contract"]
    k30_config_path = Path(source["k30_config"])
    k30_result_path = Path(source["k30_result"])
    k33_result_path = Path(source["k33_result"])
    k30_config = k30.load_protocol(k30_config_path)
    k30_result = json.loads(k30_result_path.read_text(encoding="utf-8"))
    k33_result = json.loads(k33_result_path.read_text(encoding="utf-8"))
    extra_hashes = {
        "k30_config": preflight.sha256_file(k30_config_path),
        "k30_result": preflight.sha256_file(k30_result_path),
        "k33_result": preflight.sha256_file(k33_result_path),
    }
    return (*context, k30_config, k30_result, k33_result, extra_hashes)


def curve_state_and_tangent(
    model: k30.InstrumentedSlopeTubeHead,
    roughness: torch.Tensor,
    side_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    a1, b1, a2, b2, deceleration, incline = model.relation_parameters
    theta = torch.deg2rad(side_state[:, 0])
    v0 = side_state[:, 1]
    mu1 = a1 * roughness + b1
    mu2 = a2 * roughness + b2
    v1 = v0 - deceleration * mu1
    denominator = incline * (torch.sin(theta) + mu2 * torch.cos(theta))
    length = v1.square() / denominator
    dv1 = -deceleration * a1
    denominator_derivative = incline * a2 * torch.cos(theta)
    dl = (
        2.0 * v1 * dv1 * denominator - v1.square() * denominator_derivative
    ) / denominator.square()
    state = torch.stack((roughness, mu1, mu2, v1, length), dim=1)
    tangent = torch.stack(
        (
            torch.ones_like(roughness),
            a1.expand_as(roughness),
            a2.expand_as(roughness),
            dv1.expand_as(roughness),
            dl,
        ),
        dim=1,
    )
    return state, tangent


def differentiable_curve_projection(
    normalized_prediction: torch.Tensor,
    side_state: torch.Tensor,
    model: k30.InstrumentedSlopeTubeHead,
    bounds: tuple[float, float],
    iterations: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if normalized_prediction.ndim != 2 or normalized_prediction.shape[1] != 5:
        raise ValueError("K34 prediction must have shape [N,5]")
    lower, upper = bounds
    physical_prediction = normalized_prediction * model.target_std + model.target_mean
    roughness = physical_prediction[:, 0].clamp(float(lower), float(upper))
    for _ in range(int(iterations)):
        state, tangent = curve_state_and_tangent(model, roughness, side_state)
        normalized_state = (state - model.target_mean) / model.target_std
        normalized_tangent = tangent / model.target_std
        residual = normalized_state - normalized_prediction
        step = (residual * normalized_tangent).sum(dim=1) / (
            normalized_tangent.square().sum(dim=1).clamp_min(1.0e-12)
        )
        roughness = (roughness - step).clamp(float(lower), float(upper))
    state, _ = curve_state_and_tangent(model, roughness, side_state)
    normalized_state = (state - model.target_mean) / model.target_std
    return normalized_state, {
        "roughness": roughness,
        "normalized_projection_gap": torch.linalg.vector_norm(
            normalized_state - normalized_prediction, dim=1
        ),
    }


def build_model(
    config: Mapping[str, Any],
    frozen: Mapping[str, Any],
    seed: int,
    device: torch.device,
) -> k30.InstrumentedSlopeTubeHead:
    return k30.build_model(config, frozen, "unbounded", seed, device)


def train_only_projection_audit(
    config: Mapping[str, Any],
    frozen: Mapping[str, Any],
    geometry_config: Mapping[str, Any],
    train: slope.GroupedFeatures,
) -> dict[str, float]:
    model = build_model(config, frozen, 3407, torch.device("cpu"))
    count = min(16, len(train.ids))
    features = torch.as_tensor(train.features[:count], dtype=torch.float32)
    side = torch.as_tensor(train.latents[:count, k27.SIDE_INDICES], dtype=torch.float32)
    repeated_side = side[:, None, :].expand(-1, 4, -1)
    with torch.no_grad():
        raw = model(
            features.reshape(-1, features.shape[-1]), repeated_side.reshape(-1, 2)
        )
        approximate, parts = differentiable_curve_projection(
            raw,
            repeated_side.reshape(-1, 2),
            model,
            tuple(float(value) for value in config["projection"]["roughness_bounds"]),
            int(config["projection"]["iterations"]),
        )
    raw_physical = raw.numpy() * frozen["target_std"] + frozen["target_mean"]
    exact, exact_audit = k28.nonlinear_curve_projection(
        raw_physical,
        repeated_side.reshape(-1, 2).numpy(),
        frozen["parameters"],
        frozen["target_std"],
        float(config["projection"]["roughness_bounds"][0]),
        float(config["projection"]["roughness_bounds"][1]),
        float(geometry_config["projection"]["nonlinear_tolerance"]),
        int(geometry_config["projection"]["nonlinear_max_iterations"]),
    )
    approximate_physical = approximate.numpy() * frozen["target_std"] + frozen["target_mean"]
    return {
        "row_count": int(len(raw)),
        "max_roughness_absolute_difference": float(
            np.max(np.abs(approximate_physical[:, 0] - exact[:, 0]))
        ),
        "max_normalized_state_l2_difference": float(
            np.max(
                np.linalg.norm(
                    (approximate_physical - exact) / frozen["target_std"], axis=1
                )
            )
        ),
        "mean_projection_gap": float(parts["normalized_projection_gap"].mean()),
        "exact_success_count": int(exact_audit["success_count"]),
    }


def preflight_main(config_path: Path, output_root: Path) -> None:
    config = load_protocol(config_path)
    (
        _, validation_result, geometry_config, interface_config, grouped, splits, label_ids,
        filter_audit, hashes, _, k30_result, k33_result, extra_hashes,
    ) = load_context(config)
    train = slope.select_ids(grouped, splits["train"])
    frozen = k30.frozen_components(train, label_ids, geometry_config)
    k30_preflight = json.loads(
        (Path(config["source_contract"]["k30_output_root"]) / "preflight" / "preflight.json")
        .read_text(encoding="utf-8")
    )
    initial_hashes: dict[str, str] = {}
    parameter_counts: dict[str, int] = {}
    initial_matches: dict[str, bool] = {}
    for seed_value in config["training"]["seeds"]:
        seed = int(seed_value)
        first = build_model(config, frozen, seed, torch.device("cpu"))
        second = build_model(config, frozen, seed, torch.device("cpu"))
        first_hash = slope.state_dict_sha256(first.state_dict())
        if first_hash != slope.state_dict_sha256(second.state_dict()):
            raise RuntimeError("K34 matched conditions do not share an initial state")
        initial_hashes[str(seed)] = first_hash
        initial_matches[str(seed)] = first_hash == k30_preflight["initial_state_sha256"][str(seed)]
        parameter_counts[str(seed)] = sum(parameter.numel() for parameter in first.parameters())
    projection_audit = train_only_projection_audit(config, frozen, geometry_config, train)
    source = config["source_contract"]
    validity = {
        "validation_config_sha_matches": hashes["validation_config"]
        == str(source["validation_config_sha256"]),
        "validation_result_sha_matches": hashes["validation_result"]
        == str(source["validation_result_sha256"]),
        "validation_result_valid": bool(validation_result["valid"]),
        "k30_config_sha_matches": extra_hashes["k30_config"]
        == str(source["k30_config_sha256"]),
        "k30_result_sha_matches": extra_hashes["k30_result"]
        == str(source["k30_result_sha256"]),
        "k33_result_sha_matches": extra_hashes["k33_result"]
        == str(source["k33_result_sha256"]),
        "k30_result_valid": bool(k30_result["valid"]),
        "k33_result_valid": bool(k33_result["valid"]),
        "prior_results_test_unread": not bool(k30_result["test_evaluated"])
        and not bool(k33_result["test_evaluated"]),
        "feature_cache_sha_matches": hashes["feature_cache"]
        == str(interface_config["runtime"]["feature_cache_sha256"]),
        "incomplete_product_filter_contract_matches": bool(filter_audit["contract_matches"]),
        "k30_initial_states_reproduced": all(initial_matches.values()),
        "parameter_counts_matched": len(set(parameter_counts.values())) == 1,
        "projection_exact_on_train_audit": projection_audit[
            "max_normalized_state_l2_difference"
        ]
        <= 1.0e-4,
        "projection_rows_all_solved": projection_audit["exact_success_count"]
        == projection_audit["row_count"],
    }
    payload = {
        "protocol_version": config["protocol_version"],
        "fact_type": "projected_training_preflight_after_known_development_validation",
        "scope": "training_ids_and_k40_labels_only_external_and_test_sealed",
        "config_sha256": preflight.sha256_file(config_path),
        "source_files_sha256": source_hashes(),
        "source_contract_hashes": {**hashes, **extra_hashes},
        "label_ids_sha256": preflight.sha256_int_array(label_ids),
        "train_ids_sha256": preflight.sha256_int_array(splits["train"]),
        "validation_ids_sha256": preflight.sha256_int_array(splits["validation"]),
        "test_ids_sha256": preflight.sha256_int_array(splits["test"]),
        "initial_state_sha256": initial_hashes,
        "k30_initial_state_matches": initial_matches,
        "parameter_counts": parameter_counts,
        "train_only_projection_audit": projection_audit,
        "development_validation_used_for_design": True,
        "external_evaluated": False,
        "test_evaluated": False,
        "validity": validity,
        "valid": bool(all(validity.values())),
        "machine_decision": "projected_training_preflight_pass"
        if all(validity.values())
        else "invalid",
    }
    output = output_root / "preflight" / "preflight.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    preflight.write_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


def load_preflight(output_root: Path, config_path: Path) -> dict[str, Any]:
    payload = json.loads(
        (output_root / "preflight" / "preflight.json").read_text(encoding="utf-8")
    )
    checks = {
        "valid": bool(payload["valid"]),
        "config_current": payload["config_sha256"] == preflight.sha256_file(config_path),
        "sources_current": payload["source_files_sha256"] == source_hashes(),
        "external_unread": not bool(payload["external_evaluated"]),
        "test_unread": not bool(payload["test_evaluated"]),
    }
    if not all(checks.values()):
        raise ValueError(f"invalid K34 preflight: {checks}")
    return payload


def train_one(
    config: Mapping[str, Any],
    frozen: Mapping[str, Any],
    train: slope.GroupedFeatures,
    condition: str,
    seed: int,
    device: torch.device,
) -> tuple[k30.InstrumentedSlopeTubeHead, list[dict[str, float]], str, str]:
    features = torch.as_tensor(train.features, dtype=torch.float32, device=device)
    side = torch.as_tensor(train.latents[:, k27.SIDE_INDICES], dtype=torch.float32, device=device)
    targets = torch.as_tensor(
        (train.latents[:, k27.TARGET_INDICES] - frozen["target_mean"])
        / frozen["target_std"],
        dtype=torch.float32,
        device=device,
    )
    label_rows = torch.as_tensor(frozen["label_rows"], dtype=torch.long, device=device)
    model = build_model(config, frozen, seed, device)
    initial_hash = slope.state_dict_sha256(model.state_dict())
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    weights = config["loss_weights"]
    point_mix = config["point_loss"][condition]
    bounds = tuple(float(value) for value in config["projection"]["roughness_bounds"])
    repeated_side = side[:, None, :].expand(-1, 4, -1)
    history: list[dict[str, float]] = []
    started = time.time()
    slope.set_global_seed(seed)
    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        model.train()
        raw = model(
            features.reshape(-1, features.shape[-1]), repeated_side.reshape(-1, 2)
        )
        projected, projection_parts = differentiable_curve_projection(
            raw,
            repeated_side.reshape(-1, 2),
            model,
            bounds,
            int(config["projection"]["iterations"]),
        )
        raw = raw.reshape(len(train.ids), 4, 5)
        projected = projected.reshape(len(train.ids), 4, 5)
        common = k30.generic_multiview_losses(
            raw, float(config["loss_weights"]["range_zscore_limit"])
        )
        expanded_targets = targets[label_rows, None, :].expand(-1, 4, -1)
        raw_point = F.mse_loss(raw[label_rows], expanded_targets)
        projected_point = F.mse_loss(projected[label_rows], expanded_targets)
        point = float(point_mix["raw_weight"]) * raw_point + float(
            point_mix["projected_weight"]
        ) * projected_point
        total = (
            float(weights["view"]) * common["view"]
            + float(weights["variance"]) * common["variance"]
            + float(weights["covariance"]) * common["covariance"]
            + float(weights["point"]) * point
            + float(weights["range"]) * common["range"]
        )
        if not torch.isfinite(total):
            raise FloatingPointError(f"non-finite K34 loss at epoch {epoch}")
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        optimizer.step()
        if epoch == 1 or epoch == int(config["training"]["epochs"]) or epoch % 50 == 0:
            roughness = projection_parts["roughness"]
            record = {
                "epoch": epoch,
                "total": float(total.detach().cpu()),
                "point": float(point.detach().cpu()),
                "raw_point": float(raw_point.detach().cpu()),
                "projected_point": float(projected_point.detach().cpu()),
                "projection_gap": float(
                    projection_parts["normalized_projection_gap"].detach().mean().cpu()
                ),
                "projected_roughness_min": float(roughness.detach().min().cpu()),
                "projected_roughness_max": float(roughness.detach().max().cpu()),
                **{name: float(value.detach().cpu()) for name, value in common.items()},
                "elapsed_seconds": time.time() - started,
            }
            history.append(record)
            print(
                json.dumps({"seed": seed, "condition": condition, **record}, sort_keys=True),
                flush=True,
            )
    return model, history, initial_hash, slope.state_dict_sha256(model.state_dict())


def train_main(config_path: Path, output_root: Path, condition: str, seed: int) -> None:
    config = load_protocol(config_path)
    if condition not in CONDITIONS or seed not in [int(v) for v in config["training"]["seeds"]]:
        raise ValueError("unregistered K34 train request")
    preflight_payload = load_preflight(output_root, config_path)
    (
        _, _, geometry_config, _, grouped, splits, label_ids, _, _, _, _, _, _,
    ) = load_context(config)
    train = slope.select_ids(grouped, splits["train"])
    frozen = k30.frozen_components(train, label_ids, geometry_config)
    torch.set_num_threads(int(config["runtime"]["num_threads"]))
    torch.set_float32_matmul_precision("high")
    device = torch.device(config["runtime"]["device"] if torch.cuda.is_available() else "cpu")
    model, history, initial_hash, final_hash = train_one(
        config, frozen, train, condition, seed, device
    )
    if initial_hash != preflight_payload["initial_state_sha256"][str(seed)]:
        raise RuntimeError("K34 initial state drifted after preflight")
    output_dir = output_root / "formal" / f"seed{seed}" / condition
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "training_results.json"
    checkpoint_path = output_dir / f"{condition}_seed{seed}.pt"
    lock_path = output_dir / "training_lock.json"
    if result_path.exists() or checkpoint_path.exists() or lock_path.exists():
        raise FileExistsError(f"refusing to overwrite K34 seed{seed} {condition}")
    torch.save(
        {
            "protocol_version": config["protocol_version"],
            "seed": seed,
            "condition": condition,
            "state_dict": model.state_dict(),
        },
        checkpoint_path,
    )
    checkpoint_sha = preflight.sha256_file(checkpoint_path)
    validity = {
        "preflight_valid": bool(preflight_payload["valid"]),
        "initial_state_exact": initial_hash
        == preflight_payload["initial_state_sha256"][str(seed)],
        "epoch_exact": history[-1]["epoch"] == int(config["training"]["epochs"]),
        "history_finite": all(
            np.isfinite(value)
            for record in history
            for key, value in record.items()
            if key != "epoch"
        ),
        "parameter_count_exact": sum(parameter.numel() for parameter in model.parameters())
        == int(preflight_payload["parameter_counts"][str(seed)]),
        "projected_roughness_inside_bounds": all(
            record["projected_roughness_min"] >= -1.0e-7
            and record["projected_roughness_max"] <= 1.0 + 1.0e-7
            for record in history
        ),
    }
    lock = {
        "status": "locked_before_joint_development_readout",
        "protocol_version": config["protocol_version"],
        "config_sha256": preflight.sha256_file(config_path),
        "source_files_sha256": source_hashes(),
        "seed": seed,
        "condition": condition,
        "initial_state_sha256": initial_hash,
        "final_state_sha256": final_hash,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "validity": validity,
        "external_evaluated": False,
        "test_evaluated": False,
    }
    result = {
        "protocol_version": config["protocol_version"],
        "fact_type": "formal_projected_train_after_known_development_validation",
        "seed": seed,
        "condition": condition,
        "config_sha256": preflight.sha256_file(config_path),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "initial_state_sha256": initial_hash,
        "final_state_sha256": final_hash,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "history": history,
        "validity": validity,
        "valid": bool(all(validity.values())),
        "external_evaluated": False,
        "test_evaluated": False,
    }
    preflight.write_json(lock_path, lock)
    preflight.write_json(result_path, result)
    print(json.dumps({"result": str(result_path), "lock": str(lock_path)}, sort_keys=True))


def load_model_checkpoint(
    config: Mapping[str, Any],
    frozen: Mapping[str, Any],
    output_root: Path,
    condition: str,
    seed: int,
    device: torch.device,
) -> k30.InstrumentedSlopeTubeHead:
    output_dir = output_root / "formal" / f"seed{seed}" / condition
    lock = json.loads((output_dir / "training_lock.json").read_text(encoding="utf-8"))
    checkpoint_path = Path(lock["checkpoint"])
    checks = {
        "lock_valid": all(lock["validity"].values()),
        "identity": lock["condition"] == condition and int(lock["seed"]) == seed,
        "checkpoint_current": preflight.sha256_file(checkpoint_path)
        == lock["checkpoint_sha256"],
        "sources_current": lock["source_files_sha256"] == source_hashes(),
        "external_unread": not bool(lock["external_evaluated"]),
        "test_unread": not bool(lock["test_evaluated"]),
    }
    if not all(checks.values()):
        raise ValueError(f"invalid K34 checkpoint lock: {checks}")
    model = build_model(config, frozen, seed, device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    if slope.state_dict_sha256(model.state_dict()) != lock["final_state_sha256"]:
        raise ValueError("K34 final state hash mismatch")
    return model


def exact_curve_prediction(
    raw_prediction: np.ndarray,
    side: np.ndarray,
    frozen: Mapping[str, Any],
    geometry_config: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    lower, upper = (float(value) for value in geometry_config["calibration"]["roughness_physical_bounds"])
    return k28.nonlinear_curve_projection(
        raw_prediction,
        side,
        frozen["parameters"],
        frozen["target_std"],
        lower,
        upper,
        float(geometry_config["projection"]["nonlinear_tolerance"]),
        int(geometry_config["projection"]["nonlinear_max_iterations"]),
    )


def evaluate_main(config_path: Path, output_root: Path) -> None:
    config = load_protocol(config_path)
    preflight_payload = load_preflight(output_root, config_path)
    (
        _, _, geometry_config, _, grouped, splits, label_ids, _, _,
        k30_config, k30_result, _, _,
    ) = load_context(config)
    train = slope.select_ids(grouped, splits["train"])
    validation = slope.select_ids(grouped, splits["validation"])
    frozen = k30.frozen_components(train, label_ids, geometry_config)
    truth = validation.latents[:, k27.TARGET_INDICES]
    side = validation.latents[:, k27.SIDE_INDICES]
    device = torch.device(config["runtime"]["device"] if torch.cuda.is_available() else "cpu")
    k30_root = Path(config["source_contract"]["k30_output_root"])
    seed_results: dict[str, Any] = {}
    predictions_by_name: dict[str, list[np.ndarray]] = {
        name: [] for name in config["evaluation"]["candidates"]
    }
    primary_directional: list[bool] = []
    baseline_reproduced: list[bool] = []
    pairs = {
        "joint_projected_minus_k30_posthoc": (
            "joint_projected_curve",
            "k30_posthoc_curve",
        ),
        "projected_only_minus_k30_posthoc": (
            "projected_only_curve",
            "k30_posthoc_curve",
        ),
        "joint_projected_minus_projected_only": (
            "joint_projected_curve",
            "projected_only_curve",
        ),
    }
    for seed_value in config["training"]["seeds"]:
        seed = int(seed_value)
        baseline_model = k30.load_model_checkpoint(
            k30_config, frozen, k30_root, "unbounded", seed, device
        )
        baseline_raw, baseline_model_audit = k30.predict_model(
            baseline_model, validation, frozen, device
        )
        baseline_curve, baseline_curve_audit = exact_curve_prediction(
            baseline_raw, side, frozen, geometry_config
        )
        predictions: dict[str, np.ndarray] = {"k30_posthoc_curve": baseline_curve}
        audits: dict[str, Any] = {
            "k30_posthoc_curve": {
                "raw_model": baseline_model_audit,
                "exact_curve": baseline_curve_audit,
            }
        }
        for condition in CONDITIONS:
            model = load_model_checkpoint(
                config, frozen, output_root, condition, seed, device
            )
            raw, raw_audit = k30.predict_model(model, validation, frozen, device)
            curve, curve_audit = exact_curve_prediction(raw, side, frozen, geometry_config)
            name = f"{condition}_curve"
            predictions[name] = curve
            audits[name] = {"raw_model": raw_audit, "exact_curve": curve_audit}
            del model
        metrics = {name: k27.direct_metrics(truth, value) for name, value in predictions.items()}
        recorded = k30_result["seed_results"][str(seed)]["metrics"][
            "nonlinear_curve_from_unbounded"
        ]
        reproduction = all(
            abs(float(metrics["k30_posthoc_curve"][key]) - float(recorded[key])) <= 1.0e-10
            for key in ("mean_abs_correlation", "mean_r2", "mean_normalized_rmse")
        )
        baseline_reproduced.append(reproduction)
        deltas: dict[str, dict[str, float]] = {}
        bootstrap: dict[str, Any] = {}
        for name, (candidate, baseline_name) in pairs.items():
            deltas[name] = {
                "mean_abs_correlation": metrics[candidate]["mean_abs_correlation"]
                - metrics[baseline_name]["mean_abs_correlation"],
                "mean_r2": metrics[candidate]["mean_r2"] - metrics[baseline_name]["mean_r2"],
            }
            bootstrap[name] = k29.paired_bootstrap(
                truth,
                predictions[candidate],
                predictions[baseline_name],
                int(config["evaluation"]["bootstrap_repetitions"]),
                int(config["evaluation"]["bootstrap_seed"]),
                float(config["evaluation"]["bootstrap_confidence"]),
            )
        primary_directional.append(
            all(value > 0.0 for value in deltas["joint_projected_minus_k30_posthoc"].values())
        )
        for name, prediction in predictions.items():
            predictions_by_name[name].append(prediction)
        seed_results[str(seed)] = {
            "metrics": metrics,
            "deltas": deltas,
            "paired_bootstrap": bootstrap,
            "projection_audits": audits,
            "decision_checks": {
                "k30_baseline_reproduced": reproduction,
                "primary_directional_both": primary_directional[-1],
            },
        }
        del baseline_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    mean_predictions = {
        name: np.mean(np.stack(values, axis=0), axis=0)
        for name, values in predictions_by_name.items()
    }
    aggregate_metrics = {
        name: k27.direct_metrics(truth, prediction)
        for name, prediction in mean_predictions.items()
    }
    aggregate_deltas = {
        name: {
            "mean_abs_correlation": aggregate_metrics[candidate]["mean_abs_correlation"]
            - aggregate_metrics[baseline_name]["mean_abs_correlation"],
            "mean_r2": aggregate_metrics[candidate]["mean_r2"]
            - aggregate_metrics[baseline_name]["mean_r2"],
        }
        for name, (candidate, baseline_name) in pairs.items()
    }
    aggregate_bootstrap = {
        name: k29.paired_bootstrap(
            truth,
            mean_predictions[candidate],
            mean_predictions[baseline_name],
            int(config["evaluation"]["bootstrap_repetitions"]),
            int(config["evaluation"]["bootstrap_seed"]),
            float(config["evaluation"]["bootstrap_confidence"]),
        )
        for name, (candidate, baseline_name) in pairs.items()
    }
    primary_aggregate_ci = k29.confirmed(
        aggregate_deltas["joint_projected_minus_k30_posthoc"],
        aggregate_bootstrap["joint_projected_minus_k30_posthoc"],
    )
    primary_pass = all(primary_directional) and primary_aggregate_ci
    validity = {
        "preflight_valid": bool(preflight_payload["valid"]),
        "all_six_training_locks_loaded": len(seed_results) == 3,
        "k30_baseline_reproduced_all_seeds": all(baseline_reproduced),
        "validation_ids_exact": set(int(value) for value in validation.ids)
        == set(int(value) for value in splits["validation"]),
        "test_ids_not_evaluated": not bool(
            set(int(value) for value in validation.ids)
            & set(int(value) for value in splits["test"])
        ),
        "all_metrics_finite": all(
            np.isfinite(value)
            for seed_result in seed_results.values()
            for condition_metrics in seed_result["metrics"].values()
            for value in (
                condition_metrics["mean_abs_correlation"],
                condition_metrics["mean_r2"],
                condition_metrics["mean_normalized_rmse"],
            )
        ),
    }
    valid = bool(all(validity.values()))
    if not valid:
        decision = "invalid"
    elif primary_pass:
        decision = "projected_training_advantage_confirmed_external_unlocked"
    elif all(primary_directional):
        decision = "projected_training_directional_only_external_locked"
    else:
        decision = "projected_training_not_confirmed_external_locked"
    payload = {
        "protocol_version": config["protocol_version"],
        "fact_type": "locked_projected_training_development_validation_multiseed",
        "scope": "known_development_validation_only_external_and_test_sealed",
        "config_sha256": preflight.sha256_file(config_path),
        "source_files_sha256": source_hashes(),
        "validation_count": int(len(validation.ids)),
        "validation_ids_sha256": preflight.sha256_int_array(validation.ids),
        "test_ids_sha256": preflight.sha256_int_array(splits["test"]),
        "seed_results": seed_results,
        "aggregate_seed_ensemble": {
            "metrics": aggregate_metrics,
            "deltas": aggregate_deltas,
            "paired_bootstrap": aggregate_bootstrap,
        },
        "decision_checks": {
            "primary_directional_seed_count": int(sum(primary_directional)),
            "primary_aggregate_ci": primary_aggregate_ci,
            "primary_pass": primary_pass,
            "external_evaluation_unlocked": primary_pass,
        },
        "validity": validity,
        "valid": valid,
        "development_validation_used_for_design": True,
        "external_evaluated": False,
        "test_evaluated": False,
        "machine_decision": decision,
    }
    output = output_root / "evaluation" / "validation_results.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    preflight.write_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "train", "evaluate"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--condition", choices=CONDITIONS)
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_protocol(args.config)
    output_root = args.output_root or Path(config["runtime"]["output_root"])
    if args.mode == "preflight":
        preflight_main(args.config, output_root)
    elif args.mode == "train":
        if args.condition is None or args.seed is None:
            raise ValueError("train mode requires --condition and --seed")
        train_main(args.config, output_root, args.condition, args.seed)
    else:
        evaluate_main(args.config, output_root)


if __name__ == "__main__":
    main()
