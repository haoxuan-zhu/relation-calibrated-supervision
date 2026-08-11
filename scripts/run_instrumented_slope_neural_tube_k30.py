"""Preflight, train, and evaluate the Instrumented Slope K30 neural tube."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml
from torch import nn
from torch.nn import functional as F

import audit_instrumented_slope_geometry_k28 as k28
import audit_instrumented_slope_interface_k27 as k27
import causalverse_slope_preflight as slope
import evaluate_instrumented_slope_validation_k29 as k29
import run_causalverse_slope_preflight as preflight


CONDITIONS = ("unbounded", "isotropic")


def source_hashes() -> dict[str, str]:
    paths = {
        "k30": Path(__file__),
        "k29": Path(k29.__file__),
        "k28": Path(k28.__file__),
        "k27": Path(k27.__file__),
        "slope": Path(slope.__file__),
        "preflight": Path(preflight.__file__),
    }
    return {name: preflight.sha256_file(path) for name, path in paths.items()}


def generic_multiview_losses(
    predictions: torch.Tensor, range_limit: float
) -> dict[str, torch.Tensor]:
    if predictions.ndim != 3 or predictions.shape[1] != 4 or predictions.shape[2] != 5:
        raise ValueError("predictions must have shape [N,4,5]")
    id_mean = predictions.mean(dim=1)
    std = torch.sqrt(id_mean.var(dim=0, unbiased=False) + 1e-4)
    centered = id_mean - id_mean.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / max(len(id_mean) - 1, 1)
    off_diagonal = covariance - torch.diag(torch.diagonal(covariance))
    return {
        "view": (predictions - id_mean[:, None, :]).square().mean(),
        "variance": F.relu(1.0 - std).mean(),
        "covariance": off_diagonal.square().sum() / predictions.shape[2],
        "range": F.relu(id_mean.abs() - float(range_limit)).square().mean(),
    }


class InstrumentedSlopeTubeHead(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dims: Sequence[int],
        condition: str,
        parameters: slope.RelationParameters,
        mean_roughness: float,
        target_mean: np.ndarray,
        target_std: np.ndarray,
        side_mean: np.ndarray,
        side_std: np.ndarray,
        radius: float,
    ) -> None:
        super().__init__()
        if condition not in CONDITIONS or len(hidden_dims) != 2:
            raise ValueError("unexpected neural-tube model contract")
        first, second = (int(value) for value in hidden_dims)
        self.condition = condition
        self.visual = nn.Sequential(
            nn.Linear(int(feature_dim), first),
            nn.LayerNorm(first),
            nn.GELU(),
            nn.Linear(first, second),
            nn.LayerNorm(second),
            nn.GELU(),
        )
        self.film = nn.Linear(2, 2 * second)
        self.output = nn.Linear(second, 5)
        self.register_buffer(
            "relation_parameters",
            torch.as_tensor(parameters.as_array(), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "mean_roughness", torch.as_tensor(float(mean_roughness), dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "target_mean", torch.as_tensor(target_mean, dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "target_std", torch.as_tensor(target_std, dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "side_mean", torch.as_tensor(side_mean, dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "side_std", torch.as_tensor(side_std, dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "radius", torch.as_tensor(float(radius), dtype=torch.float32), persistent=False
        )

    def center_physical(self, side_state: torch.Tensor) -> torch.Tensor:
        a1, b1, a2, b2, deceleration, incline = self.relation_parameters
        theta = torch.deg2rad(side_state[:, 0])
        v0 = side_state[:, 1]
        roughness = self.mean_roughness.expand_as(v0)
        mu1 = a1 * roughness + b1
        mu2 = a2 * roughness + b2
        v1 = v0 - deceleration * mu1
        denominator = incline * (torch.sin(theta) + mu2 * torch.cos(theta))
        length = v1.square() / denominator
        return torch.stack((roughness, mu1, mu2, v1, length), dim=1)

    def raw_residual(self, features: torch.Tensor, side_state: torch.Tensor) -> torch.Tensor:
        visual = self.visual(features)
        side = (side_state - self.side_mean) / self.side_std
        gamma, beta = self.film(side).chunk(2, dim=1)
        return self.output(visual * (1.0 + gamma) + beta)

    def components(
        self, features: torch.Tensor, side_state: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        center = (self.center_physical(side_state) - self.target_mean) / self.target_std
        raw = self.raw_residual(features, side_state)
        norms = torch.linalg.vector_norm(raw, dim=1)
        if self.condition == "isotropic":
            scales = (self.radius / norms.clamp_min(1e-12)).clamp(max=1.0)
        else:
            scales = torch.ones_like(norms)
        projected = raw * scales[:, None]
        return center + projected, {
            "center": center,
            "raw_residual": raw,
            "projected_residual": projected,
            "projection_scale": scales,
            "boundary_active": norms > self.radius,
        }

    def forward(self, features: torch.Tensor, side_state: torch.Tensor) -> torch.Tensor:
        return self.components(features, side_state)[0]


def load_protocol(config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["protocol_version"] != "instrumented_slope_neural_tube_k30":
        raise ValueError("unexpected K30 protocol")
    if tuple(config["model"]["conditions"]) != CONDITIONS:
        raise ValueError("K30 condition registry changed")
    if [int(value) for value in config["training"]["seeds"]] != [3407, 42, 0]:
        raise ValueError("K30 seed registry changed")
    if int(config["training"]["epochs"]) != 500:
        raise ValueError("K30 formal epoch count changed")
    return config


def load_context_chain(config: Mapping[str, Any]):
    validation_config_path = Path(config["source_contract"]["validation_config"])
    validation_result_path = Path(config["source_contract"]["validation_result"])
    validation_config = yaml.safe_load(validation_config_path.read_text(encoding="utf-8"))
    validation_result = json.loads(validation_result_path.read_text(encoding="utf-8"))
    geometry_config_path = Path(validation_config["source_contract"]["geometry_config"])
    geometry_config = yaml.safe_load(geometry_config_path.read_text(encoding="utf-8"))
    interface_config_path = Path(geometry_config["source_contract"]["interface_config"])
    interface_config, grouped, splits, label_ids, filter_audit = preflight.load_context(
        interface_config_path
    )
    hashes = {
        "validation_config": preflight.sha256_file(validation_config_path),
        "validation_result": preflight.sha256_file(validation_result_path),
        "geometry_config": preflight.sha256_file(geometry_config_path),
        "interface_config": preflight.sha256_file(interface_config_path),
        "feature_cache": preflight.sha256_file(Path(interface_config["runtime"]["feature_cache"])),
    }
    return (
        validation_config,
        validation_result,
        geometry_config,
        interface_config,
        grouped,
        splits,
        label_ids,
        filter_audit,
        hashes,
    )


def frozen_components(
    train: slope.GroupedFeatures,
    label_ids: np.ndarray,
    geometry_config: Mapping[str, Any],
) -> dict[str, Any]:
    label_rows = slope.ordered_label_indices(train.ids, label_ids)
    labels = train.latents[label_rows]
    targets = labels[:, k27.TARGET_INDICES]
    side = labels[:, k27.SIDE_INDICES]
    parameters, mean_roughness, target_std, geometry = k28.leave_one_out_geometry(
        labels, float(geometry_config["calibration"]["coverage"])
    )
    side_std = side.std(axis=0)
    if np.any(side_std <= 1e-8):
        raise ValueError("K40 side-state scale is degenerate")
    return {
        "label_rows": label_rows,
        "parameters": parameters,
        "mean_roughness": mean_roughness,
        "target_mean": targets.mean(axis=0),
        "target_std": target_std,
        "side_mean": side.mean(axis=0),
        "side_std": side_std,
        "geometry": geometry,
    }


def build_model(
    config: Mapping[str, Any], frozen: Mapping[str, Any], condition: str, seed: int, device: torch.device
) -> InstrumentedSlopeTubeHead:
    slope.set_global_seed(seed)
    model = InstrumentedSlopeTubeHead(
        int(config["model"]["feature_dim"]),
        config["model"]["hidden_dims"],
        condition,
        frozen["parameters"],
        float(frozen["mean_roughness"]),
        np.asarray(frozen["target_mean"]),
        np.asarray(frozen["target_std"]),
        np.asarray(frozen["side_mean"]),
        np.asarray(frozen["side_std"]),
        float(frozen["geometry"]["isotropic"]["radius"]),
    )
    return model.to(device)


def preflight_main(config_path: Path, output_root: Path) -> None:
    config = load_protocol(config_path)
    (
        _, validation_result, geometry_config, interface_config, grouped, splits, label_ids,
        filter_audit, hashes,
    ) = load_context_chain(config)
    train = slope.select_ids(grouped, splits["train"])
    frozen = frozen_components(train, label_ids, geometry_config)
    initial_hashes: dict[str, str] = {}
    parameter_counts: dict[str, int] = {}
    device = torch.device("cpu")
    for seed in config["training"]["seeds"]:
        unbounded = build_model(config, frozen, "unbounded", int(seed), device)
        isotropic = build_model(config, frozen, "isotropic", int(seed), device)
        unbounded_hash = slope.state_dict_sha256(unbounded.state_dict())
        isotropic_hash = slope.state_dict_sha256(isotropic.state_dict())
        if unbounded_hash != isotropic_hash:
            raise RuntimeError("matched K30 conditions do not share an initial state")
        initial_hashes[str(seed)] = unbounded_hash
        parameter_counts[str(seed)] = sum(parameter.numel() for parameter in unbounded.parameters())
    validity = {
        "validation_config_sha_matches": hashes["validation_config"]
        == str(config["source_contract"]["validation_config_sha256"]),
        "validation_result_sha_matches": hashes["validation_result"]
        == str(config["source_contract"]["validation_result_sha256"]),
        "validation_result_valid": bool(validation_result["valid"]),
        "validation_result_test_unread": not bool(validation_result["test_evaluated"]),
        "feature_cache_sha_matches": hashes["feature_cache"]
        == str(interface_config["runtime"]["feature_cache_sha256"]),
        "incomplete_product_filter_contract_matches": bool(filter_audit["contract_matches"]),
        "initial_states_registered": len(initial_hashes) == 3 and len(set(initial_hashes.values())) == 3,
        "parameter_counts_matched": len(set(parameter_counts.values())) == 1,
        "calibration_rank_exact": frozen["geometry"]["isotropic"]["order_index_one_based"] == 39,
    }
    payload = {
        "protocol_version": config["protocol_version"],
        "fact_type": "train_only_neural_tube_preflight",
        "scope": "training_ids_and_k40_labels_only_no_validation_or_test_semantics",
        "config_sha256": preflight.sha256_file(config_path),
        "source_files_sha256": source_hashes(),
        "source_contract_hashes": hashes,
        "label_ids_sha256": preflight.sha256_int_array(label_ids),
        "train_ids_sha256": preflight.sha256_int_array(splits["train"]),
        "validation_ids_sha256": preflight.sha256_int_array(splits["validation"]),
        "test_ids_sha256": preflight.sha256_int_array(splits["test"]),
        "initial_state_sha256": initial_hashes,
        "parameter_counts": parameter_counts,
        "calibration_geometry": frozen["geometry"],
        "validity": validity,
        "valid": bool(all(validity.values())),
        "semantic_validation_evaluated": False,
        "test_evaluated": False,
        "machine_decision": "neural_tube_preflight_pass" if all(validity.values()) else "invalid",
    }
    output = output_root / "preflight" / "preflight.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    preflight.write_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


def train_one(
    config: Mapping[str, Any],
    frozen: Mapping[str, Any],
    train: slope.GroupedFeatures,
    condition: str,
    seed: int,
    device: torch.device,
) -> tuple[InstrumentedSlopeTubeHead, list[dict[str, float]], str, str]:
    features = torch.as_tensor(train.features, dtype=torch.float32, device=device)
    side = torch.as_tensor(train.latents[:, k27.SIDE_INDICES], dtype=torch.float32, device=device)
    targets = torch.as_tensor(
        (train.latents[:, k27.TARGET_INDICES] - frozen["target_mean"]) / frozen["target_std"],
        dtype=torch.float32,
        device=device,
    )
    label_rows = torch.as_tensor(frozen["label_rows"], dtype=torch.long, device=device)
    model = build_model(config, frozen, condition, seed, device)
    initial_hash = slope.state_dict_sha256(model.state_dict())
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    weights = config["loss_weights"]
    history: list[dict[str, float]] = []
    slope.set_global_seed(seed)
    repeated_side = side[:, None, :].expand(-1, 4, -1)
    started = time.time()
    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        model.train()
        prediction = model(
            features.reshape(-1, features.shape[-1]), repeated_side.reshape(-1, 2)
        ).reshape(len(train.ids), 4, 5)
        common = generic_multiview_losses(
            prediction, float(config["loss_weights"]["range_zscore_limit"])
        )
        point = F.mse_loss(
            prediction[label_rows], targets[label_rows, None, :].expand(-1, 4, -1)
        )
        total = (
            float(weights["view"]) * common["view"]
            + float(weights["variance"]) * common["variance"]
            + float(weights["covariance"]) * common["covariance"]
            + float(weights["point"]) * point
            + float(weights["range"]) * common["range"]
        )
        if not torch.isfinite(total):
            raise FloatingPointError(f"non-finite K30 loss at epoch {epoch}")
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        optimizer.step()
        if epoch == 1 or epoch == int(config["training"]["epochs"]) or epoch % 50 == 0:
            record = {
                "epoch": epoch,
                "total": float(total.detach().cpu()),
                "point": float(point.detach().cpu()),
                **{name: float(value.detach().cpu()) for name, value in common.items()},
                "elapsed_seconds": time.time() - started,
            }
            history.append(record)
            print(json.dumps({"seed": seed, "condition": condition, **record}, sort_keys=True), flush=True)
    final_hash = slope.state_dict_sha256(model.state_dict())
    return model, history, initial_hash, final_hash


def load_preflight(output_root: Path, config_path: Path) -> dict[str, Any]:
    path = output_root / "preflight" / "preflight.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    checks = {
        "valid": bool(payload["valid"]),
        "config_current": payload["config_sha256"] == preflight.sha256_file(config_path),
        "sources_current": payload["source_files_sha256"] == source_hashes(),
        "validation_unread": not bool(payload["semantic_validation_evaluated"]),
        "test_unread": not bool(payload["test_evaluated"]),
    }
    if not all(checks.values()):
        raise ValueError(f"invalid K30 preflight: {checks}")
    return payload


def train_main(
    config_path: Path, output_root: Path, condition: str, seed: int
) -> None:
    config = load_protocol(config_path)
    if condition not in CONDITIONS or seed not in [int(value) for value in config["training"]["seeds"]]:
        raise ValueError("unregistered K30 train request")
    preflight_payload = load_preflight(output_root, config_path)
    (
        _, _, geometry_config, _, grouped, splits, label_ids, _, _,
    ) = load_context_chain(config)
    train = slope.select_ids(grouped, splits["train"])
    frozen = frozen_components(train, label_ids, geometry_config)
    torch.set_num_threads(int(config["runtime"]["num_threads"]))
    torch.set_float32_matmul_precision("high")
    device = torch.device(config["runtime"]["device"] if torch.cuda.is_available() else "cpu")
    model, history, initial_hash, final_hash = train_one(
        config, frozen, train, condition, seed, device
    )
    if initial_hash != preflight_payload["initial_state_sha256"][str(seed)]:
        raise RuntimeError("K30 initial state drifted after preflight")
    output_dir = output_root / "formal" / f"seed{seed}" / condition
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "training_results.json"
    checkpoint_path = output_dir / f"{condition}_seed{seed}.pt"
    lock_path = output_dir / "training_lock.json"
    if result_path.exists() or checkpoint_path.exists() or lock_path.exists():
        raise FileExistsError(f"refusing to overwrite K30 seed{seed} {condition}")
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
        "initial_state_exact": initial_hash == preflight_payload["initial_state_sha256"][str(seed)],
        "epoch_exact": history[-1]["epoch"] == int(config["training"]["epochs"]),
        "history_finite": all(
            np.isfinite(value)
            for record in history
            for key, value in record.items()
            if key != "epoch"
        ),
        "parameter_count_exact": sum(parameter.numel() for parameter in model.parameters())
        == int(preflight_payload["parameter_counts"][str(seed)]),
    }
    lock = {
        "status": "locked_before_joint_validation_readout",
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
        "semantic_validation_evaluated": False,
        "test_evaluated": False,
    }
    preflight.write_json(lock_path, lock)
    result = {
        "protocol_version": config["protocol_version"],
        "fact_type": "formal_train_without_validation_or_test_semantics",
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
        "semantic_validation_evaluated": False,
        "test_evaluated": False,
    }
    preflight.write_json(result_path, result)
    print(json.dumps({"result": str(result_path), "lock": str(lock_path)}, sort_keys=True))


def load_model_checkpoint(
    config: Mapping[str, Any],
    frozen: Mapping[str, Any],
    output_root: Path,
    condition: str,
    seed: int,
    device: torch.device,
) -> InstrumentedSlopeTubeHead:
    output_dir = output_root / "formal" / f"seed{seed}" / condition
    lock = json.loads((output_dir / "training_lock.json").read_text(encoding="utf-8"))
    checkpoint_path = Path(lock["checkpoint"])
    checks = {
        "lock_valid": all(lock["validity"].values()),
        "identity": lock["condition"] == condition and int(lock["seed"]) == seed,
        "checkpoint_current": preflight.sha256_file(checkpoint_path) == lock["checkpoint_sha256"],
        "validation_unread": not bool(lock["semantic_validation_evaluated"]),
        "test_unread": not bool(lock["test_evaluated"]),
        "sources_current": lock["source_files_sha256"] == source_hashes(),
    }
    if not all(checks.values()):
        raise ValueError(f"invalid K30 checkpoint lock: {checks}")
    model = build_model(config, frozen, condition, seed, device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    if slope.state_dict_sha256(model.state_dict()) != lock["final_state_sha256"]:
        raise ValueError("K30 final state hash mismatch")
    return model


def predict_model(
    model: InstrumentedSlopeTubeHead,
    grouped: slope.GroupedFeatures,
    frozen: Mapping[str, Any],
    device: torch.device,
) -> tuple[np.ndarray, dict[str, float]]:
    model.eval()
    features = torch.as_tensor(grouped.features, dtype=torch.float32, device=device)
    side = torch.as_tensor(grouped.latents[:, k27.SIDE_INDICES], dtype=torch.float32, device=device)
    repeated_side = side[:, None, :].expand(-1, 4, -1)
    with torch.no_grad():
        normalized, parts = model.components(
            features.reshape(-1, features.shape[-1]), repeated_side.reshape(-1, 2)
        )
    normalized = normalized.reshape(len(grouped.ids), 4, 5).mean(dim=1)
    prediction = normalized.cpu().numpy() * frozen["target_std"] + frozen["target_mean"]
    boundary = parts["boundary_active"].reshape(len(grouped.ids), 4).float().mean().item()
    scale = parts["projection_scale"].mean().item()
    return prediction, {
        "boundary_active_fraction": float(boundary),
        "mean_projection_scale": float(scale),
    }


def evaluate_main(config_path: Path, output_root: Path) -> None:
    config = load_protocol(config_path)
    preflight_payload = load_preflight(output_root, config_path)
    (
        _, _, geometry_config, _, grouped, splits, label_ids, _, _,
    ) = load_context_chain(config)
    train = slope.select_ids(grouped, splits["train"])
    validation = slope.select_ids(grouped, splits["validation"])
    frozen = frozen_components(train, label_ids, geometry_config)
    device = torch.device(config["runtime"]["device"] if torch.cuda.is_available() else "cpu")
    truth = validation.latents[:, k27.TARGET_INDICES]
    side = validation.latents[:, k27.SIDE_INDICES]
    center = k28.physical_state_from_roughness(
        side, float(frozen["mean_roughness"]), frozen["parameters"]
    )
    lower, upper = (float(value) for value in geometry_config["calibration"]["roughness_physical_bounds"])
    scalar_span = float(frozen["geometry"]["roughness_scalar"]["radius"]) * float(frozen["target_std"][0])
    rank_lower = max(lower, float(frozen["mean_roughness"]) - scalar_span)
    rank_upper = min(upper, float(frozen["mean_roughness"]) + scalar_span)

    seed_results: dict[str, Any] = {}
    all_isotropic_directional: list[bool] = []
    all_isotropic_ci: list[bool] = []
    all_curve_directional: list[bool] = []
    all_curve_ci: list[bool] = []
    for seed_value in config["training"]["seeds"]:
        seed = int(seed_value)
        unbounded_model = load_model_checkpoint(
            config, frozen, output_root, "unbounded", seed, device
        )
        isotropic_model = load_model_checkpoint(
            config, frozen, output_root, "isotropic", seed, device
        )
        unbounded, unbounded_audit = predict_model(unbounded_model, validation, frozen, device)
        isotropic, isotropic_audit = predict_model(isotropic_model, validation, frozen, device)
        formula = k28.physical_state_from_roughness(
            side, np.clip(unbounded[:, 0], lower, upper), frozen["parameters"]
        )
        curve, curve_audit = k28.nonlinear_curve_projection(
            unbounded,
            side,
            frozen["parameters"],
            frozen["target_std"],
            rank_lower,
            rank_upper,
            float(geometry_config["projection"]["nonlinear_tolerance"]),
            int(geometry_config["projection"]["nonlinear_max_iterations"]),
        )
        predictions = {
            "physical_center": center,
            "unbounded": unbounded,
            "isotropic": isotropic,
            "formula_from_unbounded_roughness": formula,
            "nonlinear_curve_from_unbounded": curve,
        }
        if list(predictions) != list(config["evaluation"]["candidates"]):
            raise ValueError("K30 evaluation candidate registry drifted")
        metrics = {name: k27.direct_metrics(truth, value) for name, value in predictions.items()}
        pairs = {
            "isotropic_minus_unbounded": ("isotropic", "unbounded"),
            "formula_minus_unbounded": ("formula_from_unbounded_roughness", "unbounded"),
            "curve_minus_unbounded": ("nonlinear_curve_from_unbounded", "unbounded"),
            "curve_minus_formula": ("nonlinear_curve_from_unbounded", "formula_from_unbounded_roughness"),
        }
        deltas: dict[str, dict[str, float]] = {}
        bootstrap: dict[str, Any] = {}
        for name, (candidate, baseline) in pairs.items():
            deltas[name] = {
                "mean_abs_correlation": metrics[candidate]["mean_abs_correlation"]
                - metrics[baseline]["mean_abs_correlation"],
                "mean_r2": metrics[candidate]["mean_r2"] - metrics[baseline]["mean_r2"],
            }
            bootstrap[name] = k29.paired_bootstrap(
                truth,
                predictions[candidate],
                predictions[baseline],
                int(config["evaluation"]["bootstrap_repetitions"]),
                int(config["evaluation"]["bootstrap_seed"]),
                float(config["evaluation"]["bootstrap_confidence"]),
            )
        iso_directional = all(value > 0.0 for value in deltas["isotropic_minus_unbounded"].values())
        iso_ci = k29.confirmed(
            deltas["isotropic_minus_unbounded"], bootstrap["isotropic_minus_unbounded"]
        )
        curve_directional = all(value > 0.0 for value in deltas["curve_minus_formula"].values())
        curve_ci = k29.confirmed(deltas["curve_minus_formula"], bootstrap["curve_minus_formula"])
        all_isotropic_directional.append(iso_directional)
        all_isotropic_ci.append(iso_ci)
        all_curve_directional.append(curve_directional)
        all_curve_ci.append(curve_ci)
        seed_results[str(seed)] = {
            "metrics": metrics,
            "deltas": deltas,
            "paired_bootstrap": bootstrap,
            "projection_audits": {
                "unbounded": unbounded_audit,
                "isotropic": isotropic_audit,
                "nonlinear_curve": curve_audit,
            },
            "decision_checks": {
                "isotropic_directional_both": iso_directional,
                "isotropic_joint_ci_confirmed": iso_ci,
                "curve_directional_both_vs_formula": curve_directional,
                "curve_joint_ci_confirmed_vs_formula": curve_ci,
            },
        }
        del unbounded_model, isotropic_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    validity = {
        "preflight_valid": bool(preflight_payload["valid"]),
        "all_six_training_locks_loaded": len(seed_results) == 3,
        "validation_ids_exact": set(int(value) for value in validation.ids)
        == set(int(value) for value in splits["validation"]),
        "test_ids_not_evaluated": not bool(
            set(int(value) for value in validation.ids) & set(int(value) for value in splits["test"])
        ),
        "all_metrics_finite": all(
            np.isfinite(metric)
            for seed_result in seed_results.values()
            for condition_metrics in seed_result["metrics"].values()
            for metric in (
                condition_metrics["mean_abs_correlation"],
                condition_metrics["mean_r2"],
                condition_metrics["mean_normalized_rmse"],
            )
        ),
    }
    valid = bool(all(validity.values()))
    if not valid:
        decision = "invalid"
    elif all(all_isotropic_ci) and all(all_curve_ci):
        decision = "neural_isotropic_and_curve_multiseed_ci_confirmed"
    elif all(all_isotropic_ci):
        decision = "neural_isotropic_multiseed_ci_confirmed"
    elif all(all_curve_ci):
        decision = "neural_curve_multiseed_ci_confirmed"
    elif all(all_isotropic_directional) and all(all_curve_directional):
        decision = "neural_isotropic_and_curve_multiseed_directional"
    elif all(all_isotropic_directional):
        decision = "neural_isotropic_multiseed_directional"
    elif all(all_curve_directional):
        decision = "neural_curve_multiseed_directional"
    else:
        decision = "neural_instrumented_slope_mixed"
    payload = {
        "protocol_version": config["protocol_version"],
        "fact_type": "locked_neural_tube_validation_multiseed",
        "scope": "registered_validation_ids_only_test_sealed",
        "config_sha256": preflight.sha256_file(config_path),
        "source_files_sha256": source_hashes(),
        "validation_count": int(len(validation.ids)),
        "validation_ids_sha256": preflight.sha256_int_array(validation.ids),
        "test_ids_sha256": preflight.sha256_int_array(splits["test"]),
        "seed_results": seed_results,
        "aggregate_checks": {
            "isotropic_directional_seed_count": int(sum(all_isotropic_directional)),
            "isotropic_ci_seed_count": int(sum(all_isotropic_ci)),
            "curve_directional_seed_count": int(sum(all_curve_directional)),
            "curve_ci_seed_count": int(sum(all_curve_ci)),
        },
        "validity": validity,
        "valid": valid,
        "semantic_validation_evaluated": True,
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
