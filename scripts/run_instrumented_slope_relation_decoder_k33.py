"""Train and evaluate the Instrumented Slope K33 relation decoder."""

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
import run_instrumented_slope_neural_tube_k30 as k30


CONDITIONS = ("roughness_point", "relation_point")


def source_hashes() -> dict[str, str]:
    paths = {
        "k33": Path(__file__),
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
    if config["protocol_version"] != "instrumented_slope_relation_decoder_k33":
        raise ValueError("unexpected K33 protocol")
    if tuple(config["model"]["conditions"]) != CONDITIONS:
        raise ValueError("K33 condition registry changed")
    if [int(value) for value in config["training"]["seeds"]] != [3407, 42, 0]:
        raise ValueError("K33 seed registry changed")
    if int(config["training"]["epochs"]) != 500:
        raise ValueError("K33 epoch registry changed")
    if tuple(config["evaluation"]["candidates"]) != (
        "k30_posthoc_curve",
        "roughness_point",
        "relation_point",
    ):
        raise ValueError("K33 evaluation registry changed")
    return config


class InstrumentedSlopeRelationDecoder(nn.Module):
    """Predict one hidden roughness coordinate and decode the physical state."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dims: Sequence[int],
        parameters: slope.RelationParameters,
        mean_roughness: float,
        target_mean: np.ndarray,
        target_std: np.ndarray,
        side_mean: np.ndarray,
        side_std: np.ndarray,
        roughness_bounds: Sequence[float],
    ) -> None:
        super().__init__()
        if len(hidden_dims) != 2:
            raise ValueError("K33 requires two hidden dimensions")
        lower, upper = (float(value) for value in roughness_bounds)
        if not lower < float(mean_roughness) < upper:
            raise ValueError("mean roughness must be inside the physical interval")
        first, second = (int(value) for value in hidden_dims)
        self.visual = nn.Sequential(
            nn.Linear(int(feature_dim), first),
            nn.LayerNorm(first),
            nn.GELU(),
            nn.Linear(first, second),
            nn.LayerNorm(second),
            nn.GELU(),
        )
        self.film = nn.Linear(2, 2 * second)
        self.output = nn.Linear(second, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        scaled_mean = (float(mean_roughness) - lower) / (upper - lower)
        base_logit = np.log(scaled_mean / (1.0 - scaled_mean))
        self.register_buffer(
            "relation_parameters",
            torch.as_tensor(parameters.as_array(), dtype=torch.float32),
            persistent=False,
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
            "roughness_lower", torch.as_tensor(lower, dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "roughness_upper", torch.as_tensor(upper, dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "roughness_base_logit",
            torch.as_tensor(base_logit, dtype=torch.float32),
            persistent=False,
        )

    def raw_coordinate(self, features: torch.Tensor, side_state: torch.Tensor) -> torch.Tensor:
        visual = self.visual(features)
        side = (side_state - self.side_mean) / self.side_std
        gamma, beta = self.film(side).chunk(2, dim=1)
        return self.output(visual * (1.0 + gamma) + beta).squeeze(1)

    def physical_decode(
        self, roughness: torch.Tensor, side_state: torch.Tensor
    ) -> torch.Tensor:
        a1, b1, a2, b2, deceleration, incline = self.relation_parameters
        theta = torch.deg2rad(side_state[:, 0])
        v0 = side_state[:, 1]
        mu1 = a1 * roughness + b1
        mu2 = a2 * roughness + b2
        v1 = v0 - deceleration * mu1
        denominator = incline * (torch.sin(theta) + mu2 * torch.cos(theta))
        length = v1.square() / denominator
        return torch.stack((roughness, mu1, mu2, v1, length), dim=1)

    def components(
        self, features: torch.Tensor, side_state: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        raw = self.raw_coordinate(features, side_state)
        fraction = torch.sigmoid(self.roughness_base_logit + raw)
        roughness = self.roughness_lower + (
            self.roughness_upper - self.roughness_lower
        ) * fraction
        physical = self.physical_decode(roughness, side_state)
        normalized = (physical - self.target_mean) / self.target_std
        return normalized, {
            "raw_coordinate": raw,
            "roughness": roughness,
            "physical": physical,
        }

    def forward(self, features: torch.Tensor, side_state: torch.Tensor) -> torch.Tensor:
        return self.components(features, side_state)[0]


def load_context(config: Mapping[str, Any]):
    context = k30.load_context_chain(config)
    k30_config_path = Path(config["source_contract"]["k30_config"])
    k30_result_path = Path(config["source_contract"]["k30_result"])
    k30_config = k30.load_protocol(k30_config_path)
    k30_result = json.loads(k30_result_path.read_text(encoding="utf-8"))
    extra_hashes = {
        "k30_config": preflight.sha256_file(k30_config_path),
        "k30_result": preflight.sha256_file(k30_result_path),
    }
    return (*context, k30_config, k30_result, extra_hashes)


def build_model(
    config: Mapping[str, Any],
    frozen: Mapping[str, Any],
    seed: int,
    roughness_bounds: Sequence[float],
    device: torch.device,
) -> InstrumentedSlopeRelationDecoder:
    slope.set_global_seed(seed)
    model = InstrumentedSlopeRelationDecoder(
        int(config["model"]["feature_dim"]),
        config["model"]["hidden_dims"],
        frozen["parameters"],
        float(frozen["mean_roughness"]),
        np.asarray(frozen["target_mean"]),
        np.asarray(frozen["target_std"]),
        np.asarray(frozen["side_mean"]),
        np.asarray(frozen["side_std"]),
        roughness_bounds,
    )
    return model.to(device)


def preflight_main(config_path: Path, output_root: Path) -> None:
    config = load_protocol(config_path)
    (
        _, validation_result, geometry_config, _, grouped, splits, label_ids,
        filter_audit, hashes, _, k30_result, extra_hashes,
    ) = load_context(config)
    train = slope.select_ids(grouped, splits["train"])
    frozen = k30.frozen_components(train, label_ids, geometry_config)
    bounds = geometry_config["calibration"]["roughness_physical_bounds"]
    initial_hashes: dict[str, str] = {}
    parameter_counts: dict[str, int] = {}
    for seed_value in config["training"]["seeds"]:
        seed = int(seed_value)
        first = build_model(config, frozen, seed, bounds, torch.device("cpu"))
        second = build_model(config, frozen, seed, bounds, torch.device("cpu"))
        first_hash = slope.state_dict_sha256(first.state_dict())
        second_hash = slope.state_dict_sha256(second.state_dict())
        if first_hash != second_hash:
            raise RuntimeError("K33 matched conditions do not share initial state")
        initial_hashes[str(seed)] = first_hash
        parameter_counts[str(seed)] = sum(parameter.numel() for parameter in first.parameters())
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
        "k30_result_valid": bool(k30_result["valid"]),
        "k30_test_unread": not bool(k30_result["test_evaluated"]),
        "feature_cache_sha_matches": hashes["feature_cache"]
        == str(k30.load_context_chain(config)[3]["runtime"]["feature_cache_sha256"]),
        "incomplete_product_filter_contract_matches": bool(filter_audit["contract_matches"]),
        "initial_states_registered": len(initial_hashes) == 3
        and len(set(initial_hashes.values())) == 3,
        "parameter_counts_matched": len(set(parameter_counts.values())) == 1,
        "roughness_interval_exact": [float(value) for value in bounds] == [0.0, 1.0],
    }
    payload = {
        "protocol_version": config["protocol_version"],
        "fact_type": "relation_decoder_preflight_after_known_development_validation",
        "scope": "training_ids_and_k40_labels_only_external_and_test_sealed",
        "config_sha256": preflight.sha256_file(config_path),
        "source_files_sha256": source_hashes(),
        "source_contract_hashes": {**hashes, **extra_hashes},
        "label_ids_sha256": preflight.sha256_int_array(label_ids),
        "train_ids_sha256": preflight.sha256_int_array(splits["train"]),
        "validation_ids_sha256": preflight.sha256_int_array(splits["validation"]),
        "test_ids_sha256": preflight.sha256_int_array(splits["test"]),
        "initial_state_sha256": initial_hashes,
        "parameter_counts": parameter_counts,
        "roughness_bounds": [float(value) for value in bounds],
        "development_validation_used_for_design": True,
        "external_evaluated": False,
        "test_evaluated": False,
        "validity": validity,
        "valid": bool(all(validity.values())),
        "machine_decision": "relation_decoder_preflight_pass"
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
        raise ValueError(f"invalid K33 preflight: {checks}")
    return payload


def train_one(
    config: Mapping[str, Any],
    frozen: Mapping[str, Any],
    train: slope.GroupedFeatures,
    condition: str,
    seed: int,
    bounds: Sequence[float],
    device: torch.device,
) -> tuple[InstrumentedSlopeRelationDecoder, list[dict[str, float]], str, str]:
    features = torch.as_tensor(train.features, dtype=torch.float32, device=device)
    side = torch.as_tensor(train.latents[:, k27.SIDE_INDICES], dtype=torch.float32, device=device)
    targets = torch.as_tensor(
        (train.latents[:, k27.TARGET_INDICES] - frozen["target_mean"])
        / frozen["target_std"],
        dtype=torch.float32,
        device=device,
    )
    label_rows = torch.as_tensor(frozen["label_rows"], dtype=torch.long, device=device)
    model = build_model(config, frozen, seed, bounds, device)
    initial_hash = slope.state_dict_sha256(model.state_dict())
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    weights = config["loss_weights"]
    repeated_side = side[:, None, :].expand(-1, 4, -1)
    history: list[dict[str, float]] = []
    started = time.time()
    slope.set_global_seed(seed)
    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        model.train()
        prediction, parts = model.components(
            features.reshape(-1, features.shape[-1]), repeated_side.reshape(-1, 2)
        )
        prediction = prediction.reshape(len(train.ids), 4, 5)
        roughness = parts["roughness"].reshape(len(train.ids), 4)
        common = k30.generic_multiview_losses(
            prediction, float(config["loss_weights"]["range_zscore_limit"])
        )
        if condition == "roughness_point":
            point = F.mse_loss(
                prediction[label_rows, :, 0],
                targets[label_rows, None, 0].expand(-1, 4),
            )
        elif condition == "relation_point":
            point = F.mse_loss(
                prediction[label_rows], targets[label_rows, None, :].expand(-1, 4, -1)
            )
        else:
            raise ValueError(f"unknown K33 condition {condition}")
        total = (
            float(weights["view"]) * common["view"]
            + float(weights["variance"]) * common["variance"]
            + float(weights["covariance"]) * common["covariance"]
            + float(weights["point"]) * point
            + float(weights["range"]) * common["range"]
        )
        if not torch.isfinite(total):
            raise FloatingPointError(f"non-finite K33 loss at epoch {epoch}")
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        optimizer.step()
        if epoch == 1 or epoch == int(config["training"]["epochs"]) or epoch % 50 == 0:
            record = {
                "epoch": epoch,
                "total": float(total.detach().cpu()),
                "point": float(point.detach().cpu()),
                "roughness_min": float(roughness.detach().min().cpu()),
                "roughness_max": float(roughness.detach().max().cpu()),
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
        raise ValueError("unregistered K33 train request")
    preflight_payload = load_preflight(output_root, config_path)
    (
        _, _, geometry_config, _, grouped, splits, label_ids, _, _, _, _, _,
    ) = load_context(config)
    train = slope.select_ids(grouped, splits["train"])
    frozen = k30.frozen_components(train, label_ids, geometry_config)
    bounds = geometry_config["calibration"]["roughness_physical_bounds"]
    torch.set_num_threads(int(config["runtime"]["num_threads"]))
    torch.set_float32_matmul_precision("high")
    device = torch.device(config["runtime"]["device"] if torch.cuda.is_available() else "cpu")
    model, history, initial_hash, final_hash = train_one(
        config, frozen, train, condition, seed, bounds, device
    )
    if initial_hash != preflight_payload["initial_state_sha256"][str(seed)]:
        raise RuntimeError("K33 initial state drifted after preflight")
    output_dir = output_root / "formal" / f"seed{seed}" / condition
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "training_results.json"
    checkpoint_path = output_dir / f"{condition}_seed{seed}.pt"
    lock_path = output_dir / "training_lock.json"
    if result_path.exists() or checkpoint_path.exists() or lock_path.exists():
        raise FileExistsError(f"refusing to overwrite K33 seed{seed} {condition}")
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
        "roughness_inside_physical_interval": all(
            record["roughness_min"] >= -1.0e-7 and record["roughness_max"] <= 1.0 + 1.0e-7
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
        "fact_type": "formal_train_after_known_development_validation_external_and_test_sealed",
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
    bounds: Sequence[float],
    output_root: Path,
    condition: str,
    seed: int,
    device: torch.device,
) -> InstrumentedSlopeRelationDecoder:
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
        raise ValueError(f"invalid K33 checkpoint lock: {checks}")
    model = build_model(config, frozen, seed, bounds, device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    if slope.state_dict_sha256(model.state_dict()) != lock["final_state_sha256"]:
        raise ValueError("K33 final state hash mismatch")
    return model


def predict_model(
    model: InstrumentedSlopeRelationDecoder,
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
    roughness = parts["roughness"].reshape(len(grouped.ids), 4)
    return prediction, {
        "roughness_min": float(roughness.min().cpu()),
        "roughness_max": float(roughness.max().cpu()),
        "roughness_view_std_mean": float(roughness.std(dim=1, unbiased=False).mean().cpu()),
    }


def k30_posthoc_prediction(
    config: Mapping[str, Any],
    k30_config: Mapping[str, Any],
    frozen: Mapping[str, Any],
    geometry_config: Mapping[str, Any],
    validation: slope.GroupedFeatures,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any]]:
    k30_root = Path(config["source_contract"]["k30_output_root"])
    model = k30.load_model_checkpoint(k30_config, frozen, k30_root, "unbounded", seed, device)
    unbounded, model_audit = k30.predict_model(model, validation, frozen, device)
    side = validation.latents[:, k27.SIDE_INDICES]
    lower, upper = (
        float(value) for value in geometry_config["calibration"]["roughness_physical_bounds"]
    )
    scalar_span = (
        float(frozen["geometry"]["roughness_scalar"]["radius"])
        * float(frozen["target_std"][0])
    )
    rank_lower = max(lower, float(frozen["mean_roughness"]) - scalar_span)
    rank_upper = min(upper, float(frozen["mean_roughness"]) + scalar_span)
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
    return curve, {"unbounded_model": model_audit, "curve": curve_audit}


def evaluate_main(config_path: Path, output_root: Path) -> None:
    config = load_protocol(config_path)
    preflight_payload = load_preflight(output_root, config_path)
    (
        _, _, geometry_config, _, grouped, splits, label_ids, _, _,
        k30_config, k30_result, _,
    ) = load_context(config)
    train = slope.select_ids(grouped, splits["train"])
    validation = slope.select_ids(grouped, splits["validation"])
    frozen = k30.frozen_components(train, label_ids, geometry_config)
    bounds = geometry_config["calibration"]["roughness_physical_bounds"]
    truth = validation.latents[:, k27.TARGET_INDICES]
    device = torch.device(config["runtime"]["device"] if torch.cuda.is_available() else "cpu")
    seed_results: dict[str, Any] = {}
    predictions_by_name: dict[str, list[np.ndarray]] = {
        name: [] for name in config["evaluation"]["candidates"]
    }
    propagation_directional: list[bool] = []
    training_directional: list[bool] = []
    baseline_reproduced: list[bool] = []
    pairs = {
        "relation_point_minus_k30_posthoc_curve": ("relation_point", "k30_posthoc_curve"),
        "relation_point_minus_roughness_point": ("relation_point", "roughness_point"),
        "roughness_point_minus_k30_posthoc_curve": ("roughness_point", "k30_posthoc_curve"),
    }
    for seed_value in config["training"]["seeds"]:
        seed = int(seed_value)
        baseline, baseline_audit = k30_posthoc_prediction(
            config, k30_config, frozen, geometry_config, validation, seed, device
        )
        predictions: dict[str, np.ndarray] = {"k30_posthoc_curve": baseline}
        model_audits: dict[str, Any] = {"k30_posthoc_curve": baseline_audit}
        for condition in CONDITIONS:
            model = load_model_checkpoint(
                config, frozen, bounds, output_root, condition, seed, device
            )
            predictions[condition], model_audits[condition] = predict_model(
                model, validation, frozen, device
            )
            del model
        metrics = {name: k27.direct_metrics(truth, prediction) for name, prediction in predictions.items()}
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
        propagation_directional.append(
            all(value > 0.0 for value in deltas["relation_point_minus_roughness_point"].values())
        )
        training_directional.append(
            all(value > 0.0 for value in deltas["relation_point_minus_k30_posthoc_curve"].values())
        )
        for name, prediction in predictions.items():
            predictions_by_name[name].append(prediction)
        seed_results[str(seed)] = {
            "metrics": metrics,
            "deltas": deltas,
            "paired_bootstrap": bootstrap,
            "model_audits": model_audits,
            "decision_checks": {
                "k30_baseline_reproduced": reproduction,
                "relation_label_propagation_directional": propagation_directional[-1],
                "training_time_advantage_directional": training_directional[-1],
            },
        }
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
    aggregate_deltas = {
        name: {
            "mean_abs_correlation": aggregate_metrics[candidate]["mean_abs_correlation"]
            - aggregate_metrics[baseline_name]["mean_abs_correlation"],
            "mean_r2": aggregate_metrics[candidate]["mean_r2"]
            - aggregate_metrics[baseline_name]["mean_r2"],
        }
        for name, (candidate, baseline_name) in pairs.items()
    }
    propagation_aggregate_ci = k29.confirmed(
        aggregate_deltas["relation_point_minus_roughness_point"],
        aggregate_bootstrap["relation_point_minus_roughness_point"],
    )
    training_aggregate_ci = k29.confirmed(
        aggregate_deltas["relation_point_minus_k30_posthoc_curve"],
        aggregate_bootstrap["relation_point_minus_k30_posthoc_curve"],
    )
    propagation_pass = all(propagation_directional) and propagation_aggregate_ci
    training_pass = all(training_directional) and training_aggregate_ci
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
    elif propagation_pass and training_pass:
        decision = "relation_propagation_and_training_advantage_confirmed"
    elif propagation_pass:
        decision = "relation_propagation_only_posthoc_remains_stronger"
    elif training_pass:
        decision = "training_advantage_without_relation_label_ablation_support"
    else:
        decision = "relation_decoder_not_confirmed"
    payload = {
        "protocol_version": config["protocol_version"],
        "fact_type": "locked_relation_decoder_development_validation_multiseed",
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
            "relation_label_propagation_directional_seed_count": int(
                sum(propagation_directional)
            ),
            "relation_label_propagation_aggregate_ci": propagation_aggregate_ci,
            "relation_label_propagation_pass": propagation_pass,
            "training_time_advantage_directional_seed_count": int(sum(training_directional)),
            "training_time_advantage_aggregate_ci": training_aggregate_ci,
            "training_time_advantage_pass": training_pass,
            "external_evaluation_unlocked": training_pass,
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
