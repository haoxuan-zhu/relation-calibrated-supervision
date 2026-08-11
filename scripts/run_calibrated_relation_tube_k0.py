"""Validation-only K0 for a calibrated relation tube in CRL output space.

The raw-ridge relation is used as a frozen center.  A train-only leave-one-out
residual geometry bounds the trainable image residual through an ellipsoidal
radial projection.  Candidate semantic validation is read only after all four
condition locks exist; test rows are never evaluated by this runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml

import audit_physics_functional_anchor as physics_audit
import raw_ridge_propagation as raw
import run_clamped_anchor_diagnostic as v2
import run_conditioned_film_diagnostic as v3
import run_conflict_projected_physics_from_scratch as v16
import run_conflict_projected_physics_k80_closed as v17
import run_diagnostic as base
import run_dynamic_residual_propagation_k0 as dynamic
import run_physics_functional_anchor_training as v11
import run_supervised_continuation_diagnostic as v6


PROTOCOL = "calibrated_relation_tube_k0_v1"
CONFIRMATION_PROTOCOL = "calibrated_relation_tube_multiseed_k3_v1"
REGISTERED_INITIAL_STATE_HASHES = {
    0: "6037afc85fa38492ba8b6854e7467d610016d6bfe4a06e96276c11a1c302a7b2",
    42: "9e21bbf9d3cd2c6abefe2029a6e845c2a12175b0a49ec445234fc1258dbddc56",
    3407: "4b599031478a138433ed0208cc4871be1895b15db07576b2e13b5b9f81572731",
}
CONDITIONS = (
    "center_only_correct",
    "unbounded_correct",
    "tube_correct",
    "tube_permuted_matched",
)
SOURCE_FILES = (
    "run_calibrated_relation_tube_k0.py",
    "run_dynamic_residual_propagation_k0.py",
    "audit_physics_functional_anchor.py",
    "raw_ridge_propagation.py",
    "run_physics_functional_anchor_training.py",
    "run_supervised_continuation_diagnostic.py",
    "run_conflict_projected_physics_from_scratch.py",
    "run_conflict_projected_physics_k80_closed.py",
    "run_conditioned_film_diagnostic.py",
    "run_clamped_anchor_diagnostic.py",
    "run_diagnostic.py",
)


@dataclass(frozen=True)
class ConditionSpec:
    name: str
    center: str
    residual_mode: str
    train_encoder: bool


def condition_spec(name: str) -> ConditionSpec:
    mapping = {
        "center_only_correct": ConditionSpec(name, "correct", "center_only", False),
        "unbounded_correct": ConditionSpec(name, "correct", "unbounded", True),
        "tube_correct": ConditionSpec(name, "correct", "tube", True),
        "tube_permuted_matched": ConditionSpec(name, "permuted", "tube", True),
    }
    if name not in mapping:
        raise ValueError(f"unknown tube condition: {name}")
    return mapping[name]


def source_hashes() -> dict[str, str]:
    script_dir = Path(__file__).resolve().parent
    return {name: base.sha256_file(script_dir / name) for name in SOURCE_FILES}


def validate_config(config: dict[str, Any]) -> None:
    if config["protocol_version"] not in (PROTOCOL, CONFIRMATION_PROTOCOL):
        raise ValueError("unexpected relation-tube protocol")
    if list(config["training"]["conditions"]) != list(CONDITIONS):
        raise ValueError("relation-tube condition registry changed")
    seed = int(config["training"]["seed"])
    if seed not in REGISTERED_INITIAL_STATE_HASHES:
        raise ValueError("relation-tube seed is not registered")
    if int(config["initialization"]["seed"]) != seed:
        raise ValueError("initialization/training seed mismatch")
    if (
        config["initialization"]["expected_state_dict_sha256"]
        != REGISTERED_INITIAL_STATE_HASHES[seed]
    ):
        raise ValueError("registered seed initial-state hash changed")
    if config["protocol_version"] == PROTOCOL and seed != 3407:
        raise ValueError("the completed K0 protocol remains locked to seed3407")
    if int(config["training"]["epochs"]) != 100:
        raise ValueError("formal K0 is locked to 100 epochs")
    if int(config["training"]["shuffle_seed"]) != 20260730:
        raise ValueError("historical anchor-map shuffle seed changed")
    if int(config["calibration"]["budget"]) != 80:
        raise ValueError("K0 is locked to K80")
    if not np.isclose(float(config["calibration"]["raw_ridge_alpha"]), 0.1):
        raise ValueError("raw-ridge alpha must remain 0.1")
    if not np.isclose(float(config["calibration"]["coverage"]), 0.95):
        raise ValueError("tube coverage must remain 0.95")
    expected_index = math.ceil(
        (int(config["calibration"]["budget"]) + 1)
        * float(config["calibration"]["coverage"])
    )
    if expected_index != int(
        config["calibration"]["expected_score_order_index_one_based"]
    ):
        raise ValueError("registered tube score order index changed")
    if int(config["implementation"]["expected_parameter_count"]) != 16902384:
        raise ValueError("unexpected model parameter count contract")


def calibration_features(
    images: np.ndarray,
    raw_latents: np.ndarray,
    rows: np.ndarray,
    end: int,
) -> np.ndarray:
    means = physics_audit.image_channel_means(images, 0, 0, end)[rows]
    tau = physics_audit.malus_tau(raw_latents)[0, :end][rows]
    return raw.raw_features(means, tau)


def fit_teacher(
    features: np.ndarray,
    targets: np.ndarray,
    alpha: float,
    clip: bool,
) -> tuple[dict[str, Any], np.ndarray, dict[str, Any]]:
    count = len(features)
    predictions = np.empty_like(targets, dtype=np.float64)
    for holdout in range(count):
        keep = np.arange(count) != holdout
        fold = raw._fit_standardized_ridge(features[keep], targets[keep], alpha)
        predictions[holdout] = raw.predict_raw_ridge(
            fold, features[holdout : holdout + 1], clip=clip
        )[0]
    teacher = raw._fit_standardized_ridge(features, targets, alpha)
    fitted = raw.predict_raw_ridge(teacher, features, clip=clip)
    coefficient_sha = hashlib.sha256(
        np.ascontiguousarray(teacher["coefficients"], dtype=np.float64).tobytes()
    ).hexdigest()
    audit = {
        "alpha": float(alpha),
        "clip": bool(clip),
        "loo_count": count,
        "loo_mse": float(np.mean((predictions - targets) ** 2)),
        "full_fit_mse": float(np.mean((fitted - targets) ** 2)),
        "coefficients_sha256": coefficient_sha,
        "finite": bool(
            np.all(np.isfinite(predictions))
            and np.all(np.isfinite(fitted))
            and np.all(np.isfinite(np.asarray(teacher["coefficients"])))
        ),
    }
    return teacher, predictions, audit


def build_tube_geometry(
    normalized_loo_residuals: np.ndarray,
    coverage: float,
    diagonal_ridge: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    residuals = np.asarray(normalized_loo_residuals, dtype=np.float64)
    if residuals.ndim != 2 or residuals.shape[1] != 3:
        raise ValueError("tube residuals must be K-by-3")
    if not 0 < coverage < 1 or diagonal_ridge <= 0:
        raise ValueError("invalid tube geometry constants")
    covariance = residuals.T @ residuals / len(residuals)
    covariance = covariance + diagonal_ridge * np.eye(3, dtype=np.float64)
    eigenvalues = np.linalg.eigvalsh(covariance)
    if not np.all(np.isfinite(eigenvalues)) or eigenvalues.min() <= 0:
        raise ValueError("tube covariance is not positive definite")
    precision = np.linalg.inv(covariance)
    scores = np.einsum("ni,ij,nj->n", residuals, precision, residuals)
    order_index = min(len(scores), math.ceil((len(scores) + 1) * coverage))
    radius_squared = float(np.sort(scores)[order_index - 1])
    empirical_coverage = float(np.mean(scores <= radius_squared + 1e-12))
    geometry = {
        "covariance": covariance,
        "precision": precision,
        "radius_squared": radius_squared,
    }
    audit = {
        "residual_count": len(residuals),
        "coverage_target": float(coverage),
        "score_order_index_one_based": int(order_index),
        "empirical_coverage": empirical_coverage,
        "diagonal_ridge": float(diagonal_ridge),
        "covariance_eigenvalues": eigenvalues,
        "covariance_condition_number": float(eigenvalues.max() / eigenvalues.min()),
        "score_minimum": float(scores.min()),
        "score_median": float(np.median(scores)),
        "score_maximum": float(scores.max()),
        "radius_squared": radius_squared,
        "finite": bool(
            np.all(np.isfinite(covariance))
            and np.all(np.isfinite(precision))
            and np.all(np.isfinite(scores))
        ),
    }
    return geometry, audit


def teacher_json(teacher: dict[str, Any]) -> dict[str, Any]:
    return {
        "alpha": float(teacher["alpha"]),
        "feature_mean": np.asarray(teacher["feature_mean"], dtype=np.float64),
        "feature_scale": np.asarray(teacher["feature_scale"], dtype=np.float64),
        "coefficients": np.asarray(teacher["coefficients"], dtype=np.float64),
    }


def build_preflight_payload(
    config: dict[str, Any],
    images: np.ndarray,
    raw_latents: np.ndarray,
) -> dict[str, Any]:
    subset, subset_audit = dynamic.build_subset(config)
    permutation, permutation_audit = dynamic.build_derangement(config)
    train_end = int(config["split"]["train_end"])
    features = calibration_features(images, raw_latents, subset, train_end)
    correct_targets = raw_latents[0, subset, :3].astype(np.float64) / 255.0
    permuted_targets = correct_targets[permutation]
    alpha = float(config["calibration"]["raw_ridge_alpha"])
    clip = bool(config["calibration"]["clip_predictions_to_unit_interval"])
    correct_teacher, correct_loo, correct_audit = fit_teacher(
        features, correct_targets, alpha, clip
    )
    permuted_teacher, permuted_loo, permuted_audit = fit_teacher(
        features, permuted_targets, alpha, clip
    )
    _, latent_mean, latent_std = v17.normalize_with_k80_rgb_statistics(
        raw_latents, train_end, subset
    )
    normalized_residuals = (
        (correct_targets - correct_loo) * 255.0 / latent_std[:3]
    )
    geometry, geometry_audit = build_tube_geometry(
        normalized_residuals,
        float(config["calibration"]["coverage"]),
        float(config["calibration"]["covariance_diagonal_ridge"]),
    )
    geometry_hash = dynamic.canonical_sha256(geometry)
    checks = {
        "subset_exact": subset_audit["subset_sha256"]
        == config["calibration"]["subset_sha256"],
        "subset_unique": subset_audit["unique_rows"] == 80,
        "permutation_exact": all(permutation_audit["checks"].values()),
        "correct_teacher_finite": correct_audit["finite"],
        "permuted_teacher_finite": permuted_audit["finite"],
        "loo_complete": correct_audit["loo_count"] == 80
        and permuted_audit["loo_count"] == 80,
        "geometry_finite": geometry_audit["finite"],
        "score_index_exact": geometry_audit["score_order_index_one_based"]
        == int(config["calibration"]["expected_score_order_index_one_based"]),
        "empirical_coverage": geometry_audit["empirical_coverage"]
        >= float(config["calibration"]["coverage"]),
        "geometry_spd": bool(
            min(geometry_audit["covariance_eigenvalues"]) > 0
        ),
        "test_not_read": True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"relation-tube preflight failed: {checks}")
    return {
        "subset": subset_audit,
        "permutation": permutation_audit,
        "normalization": {"latent_mean": latent_mean, "latent_std": latent_std},
        "teachers": {
            "correct": {"model": teacher_json(correct_teacher), "audit": correct_audit},
            "permuted": {
                "model": teacher_json(permuted_teacher),
                "audit": permuted_audit,
                "loo_residual_mse_normalized": float(
                    np.mean(
                        (
                            (permuted_targets - permuted_loo)
                            * 255.0
                            / latent_std[:3]
                        )
                        ** 2
                    )
                ),
            },
        },
        "geometry": geometry,
        "geometry_audit": geometry_audit,
        "geometry_sha256": geometry_hash,
        "checks": checks,
    }


class CalibratedRelationTubeEncoder(v3.AnchorConditionedEncoder):
    def __init__(
        self,
        learned_dim: int,
        anchor_dim: int,
        hidden_channels: int,
        conv_layers: int,
        teacher: dict[str, Any],
        geometry: dict[str, Any],
        latent_mean: np.ndarray,
        latent_std: np.ndarray,
        residual_mode: str,
        clip_center: bool,
    ):
        super().__init__(learned_dim, anchor_dim, hidden_channels, conv_layers)
        if learned_dim != 3 or anchor_dim != 2:
            raise ValueError("K0 relation tube requires three RGB and two angle coordinates")
        if residual_mode not in ("center_only", "unbounded", "tube"):
            raise ValueError(residual_mode)
        self.residual_mode = residual_mode
        self.clip_center = bool(clip_center)
        self.register_buffer(
            "tube_feature_mean",
            torch.as_tensor(teacher["feature_mean"], dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "tube_feature_scale",
            torch.as_tensor(teacher["feature_scale"], dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "tube_coefficients",
            torch.as_tensor(teacher["coefficients"], dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "tube_precision",
            torch.as_tensor(geometry["precision"], dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "tube_radius_squared",
            torch.as_tensor(float(geometry["radius_squared"]), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "tube_latent_mean",
            torch.as_tensor(latent_mean, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "tube_latent_std",
            torch.as_tensor(latent_std, dtype=torch.float32),
            persistent=False,
        )

    def relation_center(self, x: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        raw_angles = (
            anchors * self.tube_latent_std[3:5] + self.tube_latent_mean[3:5]
        )
        teacher = {
            "feature_mean": self.tube_feature_mean,
            "feature_scale": self.tube_feature_scale,
            "coefficients": self.tube_coefficients,
        }
        unit_rgb = dynamic.torch_ridge_prediction(x, raw_angles, teacher)
        if self.clip_center:
            unit_rgb = unit_rgb.clamp(0.0, 1.0)
        return (
            unit_rgb * 255.0 - self.tube_latent_mean[:3]
        ) / self.tube_latent_std[:3]

    def raw_residual(self, x: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        visual = self.visual(x)
        gamma, beta = self.film(anchors).chunk(2, dim=1)
        return self.output(visual * (1.0 + gamma) + beta)

    def components(
        self, x: torch.Tensor, anchors: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        center = self.relation_center(x, anchors)
        raw_residual = self.raw_residual(x, anchors)
        score = torch.einsum(
            "ni,ij,nj->n", raw_residual, self.tube_precision, raw_residual
        )
        if self.residual_mode == "center_only":
            projected = torch.zeros_like(raw_residual)
            scale = torch.zeros_like(score)
        elif self.residual_mode == "unbounded":
            projected = raw_residual
            scale = torch.ones_like(score)
        else:
            scale = torch.sqrt(
                self.tube_radius_squared / score.clamp_min(1e-12)
            ).clamp(max=1.0)
            projected = raw_residual * scale[:, None]
        output = center + projected
        return output, {
            "center": center,
            "raw_residual": raw_residual,
            "projected_residual": projected,
            "raw_mahalanobis_score": score,
            "projection_scale": scale,
            "boundary_active": score > self.tube_radius_squared,
        }

    def forward(self, x: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        return self.components(x, anchors)[0]


class CalibratedRelationTubeModel(v3.FilmConditionedModel):
    def __init__(
        self,
        config: dict[str, Any],
        teacher: dict[str, Any],
        geometry: dict[str, Any],
        latent_mean: np.ndarray,
        latent_std: np.ndarray,
        residual_mode: str,
    ):
        nn.Module.__init__(self)
        model_cfg = config["model"]
        self.latent_dim = int(model_cfg["latent_dim"])
        self.learned_indices = list(model_cfg["learned_indices"])
        self.anchor_indices = list(model_cfg["anchor_indices"])
        self.embedding = CalibratedRelationTubeEncoder(
            len(self.learned_indices),
            len(self.anchor_indices),
            int(model_cfg["hidden_channels"]),
            int(model_cfg["conv_layers"]),
            teacher,
            geometry,
            latent_mean,
            latent_std,
            residual_mode,
            bool(config["calibration"]["clip_predictions_to_unit_interval"]),
        )
        self.parametric_part = base.ParametricPart(self.latent_dim)


def model_for_condition(
    condition: str,
    config: dict[str, Any],
    preflight: dict[str, Any],
    device: torch.device,
) -> CalibratedRelationTubeModel:
    spec = condition_spec(condition)
    model = CalibratedRelationTubeModel(
        config,
        preflight["teachers"][spec.center]["model"],
        preflight["geometry"],
        np.asarray(preflight["normalization"]["latent_mean"], dtype=np.float32),
        np.asarray(preflight["normalization"]["latent_std"], dtype=np.float32),
        spec.residual_mode,
    ).to(device)
    if not spec.train_encoder:
        for parameter in model.embedding.parameters():
            parameter.requires_grad_(False)
    return model


def load_preflight(
    root: Path, config_path: Path, config: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    payload_path = root / "preflight" / "tube_preflight.json"
    lock_path = root / "preflight" / "preflight_lock.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    checks = {
        "status": lock["status"] == "locked_train_only_tube_geometry",
        "protocol": lock["protocol_version"] == config["protocol_version"],
        "config": lock["config_sha256"] == base.sha256_file(config_path),
        "source": lock["source_files_sha256"] == source_hashes(),
        "payload": lock["payload_sha256"] == base.sha256_file(payload_path),
        "all_preflight_validity": all(payload["checks"].values()),
        "test_unread": lock["test_evaluated"] is False,
    }
    if not all(checks.values()):
        raise ValueError(f"invalid relation-tube preflight: {checks}")
    return payload, lock, lock_path


def preflight_main(args: argparse.Namespace, config: dict[str, Any]) -> None:
    if args.condition is not None or args.epochs is not None:
        raise ValueError("preflight accepts neither condition nor epochs")
    root = Path(args.output_root or config["runtime"]["output_root"]).resolve()
    output_dir = root / "preflight"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "tube_preflight.json"
    lock_path = output_dir / "preflight_lock.json"
    if output_path.exists() or lock_path.exists():
        raise FileExistsError("refusing to overwrite relation-tube preflight")
    images = np.load(Path(config["dataset"]["cache_path"]), mmap_mode="r")
    raw_latents = base.load_latents(config).astype(np.float64)
    payload = build_preflight_payload(config, images, raw_latents)
    payload.update(
        protocol_version=config["protocol_version"],
        mode="train_only_tube_geometry_preflight",
        config_path=str(args.config.resolve()),
        config_sha256=base.sha256_file(args.config.resolve()),
        source_files_sha256=source_hashes(),
        semantic_validation_evaluated=False,
        test_evaluated=False,
    )
    v11.atomic_json(output_path, payload)
    lock = {
        "status": "locked_train_only_tube_geometry",
        "protocol_version": config["protocol_version"],
        "config_sha256": base.sha256_file(args.config.resolve()),
        "source_files_sha256": source_hashes(),
        "payload_sha256": base.sha256_file(output_path),
        "geometry_sha256": payload["geometry_sha256"],
        "semantic_validation_evaluated": False,
        "test_evaluated": False,
    }
    v11.atomic_json(lock_path, lock)
    print(
        json.dumps(
            {
                "result_path": str(output_path),
                "sha256": base.sha256_file(output_path),
                "geometry_sha256": payload["geometry_sha256"],
                "radius_squared": payload["geometry"]["radius_squared"],
                "coverage": payload["geometry_audit"]["empirical_coverage"],
            },
            sort_keys=True,
        )
    )


def train_condition(
    condition: str,
    config: dict[str, Any],
    preflight: dict[str, Any],
    images: np.ndarray,
    latents: np.ndarray,
    anchor_maps: dict[str, np.ndarray],
    initial_state: dict[str, torch.Tensor],
    initial_hash: str,
    device: torch.device,
    output_dir: Path,
    epochs: int,
) -> dict[str, Any]:
    spec = condition_spec(condition)
    seed = int(config["training"]["seed"])
    base.set_seed(seed)
    model = model_for_condition(condition, config, preflight, device)
    model.load_state_dict(initial_state)
    observed_hash = v6.sha256_state_dict(model.state_dict())
    if observed_hash != initial_hash:
        raise ValueError("condition did not load the registered trainable state")
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(config["training"]["learning_rate"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=float(config["training"]["scheduler_factor"]),
        patience=int(config["training"]["scheduler_patience"]),
    )
    rows, targets = base.make_pairs(
        list(config["training"]["targets"]), 0, int(config["split"]["train_end"])
    )
    batch_size = int(config["training"]["batch_size"])
    interval = int(config["training"]["validation_interval"])
    history: list[dict[str, Any]] = []
    started = time.time()
    token = v2.Condition(condition, "oracle")
    for epoch in range(epochs):
        order = np.random.default_rng(seed + 1009 * epoch).permutation(len(rows))
        model.train()
        sums: dict[str, float] = {}
        count = 0
        for begin in range(0, len(rows), batch_size):
            indices = order[begin : begin + batch_size]
            batch_rows = rows[indices]
            batch_targets = targets[indices]
            loss, values = v2.objective_for_batch(
                model,
                images,
                latents,
                anchor_maps,
                token,
                batch_rows,
                batch_targets,
                config,
                device,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            size = len(batch_rows)
            count += size
            for name, value in values.items():
                sums[name] = sums.get(name, 0.0) + size * float(value)
        record: dict[str, Any] = {
            "epoch": epoch + 1,
            "train": base.average_metrics(sums, count),
        }
        if epoch % interval == 0 or epoch + 1 == epochs:
            validation = v2.validate(
                model, images, latents, anchor_maps, token, config, device
            )
            scheduler.step(validation["total"])
            record.update(
                validation=validation,
                learning_rate=float(optimizer.param_groups[0]["lr"]),
            )
            print(
                json.dumps(
                    {
                        "condition": condition,
                        "epoch": epoch + 1,
                        "validation_total": validation["total"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        history.append(record)
    checkpoint = output_dir / f"{condition}_seed{seed}.pt"
    torch.save(
        {
            "protocol_version": config["protocol_version"],
            "condition": condition,
            "epoch": epochs,
            "initial_state_sha256": initial_hash,
            "state_dict": model.state_dict(),
        },
        checkpoint,
    )
    result = {
        "condition": condition,
        "center": spec.center,
        "residual_mode": spec.residual_mode,
        "train_encoder": spec.train_encoder,
        "initial_state_sha256": initial_hash,
        "final_epoch": epochs,
        "duration_seconds": time.time() - started,
        "history": history,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": base.sha256_file(checkpoint),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "trainable_parameter_count": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        "semantic_truth_read_during_training": False,
        "test_evaluated": False,
    }
    if not dynamic.numeric_history_is_finite(history):
        raise FloatingPointError("tube training history contains non-finite values")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def train_main(args: argparse.Namespace, config: dict[str, Any]) -> None:
    if args.condition not in CONDITIONS:
        raise ValueError("train/smoke requires a registered condition")
    formal_epochs = int(config["training"]["epochs"])
    if args.mode == "train":
        epochs = int(args.epochs or formal_epochs)
        if epochs != formal_epochs:
            raise ValueError("formal tube train must use 100 epochs")
        smoke = False
    else:
        epochs = int(args.epochs or 2)
        if epochs != 2:
            raise ValueError("tube smoke is locked to two epochs")
        smoke = True
    root = Path(args.output_root or config["runtime"]["output_root"]).resolve()
    preflight, preflight_lock, preflight_lock_path = load_preflight(
        root, args.config.resolve(), config
    )
    output_dir = root / ("smoke" if smoke else "formal") / args.condition
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / (
        "smoke_results.json" if smoke else "training_results.json"
    )
    if result_path.exists():
        raise FileExistsError(f"refusing to overwrite tube result: {result_path}")
    torch.set_num_threads(int(config["runtime"]["num_threads"]))
    torch.set_float32_matmul_precision("high")
    device = torch.device(
        config["runtime"]["device"] if torch.cuda.is_available() else "cpu"
    )
    images = np.load(Path(config["dataset"]["cache_path"]), mmap_mode="r")
    raw_latents = base.load_latents(config)
    subset, _ = dynamic.build_subset(config)
    latents, latent_mean, latent_std = v17.normalize_with_k80_rgb_statistics(
        raw_latents, int(config["split"]["train_end"]), subset
    )
    if not np.allclose(latent_mean, preflight["normalization"]["latent_mean"]):
        raise ValueError("preflight/train latent mean mismatch")
    if not np.allclose(latent_std, preflight["normalization"]["latent_std"]):
        raise ValueError("preflight/train latent std mismatch")
    anchor_maps, anchor_audit = v2.build_anchor_maps(config)
    initial_state, initial_hash = v16.make_initial_state(config, device)
    expected_initial = str(config["initialization"]["expected_state_dict_sha256"])
    if initial_hash != expected_initial:
        raise ValueError(f"tube initial state mismatch: {initial_hash} != {expected_initial}")
    training = train_condition(
        args.condition,
        config,
        preflight,
        images,
        latents,
        anchor_maps,
        initial_state,
        initial_hash,
        device,
        output_dir,
        epochs,
    )
    checks = {
        "preflight_valid": all(preflight["checks"].values()),
        "preflight_source_current": preflight_lock["source_files_sha256"]
        == source_hashes(),
        "initial_state_exact": initial_hash == expected_initial,
        "parameter_count_exact": training["parameter_count"]
        == int(config["implementation"]["expected_parameter_count"]),
        "epoch_exact": training["final_epoch"] == epochs,
        "history_finite": dynamic.numeric_history_is_finite(training["history"]),
        "semantic_unread": not training["semantic_truth_read_during_training"],
        "test_unread": not training["test_evaluated"],
    }
    if not all(checks.values()):
        raise RuntimeError(f"tube training validity failed: {checks}")
    lock = {
        "status": "locked_before_joint_semantic_validation_readout",
        "mode": "smoke" if smoke else "formal",
        "protocol_version": config["protocol_version"],
        "condition": args.condition,
        "config_sha256": base.sha256_file(args.config.resolve()),
        "source_files_sha256": source_hashes(),
        "preflight_lock_sha256": base.sha256_file(preflight_lock_path),
        "geometry_sha256": preflight["geometry_sha256"],
        "initial_state_sha256": initial_hash,
        "checkpoint": {
            "path": training["checkpoint"],
            "sha256": training["checkpoint_sha256"],
            "epoch": training["final_epoch"],
        },
        "validity": checks,
        "semantic_validation_evaluated": False,
        "test_evaluated": False,
    }
    lock_path = output_dir / "training_lock.json"
    v11.atomic_json(lock_path, lock)
    result = {
        "protocol_version": config["protocol_version"],
        "mode": "smoke_train" if smoke else "formal_train",
        "condition": args.condition,
        "config_sha256": base.sha256_file(args.config.resolve()),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "preflight_lock_sha256": base.sha256_file(preflight_lock_path),
        "geometry_sha256": preflight["geometry_sha256"],
        "anchor_map_audit": anchor_audit,
        "training": training,
        "training_lock": str(lock_path),
        "training_lock_sha256": base.sha256_file(lock_path),
        "semantic_validation_evaluated": False,
        "test_evaluated": False,
        "verdict": "smoke_completed_no_semantic_or_test_read"
        if smoke
        else "formal_training_locked_awaiting_four_condition_readout",
    }
    v11.atomic_json(result_path, result)
    print(
        json.dumps(
            {
                "result_path": str(result_path),
                "sha256": base.sha256_file(result_path),
                "lock_path": str(lock_path),
                "lock_sha256": base.sha256_file(lock_path),
            },
            sort_keys=True,
        )
    )


def load_locked_model(
    condition: str,
    lock: dict[str, Any],
    config: dict[str, Any],
    preflight: dict[str, Any],
    device: torch.device,
) -> CalibratedRelationTubeModel:
    checkpoint_path = Path(lock["checkpoint"]["path"])
    if base.sha256_file(checkpoint_path) != lock["checkpoint"]["sha256"]:
        raise ValueError("tube checkpoint hash mismatch")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if (
        checkpoint["protocol_version"] != config["protocol_version"]
        or checkpoint["condition"] != condition
        or int(checkpoint["epoch"]) != 100
    ):
        raise ValueError("tube checkpoint identity mismatch")
    model = model_for_condition(condition, config, preflight, device)
    model.load_state_dict(checkpoint["state_dict"])
    return model


@torch.no_grad()
def representation_audit(
    model: CalibratedRelationTubeModel,
    images: np.ndarray,
    latents: np.ndarray,
    start: int,
    end: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    boundary: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    raw_norms: list[np.ndarray] = []
    projected_norms: list[np.ndarray] = []
    center_norms: list[np.ndarray] = []
    model.eval()
    for begin in range(start, end, batch_size):
        rows = np.arange(begin, min(begin + batch_size, end), dtype=np.int64)
        envs = np.zeros(len(rows), dtype=np.int64)
        x = base.image_batch(images, envs, rows, device)
        anchors = torch.from_numpy(
            np.array(latents[0, rows, 3:5], dtype=np.float32, copy=True)
        ).to(device)
        _, values = model.embedding.components(x, anchors)
        boundary.append(values["boundary_active"].cpu().numpy())
        scores.append(values["raw_mahalanobis_score"].cpu().numpy())
        raw_norms.append(torch.linalg.vector_norm(values["raw_residual"], dim=1).cpu().numpy())
        projected_norms.append(
            torch.linalg.vector_norm(values["projected_residual"], dim=1).cpu().numpy()
        )
        center_norms.append(torch.linalg.vector_norm(values["center"], dim=1).cpu().numpy())
    boundary_array = np.concatenate(boundary)
    score_array = np.concatenate(scores)
    raw_array = np.concatenate(raw_norms)
    projected_array = np.concatenate(projected_norms)
    center_array = np.concatenate(center_norms)
    return {
        "boundary_active_fraction": float(np.mean(boundary_array)),
        "raw_mahalanobis_mean": float(np.mean(score_array)),
        "raw_mahalanobis_p95": float(np.percentile(score_array, 95)),
        "raw_residual_l2_mean": float(np.mean(raw_array)),
        "projected_residual_l2_mean": float(np.mean(projected_array)),
        "center_l2_mean": float(np.mean(center_array)),
        "finite": bool(
            np.all(np.isfinite(score_array))
            and np.all(np.isfinite(raw_array))
            and np.all(np.isfinite(projected_array))
        ),
    }


def decide(
    semantic: dict[str, dict[str, Any]],
    ccrl_validation: dict[str, dict[str, float]],
    audits: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    tube = semantic["tube_correct"]
    unbounded = semantic["unbounded_correct"]
    permuted = semantic["tube_permuted_matched"]
    center = semantic["center_only_correct"]
    mechanism = {
        "correlation_above_unbounded": tube["mean_direct_abs_correlation"]
        > unbounded["mean_direct_abs_correlation"],
        "r2_above_unbounded": tube["mean_r2"] > unbounded["mean_r2"],
        "correlation_above_permuted": tube["mean_direct_abs_correlation"]
        > permuted["mean_direct_abs_correlation"],
        "r2_above_permuted": tube["mean_r2"] > permuted["mean_r2"],
        "ccrl_not_worse_than_center": ccrl_validation["tube_correct"]["total"]
        <= ccrl_validation["center_only_correct"]["total"],
        "tube_actually_active": audits["tube_correct"]["boundary_active_fraction"] > 0,
    }
    mechanism_supported = all(mechanism.values())
    paper = {
        "correlation_above_projected_physical": tube[
            "mean_direct_abs_correlation"
        ]
        > float(config["evaluation"]["projected_physical_validation_correlation"]),
        "r2_above_projected_physical": tube["mean_r2"]
        > float(config["evaluation"]["projected_physical_validation_r2"]),
    }
    paper_competitive = all(paper.values())
    if mechanism_supported and paper_competitive:
        verdict = "tube_structure_supported_paper_competitive"
    elif mechanism_supported:
        verdict = "tube_structure_supported_not_paper_competitive"
    elif any(mechanism.values()):
        verdict = "tube_structure_mixed"
    else:
        verdict = "tube_structure_not_supported"
    return {
        "verdict": verdict,
        "mechanism_checks": mechanism,
        "paper_competitive_checks": paper,
        "deltas": {
            "tube_minus_unbounded_correlation": tube["mean_direct_abs_correlation"]
            - unbounded["mean_direct_abs_correlation"],
            "tube_minus_unbounded_r2": tube["mean_r2"] - unbounded["mean_r2"],
            "tube_minus_permuted_correlation": tube["mean_direct_abs_correlation"]
            - permuted["mean_direct_abs_correlation"],
            "tube_minus_permuted_r2": tube["mean_r2"] - permuted["mean_r2"],
            "tube_minus_center_correlation": tube["mean_direct_abs_correlation"]
            - center["mean_direct_abs_correlation"],
            "tube_minus_center_r2": tube["mean_r2"] - center["mean_r2"],
            "tube_minus_projected_physical_correlation": tube[
                "mean_direct_abs_correlation"
            ]
            - float(config["evaluation"]["projected_physical_validation_correlation"]),
            "tube_minus_projected_physical_r2": tube["mean_r2"]
            - float(config["evaluation"]["projected_physical_validation_r2"]),
            "tube_minus_center_ccrl_total": ccrl_validation["tube_correct"]["total"]
            - ccrl_validation["center_only_correct"]["total"],
        },
        "test_evaluated": False,
    }


def direct_center_checks(
    semantic: dict[str, dict[str, Any]], config: dict[str, Any]
) -> dict[str, bool]:
    center = semantic["center_only_correct"]
    return {
        "correlation_matches_locked_direct_center": bool(
            np.isclose(
                center["mean_direct_abs_correlation"],
                float(config["evaluation"]["direct_center_validation_correlation"]),
                atol=5e-6,
            )
        ),
        "r2_matches_locked_direct_center": bool(
            np.isclose(
                center["mean_r2"],
                float(config["evaluation"]["direct_center_validation_r2"]),
                atol=5e-6,
            )
        ),
    }


def readout_main(args: argparse.Namespace, config: dict[str, Any]) -> None:
    if args.condition is not None or args.epochs is not None:
        raise ValueError("readout accepts neither condition nor epochs")
    root = Path(args.output_root or config["runtime"]["output_root"]).resolve()
    preflight, preflight_lock, preflight_lock_path = load_preflight(
        root, args.config.resolve(), config
    )
    formal_root = root / "formal"
    locks: dict[str, dict[str, Any]] = {}
    lock_hashes: dict[str, str] = {}
    for condition in CONDITIONS:
        lock_path = formal_root / condition / "training_lock.json"
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        checks = {
            "formal": lock["mode"] == "formal",
            "status": lock["status"]
            == "locked_before_joint_semantic_validation_readout",
            "protocol": lock["protocol_version"] == config["protocol_version"],
            "condition": lock["condition"] == condition,
            "config": lock["config_sha256"] == base.sha256_file(args.config.resolve()),
            "source": lock["source_files_sha256"] == source_hashes(),
            "preflight": lock["preflight_lock_sha256"]
            == base.sha256_file(preflight_lock_path),
            "geometry": lock["geometry_sha256"] == preflight["geometry_sha256"],
            "validity": all(lock["validity"].values()),
            "semantic_unread": lock["semantic_validation_evaluated"] is False,
            "test_unread": lock["test_evaluated"] is False,
        }
        if not all(checks.values()):
            raise ValueError(f"invalid tube lock for {condition}: {checks}")
        locks[condition] = lock
        lock_hashes[condition] = base.sha256_file(lock_path)
    shared = {
        "initial_state": len({lock["initial_state_sha256"] for lock in locks.values()})
        == 1,
        "geometry": len({lock["geometry_sha256"] for lock in locks.values()}) == 1,
        "source": len(
            {dynamic.canonical_sha256(lock["source_files_sha256"]) for lock in locks.values()}
        )
        == 1,
    }
    if not all(shared.values()):
        raise ValueError(f"tube condition locks are not matched: {shared}")
    torch.set_num_threads(int(config["runtime"]["num_threads"]))
    torch.set_float32_matmul_precision("high")
    device = torch.device(
        config["runtime"]["device"] if torch.cuda.is_available() else "cpu"
    )
    images = np.load(Path(config["dataset"]["cache_path"]), mmap_mode="r")
    raw_latents = base.load_latents(config)
    subset, _ = dynamic.build_subset(config)
    latents, _, _ = v17.normalize_with_k80_rgb_statistics(
        raw_latents, int(config["split"]["train_end"]), subset
    )
    anchor_maps, _ = v2.build_anchor_maps(config)
    start = int(config["split"]["train_end"])
    end = int(config["split"]["validation_end"])
    batch_size = int(config["training"]["batch_size"])
    semantic: dict[str, dict[str, Any]] = {}
    ccrl_validation: dict[str, dict[str, float]] = {}
    audits: dict[str, dict[str, Any]] = {}
    graph: dict[str, dict[str, Any]] = {}
    for condition in CONDITIONS:
        model = load_locked_model(condition, locks[condition], config, preflight, device)
        prediction = v6.predict_rgb(model, images, latents, start, end, batch_size, device)
        semantic[condition] = v6.regression_metrics(
            prediction, latents[0, start:end, :3]
        )
        ccrl_validation[condition] = v2.validate(
            model,
            images,
            latents,
            anchor_maps,
            v2.Condition(condition, "oracle"),
            config,
            device,
        )
        audits[condition] = representation_audit(
            model, images, latents, start, end, batch_size, device
        )
        graph[condition] = base.graph_metrics(
            model.parametric_part.A.detach().cpu().numpy(),
            float(config["evaluation"]["graph_threshold"]),
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    center_checks = direct_center_checks(semantic, config)
    if not all(center_checks.values()):
        raise RuntimeError(f"center-only comparator drifted: {center_checks}")
    result = {
        "protocol_version": config["protocol_version"],
        "mode": "four_condition_joint_validation_only_readout",
        "config_sha256": base.sha256_file(args.config.resolve()),
        "preflight_lock_sha256": base.sha256_file(preflight_lock_path),
        "training_locks": lock_hashes,
        "matched_lock_checks": shared,
        "center_comparator_checks": center_checks,
        "semantic_validation": semantic,
        "ccrl_validation": ccrl_validation,
        "representation_audits": audits,
        "graph_parameter_metrics": graph,
        "decision": decide(semantic, ccrl_validation, audits, config),
        "semantic_validation_evaluated": True,
        "test_evaluated": False,
    }
    output_path = formal_root / "validation_readout.json"
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite tube readout: {output_path}")
    v11.atomic_json(output_path, result)
    print(
        json.dumps(
            {
                "result_path": str(output_path),
                "sha256": base.sha256_file(output_path),
                "verdict": result["decision"]["verdict"],
            },
            sort_keys=True,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("preflight", "smoke", "train", "readout"), required=True
    )
    parser.add_argument("--condition", choices=CONDITIONS)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.resolve().read_text(encoding="utf-8"))
    validate_config(config)
    if args.mode == "preflight":
        preflight_main(args, config)
    elif args.mode == "readout":
        readout_main(args, config)
    else:
        train_main(args, config)


if __name__ == "__main__":
    main()
