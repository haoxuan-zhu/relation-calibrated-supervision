"""Run the frozen anchor-preserving raw-ridge propagation comparison."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import torch

import anchor_preserving_raw_ridge as anchor
import raw_ridge_propagation as raw
import run_conflict_projected_physics_from_scratch as v16
import run_conflict_projected_physics_training as v13


_BASE_SPEC = importlib.util.spec_from_file_location(
    "anchor_preserving_budget_base",
    Path(__file__).resolve().with_name("run_relation_budget_condition.py"),
)
budget = importlib.util.module_from_spec(_BASE_SPEC)
assert _BASE_SPEC.loader is not None
_BASE_SPEC.loader.exec_module(budget)


PROTOCOL = "anchor_preserving_raw_ridge_v1"
INTERNAL_CONDITION_KIND = "empirical"


def validate_registry(registry: dict[str, Any]) -> None:
    if registry["protocol_version"] != PROTOCOL:
        raise ValueError("unexpected anchor-preserving protocol")
    if list(registry["budget_curve"]["budgets"]) != [20, 40, 80, 160]:
        raise ValueError("anchor-preserving budget order changed")
    if list(registry["budget_curve"]["conditions"]) != [INTERNAL_CONDITION_KIND]:
        raise ValueError("anchor-preserving protocol accepts one condition")
    if list(registry["budget_curve"]["seeds"]) != [0, 42, 3407]:
        raise ValueError("anchor-preserving initialization order changed")
    if registry["initialization"]["mode"] != "same_seed_random_no_warm":
        raise ValueError("anchor-preserving runs must start from registered states")
    if registry["anchor_preservation"] != {
        "measured_environments": ["obs"],
        "unmeasured_target_source": "raw_ridge_relation",
        "intervention_target_source": "raw_ridge_relation",
    }:
        raise ValueError("anchor-preservation contract changed")
    if tuple(float(value) for value in registry["raw_ridge_anchor"]["alphas"]) != raw.DEFAULT_ALPHAS:
        raise ValueError("raw-ridge alpha registry changed")

    implementation = registry["implementation"]
    hashes = {
        "entrypoint_sha256": budget.base.sha256_file(Path(__file__).resolve()),
        "anchor_module_sha256": budget.base.sha256_file(Path(anchor.__file__).resolve()),
        "teacher_module_sha256": budget.base.sha256_file(Path(raw.__file__).resolve()),
    }
    for key, actual in hashes.items():
        if implementation[key] != actual:
            raise ValueError(f"{key} mismatch")


def fit_teacher_bundle(
    images: np.ndarray,
    raw_latents: np.ndarray,
    subset: np.ndarray,
    config: dict[str, Any],
    include_validation: bool,
) -> dict[str, Any]:
    teacher, teacher_audit = raw.fit_teacher_from_calibration_rows(
        images,
        raw_latents,
        subset,
        int(config["split"]["train_end"]),
        config["raw_ridge_anchor"]["alphas"],
    )
    checks = {
        "raw_design_shape_exact": teacher_audit["design_shape"]
        == [int(subset.size), 5],
        "raw_coefficients_shape_exact": teacher_audit["coefficient_shape"] == [5, 3],
        "raw_teacher_finite": all(
            np.all(np.isfinite(np.asarray(teacher[key], dtype=np.float64)))
            for key in ("feature_mean", "feature_scale", "coefficients")
        ),
        "raw_feature_scale_positive": bool(
            np.all(np.asarray(teacher["feature_scale"]) > 0)
        ),
        "leave_one_out_curve_complete": len(teacher_audit["leave_one_out_curve"])
        == len(raw.DEFAULT_ALPHAS),
    }
    bundle: dict[str, Any] = {
        "physical_fit": {"not_applicable": True, "reason": "raw_ridge_only_protocol"},
        "empirical_coefficients": teacher,
        "empirical_fit": teacher_audit,
    }
    if include_validation:
        validation = raw.validation_metrics(
            teacher,
            images,
            raw_latents,
            int(config["split"]["train_end"]),
            int(config["split"]["validation_end"]),
            bool(config["raw_ridge_anchor"]["clip_predictions_to_unit_interval"]),
        )
        bundle["post_fit_validation"] = {"raw_ridge": validation}
        checks["raw_validation_finite"] = bool(validation.get("finite", False))
    bundle["numerical_gate"] = {
        "checks": checks,
        "all_valid": bool(all(checks.values())),
        "performance_threshold_used": False,
    }
    return bundle


def audit_teachers(
    registry: dict[str, Any], config_path: Path, budget_value: int, output_dir: Path
) -> Path:
    config = budget.build_run_config(
        registry, INTERNAL_CONDITION_KIND, budget_value, 3407
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    images = np.load(Path(config["dataset"]["cache_path"]), mmap_mode="r")
    raw_latents = budget.base.load_latents(config)
    subset, subset_audit = budget.build_subset(registry, budget_value)
    _, latent_mean, latent_std = budget.v17.normalize_with_k80_rgb_statistics(
        raw_latents, int(config["split"]["train_end"]), subset
    )
    bundle = fit_teacher_bundle(images, raw_latents, subset, config, True)
    checks = {
        "subset_hash_exact": subset_audit["subset_sha256"]
        == subset_audit["expected_subset_sha256"],
        "subset_unique_exact": subset_audit["unique_rows"] == budget_value,
        "rgb_mean_finite": bool(np.all(np.isfinite(latent_mean[:3]))),
        "rgb_std_finite_positive": bool(
            np.all(np.isfinite(latent_std[:3])) and np.all(latent_std[:3] > 0)
        ),
        **bundle["numerical_gate"]["checks"],
    }
    result = {
        "protocol_version": PROTOCOL,
        "mode": "teacher_audit_train_validation_only",
        "budget": budget_value,
        "config_path": str(config_path),
        "config_sha256": budget.base.sha256_file(config_path),
        "subset": subset_audit,
        "normalization": {"latent_mean": latent_mean, "latent_std": latent_std},
        "teachers": bundle,
        "decision": {
            "checks": checks,
            "performance_threshold_used": False,
            "all_valid": bool(all(checks.values())),
        },
        "test_evaluated": False,
    }
    result_path = output_dir / "teacher_audit.json"
    budget.v11.atomic_json(result_path, result)
    return result_path


def train_one(
    condition_kind: str,
    initial_state: dict[str, torch.Tensor],
    initial_state_hash: str,
    images: np.ndarray,
    latents: np.ndarray,
    raw_latents: np.ndarray,
    anchor_maps: dict[str, np.ndarray],
    subset: np.ndarray,
    config: dict[str, Any],
    device: torch.device,
    output_dir: Path,
    teacher_bundle: dict[str, Any],
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
) -> dict[str, Any]:
    if condition_kind != INTERNAL_CONDITION_KIND:
        raise ValueError("anchor-preserving wrapper received another condition")
    condition = budget.v2.Condition(anchor.CONDITION, "oracle")
    teacher = {
        key: torch.from_numpy(np.asarray(value, dtype=np.float32)).to(device)
        for key, value in teacher_bundle["empirical_coefficients"].items()
        if key in ("feature_mean", "feature_scale", "coefficients")
    }

    def registered_loss(*args: Any, **kwargs: Any):
        return anchor.loss_components(*args, **kwargs, subset=subset)

    original = v13.loss_components
    v13.loss_components = registered_loss
    try:
        training = v16.train_condition(
            condition,
            initial_state,
            initial_state_hash,
            images,
            latents,
            raw_latents,
            anchor_maps,
            config,
            device,
            output_dir,
            teacher,
            latent_mean,
            latent_std,
        )
    finally:
        v13.loss_components = original
    training["anchor_preservation"] = {
        "measured_row_count": int(subset.size),
        "measured_environments": ["obs"],
        "intervention_truth_read": False,
        "unmeasured_targets": "raw_ridge_relation",
    }
    return training


def teacher_validation_metrics(
    teacher: dict[str, Any],
    images: np.ndarray,
    raw_latents: np.ndarray,
    train_end: int,
    validation_end: int,
    clip: bool,
) -> dict[str, Any]:
    return raw.validation_metrics(
        teacher, images, raw_latents, train_end, validation_end, clip
    )


def install_protocol_hooks() -> None:
    budget.PROTOCOL = PROTOCOL
    budget.CONDITIONS = {INTERNAL_CONDITION_KIND: anchor.CONDITION}
    budget.SOURCE_FILES = list(
        dict.fromkeys(
            [
                *budget.SOURCE_FILES,
                "raw_ridge_propagation.py",
                "anchor_preserving_raw_ridge.py",
                "run_anchor_preserving_raw_ridge_condition.py",
            ]
        )
    )
    budget.validate_registry = validate_registry
    budget.fit_teacher_bundle = fit_teacher_bundle
    budget.audit_teachers = audit_teachers
    budget.train_one = train_one
    budget.empirical.teacher_validation_metrics = teacher_validation_metrics


install_protocol_hooks()


if __name__ == "__main__":
    budget.main()
