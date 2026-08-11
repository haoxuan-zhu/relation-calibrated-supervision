"""Matched point/empirical/physical relation-supervision budget runner.

The registry fixes the complete K={20,40,80,160} curve.  Each formal
invocation trains exactly one condition from the registered random state and
writes a lock before semantic validation or test evaluation.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

import audit_physics_functional_anchor as v10
import run_clamped_anchor_diagnostic as v2
import run_conflict_projected_physics_from_scratch as v16
import run_conflict_projected_physics_k80_closed as v17
import run_conflict_projected_physics_training as v13
import run_diagnostic as base
import run_empirical_function_control_k80 as empirical
import run_matched_point_control_k80 as point
import run_physics_functional_anchor_training as v11
import run_supervised_continuation_diagnostic as v6


PROTOCOL = "relation_supervision_budget_curve_v1"
CONDITIONS = {
    "point": point.CONDITION,
    "empirical": empirical.CONDITION,
    "physical": "vanilla_correct_floor",
}
SOURCE_FILES = [
    "run_relation_budget_condition.py",
    "run_empirical_function_control_k80.py",
    "run_matched_point_control_k80.py",
    "audit_physics_functional_anchor.py",
    "run_conflict_projected_physics_k80_closed.py",
    "run_conflict_projected_physics_from_scratch.py",
    "run_conflict_projected_physics_training.py",
    "run_physics_functional_anchor_training.py",
    "run_supervised_continuation_diagnostic.py",
    "run_conditioned_film_diagnostic.py",
    "run_clamped_anchor_diagnostic.py",
    "run_diagnostic.py",
]


def keyed(mapping: dict[Any, Any], key: int) -> Any:
    """Read an integer-keyed YAML mapping robustly after JSON round trips."""
    if key in mapping:
        return mapping[key]
    if str(key) in mapping:
        return mapping[str(key)]
    raise KeyError(key)


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        base.json_ready(value), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_registry(registry: dict[str, Any]) -> None:
    if registry["protocol_version"] != PROTOCOL:
        raise ValueError("unexpected relation-budget protocol")
    if registry["initialization"]["mode"] != "same_seed_random_no_warm":
        raise ValueError("budget curve must start from random state without warm start")
    if list(registry["budget_curve"]["budgets"]) != [20, 40, 80, 160]:
        raise ValueError("registered budget order changed")
    if list(registry["budget_curve"]["conditions"]) != list(CONDITIONS):
        raise ValueError("registered condition order changed")
    if list(registry["budget_curve"]["seeds"]) != [0, 42, 3407]:
        raise ValueError("registered seed order changed")


def build_subset(registry: dict[str, Any], budget: int) -> tuple[np.ndarray, dict[str, Any]]:
    curve = registry["budget_curve"]
    if budget not in list(curve["budgets"]):
        raise ValueError(f"unregistered budget: {budget}")
    train_end = int(registry["split"]["train_end"])
    master = np.random.default_rng(int(curve["subset_seed"])).permutation(train_end)
    subset = np.ascontiguousarray(master[:budget], dtype=np.int64)
    actual_hash = v11.v9.sha256_int_array(subset)
    expected_hash = keyed(curve["subset_sha256"], budget)
    audit = {
        "budget": budget,
        "subset_seed": int(curve["subset_seed"]),
        "subset_sha256": actual_hash,
        "expected_subset_sha256": expected_hash,
        "subset_rows": subset.tolist(),
        "unique_rows": int(np.unique(subset).size),
        "is_master_prefix": bool(np.array_equal(subset, master[:budget])),
    }
    if actual_hash != expected_hash:
        raise ValueError(f"registered K={budget} subset hash mismatch")
    if audit["unique_rows"] != budget:
        raise ValueError("budget subset contains duplicate rows")
    return subset, audit


def build_run_config(
    registry: dict[str, Any], condition_kind: str, budget: int, seed: int
) -> dict[str, Any]:
    validate_registry(registry)
    if condition_kind not in CONDITIONS:
        raise ValueError(condition_kind)
    if seed not in list(registry["budget_curve"]["seeds"]):
        raise ValueError(f"unregistered seed: {seed}")
    _, subset_audit = build_subset(registry, budget)
    config = copy.deepcopy(registry)
    config["selected_run"] = {
        "condition_kind": condition_kind,
        "condition_name": CONDITIONS[condition_kind],
        "budget": budget,
        "seed": seed,
    }
    config["initialization"] = {
        "mode": "same_seed_random_no_warm",
        "expected_state_dict_sha256": keyed(
            registry["initialization"]["expected_state_dict_sha256"], seed
        ),
        "provenance": "registered_same_seed_random_state",
    }
    config["normalization"] = {
        "rgb_statistics_source": keyed(
            registry["budget_curve"]["rgb_statistics_source"], budget
        ),
        "rgb_statistics_budget": budget,
        "subset_sha256": subset_audit["subset_sha256"],
        "angle_statistics_source": registry["normalization"][
            "angle_statistics_source"
        ],
    }
    config["training"]["seed"] = seed
    config["training"]["conditions"] = [CONDITIONS[condition_kind]]
    config["point_anchor"].update(
        {"budget": budget, "subset_seed": registry["budget_curve"]["subset_seed"]}
    )
    config["functional_anchor"].update(
        {"budget": budget, "subset_seed": registry["budget_curve"]["subset_seed"]}
    )
    config["empirical_anchor"]["budget"] = budget
    config["runtime"]["output_dir"] = str(
        Path(registry["runtime"]["output_root"])
        / f"k{budget}"
        / f"seed{seed}"
        / condition_kind
    )
    return config


def physical_teacher_validation_metrics(
    coefficients: np.ndarray,
    images: np.ndarray,
    raw_latents: np.ndarray,
    train_end: int,
    validation_end: int,
    rcond: float,
    clip: bool,
) -> tuple[dict[str, Any], np.ndarray]:
    means = np.stack(
        [
            v10.image_channel_means(images, env, train_end, validation_end)
            for env in range(images.shape[0])
        ],
        axis=0,
    )
    tau = v10.malus_tau(raw_latents)[:, train_end:validation_end]
    truth = raw_latents[:, train_end:validation_end, :3].astype(np.float64) / 255.0
    prediction, condition_numbers = v10.invert_forward(
        coefficients,
        means.reshape(-1, 3),
        tau.reshape(-1),
        "full_malus_forward",
        rcond,
        clip,
    )
    return (
        v10.regression_metrics(prediction, truth.reshape(-1, 3)),
        condition_numbers,
    )


def fit_teacher_bundle(
    images: np.ndarray,
    raw_latents: np.ndarray,
    subset: np.ndarray,
    config: dict[str, Any],
    include_validation: bool,
) -> dict[str, Any]:
    train_end = int(config["split"]["train_end"])
    validation_end = int(config["split"]["validation_end"])
    spec, calibration_subset = v11.build_calibration_spec(config)
    if not np.array_equal(spec["subset"], subset):
        raise ValueError("teacher and registered budget subsets differ")
    physical_coefficients, physical_fit = v11.calibrate_teachers(
        images, raw_latents, spec, config
    )
    empirical_coefficients, empirical_fit = empirical.fit_empirical_teacher(
        images, raw_latents, subset, train_end
    )

    unit_rgb = raw_latents[0, subset, :3].astype(np.float64) / 255.0
    tau = v10.malus_tau(raw_latents)[0, subset]
    train_means = v10.image_channel_means(images, 0, 0, train_end)[subset]
    physical_design = v10.full_forward_design(unit_rgb, tau)
    empirical_design = np.concatenate(
        [
            np.ones((len(subset), 1)),
            train_means,
            train_means * tau[:, None],
        ],
        axis=1,
    )
    numerical = {
        "physical_design_shape": list(physical_design.shape),
        "physical_design_rank": int(np.linalg.matrix_rank(physical_design)),
        "empirical_design_shape": list(empirical_design.shape),
        "empirical_design_rank": int(np.linalg.matrix_rank(empirical_design)),
        "physical_coefficients_finite": bool(
            np.all(np.isfinite(physical_coefficients["correct"]))
        ),
        "empirical_coefficients_finite": bool(
            np.all(np.isfinite(empirical_coefficients))
        ),
    }
    bundle: dict[str, Any] = {
        "calibration_subset": calibration_subset,
        "physical_fit": physical_fit["correct"],
        "empirical_fit": empirical_fit,
        "numerical": numerical,
        "physical_coefficients": physical_coefficients,
        "empirical_coefficients": empirical_coefficients,
    }
    if include_validation:
        physical_validation, condition_numbers = physical_teacher_validation_metrics(
            physical_coefficients["correct"],
            images,
            raw_latents,
            train_end,
            validation_end,
            float(config["functional_anchor"]["pinv_rcond"]),
            bool(
                config["functional_anchor"][
                    "clip_predictions_to_unit_interval"
                ]
            ),
        )
        empirical_validation = empirical.teacher_validation_metrics(
            empirical_coefficients,
            images,
            raw_latents,
            train_end,
            validation_end,
            bool(config["empirical_anchor"]["clip_predictions_to_unit_interval"]),
        )
        numerical.update(
            {
                "physical_validation_finite": bool(
                    physical_validation.get("finite", False)
                ),
                "empirical_validation_finite": bool(
                    empirical_validation.get("finite", False)
                ),
                "physical_condition_numbers_finite": bool(
                    np.all(np.isfinite(condition_numbers))
                ),
            }
        )
        bundle["post_fit_validation"] = {
            "physical": physical_validation,
            "empirical": empirical_validation,
            "physical_condition_numbers": v10.summarize_distribution(
                condition_numbers
            ),
        }
    gate = {
        "physical_design_rank_7": numerical["physical_design_rank"] == 7,
        "empirical_design_rank_7": numerical["empirical_design_rank"] == 7,
        "physical_coefficients_finite": numerical[
            "physical_coefficients_finite"
        ],
        "empirical_coefficients_finite": numerical[
            "empirical_coefficients_finite"
        ],
    }
    if include_validation:
        gate.update(
            {
                "physical_validation_finite": numerical[
                    "physical_validation_finite"
                ],
                "empirical_validation_finite": numerical[
                    "empirical_validation_finite"
                ],
                "physical_condition_numbers_finite": numerical[
                    "physical_condition_numbers_finite"
                ],
            }
        )
    bundle["numerical_gate"] = {
        "checks": gate,
        "all_valid": bool(all(gate.values())),
        "performance_threshold_used": False,
    }
    return bundle


def audit_teachers(
    registry: dict[str, Any], config_path: Path, budget: int, output_dir: Path
) -> Path:
    config = build_run_config(registry, "physical", budget, 3407)
    output_dir.mkdir(parents=True, exist_ok=True)
    images = np.load(Path(config["dataset"]["cache_path"]), mmap_mode="r")
    raw_latents = base.load_latents(config)
    subset, subset_audit = build_subset(registry, budget)
    _, latent_mean, latent_std = v17.normalize_with_k80_rgb_statistics(
        raw_latents, int(config["split"]["train_end"]), subset
    )
    bundle = fit_teacher_bundle(
        images, raw_latents, subset, config, include_validation=True
    )
    normalization_gate = {
        "rgb_mean_finite": bool(np.all(np.isfinite(latent_mean[:3]))),
        "rgb_std_finite_positive": bool(
            np.all(np.isfinite(latent_std[:3])) and np.all(latent_std[:3] > 0)
        ),
    }
    decision = {
        "checks": {
            "subset_hash_exact": subset_audit["subset_sha256"]
            == subset_audit["expected_subset_sha256"],
            "subset_unique_exact": subset_audit["unique_rows"] == budget,
            **normalization_gate,
            **bundle["numerical_gate"]["checks"],
        },
        "performance_threshold_used": False,
    }
    decision["all_valid"] = bool(all(decision["checks"].values()))
    result = {
        "protocol_version": PROTOCOL,
        "mode": "teacher_audit_train_validation_only",
        "budget": budget,
        "config_path": str(config_path),
        "config_sha256": base.sha256_file(config_path),
        "subset": subset_audit,
        "normalization": {"latent_mean": latent_mean, "latent_std": latent_std},
        "teachers": bundle,
        "decision": decision,
        "test_evaluated": False,
    }
    result_path = output_dir / "teacher_audit.json"
    v11.atomic_json(result_path, result)
    return result_path


def load_preflight_audit(path: Path, budget: int) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value["protocol_version"] != PROTOCOL or int(value["budget"]) != budget:
        raise ValueError("teacher preflight does not match formal budget")
    if value.get("test_evaluated") is not False:
        raise ValueError("teacher preflight unexpectedly read test")
    return value


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
    if condition_kind == "point":
        return point.train_condition(
            initial_state,
            initial_state_hash,
            images,
            latents,
            anchor_maps,
            subset,
            config,
            device,
            output_dir,
        )

    condition = v2.Condition(CONDITIONS[condition_kind], "oracle")
    if condition_kind == "physical":
        coefficients = {
            name: torch.from_numpy(value).to(device)
            for name, value in teacher_bundle["physical_coefficients"].items()
        }
        return v16.train_condition(
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
            coefficients,
            latent_mean,
            latent_std,
        )

    coefficients = {
        "correct": torch.from_numpy(
            teacher_bundle["empirical_coefficients"]
        ).to(device)
    }
    original_loss_components = v13.loss_components
    v13.loss_components = empirical.empirical_loss_components
    try:
        return v16.train_condition(
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
            coefficients,
            latent_mean,
            latent_std,
        )
    finally:
        v13.loss_components = original_loss_components


def decide_validity(
    condition_kind: str,
    config: dict[str, Any],
    subset_audit: dict[str, Any],
    preflight: dict[str, Any],
    training: dict[str, Any],
    generated_initial_state_hash: str,
    validation: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    expected_initial = config["initialization"]["expected_state_dict_sha256"]
    budget = int(config["selected_run"]["budget"])
    validation_value = float(validation["mean_direct_abs_correlation"])
    test_value = float(np.mean(metrics["direct_abs_correlation"][:3]))
    checks = {
        "random_initial_state_exact": generated_initial_state_hash
        == expected_initial,
        "same_initial_state_loaded": training["initial_state_sha256"]
        == expected_initial,
        "no_warm_checkpoint_in_config": "warm_start" not in config,
        "subset_budget_exact": subset_audit["budget"]
        == subset_audit["unique_rows"]
        == budget,
        "subset_hash_exact": subset_audit["subset_sha256"]
        == config["normalization"]["subset_sha256"],
        "normalization_budget_exact": int(
            config["normalization"]["rgb_statistics_budget"]
        )
        == budget,
        "preflight_budget_exact": int(preflight["budget"]) == budget,
        "preflight_numerically_valid": bool(preflight["decision"]["all_valid"]),
        "same_parameter_count": training["parameter_count"]
        == int(config["implementation"]["expected_parameter_count"]),
        "floor_exact": bool(
            np.isclose(training["final_supervision_weight"], 0.1)
        ),
        "no_semantic_truth_before_lock": all(
            "semantic_validation" not in record for record in training["history"]
        )
        and not training["semantic_truth_read_during_training"],
        "anchors_exact": metrics["anchor_mean_direct_r2"]
        >= float(config["evaluation"]["minimum_oracle_anchor_r2"]),
        "training_values_finite": point.numeric_training_values_are_finite(training),
        "semantic_metrics_finite": bool(
            np.isfinite(validation_value) and np.isfinite(test_value)
        ),
    }
    return {
        "verdict": (
            "relation_budget_condition_valid"
            if all(checks.values())
            else "relation_budget_condition_invalid"
        ),
        "validity": checks,
        "condition_kind": condition_kind,
        "budget": budget,
        "seed": int(config["training"]["seed"]),
        "readout": {
            "validation_rgb_mean_direct_abs_correlation": validation_value,
            "test_rgb_mean_direct_abs_correlation": test_value,
        },
        "effect_judgment_deferred_to_budget_aggregate": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--condition-kind", choices=list(CONDITIONS))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--teacher-audit", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()

    config_path = args.config.resolve()
    registry = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_registry(registry)
    actual_runner_hash = base.sha256_file(Path(__file__).resolve())
    expected_runner_hash = registry["implementation"]["runner_sha256"]
    if actual_runner_hash != expected_runner_hash:
        raise ValueError(
            f"runner hash mismatch: {actual_runner_hash} != {expected_runner_hash}"
        )

    if args.audit_only:
        if args.condition_kind is not None or args.seed is not None or args.smoke:
            raise ValueError("teacher audit accepts only config, budget and output-dir")
        output_dir = (
            args.output_dir
            or Path(registry["runtime"]["output_root"])
            / "teacher_audits"
            / f"k{args.budget}"
        ).resolve()
        result_path = audit_teachers(registry, config_path, args.budget, output_dir)
        print(
            json.dumps(
                {
                    "result_path": str(result_path),
                    "sha256": base.sha256_file(result_path),
                },
                sort_keys=True,
            )
        )
        return

    if args.condition_kind is None or args.seed is None:
        raise ValueError("formal/smoke run requires condition-kind and seed")
    config = build_run_config(
        registry, args.condition_kind, args.budget, args.seed
    )
    formal_epochs = int(config["training"]["epochs"])
    epochs = int(args.epochs or formal_epochs)
    if not args.smoke and epochs != formal_epochs:
        raise ValueError("formal run must use frozen epochs")
    config["training"]["epochs"] = epochs

    if args.teacher_audit is None:
        raise ValueError("formal/smoke run requires the matching teacher audit")
    teacher_audit_path = args.teacher_audit.resolve()
    preflight = load_preflight_audit(teacher_audit_path, args.budget)
    if args.condition_kind != "point" and not preflight["decision"]["all_valid"]:
        raise ValueError("relation teacher failed the frozen numerical gate")

    output_dir = (args.output_dir or Path(config["runtime"]["output_dir"])).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_name = "smoke_results.json" if args.smoke else "diagnostic_results.json"
    result_path = output_dir / result_name
    if result_path.exists():
        raise FileExistsError(f"refusing to overwrite existing result: {result_path}")

    torch.set_num_threads(int(config["runtime"]["num_threads"]))
    torch.set_float32_matmul_precision("high")
    device = torch.device(
        config["runtime"]["device"] if torch.cuda.is_available() else "cpu"
    )
    images = np.load(Path(config["dataset"]["cache_path"]), mmap_mode="r")
    raw_latents = base.load_latents(config)
    subset, subset_audit = build_subset(registry, args.budget)
    latents, latent_mean_np, latent_std_np = v17.normalize_with_k80_rgb_statistics(
        raw_latents, int(config["split"]["train_end"]), subset
    )
    latent_mean = torch.from_numpy(latent_mean_np).to(device)
    latent_std = torch.from_numpy(latent_std_np).to(device)
    teacher_bundle = fit_teacher_bundle(
        images, raw_latents, subset, config, include_validation=False
    )
    if not teacher_bundle["numerical_gate"]["all_valid"]:
        raise ValueError("train-only teacher fit failed numerical validity")
    anchor_maps, anchor_map_audit = v2.build_anchor_maps(config)
    initial_state, initial_state_hash = v16.make_initial_state(config, device)
    expected_initial = config["initialization"]["expected_state_dict_sha256"]
    if initial_state_hash != expected_initial:
        raise ValueError(
            f"initial state hash mismatch: {initial_state_hash} != {expected_initial}"
        )
    initial_path = output_dir / f"initial_state_seed{args.seed}.pt"
    torch.save(
        {
            "seed": args.seed,
            "state_dict_sha256": initial_state_hash,
            "state_dict": copy.deepcopy(initial_state),
        },
        initial_path,
    )

    started = time.time()
    training = train_one(
        args.condition_kind,
        initial_state,
        initial_state_hash,
        images,
        latents,
        raw_latents,
        anchor_maps,
        subset,
        config,
        device,
        output_dir,
        teacher_bundle,
        latent_mean,
        latent_std,
    )
    source_hashes = {
        name: base.sha256_file(Path(__file__).resolve().parent / name)
        for name in SOURCE_FILES
    }
    training_lock = {
        "status": "locked_before_semantic_validation_and_test_evaluation",
        "registry_config_sha256": base.sha256_file(config_path),
        "derived_config_sha256": canonical_sha256(config),
        "source_files_sha256": source_hashes,
        "initial_state": {
            "path": str(initial_path),
            "file_sha256": base.sha256_file(initial_path),
            "state_dict_sha256": initial_state_hash,
        },
        "normalization": {
            "rgb_statistics_source": config["normalization"][
                "rgb_statistics_source"
            ],
            "subset_sha256": subset_audit["subset_sha256"],
            "latent_mean": latent_mean_np,
            "latent_std": latent_std_np,
        },
        "teacher_preflight": {
            "path": str(teacher_audit_path),
            "sha256": base.sha256_file(teacher_audit_path),
            "all_valid": preflight["decision"]["all_valid"],
        },
        "train_only_teacher_fit": {
            "physical": teacher_bundle["physical_fit"],
            "empirical": teacher_bundle["empirical_fit"],
            "numerical_gate": teacher_bundle["numerical_gate"],
        },
        "subset": subset_audit,
        "anchor_map_audit": anchor_map_audit,
        "checkpoint": {
            "path": training["checkpoint"],
            "sha256": training["checkpoint_sha256"],
            "initial_state_sha256": training["initial_state_sha256"],
            "final_supervision_weight": training["final_supervision_weight"],
        },
        "semantic_validation_evaluated": False,
        "test_evaluated": False,
    }
    lock_path = output_dir / "training_lock.json"
    v11.atomic_json(lock_path, training_lock)
    lock_hash = base.sha256_file(lock_path)
    common = {
        "protocol_version": PROTOCOL,
        "mode": "smoke" if args.smoke else "formal",
        "registry_config_path": str(config_path),
        "registry_config_sha256": base.sha256_file(config_path),
        "derived_config": config,
        "derived_config_sha256": canonical_sha256(config),
        "source_files_sha256": source_hashes,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "condition_kind": args.condition_kind,
        "budget": args.budget,
        "seed": args.seed,
        "subset": subset_audit,
        "initial_state": training_lock["initial_state"],
        "normalization": training_lock["normalization"],
        "teacher_preflight": training_lock["teacher_preflight"],
        "train_only_teacher_fit": training_lock["train_only_teacher_fit"],
        "training": training,
        "training_lock": str(lock_path),
        "training_lock_sha256_before_semantic_and_test": lock_hash,
        "duration_seconds_before_semantic_and_test": time.time() - started,
    }
    if args.smoke:
        result = {
            **common,
            "verdict": "smoke_completed_no_semantic_or_test_read",
            "semantic_validation_evaluated": False,
            "test_evaluated": False,
        }
    else:
        model = v13.load_model(Path(training["checkpoint"]), config, device)
        train_end = int(config["split"]["train_end"])
        validation_end = int(config["split"]["validation_end"])
        prediction = v6.predict_rgb(
            model,
            images,
            latents,
            train_end,
            validation_end,
            int(config["training"]["batch_size"]),
            device,
        )
        validation = v6.regression_metrics(
            prediction, latents[0, train_end:validation_end, :3]
        )
        condition = v2.Condition(CONDITIONS[args.condition_kind], "oracle")
        metrics = v2.evaluate_model(
            model, images, latents, anchor_maps, condition, config, device
        )
        teacher_validation = None
        if args.condition_kind == "physical":
            teacher_validation, condition_numbers = physical_teacher_validation_metrics(
                teacher_bundle["physical_coefficients"]["correct"],
                images,
                raw_latents,
                train_end,
                validation_end,
                float(config["functional_anchor"]["pinv_rcond"]),
                bool(
                    config["functional_anchor"][
                        "clip_predictions_to_unit_interval"
                    ]
                ),
            )
            teacher_validation["condition_numbers"] = v10.summarize_distribution(
                condition_numbers
            )
        elif args.condition_kind == "empirical":
            teacher_validation = empirical.teacher_validation_metrics(
                teacher_bundle["empirical_coefficients"],
                images,
                raw_latents,
                train_end,
                validation_end,
                bool(
                    config["empirical_anchor"][
                        "clip_predictions_to_unit_interval"
                    ]
                ),
            )
        decision = decide_validity(
            args.condition_kind,
            config,
            subset_audit,
            preflight,
            training,
            initial_state_hash,
            validation,
            metrics,
        )
        result = {
            **common,
            "post_lock_teacher_validation": teacher_validation,
            "post_lock_validation_semantics": validation,
            "metrics": metrics,
            "decision": decision,
            "duration_seconds": time.time() - started,
            "semantic_validation_evaluated": True,
            "test_evaluated": True,
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    v11.atomic_json(result_path, result)
    print(
        json.dumps(
            {"result_path": str(result_path), "sha256": base.sha256_file(result_path)},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
