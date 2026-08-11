"""Read-only integrity and scale-sensitive metric audit for the locked budget curve.

This script never trains or selects a checkpoint.  It verifies the archived
aggregate against its source JSON files and, when the locked checkpoints are
available, recomputes validation/test direct R2 with the frozen evaluator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

import run_clamped_anchor_diagnostic as v2
import run_conflict_projected_physics_training as v13
import run_diagnostic as base
import run_supervised_continuation_diagnostic as v6


KINDS = ("point", "empirical", "physical")
SEEDS = (0, 42, 3407)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_result_path(project_root: Path, recorded: str) -> Path:
    path = Path(recorded)
    if path.exists():
        return path.resolve()
    # Aggregation may have happened on Windows while the locked checkpoints are
    # audited on Linux.  Normalize separators before relocating under outputs/.
    parts = [part for part in str(recorded).replace("\\", "/").split("/") if part]
    if "outputs" not in parts:
        raise FileNotFoundError(f"cannot relocate result outside outputs/: {recorded}")
    relocated = project_root.joinpath(*parts[parts.index("outputs") :]).resolve()
    if not relocated.exists():
        raise FileNotFoundError(relocated)
    return relocated


def unpack_condition(raw: dict[str, Any], kind: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = raw.get("derived_config", raw.get("config"))
    if not isinstance(config, dict):
        raise ValueError("result has no embedded config")
    if kind == "physical" and isinstance(raw.get("training"), dict) and "vanilla_correct_floor" in raw["training"]:
        return (
            config,
            raw["training"]["vanilla_correct_floor"],
            raw["post_lock_validation_semantics"]["vanilla_correct_floor"],
            raw["metrics"]["vanilla_correct_floor"],
        )
    return config, raw["training"], raw["post_lock_validation_semantics"], raw["metrics"]


def subset_hash(raw: dict[str, Any], kind: str) -> str:
    if "subset" in raw:
        return str(raw["subset"]["subset_sha256"])
    if kind == "point":
        return str(raw["point_anchor"]["subset_sha256"])
    if kind == "empirical":
        return str(raw["calibration_subset"]["subset_sha256"])
    return str(raw["calibration"]["subset_sha256"])


def normalization_arrays(raw: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(raw["normalization"]["latent_mean"], dtype=np.float32)
    std = np.asarray(raw["normalization"]["latent_std"], dtype=np.float32)
    if mean.shape != (5,) or std.shape != (5,) or np.any(std <= 0) or not np.isfinite(std).all():
        raise ValueError("invalid stored normalization")
    return mean, std


def common_contract_checks(records: dict[str, dict[str, Any]]) -> dict[str, bool]:
    configs = {kind: unpack_condition(records[kind], kind)[0] for kind in KINDS}
    trainings = {kind: unpack_condition(records[kind], kind)[1] for kind in KINDS}
    initial_hashes = {records[kind]["initial_state"]["state_dict_sha256"] for kind in KINDS}
    subset_hashes = {subset_hash(records[kind], kind) for kind in KINDS}
    means = [normalization_arrays(records[kind])[0] for kind in KINDS]
    stds = [normalization_arrays(records[kind])[1] for kind in KINDS]
    model_payloads = [configs[kind]["model"] for kind in KINDS]
    training_keys = (
        "epochs",
        "batch_size",
        "learning_rate",
        "scheduler_factor",
        "scheduler_patience",
        "validation_interval",
        "targets",
        "kappa",
        "eta",
        "mu",
    )
    return {
        "shared_initial_state": len(initial_hashes) == 1,
        "shared_subset": len(subset_hashes) == 1,
        "shared_normalization_mean": all(np.array_equal(means[0], value) for value in means[1:]),
        "shared_normalization_std": all(np.array_equal(stds[0], value) for value in stds[1:]),
        "shared_model": all(model_payloads[0] == value for value in model_payloads[1:]),
        "shared_training_contract": all(
            configs["point"]["training"][key] == configs[kind]["training"][key]
            for kind in KINDS[1:]
            for key in training_keys
        ),
        "no_warm_start": all("warm_start" not in configs[kind] for kind in KINDS),
        "same_parameter_count": len({int(trainings[kind]["parameter_count"]) for kind in KINDS}) == 1,
        "same_final_floor": len({float(trainings[kind]["final_supervision_weight"]) for kind in KINDS}) == 1,
        "no_semantic_truth_during_training": all(
            not bool(trainings[kind].get("semantic_truth_read_during_training", False)) for kind in KINDS
        ),
    }


def direct_r2(true: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    denominator = np.sum((true - true.mean(axis=0, keepdims=True)) ** 2, axis=0)
    return 1.0 - np.sum((true - predicted) ** 2, axis=0) / np.maximum(denominator, 1e-12)


def prediction_calibration(true: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    """Describe the affine scale/bias of predictions without recalibrating them.

    The fitted convention is ``predicted = slope * true + intercept`` for each
    direct coordinate.  These are diagnostics for why direct R2 and Pearson
    correlation may disagree; they are not used to select or modify a model.
    """
    if true.shape != predicted.shape or true.ndim != 2:
        raise ValueError("true and predicted must be matching 2-D arrays")
    true64 = np.asarray(true, dtype=np.float64)
    predicted64 = np.asarray(predicted, dtype=np.float64)
    true_centered = true64 - true64.mean(axis=0, keepdims=True)
    predicted_centered = predicted64 - predicted64.mean(axis=0, keepdims=True)
    denominator = np.sum(true_centered**2, axis=0)
    if np.any(denominator <= 1e-12):
        raise ValueError("calibration is undefined for a constant true coordinate")
    slope = np.sum(true_centered * predicted_centered, axis=0) / denominator
    intercept = predicted64.mean(axis=0) - slope * true64.mean(axis=0)
    return {
        "prediction_on_truth_slope": slope.tolist(),
        "prediction_on_truth_intercept": intercept.tolist(),
        "mean_abs_slope_error_from_one": float(np.mean(np.abs(slope - 1.0))),
        "mean_abs_intercept": float(np.mean(np.abs(intercept))),
    }


def recompute_split(
    raw: dict[str, Any],
    kind: str,
    split: str,
    device: torch.device,
) -> dict[str, Any]:
    config, training, _, _ = unpack_condition(raw, kind)
    checkpoint = Path(training["checkpoint"])
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    if base.sha256_file(checkpoint) != training["checkpoint_sha256"]:
        raise ValueError(f"checkpoint hash mismatch: {checkpoint}")
    images = np.load(Path(config["dataset"]["cache_path"]), mmap_mode="r")
    raw_latents = base.load_latents(config)
    mean, std = normalization_arrays(raw)
    latents = (raw_latents - mean[None, None, :]) / std[None, None, :]
    train_end = int(config["split"]["train_end"])
    validation_end = int(config["split"]["validation_end"])
    test_end = int(config["split"]["test_end"])
    start, end = (train_end, validation_end) if split == "validation" else (validation_end, test_end)
    model = v13.load_model(checkpoint, config, device)
    prediction = v6.predict_rgb(
        model,
        images,
        latents,
        start,
        end,
        int(config["training"]["batch_size"]),
        device,
    )
    truth = latents[0, start:end, :3]
    metrics = v6.regression_metrics(prediction, truth)
    r2 = direct_r2(truth, prediction)
    result = {
        "direct_abs_correlation": metrics["direct_abs_correlation"],
        "mean_direct_abs_correlation": float(metrics["mean_direct_abs_correlation"]),
        "direct_r2": r2.tolist(),
        "mean_direct_r2": float(np.mean(r2)),
        "normalized_rmse": float(np.sqrt(np.mean((truth - prediction) ** 2))),
        **prediction_calibration(truth, prediction),
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def recompute_k80_physical_condition(
    raw: dict[str, Any],
    condition_name: str,
    split: str,
    device: torch.device,
) -> dict[str, Any]:
    config = raw["config"]
    training = raw["training"][condition_name]
    checkpoint = Path(training["checkpoint"])
    if base.sha256_file(checkpoint) != training["checkpoint_sha256"]:
        raise ValueError(f"checkpoint hash mismatch: {checkpoint}")
    images = np.load(Path(config["dataset"]["cache_path"]), mmap_mode="r")
    raw_latents = base.load_latents(config)
    mean, std = normalization_arrays(raw)
    latents = (raw_latents - mean[None, None, :]) / std[None, None, :]
    train_end = int(config["split"]["train_end"])
    validation_end = int(config["split"]["validation_end"])
    test_end = int(config["split"]["test_end"])
    start, end = (train_end, validation_end) if split == "validation" else (validation_end, test_end)
    model = v13.load_model(checkpoint, config, device)
    prediction = v6.predict_rgb(
        model, images, latents, start, end, int(config["training"]["batch_size"]), device
    )
    truth = latents[0, start:end, :3]
    r2 = direct_r2(truth, prediction)
    correlations = np.asarray(v6.regression_metrics(prediction, truth)["direct_abs_correlation"])
    result = {
        "direct_abs_correlation": correlations.tolist(),
        "mean_direct_abs_correlation": float(correlations.mean()),
        "direct_r2": r2.tolist(),
        "mean_direct_r2": float(r2.mean()),
        **prediction_calibration(truth, prediction),
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def mean_sd(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "population_sd": float(array.std(ddof=0))}


def summarize(records: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for kind in KINDS:
        output[kind] = {
            split: mean_sd([records[kind][str(seed)][split]["mean_direct_r2"] for seed in SEEDS])
            for split in ("validation", "test")
        }
    for left, right, name in (
        ("physical", "point", "physical_minus_point"),
        ("empirical", "point", "empirical_minus_point"),
        ("physical", "empirical", "physical_minus_empirical"),
    ):
        output[name] = {}
        for split in ("validation", "test"):
            values = [
                records[left][str(seed)][split]["mean_direct_r2"]
                - records[right][str(seed)][split]["mean_direct_r2"]
                for seed in SEEDS
            ]
            output[name][split] = {"by_seed": dict(zip(map(str, SEEDS), values)), **mean_sd(values)}
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", required=True, type=Path)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-checkpoint-evaluation", action="store_true")
    args = parser.parse_args()

    aggregate = load_json(args.aggregate.resolve())
    project_root = args.project_root.resolve()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    contracts: dict[str, Any] = {}
    locked_metrics: dict[str, Any] = {}
    k80_four_cell_metrics: dict[str, Any] = {
        name: {} for name in (
            "vanilla_correct_floor",
            "vanilla_permuted_floor",
            "projected_correct_floor",
            "projected_permuted_floor",
        )
    }
    source_integrity: dict[str, bool] = {}

    for budget in (20, 40, 80, 160):
        budget_key = str(budget)
        contracts[budget_key] = {}
        locked_metrics[budget_key] = {kind: {} for kind in KINDS}
        for seed in SEEDS:
            raw_by_kind: dict[str, dict[str, Any]] = {}
            for kind in KINDS:
                entry = aggregate["entries"][budget_key][kind][str(seed)]
                path = resolve_result_path(project_root, entry["result_path"])
                source_integrity[f"k{budget}/seed{seed}/{kind}"] = (
                    base.sha256_file(path) == entry["result_sha256"]
                )
                raw = load_json(path)
                raw_by_kind[kind] = raw
                _, _, stored_validation, stored_test = unpack_condition(raw, kind)
                if args.skip_checkpoint_evaluation:
                    locked_metrics[budget_key][kind][str(seed)] = {
                        "validation": {
                            "mean_direct_abs_correlation": float(stored_validation["mean_direct_abs_correlation"]),
                            "mean_direct_r2": float(stored_validation["mean_r2"]),
                        },
                        "test": {"mean_direct_abs_correlation": float(np.mean(stored_test["direct_abs_correlation"][:3]))},
                    }
                else:
                    validation = recompute_split(raw, kind, "validation", device)
                    test = recompute_split(raw, kind, "test", device)
                    validation["stored_correlation_abs_error"] = abs(
                        validation["mean_direct_abs_correlation"]
                        - float(stored_validation["mean_direct_abs_correlation"])
                    )
                    validation["stored_r2_abs_error"] = abs(
                        validation["mean_direct_r2"] - float(stored_validation["mean_r2"])
                    )
                    test["stored_correlation_abs_error"] = abs(
                        test["mean_direct_abs_correlation"]
                        - float(np.mean(stored_test["direct_abs_correlation"][:3]))
                    )
                    locked_metrics[budget_key][kind][str(seed)] = {
                        "validation": validation,
                        "test": test,
                    }
            contracts[budget_key][str(seed)] = common_contract_checks(raw_by_kind)
            if budget == 80 and not args.skip_checkpoint_evaluation:
                physical_raw = raw_by_kind["physical"]
                for name in k80_four_cell_metrics:
                    k80_four_cell_metrics[name][str(seed)] = {
                        split: recompute_k80_physical_condition(physical_raw, name, split, device)
                        for split in ("validation", "test")
                    }

    summaries = None if args.skip_checkpoint_evaluation else {
        str(budget): summarize(locked_metrics[str(budget)]) for budget in (20, 40, 80, 160)
    }
    k80_four_cell_deltas = None
    if not args.skip_checkpoint_evaluation:
        k80_four_cell_deltas = {}
        for left, right, name in (
            ("vanilla_correct_floor", "vanilla_permuted_floor", "vanilla_correct_minus_permuted"),
            ("projected_correct_floor", "projected_permuted_floor", "projected_correct_minus_permuted"),
            ("projected_correct_floor", "vanilla_correct_floor", "projected_minus_vanilla_correct"),
        ):
            k80_four_cell_deltas[name] = {}
            for split in ("validation", "test"):
                k80_four_cell_deltas[name][split] = {}
                for metric in ("mean_direct_abs_correlation", "mean_direct_r2"):
                    values = [
                        k80_four_cell_metrics[left][str(seed)][split][metric]
                        - k80_four_cell_metrics[right][str(seed)][split][metric]
                        for seed in SEEDS
                    ]
                    k80_four_cell_deltas[name][split][metric] = {
                        "by_seed": dict(zip(map(str, SEEDS), values)),
                        **mean_sd(values),
                    }
    result = {
        "protocol_version": "relation_budget_locked_metric_integrity_audit_k0_v1",
        "fact_type": "retrospective_audit",
        "scope": "locked_checkpoints_only_no_training_no_model_selection",
        "formal_preregistered_decision_unchanged": True,
        "source_aggregate": str(args.aggregate.resolve()),
        "source_integrity": source_integrity,
        "all_source_hashes_match": bool(all(source_integrity.values())),
        "contracts": contracts,
        "all_contract_checks_pass": bool(
            all(all(checks.values()) for by_seed in contracts.values() for checks in by_seed.values())
        ),
        "locked_metrics": locked_metrics,
        "scale_sensitive_summaries": summaries,
        "k80_four_cell_metrics": k80_four_cell_metrics,
        "k80_four_cell_deltas": k80_four_cell_deltas,
        "interpretation_boundary": (
            "R2 was not the preregistered primary endpoint. This audit can reveal hidden calibration effects "
            "and metric mismatch, but any upgraded claim requires a prospectively frozen replication."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(base.json_ready(result), indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "all_source_hashes_match": result["all_source_hashes_match"],
        "all_contract_checks_pass": result["all_contract_checks_pass"],
        "output": str(args.output),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
