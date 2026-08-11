"""Audit K25 against unbounded variants of published physical output projection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.optimize import least_squares

import audit_causalverse_slope_manifold_projection as bounded_projection
import causalverse_slope_preflight as slope
import evaluate_causalverse_slope_heldout_k26 as heldout
import run_causalverse_slope_preflight as preflight


UNBOUNDED_VARIANTS = ("unbounded_k40_std", "unbounded_k40_minmax")


def project_unbounded(
    predicted: np.ndarray,
    parameters: slope.RelationParameters,
    coordinate_scale: np.ndarray,
    max_nfev: int = 400,
    tolerance: float = 1e-10,
) -> tuple[np.ndarray, dict[str, object]]:
    """Project outputs onto the exact Slope manifold without free-coordinate bounds."""

    values = np.asarray(predicted, dtype=np.float64)
    scale = np.asarray(coordinate_scale, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 7 or scale.shape != (7,):
        raise ValueError("projection expects [N,7] predictions and seven coordinate scales")
    if np.any(~np.isfinite(values)) or np.any(~np.isfinite(scale)) or np.any(scale <= 1e-8):
        raise ValueError("projection inputs or coordinate scales are invalid")

    projected = np.empty_like(values)
    successes: list[bool] = []
    nfev: list[int] = []
    objectives: list[float] = []
    free_outside_k40_box = 0
    for index, target in enumerate(values):
        start = target[:3].copy()

        def residual(free: np.ndarray) -> np.ndarray:
            return (bounded_projection.state_from_free(free, parameters) - target) / scale

        result = least_squares(
            residual,
            start,
            max_nfev=max_nfev,
            ftol=tolerance,
            xtol=tolerance,
            gtol=tolerance,
        )
        state = bounded_projection.state_from_free(result.x, parameters)
        if not np.isfinite(state).all():
            raise ValueError(f"unbounded projection produced non-finite state at row {index}")
        projected[index] = state
        successes.append(bool(result.success))
        nfev.append(int(result.nfev))
        objectives.append(float(np.mean(residual(result.x) ** 2)))

    return projected, {
        "row_count": int(len(values)),
        "success_count": int(sum(successes)),
        "mean_nfev": float(np.mean(nfev)),
        "max_nfev_observed": int(max(nfev)),
        "mean_scaled_objective_after": float(np.mean(objectives)),
        "free_outside_k40_box_count": free_outside_k40_box,
    }


def seed_outcome(
    correct_metrics: dict[str, object],
    projection_metrics: dict[str, dict[str, object]],
    bootstrap: dict[str, dict[str, dict[str, float]]],
) -> dict[str, object]:
    directional: dict[str, bool] = {}
    ci_positive: dict[str, bool] = {}
    projection_matches: dict[str, bool] = {}
    for name in UNBOUNDED_VARIANTS:
        baseline = projection_metrics[name]
        directional[name] = bool(
            correct_metrics["mean_direct_abs_correlation_relation4"]
            > baseline["mean_direct_abs_correlation_relation4"]
            and correct_metrics["mean_direct_r2_relation4"]
            > baseline["mean_direct_r2_relation4"]
        )
        ci_positive[name] = bool(
            bootstrap[name]["relation4_correlation"]["lower"] > 0.0
            and bootstrap[name]["relation4_r2"]["lower"] > 0.0
        )
        projection_matches[name] = bool(
            baseline["mean_direct_abs_correlation_relation4"]
            >= correct_metrics["mean_direct_abs_correlation_relation4"]
            and baseline["mean_direct_r2_relation4"]
            >= correct_metrics["mean_direct_r2_relation4"]
        )
    return {
        "directional": directional,
        "ci_positive": ci_positive,
        "projection_matches_or_exceeds": projection_matches,
    }


def aggregate_decision(valid: bool, outcomes: dict[str, dict[str, object]]) -> str:
    if not valid:
        return "invalid"
    all_directional = all(
        value
        for outcome in outcomes.values()
        for value in outcome["directional"].values()
    )
    all_ci = all(
        value
        for outcome in outcomes.values()
        for value in outcome["ci_positive"].values()
    )
    any_projection_matches = any(
        value
        for outcome in outcomes.values()
        for value in outcome["projection_matches_or_exceeds"].values()
    )
    if all_directional and all_ci:
        return "published_projection_variants_beaten_all_seeds_ci"
    if all_directional:
        return "published_projection_variants_beaten_directionally"
    if any_projection_matches:
        return "published_projection_matches_or_exceeds_relation"
    return "mixed_relation_vs_published_projection"


def run(config_path: Path, output_path: Path, device: torch.device) -> None:
    protocol = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    parent = protocol["parent"]
    parent_paths = {
        "config": Path(parent["config"]),
        "contract_audit": Path(parent["contract_audit"]),
        "confirmation_result": Path(parent["confirmation_result"]),
        "bounded_projection_result": Path(parent["bounded_projection_result"]),
    }
    dependencies = {
        name: preflight.sha256_file(path) == parent[f"{name}_sha256"]
        for name, path in parent_paths.items()
    }
    confirmation = json.loads(parent_paths["confirmation_result"].read_text(encoding="utf-8"))
    contract_audit = json.loads(parent_paths["contract_audit"].read_text(encoding="utf-8"))
    bounded_result = json.loads(parent_paths["bounded_projection_result"].read_text(encoding="utf-8"))
    dependencies.update(
        {
            "confirmation_valid_validation_only": bool(
                confirmation["valid"] and not confirmation["test_evaluated"]
            ),
            "contract_valid": bool(contract_audit["valid"]),
            "bounded_projection_valid_validation_only": bool(
                bounded_result["valid"] and not bounded_result["test_evaluated"]
            ),
        }
    )
    if not all(dependencies.values()):
        raise ValueError(f"K32 dependency contract failed: {dependencies}")

    config, grouped, splits, label_ids, filter_audit = preflight.load_context(parent_paths["config"])
    seeds = [int(value) for value in config["training"]["seeds"]]
    config_sha = preflight.sha256_file(parent_paths["config"])
    feature_sha = preflight.sha256_file(Path(config["runtime"]["feature_cache"]))
    source_sha = {
        "core": preflight.sha256_file(Path(slope.__file__)),
        "runner": preflight.sha256_file(Path(preflight.__file__)),
    }
    checkpoint_root = Path(parent["checkpoint_root"])
    assets: dict[tuple[int, str], Path] = {}
    lock_checks: list[bool] = []
    for seed in seeds:
        for condition in ("point", "correct_relation"):
            directory = checkpoint_root / f"seed{seed}" / condition
            lock_path = directory / "training_lock.json"
            checkpoint_path = directory / "checkpoint.pt"
            lock = json.loads(lock_path.read_text(encoding="utf-8"))
            lock_checks.append(
                lock["condition"] == condition
                and int(lock["seed"]) == seed
                and lock["config_sha256"] == config_sha
                and lock["feature_cache_sha256"] == feature_sha
                and lock["checkpoint_sha256"] == preflight.sha256_file(checkpoint_path)
                and lock["source_sha256"] == source_sha
            )
            assets[(seed, condition)] = checkpoint_path

    train = slope.select_ids(grouped, splits["train"])
    labels = train.latents[slope.ordered_label_indices(train.ids, label_ids)]
    parameters = slope.fit_relation_parameters(labels)
    std_scale = labels.std(axis=0)
    minmax_scale = (labels.max(axis=0) - labels.min(axis=0)) / 2.0
    lower = labels[:, :3].min(axis=0)
    upper = labels[:, :3].max(axis=0)
    validation = slope.select_ids(grouped, splits[protocol["projection"]["split"]])

    metrics: dict[str, dict[str, dict[str, object]]] = {}
    deltas: dict[str, dict[str, dict[str, float]]] = {}
    bootstraps: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    diagnostics: dict[str, dict[str, dict[str, object]]] = {}
    outcomes: dict[str, dict[str, object]] = {}
    replay_checks: list[bool] = []
    bootstrap_config = protocol["bootstrap"]
    projection_config = protocol["projection"]

    for seed_index, seed in enumerate(seeds):
        predictions: dict[str, np.ndarray] = {}
        for condition in ("point", "correct_relation"):
            checkpoint = torch.load(assets[(seed, condition)], map_location=device, weights_only=False)
            model = slope.build_head(
                int(config["backbone"]["feature_dim"]), config["model"]["hidden_dims"], seed, device
            )
            model.load_state_dict(checkpoint["state_dict"])
            predictions[condition] = slope.predict_physical(
                model,
                validation,
                int(config["backbone"]["feature_dim"]),
                np.asarray(checkpoint["normalization_mean"]),
                np.asarray(checkpoint["normalization_std"]),
                device,
            )

        predictions["bounded_k40_std_replay"], bounded_diag = bounded_projection.project_to_manifold(
            predictions["point"], parameters, std_scale, lower, upper
        )
        predictions["unbounded_k40_std"], unbounded_std_diag = project_unbounded(
            predictions["point"],
            parameters,
            std_scale,
            int(projection_config["max_nfev"]),
            float(projection_config["tolerance"]),
        )
        predictions["unbounded_k40_minmax"], unbounded_minmax_diag = project_unbounded(
            predictions["point"],
            parameters,
            minmax_scale,
            int(projection_config["max_nfev"]),
            float(projection_config["tolerance"]),
        )
        for name in UNBOUNDED_VARIANTS:
            free = predictions[name][:, :3]
            diagnostics_box = np.any((free < lower) | (free > upper), axis=1)
            if name == "unbounded_k40_std":
                unbounded_std_diag["free_outside_k40_box_count"] = int(diagnostics_box.sum())
            else:
                unbounded_minmax_diag["free_outside_k40_box_count"] = int(diagnostics_box.sum())

        seed_key = str(seed)
        diagnostics[seed_key] = {
            "bounded_k40_std_replay": bounded_diag,
            "unbounded_k40_std": unbounded_std_diag,
            "unbounded_k40_minmax": unbounded_minmax_diag,
        }
        seed_metrics = {
            name: slope.evaluate_predictions(validation.latents, predicted)
            for name, predicted in predictions.items()
        }
        metrics[seed_key] = seed_metrics
        original = bounded_result["metrics_by_seed"][seed_key]["all_coordinate_manifold_projection"]
        replay = seed_metrics["bounded_k40_std_replay"]
        tolerance = float(projection_config["replay_metric_tolerance"])
        replay_checks.append(
            abs(
                replay["mean_direct_abs_correlation_relation4"]
                - original["mean_direct_abs_correlation_relation4"]
            )
            <= tolerance
            and abs(replay["mean_direct_r2_relation4"] - original["mean_direct_r2_relation4"])
            <= tolerance
        )

        correct = seed_metrics["correct_relation"]
        deltas[seed_key] = {}
        bootstraps[seed_key] = {}
        for variant_index, name in enumerate(UNBOUNDED_VARIANTS):
            baseline = seed_metrics[name]
            deltas[seed_key][f"correct_minus_{name}"] = {
                "relation4_correlation": float(
                    correct["mean_direct_abs_correlation_relation4"]
                    - baseline["mean_direct_abs_correlation_relation4"]
                ),
                "relation4_r2": float(
                    correct["mean_direct_r2_relation4"]
                    - baseline["mean_direct_r2_relation4"]
                ),
                "free3_correlation": float(
                    correct["mean_direct_abs_correlation_free3"]
                    - baseline["mean_direct_abs_correlation_free3"]
                ),
            }
            bootstraps[seed_key][name] = heldout.paired_bootstrap(
                validation.latents,
                predictions["correct_relation"],
                predictions[name],
                int(bootstrap_config["resamples"]),
                int(bootstrap_config["seed"]) + 100 * seed_index + variant_index,
                float(bootstrap_config["confidence"]),
            )
        outcomes[seed_key] = seed_outcome(correct, seed_metrics, bootstraps[seed_key])

    convergence_checks = [
        item[variant]["success_count"] == item[variant]["row_count"]
        for item in diagnostics.values()
        for variant in protocol["projection"]["variants"]
    ]
    validity = {
        **dependencies,
        "all_six_checkpoint_locks_match": bool(all(lock_checks) and len(lock_checks) == 6),
        "incomplete_product_filter_contract_matches": bool(filter_audit["contract_matches"]),
        "bounded_replay_matches_k25": bool(all(replay_checks) and len(replay_checks) == 3),
        "all_projection_rows_converged": bool(all(convergence_checks)),
        "validation_nonempty": bool(len(validation.ids) > 0),
        "test_unread": True,
    }
    valid = bool(all(validity.values()))
    payload = {
        "protocol_version": protocol["protocol_version"],
        "fact_type": "validation_published_output_projection_parity_audit",
        "valid": valid,
        "validity": validity,
        "machine_decision": aggregate_decision(valid, outcomes),
        "projection_contract": {
            "std_scale_k40": std_scale.tolist(),
            "minmax_half_range_k40": minmax_scale.tolist(),
            "bounded_free_lower_k40": lower.tolist(),
            "bounded_free_upper_k40": upper.tolist(),
            "unbounded_variants": list(UNBOUNDED_VARIANTS),
            "max_nfev": int(projection_config["max_nfev"]),
            "tolerance": float(projection_config["tolerance"]),
        },
        "metrics_by_seed": metrics,
        "deltas_by_seed": deltas,
        "paired_bootstrap": bootstraps,
        "seed_outcomes": outcomes,
        "projection_diagnostics": diagnostics,
        "validation_id_count": int(len(validation.ids)),
        "validation_ids_sha256": preflight.sha256_int_array(validation.ids),
        "config_sha256": preflight.sha256_file(config_path),
        "evaluator_sha256": preflight.sha256_file(Path(__file__)),
        "test_evaluated": False,
    }
    preflight.write_json(output_path, payload)
    print(json.dumps({"valid": valid, "machine_decision": payload["machine_decision"], "output": str(output_path)}))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    selected_device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    run(args.config, args.output, selected_device)
