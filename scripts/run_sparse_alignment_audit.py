"""Few-label equivalence-class alignment audit for frozen v3 checkpoints."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

import run_clamped_anchor_diagnostic as v2
import run_conditioned_film_diagnostic as v3
import run_diagnostic as base


def fit_affine_ridge(x: np.ndarray, y: np.ndarray, ridge: float) -> tuple[np.ndarray, np.ndarray]:
    x_mean = x.mean(axis=0, keepdims=True)
    x_std = np.maximum(x.std(axis=0, keepdims=True), 1e-6)
    scaled = (x - x_mean) / x_std
    design = np.concatenate([np.ones((len(x), 1)), scaled], axis=1)
    penalty = np.eye(design.shape[1]) * ridge
    penalty[0, 0] = 0.0
    weights = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return weights, np.concatenate([x_mean, x_std], axis=0)


def apply_affine_ridge(x: np.ndarray, weights: np.ndarray, stats: np.ndarray) -> np.ndarray:
    scaled = (x - stats[0:1]) / stats[1:2]
    design = np.concatenate([np.ones((len(x), 1)), scaled], axis=1)
    return design @ weights


def direct_metrics(prediction: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    residual = np.sum((truth - prediction) ** 2, axis=0)
    total = np.sum((truth - truth.mean(axis=0)) ** 2, axis=0)
    r2 = 1.0 - residual / np.maximum(total, 1e-12)
    correlations = []
    for index in range(truth.shape[1]):
        correlations.append(
            float(abs(np.corrcoef(prediction[:, index], truth[:, index])[0, 1]))
        )
    return {
        "r2": r2.tolist(),
        "mean_r2": float(r2.mean()),
        "direct_abs_correlation": correlations,
        "mean_direct_abs_correlation": float(np.mean(correlations)),
    }


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "p10": float(np.percentile(array, 10)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def load_embeddings(
    run_name: str,
    run_cfg: dict[str, Any],
    images: np.ndarray,
    latents: np.ndarray,
    anchor_maps: dict[str, np.ndarray],
    config: dict[str, Any],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    model_cfg = config["model"]
    model = v3.FilmConditionedModel(
        int(model_cfg["latent_dim"]),
        list(model_cfg["learned_indices"]),
        list(model_cfg["anchor_indices"]),
        int(model_cfg["hidden_channels"]),
        int(model_cfg["conv_layers"]),
    ).to(device)
    checkpoint_path = Path(run_cfg["checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    condition = v2.Condition(run_name, str(run_cfg["anchor_mode"]))
    split = config["split"]
    validation = v2.encode_rows(
        model,
        images,
        latents,
        anchor_maps,
        condition,
        0,
        int(split["train_end"]),
        int(split["validation_end"]),
        512,
        device,
    )
    test = v2.encode_rows(
        model,
        images,
        latents,
        anchor_maps,
        condition,
        0,
        int(split["validation_end"]),
        int(split["test_end"]),
        512,
        device,
    )
    metadata = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": base.sha256_file(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "anchor_mode": condition.anchor_mode,
    }
    return validation[:, :3], test[:, :3], metadata


def decide(results: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    audit = config["audit"]
    budget = str(int(audit["decision_budget"]))
    run_pairs = [("original_oracle", "original_shuffled"), ("replay_oracle", "replay_shuffled")]
    checks = {}
    for oracle_name, shuffled_name in run_pairs:
        oracle = results[oracle_name]["budgets"][budget]
        shuffled = results[shuffled_name]["budgets"][budget]
        paired = np.asarray(oracle["repeat_mean_r2"]) - np.asarray(
            shuffled["repeat_mean_r2"]
        )
        prefix = oracle_name.removesuffix("_oracle")
        checks[prefix] = {
            "oracle_median": oracle["mean_r2_summary"]["median"],
            "oracle_p10": oracle["mean_r2_summary"]["p10"],
            "paired_oracle_minus_shuffle_median": float(np.median(paired)),
            "passes": {
                "median": oracle["mean_r2_summary"]["median"]
                >= float(audit["minimum_oracle_median_mean_r2"]),
                "p10": oracle["mean_r2_summary"]["p10"]
                >= float(audit["minimum_oracle_p10_mean_r2"]),
                "paired_gap": float(np.median(paired))
                >= float(audit["minimum_paired_oracle_minus_shuffle_median_r2"]),
            },
        }
    supported = all(all(item["passes"].values()) for item in checks.values())
    return {
        "verdict": (
            "sparse_alignment_diagnostic_supported"
            if supported
            else "sparse_alignment_diagnostic_not_supported"
        ),
        "decision_budget": int(budget),
        "checks": checks,
        "next_action": (
            "preregister_sparse_calibration_continuation"
            if supported
            else "do_not_rescue_with_more_labels"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    torch.set_num_threads(int(config["runtime"]["num_threads"]))
    device = torch.device(config["runtime"]["device"] if torch.cuda.is_available() else "cpu")
    images = np.load(Path(config["dataset"]["cache_path"]), mmap_mode="r")
    raw_latents = base.load_latents(config)
    latents, latent_mean, latent_std = base.normalize_latents(
        raw_latents, int(config["split"]["train_end"])
    )
    anchor_maps, shuffle_checks = v2.build_anchor_maps(config)
    validation_truth = latents[
        0, int(config["split"]["train_end"]) : int(config["split"]["validation_end"]), :3
    ]
    test_truth = latents[
        0, int(config["split"]["validation_end"]) : int(config["split"]["test_end"]), :3
    ]
    audit = config["audit"]
    results = {}
    for run_name, run_cfg in config["runs"].items():
        validation, test, metadata = load_embeddings(
            run_name, run_cfg, images, latents, anchor_maps, config, device
        )
        run_result = {"metadata": metadata, "budgets": {}}
        for budget in audit["label_budgets"]:
            repeat_metrics = []
            for repeat in range(int(audit["repeats"])):
                rng = np.random.default_rng(
                    int(audit["selection_seed"]) + 104729 * int(budget) + repeat
                )
                selection = rng.choice(len(validation), size=int(budget), replace=False)
                weights, stats = fit_affine_ridge(
                    validation[selection], validation_truth[selection], float(audit["ridge"])
                )
                repeat_metrics.append(
                    direct_metrics(apply_affine_ridge(test, weights, stats), test_truth)
                )
            mean_r2 = [item["mean_r2"] for item in repeat_metrics]
            mean_corr = [item["mean_direct_abs_correlation"] for item in repeat_metrics]
            run_result["budgets"][str(budget)] = {
                "repeat_metrics": repeat_metrics,
                "repeat_mean_r2": mean_r2,
                "mean_r2_summary": summarize(mean_r2),
                "mean_direct_abs_correlation_summary": summarize(mean_corr),
            }
        results[run_name] = run_result

    result = {
        "protocol_version": config["protocol_version"],
        "config": config,
        "config_sha256": base.sha256_file(config_path),
        "source_sha256": base.sha256_file(Path(__file__).resolve()),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "data": {
            "latent_mean": latent_mean,
            "latent_std": latent_std,
            "shuffle_checks": shuffle_checks,
            "test_used_for_evaluation_only": True,
        },
        "runs": results,
    }
    result["decision"] = decide(results, config)
    output_dir = Path(config["runtime"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "alignment_audit.json"
    output_path.write_text(
        json.dumps(base.json_ready(result), indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"decision": result["decision"], "output": str(output_path)}))


if __name__ == "__main__":
    main()
