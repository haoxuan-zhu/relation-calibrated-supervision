"""Retrospective evaluation of anchor-preserving relation propagation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from relation_tube import welding
from relation_tube.anchor_propagation import preserve_observed_targets


PROTOCOL = "resistance_spot_welding_anchor_preserving_v1"
EXPECTED_BUDGETS = [20, 40, 80, 160]
EXPECTED_SEEDS = [3407, 42, 0]


def validate_config(config: dict[str, Any]) -> None:
    if config.get("protocol_version") != PROTOCOL:
        raise ValueError("unexpected anchor-preserving protocol")
    if config.get("status") != "retrospective_after_original_heldout_read":
        raise ValueError("retrospective status must remain explicit")
    if config["task"].get("target") != "PullTest (N)":
        raise ValueError("destructive target changed")
    if config["task"].get("side_features") != list(welding.SIDE_FEATURE_NAMES):
        raise ValueError("instrument relation changed")
    model = config["model"]
    if model.get("primary_modality") != "infrared":
        raise ValueError("primary modality changed")
    if float(model["relation_ridge_alpha"]) != 0.1:
        raise ValueError("relation ridge changed")
    if float(model["image_ridge_alpha"]) != 0.1:
        raise ValueError("image ridge changed")
    probe = config["probe"]
    if [int(value) for value in probe["budgets"]] != EXPECTED_BUDGETS:
        raise ValueError("label budgets changed")
    if [int(value) for value in probe["seeds"]] != EXPECTED_SEEDS:
        raise ValueError("seeds changed")
    if int(probe["relation_derangement_seed_base"]) != 20260805:
        raise ValueError("derangement seed changed")
    if int(probe["bootstrap_replicates"]) != 5000:
        raise ValueError("bootstrap count changed")


def verify_files(config: dict[str, Any]) -> tuple[dict[str, Path], dict[str, str]]:
    paths: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    for name in ("csv", "split", "features"):
        path = Path(config["data"][f"{name}_path"])
        digest = welding.sha256_file(path)
        if digest != str(config["data"][f"{name}_sha256"]):
            raise ValueError(f"{name} hash changed")
        paths[name] = path
        hashes[str(path)] = digest
    for row in config["implementation"]["files"]:
        path = Path(row["path"])
        digest = welding.sha256_file(path)
        if digest != str(row["sha256"]):
            raise ValueError(f"implementation changed: {path}")
        hashes[str(path)] = digest
    return paths, hashes


def summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    comparison_names = list(runs[0]["comparisons"])
    for budget in EXPECTED_BUDGETS:
        selected = [run for run in runs if run["budget"] == budget]
        conditions = {}
        for name in selected[0]["conditions"]:
            values = np.asarray([run["conditions"][name]["r2"] for run in selected])
            conditions[name] = {
                "mean_r2": float(values.mean()),
                "range_r2": [float(values.min()), float(values.max())],
            }
        comparisons = {}
        for name in comparison_names:
            observed = np.asarray([run["comparisons"][name]["observed"] for run in selected])
            comparisons[name] = {
                "mean": float(observed.mean()),
                "range": [float(observed.min()), float(observed.max())],
                "positive_seeds": int(np.sum(observed > 0.0)),
                "ci95_positive_seeds": int(
                    sum(run["comparisons"][name]["ci95_low"] > 0.0 for run in selected)
                ),
            }
        summary[str(budget)] = {"conditions": conditions, "comparisons": comparisons}
    return summary


def evaluate(config: dict[str, Any], paths: dict[str, Path]) -> dict[str, Any]:
    units = welding.load_units(paths["csv"])
    split_by_id, split_record = welding.load_split(paths["split"], set(units))
    with np.load(paths["features"], allow_pickle=False) as archive:
        feature_ids = np.asarray(archive["sample_ids"])
        view_features = np.asarray(archive["view_features"])

    train_ids = sorted(sample_id for sample_id, split in split_by_id.items() if split == "train")
    train_index = {sample_id: index for index, sample_id in enumerate(train_ids)}
    train_side = np.stack([units[sample_id]["side"] for sample_id in train_ids])
    train_target = np.asarray([units[sample_id]["pull_force"] for sample_id in train_ids])
    train_images = welding.reorder_features(train_ids, feature_ids, view_features)[:, 2]
    relation_alpha = float(config["model"]["relation_ridge_alpha"])
    image_alpha = float(config["model"]["image_ridge_alpha"])
    replicates = int(config["probe"]["bootstrap_replicates"])

    evaluations: dict[str, Any] = {}
    for split_index, evaluation_split in enumerate(("validation", "heldout")):
        evaluation_ids = sorted(
            sample_id for sample_id, split in split_by_id.items() if split == evaluation_split
        )
        evaluation_target = np.asarray(
            [units[sample_id]["pull_force"] for sample_id in evaluation_ids]
        )
        evaluation_images = welding.reorder_features(
            evaluation_ids, feature_ids, view_features
        )[:, 2]
        runs = []
        for budget_index, budget in enumerate(EXPECTED_BUDGETS):
            for seed_index, seed in enumerate(EXPECTED_SEEDS):
                subset_ids = welding.calibration_subset(train_ids, units, budget, seed)
                subset = np.asarray([train_index[sample_id] for sample_id in subset_ids])
                target_mean = float(train_target[subset].mean())
                target_scale = float(train_target[subset].std())
                normalized_target = (train_target[subset] - target_mean) / target_scale

                relation = welding.StandardizedDualRidge(relation_alpha).fit(
                    train_side[subset], normalized_target
                )
                propagated = relation.predict(train_side)
                permutation = welding.derangement(
                    budget, int(config["probe"]["relation_derangement_seed_base"]) + seed
                )
                permuted_relation = welding.StandardizedDualRidge(relation_alpha).fit(
                    train_side[subset], normalized_target[permutation]
                )
                permuted = permuted_relation.predict(train_side)

                anchor_target = preserve_observed_targets(
                    propagated, subset, normalized_target
                )
                anchor_permuted_target = preserve_observed_targets(
                    permuted, subset, normalized_target
                )
                training_conditions = {
                    "point": (train_images[subset], normalized_target),
                    "relation_propagation": (train_images, propagated),
                    "anchor_preserving": (train_images, anchor_target),
                    "anchor_preserving_permuted": (train_images, anchor_permuted_target),
                }
                predictions = {}
                for name, (features, target) in training_conditions.items():
                    prediction = welding.StandardizedDualRidge(image_alpha).fit(
                        features, target
                    ).predict(evaluation_images)
                    predictions[name] = prediction * target_scale + target_mean
                conditions = {
                    name: welding.regression_metrics(evaluation_target, prediction)
                    for name, prediction in predictions.items()
                }
                base_seed = (
                    int(config["probe"]["bootstrap_seed_base"])
                    + split_index * 10_000
                    + budget_index * 100
                    + seed_index * 10
                )
                comparisons = {
                    "anchor_preserving_minus_point": welding.bootstrap_r2_delta(
                        evaluation_target,
                        predictions["anchor_preserving"],
                        predictions["point"],
                        replicates=replicates,
                        seed=base_seed,
                    ),
                    "anchor_preserving_minus_relation_propagation": welding.bootstrap_r2_delta(
                        evaluation_target,
                        predictions["anchor_preserving"],
                        predictions["relation_propagation"],
                        replicates=replicates,
                        seed=base_seed + 1,
                    ),
                    "anchor_preserving_minus_anchor_preserving_permuted": welding.bootstrap_r2_delta(
                        evaluation_target,
                        predictions["anchor_preserving"],
                        predictions["anchor_preserving_permuted"],
                        replicates=replicates,
                        seed=base_seed + 2,
                    ),
                }
                runs.append(
                    {
                        "budget": budget,
                        "seed": seed,
                        "subset_ids": subset_ids,
                        "subset_sha256": welding.stable_hash(",".join(map(str, subset_ids))),
                        "target_mean": target_mean,
                        "target_scale": target_scale,
                        "conditions": conditions,
                        "comparisons": comparisons,
                        "evaluation_predictions": {
                            "sample_ids": evaluation_ids,
                            "target": evaluation_target.tolist(),
                            **{name: value.tolist() for name, value in predictions.items()},
                        },
                    }
                )
        evaluations[evaluation_split] = {"runs": runs, "summary": summarize(runs)}

    return {
        "protocol": PROTOCOL,
        "status": config["status"],
        "decision": "retrospective_structural_rescue_observed_requires_independent_confirmation",
        "method_change": (
            "Measured targets are retained on the K labeled training units; "
            "the relation estimate supplies targets only to unlabeled units."
        ),
        "heldout_interpretation": "post_hoc_support_only_not_blind_confirmation",
        "split_protocol": split_record["protocol"],
        "evaluations": evaluations,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    validate_config(config)
    paths, hashes = verify_files(config)
    result = evaluate(config, paths)
    result["input_and_implementation_sha256"] = hashes
    output = Path(config["output"]["audit_json_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "decision": result["decision"]}))


if __name__ == "__main__":
    main()
