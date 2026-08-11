"""Evaluate sparse relation propagation on the spot-welding dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from relation_tube import welding


PROTOCOL = "resistance_spot_welding_relation_v1"
EXPECTED_BUDGETS = [20, 40, 80, 160]
EXPECTED_SEEDS = [3407, 42, 0]
EXPECTED_VIEWS = ["rgb_front", "rgb_back", "infrared"]


def validate_config(config: dict[str, Any]) -> None:
    if config.get("protocol_version") != PROTOCOL:
        raise ValueError("unexpected spot-welding protocol")
    phase = config.get("phase")
    if phase not in {"validation", "heldout"}:
        raise ValueError("phase must be validation or heldout")
    data = config["data"]
    if data.get("evaluation_split") != phase:
        raise ValueError("evaluation split does not match phase")
    if data.get("heldout_target_evaluated") is not (phase == "heldout"):
        raise ValueError("heldout read flag does not match phase")
    if config["task"].get("target") != "PullTest (N)":
        raise ValueError("primary destructive target changed")
    if config["task"].get("side_features") != list(welding.SIDE_FEATURE_NAMES):
        raise ValueError("registered instrument relation changed")
    model = config["model"]
    if model.get("primary_modality") != "infrared":
        raise ValueError("registered primary modality changed")
    if model.get("view_order") != EXPECTED_VIEWS:
        raise ValueError("frozen feature view order changed")
    if float(model["relation_ridge_alpha"]) != 0.1:
        raise ValueError("relation ridge changed")
    if model.get("image_ridge_alpha") != {"rgb": 1.0, "infrared": 0.1}:
        raise ValueError("image ridge changed")
    probe = config["probe"]
    if [int(value) for value in probe["budgets"]] != EXPECTED_BUDGETS:
        raise ValueError("registered label budgets changed")
    if [int(value) for value in probe["seeds"]] != EXPECTED_SEEDS:
        raise ValueError("registered seeds changed")
    if int(probe["relation_derangement_seed_base"]) != 20260805:
        raise ValueError("registered relation derangement changed")
    if int(probe["bootstrap_seed_base"]) != 20264300:
        raise ValueError("registered bootstrap seed changed")
    if int(probe["bootstrap_replicates"]) != 5000:
        raise ValueError("registered bootstrap count changed")
    if not config.get("implementation", {}).get("files"):
        raise ValueError("implementation hashes are missing")
    if not str(config.get("output", {}).get("audit_json_path", "")).endswith(".json"):
        raise ValueError("result path is not locked")


def verify_inputs(config: dict[str, Any]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for name in ("csv", "split", "features", "feature_manifest", "image_manifest"):
        path = Path(config["data"][f"{name}_path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        if welding.sha256_file(path) != str(config["data"][f"{name}_sha256"]):
            raise ValueError(f"{name} hash changed")
        paths[name] = path
    if config["phase"] == "heldout":
        path = Path(config["data"]["validation_result_path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        if welding.sha256_file(path) != str(config["data"]["validation_result_sha256"]):
            raise ValueError("validation result changed before heldout evaluation")
        paths["validation_result"] = path
    return paths


def verify_implementation(config: dict[str, Any]) -> dict[str, str]:
    observed = {}
    for row in config["implementation"]["files"]:
        path = Path(row["path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = welding.sha256_file(path)
        if digest != str(row["sha256"]):
            raise ValueError(f"formal implementation changed: {path}")
        observed[str(path)] = digest
    return observed


def leave_one_out_relation(
    features: np.ndarray, target: np.ndarray, alpha: float
) -> tuple[welding.StandardizedDualRidge, np.ndarray]:
    prediction = np.empty(len(target), dtype=np.float64)
    for index in range(len(target)):
        keep = np.arange(len(target)) != index
        prediction[index] = welding.StandardizedDualRidge(alpha).fit(
            features[keep], target[keep]
        ).predict(features[index : index + 1])[0]
    model = welding.StandardizedDualRidge(alpha).fit(features, target)
    return model, prediction


def build_inputs(
    ids: list[int], units: dict[int, dict[str, Any]], feature_ids: np.ndarray, views: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    side = np.stack([units[sample_id]["side"] for sample_id in ids])
    target = np.asarray([units[sample_id]["pull_force"] for sample_id in ids])
    ordered = welding.reorder_features(ids, feature_ids, views).astype(np.float64)
    images = {
        "rgb": ordered[:, :2].mean(axis=1),
        "infrared": ordered[:, 2],
    }
    return side, target, images


def evaluate(
    config: dict[str, Any], paths: dict[str, Path], implementation: dict[str, str]
) -> dict[str, Any]:
    units = welding.load_units(paths["csv"])
    split_by_id, split_record = welding.load_split(paths["split"], set(units))
    with np.load(paths["features"], allow_pickle=False) as archive:
        feature_ids = np.asarray(archive["sample_ids"])
        view_features = np.asarray(archive["view_features"])

    train_ids = sorted(sample_id for sample_id, split in split_by_id.items() if split == "train")
    evaluation_split = str(config["phase"])
    evaluation_ids = sorted(
        sample_id for sample_id, split in split_by_id.items() if split == evaluation_split
    )
    train_side, train_target, train_images = build_inputs(
        train_ids, units, feature_ids, view_features
    )
    evaluation_side, evaluation_target, evaluation_images = build_inputs(
        evaluation_ids, units, feature_ids, view_features
    )
    train_index = {sample_id: index for index, sample_id in enumerate(train_ids)}
    model_config = config["model"]
    relation_alpha = float(model_config["relation_ridge_alpha"])
    image_alpha = {key: float(value) for key, value in model_config["image_ridge_alpha"].items()}
    probe = config["probe"]

    dense_ceiling = {
        modality: {
            "labels": len(train_ids),
            "metrics": welding.regression_metrics(
                evaluation_target,
                welding.StandardizedDualRidge(image_alpha[modality])
                .fit(train_images[modality], train_target)
                .predict(evaluation_images[modality]),
            ),
        }
        for modality in ("rgb", "infrared")
    }
    runs = []
    for budget_index, budget in enumerate(int(value) for value in probe["budgets"]):
        for seed_index, seed in enumerate(int(value) for value in probe["seeds"]):
            subset_ids = welding.calibration_subset(train_ids, units, budget, seed)
            subset = np.asarray([train_index[sample_id] for sample_id in subset_ids])
            target_mean = float(train_target[subset].mean())
            target_scale = float(train_target[subset].std())
            if target_scale <= 0.0:
                raise ValueError("calibration target scale collapsed")
            normalized_target = (train_target[subset] - target_mean) / target_scale
            relation, relation_loo = leave_one_out_relation(
                train_side[subset], normalized_target, relation_alpha
            )
            relation_train = relation.predict(train_side)
            relation_evaluation = relation.predict(evaluation_side)

            permutation = welding.derangement(
                budget, int(probe["relation_derangement_seed_base"]) + seed
            )
            permuted_relation, _ = leave_one_out_relation(
                train_side[subset], normalized_target[permutation], relation_alpha
            )
            permuted_train = permuted_relation.predict(train_side)

            predictions = {
                "relation_center": relation_evaluation * target_scale + target_mean,
            }
            for modality in ("rgb", "infrared"):
                point = welding.StandardizedDualRidge(image_alpha[modality]).fit(
                    train_images[modality][subset], normalized_target
                )
                propagated = welding.StandardizedDualRidge(image_alpha[modality]).fit(
                    train_images[modality], relation_train
                )
                permuted = welding.StandardizedDualRidge(image_alpha[modality]).fit(
                    train_images[modality], permuted_train
                )
                predictions[f"point_{modality}"] = (
                    point.predict(evaluation_images[modality]) * target_scale + target_mean
                )
                predictions[f"relation_propagation_{modality}"] = (
                    propagated.predict(evaluation_images[modality]) * target_scale + target_mean
                )
                predictions[f"permuted_relation_propagation_{modality}"] = (
                    permuted.predict(evaluation_images[modality]) * target_scale + target_mean
                )

            conditions = {
                name: welding.regression_metrics(evaluation_target, prediction)
                for name, prediction in predictions.items()
            }
            comparisons = {}
            base_seed = (
                int(probe["bootstrap_seed_base"])
                + budget_index * 100
                + seed_index * 10
            )
            for modality_index, modality in enumerate(("rgb", "infrared")):
                candidate = predictions[f"relation_propagation_{modality}"]
                baselines = {
                    "point": predictions[f"point_{modality}"],
                    "permuted_relation": predictions[
                        f"permuted_relation_propagation_{modality}"
                    ],
                    "relation_center": predictions["relation_center"],
                }
                for baseline_index, (baseline_name, baseline) in enumerate(baselines.items()):
                    name = f"relation_propagation_{modality}_minus_{baseline_name}"
                    comparisons[name] = welding.bootstrap_r2_delta(
                        evaluation_target,
                        candidate,
                        baseline,
                        replicates=int(probe["bootstrap_replicates"]),
                        seed=base_seed + modality_index * 3 + baseline_index,
                    )
            runs.append(
                {
                    "budget": budget,
                    "seed": seed,
                    "subset_ids": subset_ids,
                    "subset_sha256": welding.stable_hash(",".join(map(str, subset_ids))),
                    "target_mean": target_mean,
                    "target_scale": target_scale,
                    "relation_loo_metrics": welding.regression_metrics(
                        normalized_target, relation_loo
                    ),
                    "conditions": conditions,
                    "comparisons": comparisons,
                    "evaluation_predictions": {
                        "sample_ids": evaluation_ids,
                        "target": evaluation_target.tolist(),
                        **{name: value.tolist() for name, value in predictions.items()},
                    },
                }
            )

    summary: dict[str, Any] = {}
    comparison_names = list(runs[0]["comparisons"])
    for budget in EXPECTED_BUDGETS:
        selected = [run for run in runs if run["budget"] == budget]
        summary[str(budget)] = {}
        for name in comparison_names:
            observed = np.asarray([run["comparisons"][name]["observed"] for run in selected])
            summary[str(budget)][name] = {
                "positive_seeds": int(np.sum(observed > 0.0)),
                "ci95_positive_seeds": int(
                    sum(run["comparisons"][name]["ci95_low"] > 0.0 for run in selected)
                ),
                "mean": float(observed.mean()),
                "range": [float(observed.min()), float(observed.max())],
            }

    primary_point = all(
        summary[str(budget)]["relation_propagation_infrared_minus_point"][
            "ci95_positive_seeds"
        ]
        == 3
        for budget in (20, 40)
    )
    correct_relation = all(
        summary[str(budget)]["relation_propagation_infrared_minus_permuted_relation"][
            "ci95_positive_seeds"
        ]
        == 3
        for budget in EXPECTED_BUDGETS
    )
    passed = primary_point and correct_relation
    phase = str(config["phase"])
    if phase == "validation":
        decision = (
            "validation_relation_propagation_supported_heldout_unlocked"
            if passed
            else "validation_relation_propagation_not_supported_stop"
        )
    else:
        decision = (
            "heldout_relation_propagation_confirmed"
            if passed
            else "heldout_relation_propagation_not_confirmed"
        )
    return {
        "protocol_version": PROTOCOL,
        "status": f"completed_formal_{phase}",
        "decision": decision,
        "heldout_unlocked": phase == "validation" and passed,
        "heldout_target_evaluated": phase == "heldout",
        "evaluation_split": phase,
        "input_sha256": {name: welding.sha256_file(path) for name, path in paths.items()},
        "implementation_sha256": implementation,
        "dataset": {
            "source": "Mendeley Data 10.17632/rwh8kjzdch.3",
            "sample_unit": "weld_sample_id",
            "split_protocol": split_record["protocol"],
            "counts": split_record["counts"],
            "setting_count": split_record["setting_count"],
            "time_series_rows": int(sum(unit["time_rows"] for unit in units.values())),
        },
        "task": {
            "target": config["task"]["target"],
            "side_features": config["task"]["side_features"],
            "primary_modality": model_config["primary_modality"],
            "prediction_input_at_evaluation": "image_only_for_point_and_propagation",
        },
        "model": model_config,
        "dense_image_ceiling": dense_ceiling,
        "decision_checks": {
            "k20_k40_ir_propagation_minus_point_all_ci_positive": primary_point,
            "all_budget_ir_correct_minus_permuted_all_ci_positive": correct_relation,
        },
        "summary": summary,
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite result: {args.output}")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    validate_config(config)
    if args.output != Path(config["output"]["audit_json_path"]):
        raise ValueError("CLI output differs from the locked result path")
    paths = verify_inputs(config)
    result = evaluate(config, paths, verify_implementation(config))
    result["config"] = {
        "path": str(args.config),
        "sha256": welding.sha256_file(args.config),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"decision": result["decision"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
