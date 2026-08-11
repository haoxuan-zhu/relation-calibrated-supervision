from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml

import audit_physics_gradient_alignment as v12a
import run_clamped_anchor_diagnostic as v2
import run_diagnostic as base
import run_physics_functional_anchor_training as v11


def project_to_nonconflicting(ccrl: torch.Tensor, physics: torch.Tensor) -> torch.Tensor:
    coefficient = torch.dot(ccrl, physics) / torch.dot(physics, physics).clamp_min(1e-12)
    return ccrl - torch.minimum(coefficient, torch.zeros_like(coefficient)) * physics


def audit_state(
    model: torch.nn.Module,
    images: np.ndarray,
    latents: np.ndarray,
    raw_latents: np.ndarray,
    anchor_maps: dict[str, np.ndarray],
    coefficients: torch.Tensor,
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
    batches: list[tuple[np.ndarray, np.ndarray]],
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    names = [
        "physics_oracle_cosine",
        "ccrl_oracle_cosine",
        "ccrl_physics_cosine",
        "vanilla_total_oracle_cosine",
        "projected_total_oracle_cosine",
        "projected_gain_over_vanilla",
        "physics_to_ccrl_norm_ratio",
        "weighted_physics_to_ccrl_norm_ratio",
    ]
    records = {name: [] for name in names}
    parameters = [parameter for parameter in model.embedding.parameters() if parameter.requires_grad]
    anchor_indices = list(config["model"]["anchor_indices"])
    rcond = float(config["functional_anchor"]["pinv_rcond"])
    clip = bool(config["functional_anchor"]["clip_inverse_predictions_to_unit_interval"])
    weight = float(config["audit"]["floor_weight"])
    for rows, targets in batches:
        ccrl_loss, _ = v2.objective_for_batch(
            model,
            images,
            latents,
            anchor_maps,
            v2.Condition("gradient_audit", "oracle"),
            rows,
            targets,
            config,
            device,
        )
        combined_rows = np.concatenate([rows, rows])
        environments = np.concatenate([np.zeros_like(rows), targets + 1])
        x = base.image_batch(images, environments, combined_rows, device)
        anchors = torch.from_numpy(
            np.array(latents[environments, combined_rows][:, anchor_indices], copy=True)
        ).to(device)
        raw_angles = torch.from_numpy(
            np.array(raw_latents[environments, combined_rows][:, anchor_indices], copy=True)
        ).to(device)
        truth = torch.from_numpy(
            np.array(latents[environments, combined_rows, :3], copy=True)
        ).to(device)
        prediction = model.embedding(x, anchors)
        oracle_loss = F.mse_loss(prediction, truth)
        pseudo = v11.physical_pseudo_target(
            x,
            raw_angles,
            coefficients,
            latent_mean,
            latent_std,
            rcond,
            clip,
        )
        physics_loss = F.mse_loss(prediction, pseudo)
        oracle = v12a.flatten_gradients(oracle_loss, parameters, retain_graph=True)
        physics = v12a.flatten_gradients(physics_loss, parameters, retain_graph=True)
        ccrl = v12a.flatten_gradients(ccrl_loss, parameters, retain_graph=False)
        vanilla = ccrl + weight * physics
        projected = project_to_nonconflicting(ccrl, physics) + weight * physics
        physics_oracle = float(v12a.cosine(physics[None], oracle[None]).item())
        ccrl_oracle = float(v12a.cosine(ccrl[None], oracle[None]).item())
        ccrl_physics = float(v12a.cosine(ccrl[None], physics[None]).item())
        vanilla_oracle = float(v12a.cosine(vanilla[None], oracle[None]).item())
        projected_oracle = float(v12a.cosine(projected[None], oracle[None]).item())
        physics_norm = torch.linalg.vector_norm(physics)
        ccrl_norm = torch.linalg.vector_norm(ccrl).clamp_min(1e-12)
        records["physics_oracle_cosine"].append(physics_oracle)
        records["ccrl_oracle_cosine"].append(ccrl_oracle)
        records["ccrl_physics_cosine"].append(ccrl_physics)
        records["vanilla_total_oracle_cosine"].append(vanilla_oracle)
        records["projected_total_oracle_cosine"].append(projected_oracle)
        records["projected_gain_over_vanilla"].append(projected_oracle - vanilla_oracle)
        records["physics_to_ccrl_norm_ratio"].append(float((physics_norm / ccrl_norm).item()))
        records["weighted_physics_to_ccrl_norm_ratio"].append(
            float((weight * physics_norm / ccrl_norm).item())
        )
    result = {name: v12a.summary(values) for name, values in records.items()}
    result["ccrl_physics_conflict_fraction"] = float(
        np.mean(np.asarray(records["ccrl_physics_cosine"]) < 0.0)
    )
    return result


def decide(results: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    evaluation = config["evaluation"]
    primary = results[evaluation["primary_state"]]
    gates = {
        "physics_oracle_alignment": primary["physics_oracle_cosine"]["median"]
        >= float(evaluation["minimum_physics_oracle_cosine_median"]),
        "ccrl_physics_opposed": primary["ccrl_physics_cosine"]["median"]
        <= float(evaluation["maximum_ccrl_physics_cosine_median"]),
        "conflict_fraction": primary["ccrl_physics_conflict_fraction"]
        >= float(evaluation["minimum_ccrl_physics_conflict_fraction"]),
        "projected_total_alignment": primary["projected_total_oracle_cosine"]["median"]
        >= float(evaluation["minimum_projected_total_oracle_cosine_median"]),
        "projected_gain": primary["projected_gain_over_vanilla"]["median"]
        >= float(evaluation["minimum_projected_gain_over_vanilla_median"]),
        "zero_state_replication": results["zero_final"]["physics_oracle_cosine"]["median"]
        >= float(evaluation["minimum_zero_state_physics_oracle_cosine_median"]),
    }
    verdict = (
        "physics_ccrl_gradient_conflict_supported_v12b"
        if all(gates.values())
        else "physics_ccrl_gradient_conflict_not_supported_v12b"
    )
    return {"verdict": verdict, "gates": gates}


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
    train_end = int(config["split"]["train_end"])
    latents, latent_mean_np, latent_std_np = base.normalize_latents(raw_latents, train_end)
    latent_mean = torch.from_numpy(latent_mean_np).to(device)
    latent_std = torch.from_numpy(latent_std_np).to(device)
    anchor_maps, _ = v2.build_anchor_maps(config)
    spec, calibration_audit = v11.build_calibration_spec(config)
    coefficients_np, coefficient_audit = v11.calibrate_teachers(
        images, raw_latents, spec, config
    )
    correct_coefficients = torch.from_numpy(coefficients_np["correct"]).to(device)
    rows, targets = base.make_pairs(list(config["training"]["targets"]), 0, train_end)
    order = np.random.default_rng(int(config["audit"]["batch_seed"])).permutation(len(rows))
    batch_size = int(config["training"]["batch_size"])
    selected = order[: batch_size * int(config["audit"]["batch_count"])]
    batches = [
        (rows[selected[start : start + batch_size]], targets[selected[start : start + batch_size]])
        for start in range(0, len(selected), batch_size)
    ]
    results = {}
    state_audit = {}
    for name, item in config["states"].items():
        model = v12a.load_state_model(item, config, device)
        state_audit[name] = {"path": item["path"], "sha256": base.sha256_file(Path(item["path"]))}
        results[name] = audit_state(
            model,
            images,
            latents,
            raw_latents,
            anchor_maps,
            correct_coefficients,
            latent_mean,
            latent_std,
            batches,
            config,
            device,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    result = {
        "protocol_version": config["protocol_version"],
        "scope": "train_only_no_validation_or_test_samples",
        "config": config,
        "config_sha256": base.sha256_file(config_path),
        "source_sha256": base.sha256_file(Path(__file__)),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "state_audit": state_audit,
        "calibration": calibration_audit,
        "teacher_coefficients": coefficient_audit,
        "selected_pair_indices_sha256": v11.v9.sha256_int_array(selected),
        "results": results,
        "decision": decide(results, config),
    }
    output = Path(config["runtime"]["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    v11.atomic_json(output, result)
    print(json.dumps({"result_path": str(output), "sha256": base.sha256_file(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
