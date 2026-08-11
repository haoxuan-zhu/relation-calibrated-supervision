from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

import audit_physics_functional_anchor as v10
import run_clamped_anchor_diagnostic as v2
import run_conditioned_film_diagnostic as v3
import run_diagnostic as base
import run_physics_functional_anchor_training as v11
import run_supervised_continuation_diagnostic as v6


CONSTRAINTS = ["forward_correct", "inverse_correct", "forward_permuted"]


def cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    numerator = torch.sum(a * b, dim=-1)
    denominator = torch.linalg.vector_norm(a, dim=-1) * torch.linalg.vector_norm(b, dim=-1)
    return numerator / denominator.clamp_min(eps)


def summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p10": float(np.quantile(array, 0.10)),
        "p90": float(np.quantile(array, 0.90)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "positive_fraction": float(np.mean(array > 0.0)),
    }


def forward_per_sample(
    prediction: torch.Tensor,
    images: torch.Tensor,
    raw_angles: torch.Tensor,
    coefficients: torch.Tensor,
    channel_scale: torch.Tensor,
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
) -> torch.Tensor:
    unit_rgb = (prediction * latent_std[:3] + latent_mean[:3]) / 255.0
    tau = torch.cos(torch.deg2rad(raw_angles[:, 0] - raw_angles[:, 1])).square()
    matrices = coefficients[1:4].unsqueeze(0) + tau[:, None, None] * coefficients[
        4:7
    ].unsqueeze(0)
    predicted_means = (
        torch.bmm(unit_rgb.unsqueeze(1), matrices).squeeze(1) + coefficients[0]
    )
    observed_means = images.mean(dim=(2, 3))
    return torch.mean(((predicted_means - observed_means) / channel_scale) ** 2, dim=1)


def flatten_gradients(
    loss: torch.Tensor, parameters: list[torch.nn.Parameter], retain_graph: bool
) -> torch.Tensor:
    gradients = torch.autograd.grad(loss, parameters, retain_graph=retain_graph)
    return torch.cat([gradient.reshape(-1) for gradient in gradients])


def load_state_model(
    item: dict[str, Any], config: dict[str, Any], device: torch.device
) -> v3.FilmConditionedModel:
    path = Path(item["path"])
    if base.sha256_file(path) != item["sha256"]:
        raise ValueError(f"checkpoint hash mismatch: {path}")
    model_cfg = config["model"]
    model = v3.FilmConditionedModel(
        int(model_cfg["latent_dim"]),
        list(model_cfg["learned_indices"]),
        list(model_cfg["anchor_indices"]),
        int(model_cfg["hidden_channels"]),
        int(model_cfg["conv_layers"]),
    ).to(device)
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def audit_state(
    model: v3.FilmConditionedModel,
    images: np.ndarray,
    latents: np.ndarray,
    raw_latents: np.ndarray,
    coefficients: dict[str, torch.Tensor],
    channel_scale: torch.Tensor,
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
    batches: list[tuple[np.ndarray, np.ndarray]],
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    parameter_cosines = {name: [] for name in CONSTRAINTS}
    parameter_norm_ratios = {name: [] for name in CONSTRAINTS}
    output_cosines = {name: [] for name in CONSTRAINTS}
    output_by_environment = {
        name: {environment: [] for environment in ["obs", "red", "green", "blue"]}
        for name in CONSTRAINTS
    }
    parameters = [parameter for parameter in model.embedding.parameters() if parameter.requires_grad]
    anchor_indices = list(config["model"]["anchor_indices"])
    rcond = float(config["functional_anchor"]["pinv_rcond"])
    clip = bool(config["functional_anchor"]["clip_inverse_predictions_to_unit_interval"])
    environment_names = list(config["dataset"]["environments"])
    for rows, targets in batches:
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
        oracle_per = torch.mean((prediction - truth) ** 2, dim=1)
        inverse_target = v11.physical_pseudo_target(
            x,
            raw_angles,
            coefficients["correct"],
            latent_mean,
            latent_std,
            rcond,
            clip,
        )
        per_sample = {
            "forward_correct": forward_per_sample(
                prediction,
                x,
                raw_angles,
                coefficients["correct"],
                channel_scale,
                latent_mean,
                latent_std,
            ),
            "inverse_correct": torch.mean((prediction - inverse_target) ** 2, dim=1),
            "forward_permuted": forward_per_sample(
                prediction,
                x,
                raw_angles,
                coefficients["permuted"],
                channel_scale,
                latent_mean,
                latent_std,
            ),
        }
        oracle_output_gradient = torch.autograd.grad(
            oracle_per.sum(), prediction, retain_graph=True
        )[0]
        for name in CONSTRAINTS:
            constraint_gradient = torch.autograd.grad(
                per_sample[name].sum(), prediction, retain_graph=True
            )[0]
            values = cosine(constraint_gradient, oracle_output_gradient).detach().cpu().numpy()
            output_cosines[name].extend(values.tolist())
            for index, environment in enumerate(environments):
                output_by_environment[name][environment_names[int(environment)]].append(
                    float(values[index])
                )
        oracle_parameter_gradient = flatten_gradients(
            oracle_per.mean(), parameters, retain_graph=True
        )
        oracle_norm = torch.linalg.vector_norm(oracle_parameter_gradient)
        for index, name in enumerate(CONSTRAINTS):
            gradient = flatten_gradients(
                per_sample[name].mean(),
                parameters,
                retain_graph=index < len(CONSTRAINTS) - 1,
            )
            parameter_cosines[name].append(
                float(cosine(gradient[None], oracle_parameter_gradient[None]).item())
            )
            parameter_norm_ratios[name].append(
                float((torch.linalg.vector_norm(gradient) / oracle_norm.clamp_min(1e-12)).item())
            )
    return {
        "parameter_cosine": {name: summary(values) for name, values in parameter_cosines.items()},
        "parameter_norm_ratio": {
            name: summary(values) for name, values in parameter_norm_ratios.items()
        },
        "output_cosine": {name: summary(values) for name, values in output_cosines.items()},
        "output_cosine_by_environment": {
            name: {environment: summary(values) for environment, values in groups.items()}
            for name, groups in output_by_environment.items()
        },
    }


def decide(results: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    evaluation = config["evaluation"]
    primary = results[evaluation["primary_state"]]
    parameter = primary["parameter_cosine"]
    output = primary["output_cosine"]
    forward_median = parameter["forward_correct"]["median"]
    inverse_median = parameter["inverse_correct"]["median"]
    permuted_median = parameter["forward_permuted"]["median"]
    worst_environment = min(
        item["median"]
        for item in primary["output_cosine_by_environment"]["forward_correct"].values()
    )
    gates = {
        "forward_parameter_median": forward_median
        >= float(evaluation["minimum_forward_parameter_cosine_median"]),
        "forward_parameter_p10": parameter["forward_correct"]["p10"]
        >= float(evaluation["minimum_forward_parameter_cosine_p10"]),
        "gain_over_inverse": forward_median - inverse_median
        >= float(evaluation["minimum_parameter_cosine_gain_over_inverse"]),
        "gain_over_permuted": forward_median - permuted_median
        >= float(evaluation["minimum_parameter_cosine_gain_over_permuted"]),
        "output_positive_fraction": output["forward_correct"]["positive_fraction"]
        >= float(evaluation["minimum_output_positive_fraction"]),
        "worst_environment_output_median": worst_environment
        >= float(evaluation["minimum_worst_environment_output_cosine_median"]),
        "warm_parameter_median": results["warm"]["parameter_cosine"]["forward_correct"][
            "median"
        ]
        >= float(evaluation["minimum_warm_parameter_cosine_median"]),
    }
    verdict = (
        "forward_physics_gradient_alignment_supported_v12a"
        if all(gates.values())
        else "forward_physics_gradient_alignment_not_supported_v12a"
    )
    return {
        "verdict": verdict,
        "gates": gates,
        "primary_deltas": {
            "forward_minus_inverse_parameter_cosine_median": forward_median - inverse_median,
            "forward_minus_permuted_parameter_cosine_median": forward_median - permuted_median,
        },
        "worst_environment_output_cosine_median": worst_environment,
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
    train_end = int(config["split"]["train_end"])
    latents, latent_mean_np, latent_std_np = base.normalize_latents(raw_latents, train_end)
    latent_mean = torch.from_numpy(latent_mean_np).to(device)
    latent_std = torch.from_numpy(latent_std_np).to(device)
    spec, calibration_audit = v11.build_calibration_spec(config)
    coefficients_np, coefficient_audit = v11.calibrate_teachers(
        images, raw_latents, spec, config
    )
    coefficients = {
        name: torch.from_numpy(value).to(device) for name, value in coefficients_np.items()
    }
    pooled_means = np.concatenate(
        [v10.image_channel_means(images, environment, 0, train_end) for environment in range(4)],
        axis=0,
    )
    channel_scale_np = np.std(pooled_means, axis=0).astype(np.float32)
    if np.any(channel_scale_np <= 1e-6):
        raise ValueError("degenerate image channel scale")
    channel_scale = torch.from_numpy(channel_scale_np).to(device)
    rows, targets = base.make_pairs(list(config["audit"]["pair_targets"]), 0, train_end)
    order = np.random.default_rng(int(config["audit"]["batch_seed"])).permutation(len(rows))
    batch_size = int(config["audit"]["batch_size"])
    batch_count = int(config["audit"]["batch_count"])
    selected = order[: batch_size * batch_count]
    batches = [
        (rows[selected[start : start + batch_size]], targets[selected[start : start + batch_size]])
        for start in range(0, len(selected), batch_size)
    ]
    if len(batches) != batch_count:
        raise ValueError("unexpected batch count")
    results = {}
    state_audit = {}
    for name, item in config["states"].items():
        model = load_state_model(item, config, device)
        state_audit[name] = {
            "path": item["path"],
            "sha256": base.sha256_file(Path(item["path"])),
        }
        results[name] = audit_state(
            model,
            images,
            latents,
            raw_latents,
            coefficients,
            channel_scale,
            latent_mean,
            latent_std,
            batches,
            config,
            device,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    decision = decide(results, config)
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
        "channel_scale": channel_scale_np,
        "selected_pair_indices_sha256": v11.v9.sha256_int_array(selected),
        "results": results,
        "decision": decision,
    }
    output = Path(config["runtime"]["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    v11.atomic_json(output, result)
    print(json.dumps({"result_path": str(output), "sha256": base.sha256_file(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
