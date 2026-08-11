"""Anchor-conditioned FiLM recovery pilot for partial-anchor CRL v3."""

from __future__ import annotations

import argparse
import copy
import json
import math
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml

import run_clamped_anchor_diagnostic as v2
import run_diagnostic as base


class AnchorConditionedEncoder(nn.Module):
    """Official visual bottleneck with identity-initialized anchor FiLM."""

    def __init__(
        self,
        learned_dim: int,
        anchor_dim: int,
        hidden_channels: int,
        conv_layers: int,
    ):
        super().__init__()
        blocks = []
        for index in range(conv_layers):
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(
                        3 if index == 0 else hidden_channels,
                        hidden_channels,
                        3,
                        2,
                        1,
                        bias=False,
                    ),
                    nn.GroupNorm(num_groups=8, num_channels=hidden_channels),
                    nn.SiLU(),
                    nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1, bias=False),
                    nn.GroupNorm(num_groups=8, num_channels=hidden_channels),
                    nn.SiLU(),
                )
            )
        spatial = 64 // (2**conv_layers)
        bottleneck_dim = 16 * hidden_channels
        self.visual = nn.Sequential(
            *blocks,
            nn.Flatten(),
            nn.Linear(spatial * spatial * hidden_channels, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.SiLU(),
        )
        self.film = nn.Linear(anchor_dim, 2 * bottleneck_dim)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.output = nn.Linear(bottleneck_dim, learned_dim)
        self.bottleneck_dim = bottleneck_dim

    def forward_unconditioned(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.visual(x))

    def forward(self, x: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        visual = self.visual(x)
        gamma, beta = self.film(anchors).chunk(2, dim=1)
        conditioned = visual * (1.0 + gamma) + beta
        return self.output(conditioned)

    def identity_initialization_error(self) -> float:
        return float(
            max(
                self.film.weight.detach().abs().max().cpu(),
                self.film.bias.detach().abs().max().cpu(),
            )
        )


class FilmConditionedModel(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        learned_indices: list[int],
        anchor_indices: list[int],
        hidden_channels: int,
        conv_layers: int,
    ):
        super().__init__()
        if sorted(learned_indices + anchor_indices) != list(range(latent_dim)):
            raise ValueError("coordinate blocks must partition latent space")
        self.latent_dim = latent_dim
        self.learned_indices = list(learned_indices)
        self.anchor_indices = list(anchor_indices)
        self.embedding = AnchorConditionedEncoder(
            len(learned_indices), len(anchor_indices), hidden_channels, conv_layers
        )
        self.parametric_part = base.ParametricPart(latent_dim)

    def compose_z(self, learned: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        columns: list[torch.Tensor | None] = [None] * self.latent_dim
        for source, target in enumerate(self.learned_indices):
            columns[target] = learned[:, source]
        for source, target in enumerate(self.anchor_indices):
            columns[target] = anchors[:, source]
        if any(column is None for column in columns):
            raise RuntimeError("incomplete coordinate composition")
        return torch.stack([column for column in columns if column is not None], dim=1)

    def get_z(self, x: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        return self.compose_z(self.embedding(x, anchors), anchors)

    def forward(
        self, x: torch.Tensor, anchors: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.get_z(x, anchors)
        return self.parametric_part(z, targets), z


def validate_config(config: dict[str, Any]) -> None:
    model = config["model"]
    learned = list(model["learned_indices"])
    anchors = list(model["anchor_indices"])
    if sorted(learned + anchors) != list(range(int(model["latent_dim"]))):
        raise ValueError("coordinate blocks do not partition latent space")
    if list(config["training"]["targets"]) != learned:
        raise ValueError("targets must equal learned coordinates")
    if list(config["training"]["conditions"]) != [
        "oracle_conditioned",
        "shuffled_conditioned",
    ]:
        raise ValueError("unexpected v3 condition order")
    if model["film_initialization"] != "identity_zero":
        raise ValueError("v3 requires identity-zero FiLM initialization")


def conditions_from_config(config: dict[str, Any]) -> list[v2.Condition]:
    mapping = {
        "oracle_conditioned": v2.Condition("oracle_conditioned", "oracle"),
        "shuffled_conditioned": v2.Condition("shuffled_conditioned", "shuffled"),
    }
    return [mapping[name] for name in config["training"]["conditions"]]


def train_condition(
    condition: v2.Condition,
    images: np.ndarray,
    latents: np.ndarray,
    anchor_maps: dict[str, np.ndarray],
    config: dict[str, Any],
    device: torch.device,
    epochs: int,
    output_dir: Path,
) -> tuple[FilmConditionedModel, dict[str, Any]]:
    training = config["training"]
    model_cfg = config["model"]
    seed = int(training["seed"])
    base.set_seed(seed)
    model = FilmConditionedModel(
        int(model_cfg["latent_dim"]),
        list(model_cfg["learned_indices"]),
        list(model_cfg["anchor_indices"]),
        int(model_cfg["hidden_channels"]),
        int(model_cfg["conv_layers"]),
    ).to(device)
    initial_identity_error = model.embedding.identity_initialization_error()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training["learning_rate"]))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=float(training["scheduler_factor"]),
        patience=int(training["scheduler_patience"]),
    )
    rows, targets = base.make_pairs(
        list(training["targets"]), 0, int(config["split"]["train_end"])
    )
    batch_size = int(training["batch_size"])
    validation_interval = int(training["validation_interval"])
    best_objective = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    history = []
    started = time.time()

    for epoch in range(epochs):
        generator = np.random.default_rng(seed + 1009 * epoch)
        order = generator.permutation(len(rows))
        sums: dict[str, float] = {}
        total_count = 0
        model.train()
        for start in range(0, len(rows), batch_size):
            indices = order[start : start + batch_size]
            batch_rows = rows[indices]
            batch_targets = targets[indices]
            loss, metrics = v2.objective_for_batch(
                model,
                images,
                latents,
                anchor_maps,
                condition,
                batch_rows,
                batch_targets,
                config,
                device,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            size = len(batch_rows)
            total_count += size
            for name, value in metrics.items():
                sums[name] = sums.get(name, 0.0) + size * value

        train_metrics = base.average_metrics(sums, total_count)
        should_validate = epoch % validation_interval == 0 or epoch == epochs - 1
        record: dict[str, Any] = {"epoch": epoch + 1, "train": train_metrics}
        if should_validate:
            validation_metrics = v2.validate(
                model, images, latents, anchor_maps, condition, config, device
            )
            selection_objective = validation_metrics["total"]
            scheduler.step(selection_objective)
            record["validation"] = validation_metrics
            record["learning_rate"] = optimizer.param_groups[0]["lr"]
            if selection_objective < best_objective:
                best_objective = selection_objective
                best_epoch = epoch + 1
                best_state = copy.deepcopy(model.state_dict())
            print(
                json.dumps(
                    {
                        "condition": condition.name,
                        "epoch": epoch + 1,
                        "train_total": train_metrics["total"],
                        "validation_total": validation_metrics["total"],
                        "validation_classifier": validation_metrics["classifier"],
                        "lr": optimizer.param_groups[0]["lr"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        history.append(record)

    if best_state is None:
        raise RuntimeError(f"no validation checkpoint for {condition.name}")
    model.load_state_dict(best_state)
    checkpoint = output_dir / f"{condition.name}_seed{seed}.pt"
    torch.save(
        {
            "condition": condition.name,
            "anchor_mode": condition.anchor_mode,
            "seed": seed,
            "epoch": best_epoch,
            "state_dict": best_state,
        },
        checkpoint,
    )
    return model, {
        "condition": condition.name,
        "anchor_mode": condition.anchor_mode,
        "best_epoch": best_epoch,
        "best_validation_objective": best_objective,
        "duration_seconds": time.time() - started,
        "history": history,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": base.sha256_file(checkpoint),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "initial_film_identity_error": initial_identity_error,
    }


def decide(
    metrics: dict[str, dict[str, Any]],
    training: dict[str, dict[str, Any]],
    shuffle_checks: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    evaluation = config["evaluation"]
    oracle = metrics["oracle_conditioned"]
    shuffled = metrics["shuffled_conditioned"]
    baseline = float(evaluation["clamped_v2_oracle_rgb_mcc"])
    deltas = {
        "oracle_minus_clamped_v2_rgb_mcc": oracle["unanchored_mcc"] - baseline,
        "oracle_minus_shuffled_rgb_mcc": oracle["unanchored_mcc"]
        - shuffled["unanchored_mcc"],
    }
    validity = {
        "oracle_anchor_exact": oracle["anchor_mean_direct_r2"]
        >= float(evaluation["minimum_oracle_anchor_r2"]),
        "shuffle_bijective": bool(shuffle_checks["all_bijective"]),
        "matched_parameter_count": training["oracle_conditioned"]["parameter_count"]
        == training["shuffled_conditioned"]["parameter_count"],
        "identity_initialized": training["oracle_conditioned"][
            "initial_film_identity_error"
        ]
        == 0.0
        and training["shuffled_conditioned"]["initial_film_identity_error"] == 0.0,
    }
    success = {
        "gain_over_clamped_v2": deltas["oracle_minus_clamped_v2_rgb_mcc"]
        >= float(evaluation["minimum_gain_over_clamped_v2"]),
        "gain_over_shuffle": deltas["oracle_minus_shuffled_rgb_mcc"]
        >= float(evaluation["minimum_gain_over_shuffle"]),
        "rgb_absolute": oracle["unanchored_mcc"]
        >= float(evaluation["minimum_rgb_absolute"]),
    }
    if not all(validity.values()):
        verdict = "conditioned_film_branch_invalid"
    elif all(success.values()):
        verdict = "conditioned_film_branch_supported"
    elif any(success.values()):
        verdict = "conditioned_film_branch_partial_signal"
    else:
        verdict = "conditioned_film_branch_not_supported"
    return {
        "verdict": verdict,
        "validity": validity,
        "success": success,
        "deltas": deltas,
        "next_action": (
            "independent_confirmation"
            if verdict == "conditioned_film_branch_supported"
            else "block_structural_or_capacity_attribution"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    validate_config(config)
    epochs = int(args.epochs or config["training"]["epochs"])
    output_dir = (args.output_dir or Path(config["runtime"]["output_dir"])).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(config["runtime"]["num_threads"]))
    torch.set_float32_matmul_precision("high")
    device = torch.device(config["runtime"]["device"] if torch.cuda.is_available() else "cpu")

    cache_path = Path(config["dataset"]["cache_path"])
    images = np.load(cache_path, mmap_mode="r")
    expected_shape = (
        len(config["dataset"]["environments"]),
        int(config["dataset"]["samples_per_environment"]),
        64,
        64,
        3,
    )
    if images.shape != expected_shape or images.dtype != np.uint8:
        raise ValueError(f"unexpected image cache: {images.shape}, {images.dtype}")
    raw_latents = base.load_latents(config)
    latents, latent_mean, latent_std = base.normalize_latents(
        raw_latents, int(config["split"]["train_end"])
    )
    anchor_maps, shuffle_checks = v2.build_anchor_maps(config)

    models: dict[str, FilmConditionedModel] = {}
    training_summaries: dict[str, Any] = {}
    run_started = time.time()
    for condition in conditions_from_config(config):
        model, summary = train_condition(
            condition,
            images,
            latents,
            anchor_maps,
            config,
            device,
            epochs,
            output_dir,
        )
        models[condition.name] = model
        training_summaries[condition.name] = summary

    source_path = Path(__file__).resolve()
    v2_source_path = Path(v2.__file__).resolve()
    base_source_path = Path(base.__file__).resolve()
    base_result = {
        "protocol_version": config["protocol_version"],
        "mode": "smoke" if args.smoke else "formal",
        "config": config,
        "config_path": str(config_path),
        "config_sha256": base.sha256_file(config_path),
        "source_path": str(source_path),
        "source_sha256": base.sha256_file(source_path),
        "v2_source_sha256": base.sha256_file(v2_source_path),
        "base_source_sha256": base.sha256_file(base_source_path),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "data": {
            "cache_path": str(cache_path),
            "cache_shape": images.shape,
            "cache_dtype": str(images.dtype),
            "latent_mean": latent_mean,
            "latent_std": latent_std,
            "shuffle_checks": shuffle_checks,
        },
        "epochs": epochs,
        "training": training_summaries,
        "duration_seconds": time.time() - run_started,
    }
    if args.smoke:
        result = {**base_result, "verdict": "smoke_completed_no_test_read"}
        result_path = output_dir / "smoke_results.json"
    else:
        metrics = {
            condition.name: v2.evaluate_model(
                models[condition.name],
                images,
                latents,
                anchor_maps,
                condition,
                config,
                device,
            )
            for condition in conditions_from_config(config)
        }
        result = {
            **base_result,
            "metrics": metrics,
            "decision": decide(metrics, training_summaries, shuffle_checks, config),
        }
        result_path = output_dir / "diagnostic_results.json"
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(base.json_ready(result), handle, indent=2, sort_keys=True)
    print(
        json.dumps({"result_path": str(result_path), "sha256": base.sha256_file(result_path)}),
        flush=True,
    )


if __name__ == "__main__":
    main()
