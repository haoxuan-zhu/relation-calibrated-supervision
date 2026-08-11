"""Supervised-to-unsupervised continuation upper-bound pilot for v6."""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import math
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml

import run_clamped_anchor_diagnostic as v2
import run_conditioned_film_diagnostic as v3
import run_diagnostic as base


_INITIAL_STATE: dict[str, torch.Tensor] | None = None


def sha256_state_dict(state: dict[str, torch.Tensor]) -> str:
    buffer = io.BytesIO()
    torch.save(state, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


class InitialStateModel(v3.FilmConditionedModel):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        if _INITIAL_STATE is not None:
            self.load_state_dict(_INITIAL_STATE)


def conditions_from_config(config: dict[str, Any]) -> list[v2.Condition]:
    expected = ["random_init_ccrl", "supervised_warmstart_ccrl"]
    if list(config["training"]["conditions"]) != expected:
        raise ValueError("unexpected v6 conditions")
    return [v2.Condition(name, "oracle") for name in expected]


def regression_metrics(prediction: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    residual = np.sum((truth - prediction) ** 2, axis=0)
    total = np.sum((truth - truth.mean(axis=0)) ** 2, axis=0)
    r2 = 1.0 - residual / np.maximum(total, 1e-12)
    correlations = [
        float(abs(np.corrcoef(prediction[:, i], truth[:, i])[0, 1]))
        for i in range(truth.shape[1])
    ]
    return {
        "r2": r2,
        "mean_r2": float(r2.mean()),
        "direct_abs_correlation": correlations,
        "mean_direct_abs_correlation": float(np.mean(correlations)),
        "mse": float(np.mean((truth - prediction) ** 2)),
    }


@torch.no_grad()
def predict_rgb(
    model: v3.FilmConditionedModel,
    images: np.ndarray,
    latents: np.ndarray,
    start: int,
    end: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    outputs = []
    model.eval()
    for begin in range(start, end, batch_size):
        rows = np.arange(begin, min(begin + batch_size, end), dtype=np.int64)
        envs = np.zeros(len(rows), dtype=np.int64)
        x = base.image_batch(images, envs, rows, device)
        anchors = torch.from_numpy(np.array(latents[0, rows, 3:5], copy=True)).to(device)
        outputs.append(model.embedding(x, anchors).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def pretrain_embedding(
    images: np.ndarray,
    latents: np.ndarray,
    config: dict[str, Any],
    device: torch.device,
    epochs: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    training = config["training"]
    pretraining = config["pretraining"]
    model_cfg = config["model"]
    seed = int(training["seed"])
    base.set_seed(seed)
    model = v3.FilmConditionedModel(
        int(model_cfg["latent_dim"]),
        list(model_cfg["learned_indices"]),
        list(model_cfg["anchor_indices"]),
        int(model_cfg["hidden_channels"]),
        int(model_cfg["conv_layers"]),
    ).to(device)
    initial_state = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.Adam(
        model.embedding.parameters(), lr=float(pretraining["learning_rate"])
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=float(pretraining["scheduler_factor"]),
        patience=int(pretraining["scheduler_patience"]),
    )
    train_end = int(config["split"]["train_end"])
    validation_end = int(config["split"]["validation_end"])
    batch_size = int(pretraining["batch_size"])
    interval = int(pretraining["validation_interval"])
    best_mse = math.inf
    best_epoch = -1
    best_state = None
    history = []
    started = time.time()
    for epoch in range(epochs):
        rng = np.random.default_rng(seed + 65537 * epoch)
        order = rng.permutation(train_end)
        model.train()
        total_loss = 0.0
        for begin in range(0, train_end, batch_size):
            rows = order[begin : begin + batch_size]
            envs = np.zeros(len(rows), dtype=np.int64)
            x = base.image_batch(images, envs, rows, device)
            anchors = torch.from_numpy(np.array(latents[0, rows, 3:5], copy=True)).to(device)
            target = torch.from_numpy(np.array(latents[0, rows, :3], copy=True)).to(device)
            loss = F.mse_loss(model.embedding(x, anchors), target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += len(rows) * float(loss.detach().cpu())
        record: dict[str, Any] = {
            "epoch": epoch + 1,
            "train_mse": total_loss / train_end,
        }
        if epoch % interval == 0 or epoch == epochs - 1:
            prediction = predict_rgb(
                model, images, latents, train_end, validation_end, batch_size, device
            )
            truth = latents[0, train_end:validation_end, :3]
            metrics = regression_metrics(prediction, truth)
            scheduler.step(metrics["mse"])
            record["validation"] = metrics
            record["learning_rate"] = optimizer.param_groups[0]["lr"]
            if metrics["mse"] < best_mse:
                best_mse = metrics["mse"]
                best_epoch = epoch + 1
                best_state = copy.deepcopy(model.state_dict())
            print(
                json.dumps(
                    {
                        "phase": "supervised_pretrain",
                        "epoch": epoch + 1,
                        "train_mse": record["train_mse"],
                        "validation_mse": metrics["mse"],
                        "validation_mean_r2": metrics["mean_r2"],
                    }
                ),
                flush=True,
            )
        history.append(record)
    if best_state is None:
        raise RuntimeError("no supervised checkpoint")
    model.load_state_dict(best_state)
    validation_prediction = predict_rgb(
        model, images, latents, train_end, validation_end, batch_size, device
    )
    validation_metrics = regression_metrics(
        validation_prediction, latents[0, train_end:validation_end, :3]
    )
    return best_state, {
        "best_epoch": best_epoch,
        "best_validation_mse": best_mse,
        "best_validation_metrics": validation_metrics,
        "history": history,
        "duration_seconds": time.time() - started,
        "base_initial_state_sha256": sha256_state_dict(initial_state),
        "warm_state_sha256": sha256_state_dict(best_state),
    }


def decide(
    metrics: dict[str, dict[str, Any]],
    training: dict[str, dict[str, Any]],
    pretrain: dict[str, Any],
    pretrain_test: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    evaluation = config["evaluation"]
    control = metrics["random_init_ccrl"]
    warm = metrics["supervised_warmstart_ccrl"]
    deltas = {
        "warm_minus_random_rgb_mcc": warm["unanchored_mcc"] - control["unanchored_mcc"],
        "warm_minus_pretrain_test_mcc": warm["unanchored_mcc"]
        - pretrain_test["mean_direct_abs_correlation"],
    }
    validity = {
        "pretrain_learned_rgb": pretrain["best_validation_metrics"]["mean_r2"]
        >= float(config["pretraining"]["minimum_validation_mean_direct_r2"]),
        "anchors_exact": min(
            control["anchor_mean_direct_r2"], warm["anchor_mean_direct_r2"]
        )
        >= float(evaluation["minimum_oracle_anchor_r2"]),
        "matched_parameter_count": training["random_init_ccrl"]["parameter_count"]
        == training["supervised_warmstart_ccrl"]["parameter_count"],
    }
    success = {
        "gain_over_random": deltas["warm_minus_random_rgb_mcc"]
        >= float(evaluation["minimum_gain_over_random_control"]),
        "rgb_absolute": warm["unanchored_mcc"]
        >= float(evaluation["minimum_warmstart_rgb_mcc"]),
        "retains_pretrained_alignment": deltas["warm_minus_pretrain_test_mcc"]
        >= -float(evaluation["maximum_drop_from_pretrain_test_mcc"]),
    }
    if not all(validity.values()):
        verdict = "supervised_continuation_upper_bound_invalid"
    elif all(success.values()):
        verdict = "supervised_continuation_upper_bound_supported"
    elif any(success.values()):
        verdict = "supervised_continuation_upper_bound_partial_signal"
    else:
        verdict = "supervised_continuation_upper_bound_not_supported"
    return {
        "verdict": verdict,
        "validity": validity,
        "success": success,
        "deltas": deltas,
        "next_action": (
            "preregister_supervision_budget_release"
            if verdict == "supervised_continuation_upper_bound_supported"
            else "nonzero_decay_constraint_or_block_structural_objective"
        ),
    }


def main() -> None:
    global _INITIAL_STATE
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pretrain-epochs", type=int)
    parser.add_argument("--ccrl-epochs", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    conditions = conditions_from_config(config)
    pretrain_epochs = int(args.pretrain_epochs or config["pretraining"]["epochs"])
    ccrl_epochs = int(args.ccrl_epochs or config["training"]["epochs"])
    output_dir = (args.output_dir or Path(config["runtime"]["output_dir"])).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(config["runtime"]["num_threads"]))
    torch.set_float32_matmul_precision("high")
    device = torch.device(config["runtime"]["device"] if torch.cuda.is_available() else "cpu")
    images = np.load(Path(config["dataset"]["cache_path"]), mmap_mode="r")
    raw_latents = base.load_latents(config)
    latents, latent_mean, latent_std = base.normalize_latents(
        raw_latents, int(config["split"]["train_end"])
    )
    anchor_maps, _ = v2.build_anchor_maps(config)
    warm_state, pretrain_summary = pretrain_embedding(
        images, latents, config, device, pretrain_epochs
    )

    original_class = v3.FilmConditionedModel
    v3.FilmConditionedModel = InitialStateModel
    models = {}
    training_summaries = {}
    started = time.time()
    try:
        for condition in conditions:
            _INITIAL_STATE = (
                warm_state if condition.name == "supervised_warmstart_ccrl" else None
            )
            model, summary = v3.train_condition(
                condition,
                images,
                latents,
                anchor_maps,
                config,
                device,
                ccrl_epochs,
                output_dir,
            )
            models[condition.name] = model
            training_summaries[condition.name] = summary
    finally:
        _INITIAL_STATE = None
        v3.FilmConditionedModel = original_class

    pretrain_test_metrics = None
    if not args.smoke:
        pretrain_model = v3.FilmConditionedModel(
            5,
            [0, 1, 2],
            [3, 4],
            int(config["model"]["hidden_channels"]),
            int(config["model"]["conv_layers"]),
        ).to(device)
        pretrain_model.load_state_dict(warm_state)
        test_start = int(config["split"]["validation_end"])
        test_end = int(config["split"]["test_end"])
        pretrain_test_prediction = predict_rgb(
            pretrain_model, images, latents, test_start, test_end, 512, device
        )
        pretrain_test_metrics = regression_metrics(
            pretrain_test_prediction, latents[0, test_start:test_end, :3]
        )
    result_base = {
        "protocol_version": config["protocol_version"],
        "mode": "smoke" if args.smoke else "formal",
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
        "data": {"latent_mean": latent_mean, "latent_std": latent_std},
        "pretraining": pretrain_summary,
        "pretrain_test_metrics": pretrain_test_metrics,
        "training": training_summaries,
        "duration_seconds": time.time() - started + pretrain_summary["duration_seconds"],
    }
    if args.smoke:
        result = {**result_base, "verdict": "smoke_completed_no_test_read"}
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
            for condition in conditions
        }
        result = {
            **result_base,
            "metrics": metrics,
            "decision": decide(
                metrics, training_summaries, pretrain_summary, pretrain_test_metrics, config
            ),
        }
        result_path = output_dir / "diagnostic_results.json"
    result_path.write_text(
        json.dumps(base.json_ready(result), indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"result_path": str(result_path), "sha256": base.sha256_file(result_path)}))


if __name__ == "__main__":
    main()
