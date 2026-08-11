"""Version-aware held-out audit for the frozen K=80 residual checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

import audit_bounded_unbounded_heldout_k45 as k45
import run_calibrated_relation_tube_k0 as tube
import run_conflict_projected_physics_k80_closed as v17
import run_diagnostic as base
import run_dynamic_residual_propagation_k0 as dynamic
import run_physics_functional_anchor_training as v11
import run_supervised_continuation_diagnostic as v6


PROTOCOL = "bounded_unbounded_heldout_k45_v2"
SEEDS = (0, 42, 3407)
CONDITIONS = ("unbounded_correct", "tube_correct")
SOURCE_FILES = (
    "audit_bounded_unbounded_heldout_k45_v2.py",
    "audit_bounded_unbounded_heldout_k45.py",
    "run_calibrated_relation_tube_k0.py",
    "run_conflict_projected_physics_k80_closed.py",
    "run_dynamic_residual_propagation_k0.py",
    "run_supervised_continuation_diagnostic.py",
    "run_diagnostic.py",
)


def source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    return {name: base.sha256_file(root / name) for name in SOURCE_FILES}


def validate_contract(master: dict[str, Any], config_path: Path) -> None:
    if master["protocol_version"] != PROTOCOL:
        raise ValueError("unexpected K45-v2 protocol")
    evaluation = master["evaluation"]
    if tuple(int(value) for value in evaluation["seeds"]) != SEEDS:
        raise ValueError("K45-v2 seed registry changed")
    if tuple(evaluation["conditions"]) != CONDITIONS:
        raise ValueError("K45-v2 condition registry changed")
    if (int(evaluation["test_start"]), int(evaluation["test_end"])) != (9000, 10000):
        raise ValueError("K45-v2 held-out rows changed")
    if evaluation["training_allowed"] is not False:
        raise ValueError("K45-v2 cannot train")
    if evaluation["selection_allowed"] is not False:
        raise ValueError("K45-v2 cannot select a residual geometry")
    if evaluation["test_evaluation_authorized"] is not True:
        raise ValueError("K45-v2 held-out read is not authorized")
    if master["implementation"]["entrypoint_sha256"] != base.sha256_file(
        Path(__file__).resolve()
    ):
        raise ValueError("K45-v2 entrypoint hash mismatch")
    if master["implementation"]["failed_predecessor_config_sha256"] != base.sha256_file(
        Path(master["implementation"]["failed_predecessor_config"])
    ):
        raise ValueError("K45-v2 predecessor config hash mismatch")
def git_blob(repository: Path, commit: str, relative_path: str) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(repository), "show", f"{commit}:{relative_path}"],
        check=True,
        capture_output=True,
    )
    return completed.stdout


def resolve_locked_sources(
    repository: Path, entry: dict[str, Any], locked: dict[str, str]
) -> dict[str, Any]:
    scripts = Path(__file__).resolve().parent
    commit = entry["source_commit"]
    resolved: dict[str, Any] = {}
    for name, expected in sorted(locked.items()):
        current = scripts / name
        if current.is_file() and base.sha256_file(current) == expected:
            resolved[name] = {"source": "current_tree", "sha256": expected}
            continue
        relative = f"scripts/{name}"
        snapshot_root = entry.get("source_snapshot_root")
        if (repository / ".git").exists():
            payload = git_blob(repository, commit, relative)
            source = {"source": "git_commit", "commit": commit, "path": relative}
        elif snapshot_root is not None:
            snapshot = Path(snapshot_root) / name
            payload = snapshot.read_bytes()
            source = {"source": "registered_snapshot", "path": str(snapshot)}
        else:
            raise ValueError(f"no registered historical source is available for {name}")
        observed = hashlib.sha256(payload).hexdigest()
        if observed != expected:
            raise ValueError(
                f"locked source unavailable for {name}: expected {expected}, got {observed}"
            )
        resolved[name] = {**source, "sha256": observed}
    return resolved


def semantic_metrics_close(
    observed: dict[str, Any], expected: dict[str, Any], atol: float
) -> bool:
    scalar_keys = ("mean_direct_abs_correlation", "mean_r2", "mse")
    vector_keys = ("direct_abs_correlation", "r2")
    return all(
        np.isclose(float(observed[key]), float(expected[key]), atol=atol, rtol=0.0)
        for key in scalar_keys
    ) and all(
        np.allclose(
            np.asarray(observed[key], dtype=np.float64),
            np.asarray(expected[key], dtype=np.float64),
            atol=atol,
            rtol=0.0,
        )
        for key in vector_keys
    )


def load_compatible_model(
    master: dict[str, Any],
    entry: dict[str, Any],
    condition: str,
    seed: int,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any], dict[str, Any]]:
    config_path = Path(entry["config_path"])
    if base.sha256_file(config_path) != entry["config_sha256"]:
        raise ValueError(f"K45-v2 seed {seed} config hash mismatch")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    tube.validate_config(config)
    if int(config["training"]["seed"]) != seed:
        raise ValueError("K45-v2 seed/config mismatch")
    if tuple(int(config["split"][key]) for key in ("train_end", "validation_end", "test_end")) != (8000, 9000, 10000):
        raise ValueError("K45-v2 source split changed")

    root = Path(entry["output_root"])
    payload_path = root / "preflight" / "tube_preflight.json"
    preflight_lock_path = root / "preflight" / "preflight_lock.json"
    if base.sha256_file(preflight_lock_path) != entry["preflight_lock_sha256"]:
        raise ValueError(f"K45-v2 seed {seed} preflight lock hash mismatch")
    preflight = json.loads(payload_path.read_text(encoding="utf-8"))
    preflight_lock = json.loads(preflight_lock_path.read_text(encoding="utf-8"))
    preflight_checks = {
        "status": preflight_lock["status"] == "locked_train_only_tube_geometry",
        "protocol": preflight_lock["protocol_version"] == config["protocol_version"],
        "config": preflight_lock["config_sha256"] == entry["config_sha256"],
        "payload": preflight_lock["payload_sha256"] == base.sha256_file(payload_path),
        "validity": all(preflight["checks"].values()),
        "test_unread": preflight_lock["test_evaluated"] is False,
    }
    if not all(preflight_checks.values()):
        raise ValueError(f"invalid K45-v2 preflight seed={seed}: {preflight_checks}")
    source_resolution = resolve_locked_sources(
        Path(master["implementation"]["repository_root"]),
        entry,
        preflight_lock["source_files_sha256"],
    )

    lock_path = root / "formal" / condition / "training_lock.json"
    if base.sha256_file(lock_path) != entry["training_lock_sha256"][condition]:
        raise ValueError(f"K45-v2 seed {seed} {condition} lock hash mismatch")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock_checks = {
        "formal": lock["mode"] == "formal",
        "status": lock["status"] == "locked_before_joint_semantic_validation_readout",
        "condition": lock["condition"] == condition,
        "config": lock["config_sha256"] == entry["config_sha256"],
        "source": lock["source_files_sha256"] == preflight_lock["source_files_sha256"],
        "preflight": lock["preflight_lock_sha256"] == entry["preflight_lock_sha256"],
        "validity": all(lock["validity"].values()),
        "test_unread_at_lock": lock["test_evaluated"] is False,
    }
    if not all(lock_checks.values()):
        raise ValueError(
            f"invalid K45-v2 lock seed={seed} condition={condition}: {lock_checks}"
        )

    checkpoint_path = Path(lock["checkpoint"]["path"])
    if base.sha256_file(checkpoint_path) != lock["checkpoint"]["sha256"]:
        raise ValueError("K45-v2 checkpoint hash mismatch")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if (
        checkpoint["protocol_version"] != config["protocol_version"]
        or checkpoint["condition"] != condition
        or int(checkpoint["epoch"]) != 100
    ):
        raise ValueError("K45-v2 checkpoint identity mismatch")
    model = tube.model_for_condition(condition, config, preflight, device)
    model.load_state_dict(checkpoint["state_dict"])
    audit = {
        "preflight_checks": preflight_checks,
        "lock_checks": lock_checks,
        "source_resolution": source_resolution,
        "training_lock_sha256": base.sha256_file(lock_path),
        "checkpoint_sha256": lock["checkpoint"]["sha256"],
    }
    return model, config, preflight, audit


def reproduce_validation(
    model: torch.nn.Module,
    config: dict[str, Any],
    entry: dict[str, Any],
    condition: str,
    device: torch.device,
    atol: float,
) -> dict[str, Any]:
    readout_path = Path(entry["validation_readout_path"])
    if base.sha256_file(readout_path) != entry["validation_readout_sha256"]:
        raise ValueError("K45-v2 validation readout hash mismatch")
    archived = json.loads(readout_path.read_text(encoding="utf-8"))
    images = np.load(Path(config["dataset"]["cache_path"]), mmap_mode="r")
    raw_latents = base.load_latents(config)
    subset, _ = dynamic.build_subset(config)
    latents, _, _ = v17.normalize_with_k80_rgb_statistics(
        raw_latents, int(config["split"]["train_end"]), subset
    )
    start = int(config["split"]["train_end"])
    end = int(config["split"]["validation_end"])
    prediction = v6.predict_rgb(
        model, images, latents, start, end, int(config["training"]["batch_size"]), device
    )
    observed = v6.regression_metrics(prediction, latents[0, start:end, :3])
    expected = archived["semantic_validation"][condition]
    reproduced = semantic_metrics_close(observed, expected, atol)
    if not reproduced:
        raise ValueError(f"K45-v2 validation reproduction failed for {condition}")
    return {
        "archived_readout_sha256": entry["validation_readout_sha256"],
        "atol": atol,
        "reproduced": True,
        "observed": observed,
        "expected": expected,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config_path = args.config.resolve()
    master = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_contract(master, config_path)
    torch.set_num_threads(int(master["runtime"]["num_threads"]))
    torch.set_float32_matmul_precision("high")
    device = torch.device(
        master["runtime"]["device"] if torch.cuda.is_available() else "cpu"
    )
    output = Path(args.output or master["runtime"]["output_path"]).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite K45-v2 result: {output}")

    runs: dict[str, Any] = {}
    atol = float(master["evaluation"]["validation_reproduction_atol"])
    for seed in SEEDS:
        entry = k45.keyed(master["upstream"], seed)
        runs[str(seed)] = {}
        for condition in CONDITIONS:
            model, config, _, provenance = load_compatible_model(
                master, entry, condition, seed, device
            )
            validation = reproduce_validation(
                model, config, entry, condition, device, atol
            )
            run = k45.evaluate_model(model, config, master, device)
            run["provenance"] = provenance
            run["validation_reproduction"] = validation
            runs[str(seed)][condition] = run
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    result = {
        "protocol_version": PROTOCOL,
        "mode": "version_aware_post_lock_k80_bounded_unbounded_heldout_readout",
        "config_path": str(config_path),
        "config_sha256": base.sha256_file(config_path),
        "source_files_sha256": source_hashes(),
        "runs": runs,
        "comparison": k45.summarize(runs),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "training_performed": False,
        "selection_performed": False,
        "test_rows": [
            int(master["evaluation"]["test_start"]),
            int(master["evaluation"]["test_end"]),
        ],
        "test_evaluated": True,
        "failed_predecessor_preserved": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    v11.atomic_json(output, result)
    print(json.dumps({"result_path": str(output), "sha256": base.sha256_file(output)}))


if __name__ == "__main__":
    main()
