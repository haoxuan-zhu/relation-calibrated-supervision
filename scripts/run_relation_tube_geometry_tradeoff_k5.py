"""Cross-seed confirmation of the K4 geometry tradeoff."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import run_relation_tube_geometry_ablation_k4 as k4


PROTOCOL = "relation_tube_geometry_tradeoff_k5_v1"
EXPECTED_ORDER = ("diagonal", "isotropic", "full_anisotropic", "rotated")
CONFIRMATION_SEEDS = (0, 42)
DEVELOPMENT_RESULT_SHA256 = (
    "048509bed8a66590b1ce56358a8b761be68dced903211f753e175439876e2bf0"
)
SOURCE_FILES = ("run_relation_tube_geometry_tradeoff_k5.py", *k4.SOURCE_FILES)
ORIGINAL_VALIDATE_CONFIG = k4.validate_config


def source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    return {name: k4.base.sha256_file(root / name) for name in SOURCE_FILES}


def validate_config(config: dict[str, Any]) -> None:
    ORIGINAL_VALIDATE_CONFIG(config)
    seed = int(config["training"]["seed"])
    if seed not in CONFIRMATION_SEEDS:
        raise ValueError("K5 is restricted to seed0/42 confirmation")
    tradeoff = config["geometry_tradeoff"]
    if tuple(tradeoff["expected_order"]) != EXPECTED_ORDER:
        raise ValueError("registered K5 tradeoff order changed")
    if tradeoff["development_result_sha256"] != DEVELOPMENT_RESULT_SHA256:
        raise ValueError("K4 development result hash changed")


def nondominated_names(runs: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for candidate in k4.GEOMETRIES:
        candidate_r2 = float(runs[candidate]["full_affine"]["mean_r2"])
        candidate_ccrl = float(runs[candidate]["ccrl_validation"]["total"])
        dominated = False
        for other in k4.GEOMETRIES:
            if other == candidate:
                continue
            other_r2 = float(runs[other]["full_affine"]["mean_r2"])
            other_ccrl = float(runs[other]["ccrl_validation"]["total"])
            weakly_better = other_r2 >= candidate_r2 and other_ccrl <= candidate_ccrl
            strictly_better = other_r2 > candidate_r2 or other_ccrl < candidate_ccrl
            if weakly_better and strictly_better:
                dominated = True
                break
        if not dominated:
            names.append(candidate)
    return names


def decide(runs: dict[str, Any]) -> dict[str, Any]:
    r2_order = tuple(
        sorted(
            k4.GEOMETRIES,
            key=lambda name: float(runs[name]["full_affine"]["mean_r2"]),
            reverse=True,
        )
    )
    ccrl_order = tuple(
        sorted(
            k4.GEOMETRIES,
            key=lambda name: float(runs[name]["ccrl_validation"]["total"]),
            reverse=True,
        )
    )
    nondominated = tuple(nondominated_names(runs))
    checks = {
        "full_affine_r2_order_exact": r2_order == EXPECTED_ORDER,
        "ccrl_total_order_exact": ccrl_order == EXPECTED_ORDER,
        "all_four_nondominated": set(nondominated) == set(k4.GEOMETRIES),
        "all_boundaries_active": all(
            float(runs[name]["representation_audit"]["boundary_active_fraction"])
            > 0.0
            for name in k4.GEOMETRIES
        ),
    }
    supported = all(checks.values())
    return {
        "verdict": (
            "geometry_tradeoff_confirmed_seed"
            if supported
            else "geometry_tradeoff_not_confirmed_seed"
        ),
        "checks": checks,
        "full_affine_r2_order": list(r2_order),
        "ccrl_total_order": list(ccrl_order),
        "nondominated_geometries": list(nondominated),
        "seed_tradeoff_supported": supported,
        "test_evaluated": False,
    }


def configure_k4_engine() -> None:
    k4.PROTOCOL = PROTOCOL
    k4.source_hashes = source_hashes
    k4.validate_config = validate_config
    k4.decide = decide


def main() -> None:
    configure_k4_engine()
    k4.main()


if __name__ == "__main__":
    main()
