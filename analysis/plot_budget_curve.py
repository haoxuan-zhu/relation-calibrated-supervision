"""Render the locked budget curve from the machine aggregate JSON."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("SOURCE_DATE_EPOCH", "1785542400")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


mpl.rcParams["svg.hashsalt"] = "relation-supervision-crl-budget-v1"


COLORS = {
    "point": "#6B7280",
    "empirical": "#2563EB",
    "physical": "#DC2626",
    "raw_ridge": "#059669",
}
LABELS = {
    "point": "Point replay",
    "empirical": "Interaction OLS",
    "physical": "Physical relation",
    "raw_ridge": "Raw ridge",
}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def plot(aggregate: dict, raw_aggregate: dict, output_prefix: Path) -> None:
    summaries = aggregate["budget_summaries"]
    raw_summaries = raw_aggregate["raw_ridge"]["budget_summaries"]
    budgets = sorted(int(value) for value in summaries)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.35), constrained_layout=True)

    for kind in ("point", "empirical", "physical", "raw_ridge"):
        if kind == "raw_ridge":
            means = [raw_summaries[str(k)]["readout"][kind]["mean"] for k in budgets]
            errors = [
                raw_summaries[str(k)]["readout"][kind]["population_sd"]
                for k in budgets
            ]
        else:
            means = [summaries[str(k)]["readout"][kind]["test"]["mean"] for k in budgets]
            errors = [
                summaries[str(k)]["readout"][kind]["test"]["population_sd"]
                for k in budgets
            ]
        axes[0].errorbar(
            budgets,
            means,
            yerr=errors,
            marker="o",
            linewidth=2,
            capsize=3,
            color=COLORS[kind],
            label=LABELS[kind],
        )

    for key, color, label in [
        ("empirical_minus_point", COLORS["empirical"], "Interaction OLS − point"),
        ("physical_minus_point", COLORS["physical"], "Physical − point"),
    ]:
        means = [summaries[str(k)]["deltas"][key]["test"]["mean"] for k in budgets]
        errors = [
            summaries[str(k)]["deltas"][key]["test"]["population_sd"]
            for k in budgets
        ]
        axes[1].errorbar(
            budgets,
            means,
            yerr=errors,
            marker="o",
            linewidth=2,
            capsize=3,
            color=color,
            label=label,
        )
    raw_means = [
        raw_summaries[str(k)]["deltas"]["raw_minus_point"]["test"]["mean"]
        for k in budgets
    ]
    raw_errors = [
        raw_summaries[str(k)]["deltas"]["raw_minus_point"]["test"]["population_sd"]
        for k in budgets
    ]
    axes[1].errorbar(
        budgets,
        raw_means,
        yerr=raw_errors,
        marker="o",
        linewidth=2,
        capsize=3,
        color=COLORS["raw_ridge"],
        label="Raw ridge − point",
    )

    for axis in axes:
        axis.set_xscale("log", base=2)
        axis.set_xticks(budgets, labels=[str(k) for k in budgets])
        axis.grid(True, alpha=0.25)
        axis.set_xlabel("Number of labeled observations (K)")
        axis.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("Test direct RGB correlation")
    axes[0].set_title("Semantic coordinate recovery")
    axes[1].axhline(0.0, color="black", linewidth=1, alpha=0.6)
    axes[1].set_ylabel("Gain over matched point replay")
    axes[1].set_title("Value of calibrated propagation")

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        output_path = output_prefix.with_suffix(f".{suffix}")
        if suffix == "pdf":
            metadata = {
                "Title": "Relation supervision budget curve",
                "Author": "",
                "Subject": "Locked Light Tunnel budget comparison",
                "Creator": "RelationSupervision-CRL",
                "CreationDate": None,
                "ModDate": None,
            }
        elif suffix == "svg":
            metadata = {
                "Title": "Relation supervision budget curve",
                "Description": "Locked Light Tunnel budget comparison",
                "Creator": "RelationSupervision-CRL",
                "Date": None,
            }
        else:
            metadata = {"Software": "RelationSupervision-CRL"}
        fig.savefig(output_path, dpi=240, metadata=metadata)
        if suffix == "svg":
            svg_text = output_path.read_text(encoding="utf-8")
            output_path.write_text(
                "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n",
                encoding="utf-8",
                newline="\n",
            )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--raw-aggregate", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    plot(
        load(args.aggregate.resolve()),
        load(args.raw_aggregate.resolve()),
        args.output_prefix.resolve(),
    )


if __name__ == "__main__":
    main()
