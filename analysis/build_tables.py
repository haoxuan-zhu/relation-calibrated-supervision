"""Build paper auxiliary tables only from locked local JSON results."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import fmean, pstdev


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "generated"

FUNCTION_RUNS = {
    3407: ROOT / "outputs/conflict_projected_physics_k80_closed_v17_seed3407/formal/diagnostic_results.json",
    42: ROOT / "outputs/conflict_projected_physics_k80_closed_v18_seed42/formal/diagnostic_results.json",
    0: ROOT / "outputs/conflict_projected_physics_k80_closed_v18_seed0/formal/diagnostic_results.json",
}
POINT_RUNS = {
    3407: ROOT / "outputs/matched_point_control_k80_v1/formal_seed3407/diagnostic_results.json",
    42: ROOT / "outputs/matched_point_control_k80_v1/formal_seed42/diagnostic_results.json",
    0: ROOT / "outputs/matched_point_control_k80_v1/formal_seed0/diagnostic_results.json",
}
TEACHER_AUDIT = ROOT / "outputs/physics_functional_anchor_v10/physics_functional_anchor_audit.json"
SPRING_V4_AGGREGATE = (
    ROOT
    / "outputs/causalverse_spring_correctness_k40_v4_unread/aggregate_results.json"
)
BUDGET_AGGREGATE = ROOT / "outputs/relation_budget_curve_v1/aggregate/aggregate_results.json"
LOCKED_METRIC_AUDIT = (
    ROOT
    / "outputs/locked_metrics_2026-07-31/relation_budget_locked_metrics_v3.json"
)
RAW_AGGREGATE = (
    ROOT
    / "outputs/raw_ridge_propagation_aggregate_v1/raw/raw_aggregate_results.json"
)
RAW_LOCKED_METRIC_AUDIT = (
    ROOT
    / "outputs/raw_ridge_propagation_aggregate_v1/raw/raw_locked_metrics_audit.json"
)
RAW_TEACHER_AUDITS = {
    budget: ROOT
    / f"outputs/raw_ridge_propagation_aggregate_v1/raw/teacher_audits/k{budget}.json"
    for budget in (20, 40, 80, 160)
}
RAW_K80_DIAGNOSTICS = {
    seed: ROOT
    / f"outputs/raw_ridge_propagation_aggregate_v1/raw/diagnostic_results_k12b/k80_seed{seed}/diagnostic_results.json"
    for seed in (0, 42, 3407)
}
TUBE_AGGREGATE = (
    ROOT
    / "outputs/calibrated_relation_tube_multiseed_aggregate_k3/multiseed_aggregate.json"
)
TUBE_VALIDATION_READOUTS = {
    0: ROOT / "outputs/calibrated_relation_tube_k3_seed0/formal/validation_readout.json",
    42: ROOT / "outputs/calibrated_relation_tube_k3_seed42/formal/validation_readout.json",
    3407: ROOT
    / "outputs/calibrated_relation_tube_k0_seed3407/formal/validation_readout.json",
}
GEOMETRY_K5_AGGREGATE = (
    ROOT / "outputs/relation_tube_geometry_tradeoff_k5_aggregate/aggregate_results.json"
)
GEOMETRY_K6_AGGREGATE = (
    ROOT / "outputs/relation_tube_geometry_downstream_k6_aggregate/aggregate_results.json"
)
SUBSET_K8_AGGREGATE = (
    ROOT
    / "outputs/relation_tube_calibration_subset_k8_aggregate_v1/calibration_subset_aggregate.json"
)
GEOMETRY_K18_AGGREGATE = (
    ROOT
    / "outputs/relation_tube_geometry_crossed_k18/aggregate/crossed_geometry_aggregate.json"
)
OAS_K9_AGGREGATE = (
    ROOT
    / "outputs/relation_tube_oas_shrinkage_k9_aggregate_v1/oas_shrinkage_aggregate.json"
)
COUPLING_K10_AUDIT = (
    ROOT / "outputs/relation_tube_sign_coupling_k10/sign_coupling_audit.json"
)
ISOTROPIC_K11_AGGREGATE = (
    ROOT
    / "outputs/isotropic_relation_tube_budget_k11/aggregate/budget_validation_aggregate.json"
)
ISOTROPIC_K12_AUDIT = (
    ROOT
    / "outputs/isotropic_tube_vs_raw_propagation_k12/paired_validation_audit.json"
)
ISOTROPIC_K12B_AUDIT = (
    ROOT
    / "outputs/isotropic_tube_vs_raw_downstream_k12b/paired_validation_downstream_audit.json"
)
ISOTROPIC_K13_AGGREGATE = (
    ROOT
    / "outputs/isotropic_relation_tube_heldout_k13/aggregate/heldout_aggregate.json"
)
ISOTROPIC_K13_BASELINES = (
    ROOT / "outputs/isotropic_relation_tube_heldout_k13/baseline_comparator.json"
)
ANCHOR_PRESERVING_COMPARISON = (
    ROOT / "outputs/anchor_preserving_raw_ridge_v1/audit/comparison.json"
)
BOUNDED_UNBOUNDED_HELDOUT_K45_V2 = (
    ROOT / "outputs/bounded_unbounded_heldout_k45_v2/heldout_audit.json"
)
INSTANCE_K14_AUDIT = (
    ROOT
    / "outputs/relation_tube_instance_information_audit_k14/instance_information_audit.json"
)
RADIUS_K15_AGGREGATE = (
    ROOT
    / "outputs/relation_tube_radius_scale_ablation_k15/radius_scale_aggregate.json"
)
HALF_RADIUS_K16_AUDIT = (
    ROOT
    / "outputs/relation_tube_half_radius_instance_information_audit_k16/half_radius_instance_information_audit.json"
)
LABEL_NOISE_K17_AGGREGATE = (
    ROOT
    / "outputs/relation_tube_label_noise_robustness_k17/aggregate/noise_robustness_aggregate.json"
)
SLOPE_K26_HELDOUT = (
    ROOT / "outputs/causalverse_slope_heldout_k26/heldout_results.json"
)
SLOPE_K32_PROJECTION = (
    ROOT
    / "outputs/causalverse_slope_projection_parity_k32/projection_parity_results.json"
)
INSTRUMENTED_SLOPE_K33 = (
    ROOT
    / "outputs/instrumented_slope_relation_decoder_k33/evaluation/validation_results.json"
)
INSTRUMENTED_SLOPE_K35 = (
    ROOT
    / "outputs/instrumented_slope_projected_replication_k35/evaluation/validation_results.json"
)
INSTRUMENTED_SLOPE_K36 = (
    ROOT
    / "outputs/instrumented_slope_projected_external_k36/external_results.json"
)
MOVI_FORMAL_VALIDATION = (
    ROOT / "outputs/movi_sparse_collision_formal_v2/formal_validation_v3.json"
)
MOVI_FORMAL_HELDOUT = (
    ROOT / "outputs/movi_sparse_collision_formal_v2/formal_heldout_v3.json"
)
WELDING_ANCHOR_PRESERVING = (
    ROOT
    / "outputs/resistance_spot_welding_relation_k43/formal/retrospective_anchor_preserving_v1_full.json"
)
FUNCTION_CONDITIONS = {
    "vanilla_functional": "vanilla_correct_floor",
    "projected_functional": "projected_correct_floor",
    "permuted_functional": "projected_permuted_floor",
}


def load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def direct_rgb(metric: dict) -> float:
    return fmean(float(value) for value in metric["direct_abs_correlation"][:3])


def functional_row(
    seed: int,
    label: str,
    key: str,
    payload: dict,
    test_direct_rgb_r2: float,
) -> dict:
    validation = payload["post_lock_validation_semantics"][key]
    test = payload["metrics"][key]
    final_history = payload["training"][key]["history"][-1]
    return {
        "seed": seed,
        "condition": label,
        "validation_direct_rgb_corr": float(validation["mean_direct_abs_correlation"]),
        "validation_direct_rgb_r2": float(validation["mean_r2"]),
        "test_direct_rgb_corr": direct_rgb(test),
        "test_direct_rgb_r2": float(test_direct_rgb_r2),
        "test_rgb_hungarian_mcc": float(test["unanchored_mcc"]),
        "test_graph_edge_auroc": float(test["graph"]["edge_auroc"]),
        "test_graph_shd": int(test["graph"]["fixed_threshold_shd"]),
        "final_ccrl_validation_total": float(final_history["validation"]["total"]),
        "parameter_count": int(payload["training"][key]["parameter_count"]),
        "source": str(FUNCTION_RUNS[seed].relative_to(ROOT)).replace("\\", "/"),
    }


def point_row(seed: int, payload: dict, test_direct_rgb_r2: float) -> dict:
    validation = payload["post_lock_validation_semantics"]
    test = payload["metrics"]
    final_history = payload["training"]["history"][-1]
    return {
        "seed": seed,
        "condition": "matched_point",
        "validation_direct_rgb_corr": float(validation["mean_direct_abs_correlation"]),
        "validation_direct_rgb_r2": float(validation["mean_r2"]),
        "test_direct_rgb_corr": direct_rgb(test),
        "test_direct_rgb_r2": float(test_direct_rgb_r2),
        "test_rgb_hungarian_mcc": float(test["unanchored_mcc"]),
        "test_graph_edge_auroc": float(test["graph"]["edge_auroc"]),
        "test_graph_shd": int(test["graph"]["fixed_threshold_shd"]),
        "final_ccrl_validation_total": float(final_history["validation"]["total"]),
        "parameter_count": int(payload["training"]["parameter_count"]),
        "source": str(POINT_RUNS[seed].relative_to(ROOT)).replace("\\", "/"),
    }


def fmt(value: float) -> str:
    return f"{value:.6f}"


def write_text_lf(path: Path, value: str) -> None:
    path.write_text(value, encoding="utf-8", newline="\n")


def write_tex_rows(name: str, rows: list[str]) -> None:
    header = "% Generated by analysis/build_tables.py from locked machine results; do not edit.\n"
    write_text_lf(OUT / name, header + "\n".join(rows) + "\n")


def spring_rows(payload: dict) -> list[dict]:
    rows: list[dict] = []
    conditions = (
        "point",
        "correct_relation",
        "coefficient_permuted_relation",
        "wrong_lm_to_k_relation",
        "wrong_km_to_l_relation",
    )
    for seed in (0, 42, 3407):
        for condition in conditions:
            for split in ("validation", "test"):
                metrics = payload["metrics"][str(seed)][condition][split]
                rows.append(
                    {
                        "seed": seed,
                        "split": split,
                        "condition": condition,
                        "relation3_signed_pearson": float(
                            metrics["relation3_mean_signed_pearson"]
                        ),
                        "relation3_direct_r2": float(
                            metrics["relation3_mean_direct_r2"]
                        ),
                        "relation3_normalized_rmse": float(
                            metrics["relation3_mean_normalized_rmse"]
                        ),
                        "free2_abs_pearson": float(
                            metrics["mean_direct_abs_correlation_free2"]
                        ),
                        "free2_direct_r2": float(metrics["free2_mean_direct_r2"]),
                    }
                )
    return rows


def main() -> None:
    locked_metric_audit = load(LOCKED_METRIC_AUDIT)
    if not locked_metric_audit["all_source_hashes_match"]:
        raise ValueError("Locked metric audit source hashes do not match")
    if not locked_metric_audit["all_contract_checks_pass"]:
        raise ValueError("Locked metric audit contract checks did not pass")

    rows: list[dict] = []
    for seed in (3407, 42, 0):
        function_payload = load(FUNCTION_RUNS[seed])
        point_payload = load(POINT_RUNS[seed])
        point_test_r2 = locked_metric_audit["locked_metrics"]["80"]["point"][
            str(seed)
        ]["test"]["mean_direct_r2"]
        rows.append(point_row(seed, point_payload, point_test_r2))
        for label, key in FUNCTION_CONDITIONS.items():
            test_r2 = locked_metric_audit["k80_four_cell_metrics"][key][str(seed)][
                "test"
            ]["mean_direct_r2"]
            rows.append(
                functional_row(seed, label, key, function_payload, test_r2)
            )

    OUT.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (OUT / "condition_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Locked condition metrics",
        "",
        "Generated by `analysis/build_tables.py`; do not edit manually.",
        "",
        "| seed | condition | val direct corr | val direct R² | test direct corr | test Hungarian MCC | edge AUROC | SHD | final CCRL val |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {seed} | {condition} | {vc} | {vr2} | {tc} | {tm} | {ga} | {shd} | {cv} |".format(
                seed=row["seed"],
                condition=row["condition"],
                vc=fmt(row["validation_direct_rgb_corr"]),
                vr2=fmt(row["validation_direct_rgb_r2"]),
                tc=fmt(row["test_direct_rgb_corr"]),
                tm=fmt(row["test_rgb_hungarian_mcc"]),
                ga=fmt(row["test_graph_edge_auroc"]),
                shd=row["test_graph_shd"],
                cv=fmt(row["final_ccrl_validation_total"]),
            )
        )

    lines.extend(["", "## Three-seed summaries", ""])
    lines.append("| condition | test direct corr mean ± population SD |")
    lines.append("|---|---:|")
    for condition in ("matched_point", "vanilla_functional", "projected_functional", "permuted_functional"):
        values = [float(row["test_direct_rgb_corr"]) for row in rows if row["condition"] == condition]
        lines.append(f"| {condition} | {fmean(values):.6f} ± {pstdev(values):.6f} |")

    write_text_lf(OUT / "condition_metrics.md", "\n".join(lines) + "\n")

    condition_names = {
        "matched_point": "Point replay",
        "vanilla_functional": "Physical relation",
        "projected_functional": "Projected relation",
        "permuted_functional": "Permuted relation",
    }
    primary_rows = []
    diagnostic_rows = []
    for index, row in enumerate(rows):
        prefix = f'{row["seed"]} & {condition_names[row["condition"]]}'
        primary_rows.append(
            f'{prefix} & {fmt(row["validation_direct_rgb_corr"])} & '
            f'{fmt(row["validation_direct_rgb_r2"])} & '
            f'{fmt(row["test_direct_rgb_corr"])} & '
            f'{fmt(row["test_direct_rgb_r2"])} \\\\'
        )
        diagnostic_rows.append(
            f'{prefix} & {fmt(row["test_rgb_hungarian_mcc"])} & '
            f'{fmt(row["test_graph_edge_auroc"])} & {row["test_graph_shd"]} & '
            f'{fmt(row["final_ccrl_validation_total"])} \\\\'
        )
        if index in (3, 7):
            primary_rows.append(r"\addlinespace[2pt]")
            diagnostic_rows.append(r"\addlinespace[2pt]")
    write_tex_rows("condition_metrics_primary_rows.tex", primary_rows)
    write_tex_rows("condition_metrics_diagnostic_rows.tex", diagnostic_rows)

    teacher = load(TEACHER_AUDIT)
    raw_teacher_by_budget = {
        budget: load(path)["teachers"]["post_fit_validation"]["raw_ridge"]
        for budget, path in RAW_TEACHER_AUDITS.items()
    }
    teacher_lines = [
        "# Calibration-teacher budget audit",
        "",
        "Generated from the locked v10 validation-only audit and the frozen raw-ridge follow-up. "
        "The empirical inverse uses `[1,m,mτ]→RGB`; raw ridge uses `[1,m,τ]→RGB`.",
        "",
        "| K | affine forward | diagonal Malus | full Malus | matched empirical inverse | raw ridge |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for budget in (20, 80, 800, 8000):
        block = teacher["results"][str(budget)]["correct_labels"]
        values = []
        for key in (
            "affine_forward",
            "diagonal_malus_forward",
            "full_malus_forward",
            "matched_interaction_inverse",
        ):
            values.append(float(block[key]["pooled_inverse"]["mean_direct_correlation"]))
        raw_value = raw_teacher_by_budget.get(budget)
        raw_text = (
            f"{float(raw_value['mean_direct_correlation']):.6f}"
            if raw_value is not None
            else "--"
        )
        teacher_lines.append(
            f"| {budget} | {values[0]:.6f} | {values[1]:.6f} | {values[2]:.6f} | {values[3]:.6f} | {raw_text} |"
        )
    teacher_lines.extend(
        [
            "",
            "## K80 direct R²",
            "",
            "| teacher | pooled mean direct R² |",
            "|---|---:|",
            f"| full Malus | {float(teacher['results']['80']['correct_labels']['full_malus_forward']['pooled_inverse']['mean_direct_r2']):.6f} |",
            f"| matched empirical inverse | {float(teacher['results']['80']['correct_labels']['matched_interaction_inverse']['pooled_inverse']['mean_direct_r2']):.6f} |",
        ]
    )
    write_text_lf(OUT / "teacher_budget.md", "\n".join(teacher_lines) + "\n")

    teacher_tex_rows = []
    for budget in (20, 80, 800, 8000):
        block = teacher["results"][str(budget)]["correct_labels"]
        values = [
            float(block[key]["pooled_inverse"]["mean_direct_correlation"])
            for key in (
                "affine_forward",
                "diagonal_malus_forward",
                "full_malus_forward",
                "matched_interaction_inverse",
            )
        ]
        raw_value = raw_teacher_by_budget.get(budget)
        raw_text = (
            fmt(float(raw_value["mean_direct_correlation"]))
            if raw_value is not None
            else "--"
        )
        teacher_tex_rows.append(
            f"{budget} & {fmt(values[0])} & {fmt(values[1])} & "
            f"{fmt(values[2])} & {fmt(values[3])} & {raw_text} \\\\"
        )
    write_tex_rows("teacher_budget_rows.tex", teacher_tex_rows)

    spring = load(SPRING_V4_AGGREGATE)
    if not spring["valid"]:
        raise ValueError("Spring v4 aggregate is not valid")
    spring_table_rows = spring_rows(spring)
    spring_fields = list(spring_table_rows[0])
    with (OUT / "spring_unread_replication.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=spring_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(spring_table_rows)

    spring_lines = [
        "# CausalVerse Spring: shards fixed before download",
        "",
        "Generated by `analysis/build_tables.py` from the locked v4 aggregate; do not edit manually.",
        "",
        f"Formal decision: `{spring['machine_decision']}`; valid: `{str(spring['valid']).lower()}`.",
        "",
        "| seed | split | condition | relation-3 signed Pearson | relation-3 direct R² | relation-3 normalized RMSE | free-2 abs. Pearson | free-2 direct R² |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in spring_table_rows:
        spring_lines.append(
            "| {seed} | {split} | {condition} | {pearson} | {r2} | {nrmse} | {free2_pearson} | {free2_r2} |".format(
                seed=row["seed"],
                split=row["split"],
                condition=row["condition"],
                pearson=fmt(row["relation3_signed_pearson"]),
                r2=fmt(row["relation3_direct_r2"]),
                nrmse=fmt(row["relation3_normalized_rmse"]),
                free2_pearson=fmt(row["free2_abs_pearson"]),
                free2_r2=fmt(row["free2_direct_r2"]),
            )
        )

    spring_lines.extend(
        [
            "",
            "## Internally frozen R² contrasts",
            "",
            "| seed | split | correct−point | correct−coefficient-permuted |",
            "|---:|---|---:|---:|",
        ]
    )
    for seed in (0, 42, 3407):
        for split in ("validation", "test"):
            deltas = spring["deltas"][str(seed)][split]
            spring_lines.append(
                "| {seed} | {split} | {point} | {coefficient} |".format(
                    seed=seed,
                    split=split,
                    point=fmt(deltas["correct_minus_point"]["relation3_r2"]),
                    coefficient=fmt(
                        deltas["correct_minus_coefficient_permuted_relation"][
                            "relation3_r2"
                        ]
                    ),
                )
            )
    write_text_lf(OUT / "spring_unread_replication.md", "\n".join(spring_lines) + "\n")

    spring_names = {
        "point": "Point",
        "correct_relation": "Correct relation",
        "coefficient_permuted_relation": "Coefficient-permuted",
        "wrong_lm_to_k_relation": r"Wrong $lm\!\to\!k$",
        "wrong_km_to_l_relation": r"Wrong $km\!\to\!l$",
    }
    spring_tex_rows = []
    for index, row in enumerate(spring_table_rows):
        spring_tex_rows.append(
            f'{row["seed"]} & {row["split"].title()} & {spring_names[row["condition"]]} & '
            f'{fmt(row["relation3_signed_pearson"])} & '
            f'{fmt(row["relation3_direct_r2"])} & '
            f'{fmt(row["relation3_normalized_rmse"])} \\\\'
        )
        if index in (9, 19):
            spring_tex_rows.append(r"\addlinespace[2pt]")
    write_tex_rows("spring_full_rows.tex", spring_tex_rows)

    spring_summary_rows = []
    for condition in spring_names:
        for split in ("validation", "test"):
            selected = [
                row
                for row in spring_table_rows
                if row["condition"] == condition and row["split"] == split
            ]
            metrics = []
            for key in (
                "relation3_signed_pearson",
                "relation3_direct_r2",
                "relation3_normalized_rmse",
            ):
                values = [row[key] for row in selected]
                metrics.append(f"{fmean(values):.6f}\\pm{pstdev(values):.6f}")
            spring_summary_rows.append(
                f"{split.title()} & {spring_names[condition]} & "
                f"${metrics[0]}$ & ${metrics[1]}$ & ${metrics[2]}$ \\\\"
            )
        spring_summary_rows.append(r"\addlinespace[2pt]")
    spring_summary_rows.pop()
    write_tex_rows("spring_summary_rows.tex", spring_summary_rows)

    spring_summary_compact_rows = []
    for condition in spring_names:
        cells = []
        for split in ("validation", "test"):
            selected = [
                row
                for row in spring_table_rows
                if row["condition"] == condition and row["split"] == split
            ]
            for key in (
                "relation3_signed_pearson",
                "relation3_direct_r2",
                "relation3_normalized_rmse",
            ):
                values = [row[key] for row in selected]
                cells.append(
                    "$" + f"{fmean(values):.3f}\\pm{pstdev(values):.3f}" + "$"
                )
        spring_summary_compact_rows.append(
            f'{spring_names[condition]} & ' + " & ".join(cells) + r" \\"
        )
    write_tex_rows("spring_summary_compact_rows.tex", spring_summary_compact_rows)

    spring_main_rows = []
    spring_main_rounded_rows = []
    for seed in (0, 42, 3407):
        validation = spring["deltas"][str(seed)]["validation"]["correct_minus_point"][
            "relation3_r2"
        ]
        test = spring["deltas"][str(seed)]["test"]["correct_minus_point"][
            "relation3_r2"
        ]
        spring_main_rows.append(f"{seed} & +{validation:.6f} & +{test:.6f} \\\\")
        spring_main_rounded_rows.append(
            f"{seed} & +{validation:.3f} & +{test:.3f} \\\\"
        )
    write_tex_rows("spring_main_rows.tex", spring_main_rows)
    write_tex_rows("spring_main_rounded_rows.tex", spring_main_rounded_rows)

    spring_free2_rows = []
    spring_free2_rounded_rows = []
    for seed in (0, 42, 3407):
        validation = spring["deltas"][str(seed)]["validation"][
            "correct_minus_point_free2_pearson"
        ]
        test = spring["deltas"][str(seed)]["test"][
            "correct_minus_point_free2_pearson"
        ]
        spring_free2_rows.append(f"{seed} & +{validation:.6f} & +{test:.6f} \\\\")
        spring_free2_rounded_rows.append(
            f"{seed} & +{validation:.3f} & +{test:.3f} \\\\"
        )
    write_tex_rows("spring_free2_rows.tex", spring_free2_rows)
    write_tex_rows("spring_free2_rounded_rows.tex", spring_free2_rounded_rows)

    budget = load(BUDGET_AGGREGATE)
    raw = load(RAW_AGGREGATE)
    if raw["raw_ridge"]["decision"]["verdict"] != "generic_propagation_major_gain_supported":
        raise ValueError("Raw-ridge aggregate did not pass its frozen decision")
    budget_tex_rows = []
    budget_main_rows = []
    for value in (20, 40, 80, 160):
        summary = raw["raw_ridge"]["budget_summaries"][str(value)]
        point = summary["readout"]["point"]
        empirical = summary["readout"]["empirical"]
        physical = summary["readout"]["physical"]
        raw_ridge = summary["readout"]["raw_ridge"]
        raw_delta = summary["deltas"]["raw_minus_point"]["test"]
        budget_tex_rows.append(
            f'{value} & ${fmt(point["mean"])}\\pm{fmt(point["population_sd"])}$ & '
            f'${fmt(empirical["mean"])}\\pm{fmt(empirical["population_sd"])}$ & '
            f'${fmt(physical["mean"])}\\pm{fmt(physical["population_sd"])}$ & '
            f'$\\mathbf{{{fmt(raw_ridge["mean"])}\\pm{fmt(raw_ridge["population_sd"])}}}$ & '
            f'$\\mathbf{{+{fmt(raw_delta["mean"])}\\pm{fmt(raw_delta["population_sd"])}}}$ \\\\'
        )
    write_tex_rows("budget_summary_rows.tex", budget_tex_rows)
    for value in (20, 40, 80, 160):
        summary = raw["raw_ridge"]["budget_summaries"][str(value)]
        point = summary["readout"]["point"]
        empirical = summary["readout"]["empirical"]
        physical = summary["readout"]["physical"]
        raw_ridge = summary["readout"]["raw_ridge"]
        budget_main_rows.append(
            f'{value} & $' + f'{point["mean"]:.3f}\\pm{point["population_sd"]:.3f}'
            + '$ & $' + f'{raw_ridge["mean"]:.3f}\\pm{raw_ridge["population_sd"]:.3f}'
            + '$ & $' + f'{empirical["mean"]:.3f}\\pm{empirical["population_sd"]:.3f}'
            + '$ & $' + f'{physical["mean"]:.3f}\\pm{physical["population_sd"]:.3f}'
            + '$ \\\\'
        )
    write_tex_rows("budget_main_rows.tex", budget_main_rows)

    semantic_content_rows = []
    for seed in (0, 42, 3407):
        payload = load(FUNCTION_RUNS[seed])
        correct = direct_rgb(payload["metrics"]["projected_correct_floor"])
        permuted = direct_rgb(payload["metrics"]["projected_permuted_floor"])
        semantic_content_rows.append(
            f"{seed} & {correct:.3f} & {permuted:.3f} & "
            f"\\textbf{{+{correct - permuted:.3f}}} \\\\"
        )
    write_tex_rows("semantic_content_rows.tex", semantic_content_rows)

    raw_locked_metric_audit = load(RAW_LOCKED_METRIC_AUDIT)
    if not raw_locked_metric_audit["stored_correlation_reproduced"]:
        raise ValueError("Raw-ridge locked metric audit did not reproduce correlation")
    if not all(raw_locked_metric_audit["source_integrity"].values()):
        raise ValueError("Raw-ridge locked metric audit source integrity failed")
    budget_r2_tex_rows = []
    for value in (20, 40, 80, 160):
        summary = raw_locked_metric_audit["scale_sensitive_summaries"][str(value)]
        point = summary["point"]["test"]
        empirical = summary["empirical"]["test"]
        physical = summary["physical"]["test"]
        raw_ridge = summary["raw_ridge"]["test"]
        budget_r2_tex_rows.append(
            f'{value} & ${fmt(point["mean"])}\\pm{fmt(point["population_sd"])}$ & '
            f'${fmt(empirical["mean"])}\\pm{fmt(empirical["population_sd"])}$ & '
            f'${fmt(physical["mean"])}\\pm{fmt(physical["population_sd"])}$ & '
            f'$\\mathbf{{{fmt(raw_ridge["mean"])}\\pm{fmt(raw_ridge["population_sd"])}}}$ \\\\'
        )
    write_tex_rows("budget_r2_rows.tex", budget_r2_tex_rows)

    tube = load(TUBE_AGGREGATE)
    if tube["decision"]["verdict"] != "relation_tube_multiseed_structure_supported":
        raise ValueError("Calibrated relation tube aggregate did not pass its frozen decision")
    if tube["test_evaluated"]:
        raise ValueError("Calibrated relation tube aggregate unexpectedly contains test results")

    tube_names = {
        "center_only_correct": "Center only",
        "projected_physical_correct": "Projected physical",
        "unbounded_correct": "Unbounded residual",
        "tube_permuted_matched": "Tube, permuted center",
        "tube_correct": r"\textbf{Calibrated tube}",
    }
    tube_conditions = tuple(tube_names)
    tube_validation_payloads = {
        seed: load(path) for seed, path in TUBE_VALIDATION_READOUTS.items()
    }
    tube_readouts = {
        seed: payload["ccrl_validation"]
        for seed, payload in tube_validation_payloads.items()
    }

    tube_seed_rows = []
    tube_csv_rows: list[dict] = []
    for seed in (0, 42, 3407):
        runs = tube["seeds"][str(seed)]["runs"]
        projected = float(runs["projected_physical_correct"]["full_affine"]["mean_r2"])
        tube_r2 = float(runs["tube_correct"]["full_affine"]["mean_r2"])
        tube_seed_rows.append(
            f"{seed} & {projected:.3f} & {tube_r2:.3f} & "
            f"\\textbf{{+{tube_r2 - projected:.3f}}} \\\\"
        )
        for condition in tube_conditions:
            result = runs[condition]
            ccrl = tube_readouts[seed].get(condition, {}).get("total")
            tube_csv_rows.append(
                {
                    "seed": seed,
                    "condition": condition,
                    "raw_direct_correlation": float(
                        result["raw"]["mean_direct_abs_correlation"]
                    ),
                    "raw_direct_r2": float(result["raw"]["mean_r2"]),
                    "full_affine_direct_correlation": float(
                        result["full_affine"]["mean_direct_abs_correlation"]
                    ),
                    "full_affine_direct_r2": float(
                        result["full_affine"]["mean_r2"]
                    ),
                    "ccrl_validation_total": "" if ccrl is None else float(ccrl),
                }
            )
    write_tex_rows("tube_seed_rows.tex", tube_seed_rows)

    with (OUT / "tube_condition_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(tube_csv_rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(tube_csv_rows)

    tube_summary_rows = []
    for condition in tube_conditions:
        selected = [row for row in tube_csv_rows if row["condition"] == condition]
        raw_corr = [float(row["raw_direct_correlation"]) for row in selected]
        raw_r2 = [float(row["raw_direct_r2"]) for row in selected]
        full_corr = [float(row["full_affine_direct_correlation"]) for row in selected]
        full_r2 = [float(row["full_affine_direct_r2"]) for row in selected]
        ccrl = [
            float(row["ccrl_validation_total"])
            for row in selected
            if row["ccrl_validation_total"] != ""
        ]
        ccrl_text = (
            f"${fmean(ccrl):.3f}\\pm{pstdev(ccrl):.3f}$" if ccrl else "--"
        )
        tube_summary_rows.append(
            f'{tube_names[condition]} & ${fmean(raw_corr):.3f}\\pm{pstdev(raw_corr):.3f}$ & '
            f'${fmean(raw_r2):.3f}\\pm{pstdev(raw_r2):.3f}$ & '
            f'${fmean(full_corr):.3f}\\pm{pstdev(full_corr):.3f}$ & '
            f'${fmean(full_r2):.3f}\\pm{pstdev(full_r2):.3f}$ & '
            f'{ccrl_text} \\\\'
        )
    write_tex_rows("tube_summary_rows.tex", tube_summary_rows)

    # K13 tables come from the sealed K8--K13 records.
    k11 = load(ISOTROPIC_K11_AGGREGATE)
    k12 = load(ISOTROPIC_K12_AUDIT)
    k12b = load(ISOTROPIC_K12B_AUDIT)
    k13 = load(ISOTROPIC_K13_AGGREGATE)
    k13_baselines = load(ISOTROPIC_K13_BASELINES)
    anchor_comparison = load(ANCHOR_PRESERVING_COMPARISON)["comparison"]
    if k11["decision"]["test_evaluated"]:
        raise ValueError("K11 unexpectedly contains test results")
    if k12["overall"]["raw_correlation_strict_positive_count"] != 12:
        raise ValueError("K12 no longer has 12/12 correlation support")
    if k12["overall"]["raw_r2_strict_positive_count"] != 12:
        raise ValueError("K12 no longer has 12/12 R2 support")
    if k13["decision"]["raw_ridge_strict_positive_count"] != 12:
        raise ValueError("K13 no longer has 12/12 held-out support")
    if anchor_comparison["overall"]["tube_minus_anchor_correlation"]["strict_positive_count"] != 12:
        raise ValueError("Tube no longer has 12/12 correlation support over anchor-preserving propagation")
    if anchor_comparison["overall"]["tube_minus_anchor_r2"]["strict_positive_count"] != 12:
        raise ValueError("Tube no longer has 12/12 R2 support over anchor-preserving propagation")

    heldout_rows = []
    for value in (20, 40, 80, 160):
        block = k13["budgets"][str(value)]
        tube_corr = block["test_metrics"]["raw_correlation"]
        tube_r2 = block["test_metrics"]["raw_r2"]
        anchor_entries = anchor_comparison["entries"][str(value)]
        anchor_values = [
            float(anchor_entries[str(seed)]["anchor_preserving"]["correlation"])
            for seed in (0, 42, 3407)
        ]
        anchor_r2_values = [
            float(anchor_entries[str(seed)]["anchor_preserving"]["r2"])
            for seed in (0, 42, 3407)
        ]
        correlation_delta_values = [
            float(anchor_entries[str(seed)]["deltas"]["tube_minus_anchor_correlation"])
            for seed in (0, 42, 3407)
        ]
        r2_delta_values = [
            float(anchor_entries[str(seed)]["deltas"]["tube_minus_anchor_r2"])
            for seed in (0, 42, 3407)
        ]
        heldout_rows.append(
            f'{value} & ${fmean(anchor_values):.3f}\\pm{pstdev(anchor_values):.3f}$ & '
            f'$\\mathbf{{{tube_corr["mean"]:.3f}\\pm{tube_corr["population_std"]:.3f}}}$ & '
            f'$\\mathbf{{+{fmean(correlation_delta_values):.3f}\\pm{pstdev(correlation_delta_values):.3f}}}$ & '
            f'${fmean(anchor_r2_values):.3f}\\pm{pstdev(anchor_r2_values):.3f}$ & '
            f'$\\mathbf{{{tube_r2["mean"]:.3f}\\pm{tube_r2["population_std"]:.3f}}}$ & '
            f'$\\mathbf{{+{fmean(r2_delta_values):.3f}\\pm{pstdev(r2_delta_values):.3f}}}$ '
            + r"\\"
        )
    write_tex_rows("isotropic_heldout_rows.tex", heldout_rows)

    # A post-training affine readout answers a different question from direct
    # named-coordinate evaluation.  Keep both in one locked K=80 table so the
    # distinction is visible rather than inferred from separate appendices.
    raw_k80_runs = {seed: load(path) for seed, path in RAW_K80_DIAGNOSTICS.items()}
    posthoc_rows = []
    baseline_names = {
        "point": "Point replay",
        "empirical": "Interaction OLS",
        "physical": "Physical relation",
    }
    for condition, label in baseline_names.items():
        direct_corr = [
            float(
                k13_baselines["entries"]["80"][condition][str(seed)][
                    "mean_direct_abs_rgb_correlation"
                ]
            )
            for seed in (0, 42, 3407)
        ]
        direct_r2 = raw_locked_metric_audit["scale_sensitive_summaries"]["80"][
            condition
        ]["test"]
        affine_r2 = [
            float(
                budget["entries"]["80"][condition][str(seed)][
                    "test_linear_readout_mean_r2"
                ]
            )
            for seed in (0, 42, 3407)
        ]
        posthoc_rows.append(
            f'{label} & ${fmean(direct_corr):.3f}\\pm{pstdev(direct_corr):.3f}$ & '
            f'${direct_r2["mean"]:.3f}\\pm{direct_r2["population_sd"]:.3f}$ & '
            f'${fmean(affine_r2):.3f}\\pm{pstdev(affine_r2):.3f}$ ' + r"\\"
        )

    raw_corr = [
        float(
            k13_baselines["entries"]["80"]["raw_ridge"][str(seed)][
                "mean_direct_abs_rgb_correlation"
            ]
        )
        for seed in (0, 42, 3407)
    ]
    raw_r2 = raw_locked_metric_audit["scale_sensitive_summaries"]["80"][
        "raw_ridge"
    ]["test"]
    raw_affine = [
        float(raw_k80_runs[seed]["metrics"]["linear_readout_mean_r2"])
        for seed in (0, 42, 3407)
    ]
    posthoc_rows.append(
        f'Raw propagation & ${fmean(raw_corr):.3f}\\pm{pstdev(raw_corr):.3f}$ & '
        f'${raw_r2["mean"]:.3f}\\pm{raw_r2["population_sd"]:.3f}$ & '
        f'$\\mathbf{{{fmean(raw_affine):.3f}\\pm{pstdev(raw_affine):.3f}}}$ ' + r"\\"
    )

    tube_corr = k13["budgets"]["80"]["test_metrics"]["raw_correlation"]
    tube_r2 = k13["budgets"]["80"]["test_metrics"]["raw_r2"]
    tube_affine = k13["budgets"]["80"]["test_metrics"]["full_affine_r2"]
    posthoc_rows.append(
        f'Isotropic tube & $\\mathbf{{{tube_corr["mean"]:.3f}\\pm{tube_corr["population_std"]:.3f}}}$ & '
        f'$\\mathbf{{{tube_r2["mean"]:.3f}\\pm{tube_r2["population_std"]:.3f}}}$ & '
        f'${tube_affine["mean"]:.3f}\\pm{tube_affine["population_std"]:.3f}$ ' + r"\\"
    )
    write_tex_rows("k80_direct_vs_affine_rows.tex", posthoc_rows)

    heldout_pair_rows = []
    for value in (20, 40, 80, 160):
        for seed in (0, 42, 3407):
            anchor_entry = anchor_comparison["entries"][str(value)][str(seed)]
            propagation = float(anchor_entry["anchor_preserving"]["correlation"])
            run = k13["budgets"][str(value)]["seeds"][str(seed)]
            tube = float(run["test"]["raw_correlation"])
            gain = float(anchor_entry["deltas"]["tube_minus_anchor_correlation"])
            propagation_r2 = float(anchor_entry["anchor_preserving"]["r2"])
            tube_r2 = float(run["test"]["raw_r2"])
            heldout_pair_rows.append(
                f"{value} & {seed} & {propagation:.3f} & {tube:.3f} & {gain:+.3f} & "
                f"{propagation_r2:.3f} & {tube_r2:.3f} & {tube_r2 - propagation_r2:+.3f} "
                + r"\\"
            )
        if value != 160:
            heldout_pair_rows.append(r"\addlinespace[2pt]")
    write_tex_rows("isotropic_heldout_pair_rows.tex", heldout_pair_rows)

    validation_gain_rows = []
    for value in (20, 40, 80, 160):
        semantic = k12["budgets"][str(value)]["summary"]
        downstream = k12b["budgets"][str(value)]["summary"]
        validation_gain_rows.append(
            f'{value} & +{semantic["raw_correlation"]["delta"]["mean"]:.3f} & '
            f'+{semantic["raw_r2"]["delta"]["mean"]:.3f} & '
            f'{downstream["edge_auroc_delta"]["mean"]:+.3f} & '
            f'{downstream["shd_delta"]["mean"]:+.2f} & '
            f'{downstream["joint_graph_improvement_count"]}/3 ' + r"\\"
        )
    write_tex_rows("isotropic_validation_gain_rows.tex", validation_gain_rows)

    downstream_seed_rows = []
    for value in (20, 40, 80, 160):
        for seed in (0, 42, 3407):
            run = k12b["budgets"][str(value)]["seeds"][str(seed)]
            delta = run["tube_minus_raw_representation"]
            downstream_seed_rows.append(
                f'{value} & {seed} & {delta["edge_auroc"]:+.3f} & '
                f'{delta["shd"]:+.0f} & {delta["ccrl_total"]:+.3f} & '
                f'{"yes" if run["joint_graph_improvement"] else "no"} ' + r"\\"
            )
        if value != 160:
            downstream_seed_rows.append(r"\addlinespace[2pt]")
    write_tex_rows("isotropic_graph_seed_rows.tex", downstream_seed_rows)

    k5 = load(GEOMETRY_K5_AGGREGATE)
    k6 = load(GEOMETRY_K6_AGGREGATE)

    k3 = load(TUBE_AGGREGATE)
    bounded_control_rows = []
    condition_sources = (
        ("Center only", lambda seed: k3["seeds"][str(seed)]["runs"]["center_only_correct"]),
        ("Unbounded", lambda seed: k3["seeds"][str(seed)]["runs"]["unbounded_correct"]),
        ("Bounded full", lambda seed: k3["seeds"][str(seed)]["runs"]["tube_correct"]),
        ("Isotropic", lambda seed: k5["seeds"][str(seed)]["runs"]["isotropic"]),
    )
    for seed in (0, 42, 3407):
        for label, source in condition_sources:
            run = source(seed)
            bounded_control_rows.append(
                f'{seed} & {label} & {run["raw"]["mean_direct_abs_correlation"]:.3f} & '
                f'{run["raw"]["mean_r2"]:.3f} & {run["full_affine"]["mean_r2"]:.3f} '
                + r"\\"
            )
        if seed != 3407:
            bounded_control_rows.append(r"\addlinespace[2pt]")
    write_tex_rows("bounded_control_rows.tex", bounded_control_rows)

    heldout_bound = load(BOUNDED_UNBOUNDED_HELDOUT_K45_V2)
    if heldout_bound["training_performed"] or heldout_bound["selection_performed"]:
        raise ValueError("K45-v2 unexpectedly trained or selected a condition")
    for seed in (0, 42, 3407):
        for condition in ("unbounded_correct", "tube_correct"):
            if not heldout_bound["runs"][str(seed)][condition]["validation_reproduction"]["reproduced"]:
                raise ValueError("K45-v2 did not reproduce an archived validation readout")
    for readout in ("raw", "coordinatewise_affine", "full_affine"):
        counts = heldout_bound["comparison"]["direction_counts"][readout]
        if any(value != 3 for value in counts.values()):
            raise ValueError(f"K45-v2 lost directional support for {readout}")
    heldout_bound_rows = []
    for seed in (0, 42, 3407):
        runs = heldout_bound["runs"][str(seed)]
        unbounded = runs["unbounded_correct"]["semantic"]
        bounded = runs["tube_correct"]["semantic"]
        heldout_bound_rows.append(
            f'{seed} & {unbounded["raw"]["mean_direct_abs_correlation"]:.3f} & '
            f'{bounded["raw"]["mean_direct_abs_correlation"]:.3f} & '
            f'{bounded["raw"]["mean_direct_abs_correlation"] - unbounded["raw"]["mean_direct_abs_correlation"]:+.3f} & '
            f'{unbounded["full_affine"]["mean_r2"]:.3f} & '
            f'{bounded["full_affine"]["mean_r2"]:.3f} & '
            f'{bounded["full_affine"]["mean_r2"] - unbounded["full_affine"]["mean_r2"]:+.3f} '
            + r"\\"
        )
    write_tex_rows("bounded_unbounded_heldout_rows.tex", heldout_bound_rows)

    geometry_names = {
        "full_anisotropic": "Full covariance",
        "diagonal": "Diagonal",
        "isotropic": "Isotropic (selected)",
        "rotated": "Rotated control",
    }
    geometry_rows = []
    for geometry in geometry_names:
        raw_corr = [
            float(k5["seeds"][str(seed)]["runs"][geometry]["raw"]["mean_direct_abs_correlation"])
            for seed in (0, 42, 3407)
        ]
        raw_r2 = [
            float(k5["seeds"][str(seed)]["runs"][geometry]["raw"]["mean_r2"])
            for seed in (0, 42, 3407)
        ]
        summary = k6["geometry_summary"][geometry]
        geometry_rows.append(
            f'{geometry_names[geometry]} & {fmean(raw_corr):.3f} & {fmean(raw_r2):.3f} & '
            f'{summary["coordinatewise_affine_r2"]["mean"]:.3f} & '
            f'{summary["edge_auroc"]["mean"]:.3f} & '
            f'{summary["fixed_threshold_shd"]["mean"]:.2f} & '
            f'{summary["ccrl_total"]["mean"]:.3f} ' + r"\\"
        )
    write_tex_rows("isotropic_geometry_rows.tex", geometry_rows)

    k8 = load(SUBSET_K8_AGGREGATE)
    subset_labels = {
        "20260811": "A",
        "20260821": "B",
        "20260831": "C",
    }
    subset_rows = []
    for subset in ("20260811", "20260821", "20260831"):
        delta = k8["comparisons"]["isotropic"]["subsets"][subset]
        semantic = delta["semantic_deltas"]
        subset_rows.append(
            f'{subset_labels[subset]} & {semantic["raw_correlation"]:+.3f} & '
            f'{semantic["raw_r2"]:+.3f} & '
            f'{semantic["coordinatewise_r2"]:+.3f} & '
            f'{semantic["full_affine_r2"]:+.3f} & '
            f'{delta["edge_auroc_delta"]:+.3f} & '
            f'{delta["full_minus_candidate_shd"]:+d} ' + r"\\"
        )
    write_tex_rows("isotropic_subset_rows.tex", subset_rows)

    k18 = load(GEOMETRY_K18_AGGREGATE)
    if k18["test_evaluated"]:
        raise ValueError("Crossed geometry audit read test data")
    if k18["decision"]["verdict"] != "isotropic_crossed_directional_support_mixed":
        raise ValueError("Crossed geometry verdict drifted")
    crossed_rows = []
    for subset in ("20260811", "20260821", "20260831"):
        summary = k18["by_subset"][subset]
        correlation = summary["raw_correlation"]
        coordinatewise = summary["coordinatewise_r2"]
        crossed_rows.append(
            f'{subset_labels[subset]} & {correlation["mean"]:+.3f} & '
            f'\\multicolumn{{2}}{{c}}{{{correlation["strict_positive_count"]}/3}} & '
            f'{coordinatewise["mean"]:+.3f} & '
            f'\\multicolumn{{2}}{{c}}{{{coordinatewise["strict_positive_count"]}/3}} ' + r"\\"
        )
    write_tex_rows("axis_separable_crossed_rows.tex", crossed_rows)

    k9 = load(OAS_K9_AGGREGATE)
    oas_rows = []
    for subset in ("20260811", "20260821", "20260831"):
        delta = k9["comparisons"]["oas_minus_isotropic"]["subsets"][subset]["semantic_deltas"]
        oas_rows.append(
            f'{subset_labels[subset]} & {delta["raw_correlation"]:+.3f} & '
            f'{delta["raw_r2"]:+.3f} & {delta["coordinatewise_r2"]:+.3f} & '
            f'{delta["full_affine_r2"]:+.3f} ' + r"\\"
        )
    write_tex_rows("oas_vs_isotropic_rows.tex", oas_rows)

    k10 = load(COUPLING_K10_AUDIT)
    coupling_rows = []
    for subset in ("20260811", "20260821", "20260831"):
        audits = k10["subsets"][subset]["audits"]
        coupling_rows.append(
            f'{subset_labels[subset]} & '
            f'{audits["full_anisotropic"]["sign_orbit"]["boundary_decision_flip_fraction"]:.3f} & '
            f'{audits["oas"]["sign_orbit"]["boundary_decision_flip_fraction"]:.3f} & '
            f'{audits["full_anisotropic"]["axis_probe"]["maximum_off_diagonal_frobenius_ratio"]:.3f} & '
            f'{audits["oas"]["axis_probe"]["maximum_off_diagonal_frobenius_ratio"]:.3f} ' + r"\\"
        )
    write_tex_rows("cross_axis_coupling_rows.tex", coupling_rows)

    k14 = load(INSTANCE_K14_AUDIT)
    k15 = load(RADIUS_K15_AGGREGATE)
    k16 = load(HALF_RADIUS_K16_AUDIT)
    if k14["decision"]["verdict"] != "instance_specific_residual_signal_supported":
        raise ValueError("K14 instance-information audit no longer passes")
    if k15["decision"]["verdict"] != "alternative_radius_preferred":
        raise ValueError("K15 radius-scale verdict drifted")
    if k15["test_evaluated"]:
        raise ValueError("K15 radius-scale audit read test data")
    if k16["decision"]["verdict"] != "image_specific_signal_retained_under_half_radius":
        raise ValueError("K16 half-radius instance signal no longer passes")
    if k16["test_evaluated"]:
        raise ValueError("K16 half-radius audit read test data")

    radius_names = {
        "center_only": "Center only",
        "half_empirical": r"$0.5R_K$",
        "empirical_rank": r"Empirical $R_K$",
        "fixed_unit": "Fixed 1.0",
        "double_empirical": r"$2R_K$",
        "unbounded": "Unbounded",
    }
    radius_scale_rows = []
    for mode, label in radius_names.items():
        summary = k15["summaries"][mode]
        if mode == "center_only":
            graph_runs = [
                tube_validation_payloads[seed]["graph_parameter_metrics"][
                    "center_only_correct"
                ]
                for seed in (0, 42, 3407)
            ]
            ccrl_runs = [
                tube_validation_payloads[seed]["ccrl_validation"][
                    "center_only_correct"
                ]
                for seed in (0, 42, 3407)
            ]
        else:
            graph_runs = [
                k15["results"][str(seed)][mode]["graph"]
                for seed in (0, 42, 3407)
            ]
            ccrl_runs = [
                k15["results"][str(seed)][mode]["ccrl_validation"]
                for seed in (0, 42, 3407)
            ]
        if all(run is not None for run in graph_runs):
            edge = f'{fmean(float(run["edge_auroc"]) for run in graph_runs):.3f}'
            shd = f'{fmean(float(run["fixed_threshold_shd"]) for run in graph_runs):.2f}'
        else:
            edge = shd = "--"
        if all(run is not None for run in ccrl_runs):
            ccrl = f'{fmean(float(run["total"]) for run in ccrl_runs):.3f}'
        else:
            ccrl = "--"
        radius_scale_rows.append(
            f'{label} & {summary["raw_r2_mean"]:.3f} & '
            f'{summary["full_affine_r2_mean"]:.3f} & {edge} & {shd} & {ccrl} '
            + r"\\"
        )
    write_tex_rows("radius_scale_rows.tex", radius_scale_rows)

    half_radius_instance_rows = []
    for radius_label, audit in ((r"Empirical $R_K$", k14), (r"$0.5R_K$", k16)):
        for seed in (0, 42, 3407):
            deltas = audit["results"][str(seed)]["deltas"]
            visual = deltas["fixed_anchor_visual_derangement"]
            reassigned = deltas["projected_residual_reassignment"]
            half_radius_instance_rows.append(
                f'{radius_label} & {seed} & '
                f'{visual["correct_minus_control_output_full_affine_r2"]:+.3f} & '
                f'{visual["correct_minus_control_residual_full_affine_r2"]:+.3f} & '
                f'{reassigned["correct_minus_control_output_full_affine_r2"]:+.3f} & '
                f'{reassigned["correct_minus_control_residual_full_affine_r2"]:+.3f} '
                + r"\\"
            )
        if radius_label == r"Empirical $R_K$":
            half_radius_instance_rows.append(r"\addlinespace[2pt]")
    write_tex_rows("half_radius_instance_rows.tex", half_radius_instance_rows)

    k17 = load(LABEL_NOISE_K17_AGGREGATE)
    if k17["decision"]["verdict"] != "no_stable_paired_noise_advantage":
        raise ValueError("K17 label-noise verdict drifted")
    if k17["test_evaluated"]:
        raise ValueError("K17 label-noise audit read test data")
    label_noise_rows = []
    for key in ("sigma_0p01", "sigma_0p025", "sigma_0p05", "sigma_0p10"):
        summary = k17["summary"][key]
        label_noise_rows.append(
            f'{100 * summary["sigma_full_scale"]:.1f}\\% & '
            f'{summary["tube_minus_raw_correlation"]["mean"]:+.3f} & '
            f'{summary["raw_correlation_strict_positive_count"]}/3 & '
            f'{summary["tube_minus_raw_r2"]["mean"]:+.3f} & '
            f'{summary["raw_r2_strict_positive_count"]}/3 ' + r"\\"
        )
    write_tex_rows("label_noise_robustness_rows.tex", label_noise_rows)

    # Cross-system rows come from the locked K26/K32/K33/K35/K36 records.
    slope_k26 = load(SLOPE_K26_HELDOUT)
    if not slope_k26["valid"] or slope_k26["machine_decision"] != (
        "heldout_relation_content_and_projection_confirmed"
    ):
        raise ValueError("Slope K26 held-out result is not valid and confirmed")
    if not slope_k26["test_evaluated"]:
        raise ValueError("Slope K26 no longer records its sealed test read")

    slope_condition_names = {
        "point": "Point supervision",
        "coefficient_permuted_relation": "Permuted relation",
        "all_coordinate_manifold_projection": "Physical projection",
        "correct_relation": r"\textbf{Relation training}",
    }
    slope_heldout_rows = []
    for condition, label in slope_condition_names.items():
        correlations = [
            float(
                slope_k26["metrics_by_seed"][str(seed)][condition][
                    "mean_direct_abs_correlation_relation4"
                ]
            )
            for seed in (0, 42, 3407)
        ]
        r2_values = [
            float(
                slope_k26["metrics_by_seed"][str(seed)][condition][
                    "mean_direct_r2_relation4"
                ]
            )
            for seed in (0, 42, 3407)
        ]
        slope_heldout_rows.append(
            f"{label} & ${fmean(correlations):.3f}\\pm{pstdev(correlations):.3f}$ & "
            f"${fmean(r2_values):.3f}\\pm{pstdev(r2_values):.3f}$ " + r"\\"
        )
    write_tex_rows("slope_heldout_summary_rows.tex", slope_heldout_rows)

    slope_k32 = load(SLOPE_K32_PROJECTION)
    if not slope_k32["valid"] or slope_k32["machine_decision"] != (
        "published_projection_variants_beaten_all_seeds_ci"
    ):
        raise ValueError("Slope K32 projection-parity result is not valid")
    if slope_k32["test_evaluated"]:
        raise ValueError("Slope K32 unexpectedly re-read the sealed test split")
    slope_projection_rows = []
    for seed in (0, 42, 3407):
        metrics = slope_k32["metrics_by_seed"][str(seed)]
        projection = metrics["unbounded_k40_std"]
        relation = metrics["correct_relation"]
        slope_projection_rows.append(
            f'{seed} & {projection["mean_direct_abs_correlation_relation4"]:.3f} & '
            f'{relation["mean_direct_abs_correlation_relation4"]:.3f} & '
            f'{relation["mean_direct_abs_correlation_relation4"] - projection["mean_direct_abs_correlation_relation4"]:+.3f} & '
            f'{projection["mean_direct_r2_relation4"]:.3f} & '
            f'{relation["mean_direct_r2_relation4"]:.3f} & '
            f'{relation["mean_direct_r2_relation4"] - projection["mean_direct_r2_relation4"]:+.3f} '
            + r"\\"
        )
    write_tex_rows("slope_projection_parity_rows.tex", slope_projection_rows)

    slope_k33 = load(INSTRUMENTED_SLOPE_K33)
    if not slope_k33["valid"] or slope_k33["machine_decision"] != (
        "relation_propagation_only_posthoc_remains_stronger"
    ):
        raise ValueError("Instrumented Slope K33 result is not valid")
    decoder_names = {
        "roughness_point": "Roughness labels only",
        "relation_point": "All relation-linked labels",
        "k30_posthoc_curve": "Five outputs + physical projection",
    }
    decoder_rows = []
    for condition, label in decoder_names.items():
        correlations = [
            float(
                slope_k33["seed_results"][str(seed)]["metrics"][condition][
                    "mean_abs_correlation"
                ]
            )
            for seed in (0, 42, 3407)
        ]
        r2_values = [
            float(
                slope_k33["seed_results"][str(seed)]["metrics"][condition]["mean_r2"]
            )
            for seed in (0, 42, 3407)
        ]
        decoder_rows.append(
            f"{label} & ${fmean(correlations):.3f}\\pm{pstdev(correlations):.3f}$ & "
            f"${fmean(r2_values):.3f}\\pm{pstdev(r2_values):.3f}$ " + r"\\"
        )
    write_tex_rows("instrumented_slope_decoder_rows.tex", decoder_rows)

    slope_k35 = load(INSTRUMENTED_SLOPE_K35)
    slope_k36 = load(INSTRUMENTED_SLOPE_K36)
    if not slope_k35["valid"] or slope_k35["machine_decision"] != (
        "projected_objective_new_seed_replication_pass_external_unlocked"
    ):
        raise ValueError("Instrumented Slope K35 result is not valid")
    if not slope_k36["valid"] or slope_k36["machine_decision"] != (
        "projected_objective_external_confirmation_failed"
    ):
        raise ValueError("Instrumented Slope K36 result is not the frozen external failure")
    if slope_k36["sealed_k26_test_evaluated"]:
        raise ValueError("Instrumented Slope K36 unexpectedly read the sealed K26 test")
    projected_training_rows = []
    for seed in (123, 2026, 31415):
        development = slope_k35["seed_results"][str(seed)][
            "delta_projected_minus_raw"
        ]
        external = slope_k36["seed_results"][str(seed)][
            "delta_projected_minus_raw"
        ]
        projected_training_rows.append(
            f'{seed} & {development["mean_abs_correlation"]:+.3f} & '
            f'{development["mean_r2"]:+.3f} & '
            f'{external["mean_abs_correlation"]:+.3f} & '
            f'{external["mean_r2"]:+.3f} ' + r"\\"
        )
    write_tex_rows("instrumented_slope_projected_training_rows.tex", projected_training_rows)

    # Independent MOVi-B relation-supervision result. The table reports the
    # once-opened held-out split, while the validation checks below prevent a
    # held-out-only result from being rendered as a replicated pattern.
    movi_validation = load(MOVI_FORMAL_VALIDATION)
    movi_heldout = load(MOVI_FORMAL_HELDOUT)
    if (
        movi_validation["status"] != "completed_formal_validation_relation_evaluation"
        or movi_validation["heldout_target_evaluated"]
    ):
        raise ValueError("MOVi formal validation result is not frozen")
    if (
        movi_heldout["status"] != "completed_formal_heldout_relation_evaluation"
        or not movi_heldout["heldout_target_evaluated"]
    ):
        raise ValueError("MOVi formal held-out result is not frozen")
    for payload in (movi_validation, movi_heldout):
        for budget in (20, 40, 80, 160):
            comparisons = payload["budgets"][str(budget)]["comparisons"]
            for key in (
                "propagation_minus_point",
                "propagation_minus_deranged",
            ):
                if float(comparisons[key]["ci95_low"]) <= 0.0:
                    raise ValueError(f"MOVi {key} is not stable at K={budget}")
    movi_rows = []
    for budget in (20, 40, 80, 160):
        row = movi_heldout["budgets"][str(budget)]
        conditions = row["conditions"]
        comparisons = row["comparisons"]

        def interval(name: str) -> str:
            value = comparisons[name]
            return (
                f'{value["observed"]:+.3f} '
                f'[{value["ci95_low"]:.3f},{value["ci95_high"]:.3f}]'
            )

        movi_rows.append(
            f'{budget} & {conditions["center_propagation"]["mean_coordinate_r2"]:.3f} & '
            f'{interval("propagation_minus_point")} & '
            f'{interval("propagation_minus_deranged")} & '
            f'{conditions["center_only"]["mean_coordinate_r2"]:.3f} & '
            f'{interval("relation_minus_permuted")} ' + r"\\"
        )
    write_tex_rows("movi_heldout_relation_rows.tex", movi_rows)

    # The industrial table is derived from the archived retrospective result.
    # The paper describes it as a structural ablation because the original
    # held-out split had already been opened when anchor preservation was added.
    welding = load(WELDING_ANCHOR_PRESERVING)
    if (
        welding["status"] != "retrospective_after_original_heldout_read"
        or welding["decision"]
        != "retrospective_structural_rescue_observed_requires_independent_confirmation"
    ):
        raise ValueError("Welding anchor-preserving result is not the frozen retrospective audit")
    welding_rows = []
    for budget in (20, 40, 80, 160):
        row = welding["evaluations"]["heldout"]["summary"][str(budget)]
        conditions = row["conditions"]
        delta = row["comparisons"]["anchor_preserving_minus_point"]["mean"]
        welding_rows.append(
            f'{budget} & {conditions["point"]["mean_r2"]:.3f} & '
            f'{conditions["relation_propagation"]["mean_r2"]:.3f} & '
            f'{conditions["anchor_preserving"]["mean_r2"]:.3f} & '
            f'{delta:+.3f} ' + r"\\"
        )
    write_tex_rows("welding_heldout_rows.tex", welding_rows)

    welding_seed_rows = []
    run_index = {
        (int(run["budget"]), int(run["seed"])): run
        for run in welding["evaluations"]["heldout"]["runs"]
    }
    for budget in (20, 40, 80, 160):
        for seed in (0, 42, 3407):
            run = run_index[(budget, seed)]
            conditions = run["conditions"]
            comparisons = run["comparisons"]
            welding_seed_rows.append(
                f'{budget} & {seed} & {conditions["point"]["r2"]:.3f} & '
                f'{conditions["relation_propagation"]["r2"]:.3f} & '
                f'{conditions["anchor_preserving"]["r2"]:.3f} & '
                f'{comparisons["anchor_preserving_minus_point"]["observed"]:+.3f} & '
                f'{comparisons["anchor_preserving_minus_anchor_preserving_permuted"]["observed"]:+.3f} '
                + r"\\"
            )
    write_tex_rows("welding_heldout_seed_rows.tex", welding_seed_rows)


if __name__ == "__main__":
    main()
