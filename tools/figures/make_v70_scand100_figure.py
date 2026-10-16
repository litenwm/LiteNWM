#!/usr/bin/env python3
"""Create the publication figure for the frozen v70 SCAND-100 benchmark.

The figure is intentionally descriptive. It uses only frozen local artifacts and
does not make a production-safety claim.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np


# Publication/export contract: editable text in SVG/PDF and journal-scale type.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["font.size"] = 7
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["legend.frameon"] = False
plt.rcParams["xtick.major.width"] = 0.8
plt.rcParams["ytick.major.width"] = 0.8
plt.rcParams["xtick.major.size"] = 3
plt.rcParams["ytick.major.size"] = 3


COLORS = {
    "ink": "#193247",
    "muted": "#617184",
    "grid": "#D9E0E6",
    "baseline": "#484878",
    "v70": "#2A8C86",
    "v70_light": "#CBE7E4",
    "oracle": "#B4C0E4",
    "success": "#2A8C86",
    "unchanged": "#D5DADE",
    "harm": "#C84A4A",
    "dino_fill": "#DCEAF4",
    "dino_edge": "#3979AA",
    "ltwm_fill": "#EEEAF7",
    "ltwm_edge": "#6C5FA4",
    "arm_fill": "#DDF0EE",
    "arm_edge": "#2A8C86",
    "cem_fill": "#F7EEDC",
    "cem_edge": "#B27A28",
    "input_fill": "#EEF2F5",
    "input_edge": "#718294",
}

METRIC_NAMES = ["ATE", "RPE", "Endpoint", "Objective"]
METRIC_ORDER = {name: index for index, name in enumerate(METRIC_NAMES)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root containing outputs/ and artifacts/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <repo>/figures/v70_scand100).",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_evidence(repo_root: Path) -> dict:
    benchmark_dir = (
        repo_root
        / "outputs"
        / "project_strict_scand100_v70_dwm_scorerv2_benchmark_v2"
        / "scand"
    )
    summary_path = benchmark_dir / "matched_comparison_corrected.json"
    native_path = benchmark_dir / "dwm_plus_scorer_v2_native_greedy_selection.npz"
    selected_path = benchmark_dir / "selected_metrics.npz"
    contract_path = (
        repo_root
        / "artifacts"
        / "v70_release_20260723_final"
        / "dinov3-ltwm-scorer-v2-mainline-v70"
        / "V70_RUNTIME_CONTRACT.json"
    )

    for path in [summary_path, native_path, selected_path, contract_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    summary = read_json(summary_path)
    contract = read_json(contract_path)
    native = np.load(native_path)
    selected = np.load(selected_path)

    native_keys = native["record_key"].astype(str)
    selected_keys = selected["record_key"].astype(str)
    if not np.array_equal(native_keys, selected_keys):
        raise ValueError("Native-scorer and v70 records are not identically ordered")

    v70_metrics = selected["formal_v70_three_round_cem__absolute_metrics_m"]
    oracle_metrics = selected["existing_union_objective_oracle__absolute_metrics_m"]
    native_metrics = native["absolute_metrics_m"]
    if native_metrics.shape != (100, 4) or v70_metrics.shape != (100, 4):
        raise ValueError("Expected 100 states and four metrics")

    objective_gain = native_metrics[:, 3] - v70_metrics[:, 3]
    tol = 1e-7
    outcome = np.full(len(objective_gain), "unchanged", dtype="<U9")
    outcome[objective_gain > tol] = "improved"
    outcome[objective_gain < -tol] = "harm"
    counts = {
        label: int(np.sum(outcome == label))
        for label in ["improved", "unchanged", "harm"]
    }

    official_v70 = summary["methods"]["formal_v70_three_round_cem"]
    official_native = summary["methods"]["correct_dwm_plus_scorer_v2_native_greedy"]
    if counts != {"improved": 28, "unchanged": 70, "harm": 2}:
        raise ValueError(f"Unexpected per-state outcome counts: {counts}")
    if official_v70["promotions_vs_k17_incumbent"] != 30:
        raise ValueError("Unexpected v70 promotion count")
    if official_native["promotions_vs_k17_incumbent"] != 3:
        raise ValueError("Unexpected native-scorer promotion count")

    return {
        "summary": summary,
        "contract": contract,
        "record_key": native_keys,
        "native_metrics": native_metrics,
        "v70_metrics": v70_metrics,
        "oracle_metrics": oracle_metrics,
        "objective_gain": objective_gain,
        "outcome": outcome,
        "counts": counts,
    }


def write_source_data(output_dir: Path, evidence: dict) -> list[Path]:
    source_dir = output_dir / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    panel_b_path = source_dir / "panel_b_mean_errors.csv"
    method_arrays = {
        "DWM + scorer v2": evidence["native_metrics"],
        "v70": evidence["v70_metrics"],
        "Union oracle": evidence["oracle_metrics"],
    }
    with panel_b_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "panel",
                "method",
                "metric",
                "mean_error_m",
                "relative_to_native_percent",
                "reduction_vs_native_percent",
                "states",
            ],
        )
        writer.writeheader()
        native_mean = evidence["native_metrics"].mean(axis=0)
        for method, array in method_arrays.items():
            mean = array.mean(axis=0)
            for index, metric in enumerate(METRIC_NAMES):
                relative = 100.0 * mean[index] / native_mean[index]
                writer.writerow(
                    {
                        "panel": "b",
                        "method": method,
                        "metric": metric,
                        "mean_error_m": f"{mean[index]:.9f}",
                        "relative_to_native_percent": f"{relative:.6f}",
                        "reduction_vs_native_percent": f"{100.0 - relative:.6f}",
                        "states": len(array),
                    }
                )
    written.append(panel_b_path)

    panel_c_path = source_dir / "panel_c_per_state_objective.csv"
    with panel_c_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "panel",
                "record_key",
                "native_objective_m",
                "v70_objective_m",
                "native_minus_v70_m",
                "outcome",
            ],
        )
        writer.writeheader()
        for key, native_row, v70_row, gain, outcome in zip(
            evidence["record_key"],
            evidence["native_metrics"],
            evidence["v70_metrics"],
            evidence["objective_gain"],
            evidence["outcome"],
        ):
            writer.writerow(
                {
                    "panel": "c",
                    "record_key": key,
                    "native_objective_m": f"{native_row[3]:.9f}",
                    "v70_objective_m": f"{v70_row[3]:.9f}",
                    "native_minus_v70_m": f"{gain:.9f}",
                    "outcome": outcome,
                }
            )
    written.append(panel_c_path)

    panel_d_path = source_dir / "panel_d_selection_outcomes.csv"
    official_v70 = evidence["summary"]["methods"]["formal_v70_three_round_cem"]
    official_native = evidence["summary"]["methods"][
        "correct_dwm_plus_scorer_v2_native_greedy"
    ]
    with panel_d_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "panel",
                "method",
                "states",
                "promotions",
                "dual_successes",
                "unchanged",
                "harm_count",
                "coverage",
                "dual_precision",
                "harm_rate_among_promotions",
            ],
        )
        writer.writeheader()
        for method, values in [
            ("DWM + scorer v2", official_native),
            ("v70", official_v70),
        ]:
            writer.writerow(
                {
                    "panel": "d",
                    "method": method,
                    "states": values["states"],
                    "promotions": values["promotions_vs_k17_incumbent"],
                    "dual_successes": values["dual_successes"],
                    "unchanged": values["states"]
                    - values["promotions_vs_k17_incumbent"],
                    "harm_count": values["harm_count"],
                    "coverage": f"{values['coverage_vs_k17_incumbent']:.6f}",
                    "dual_precision": f"{values['dual_precision']:.6f}",
                    "harm_rate_among_promotions": f"{values['harm_rate']:.6f}",
                }
            )
    written.append(panel_d_path)

    contract_path = source_dir / "v70_runtime_contract_summary.csv"
    cem = evidence["contract"]["cem_contract"]
    with contract_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["parameter", "value"])
        writer.writeheader()
        for key in [
            "iterations",
            "population_challengers",
            "elite_count",
            "elite_fraction",
            "proposal_scale",
            "spacing_m",
            "support_quantile",
            "support_threshold",
            "candidate_level_safety_before_elite",
            "cross_round_safe_archive",
            "safe_shortfall_rule",
            "seed",
        ]:
            writer.writerow({"parameter": key, "value": cem[key]})
    written.append(contract_path)
    return written


def add_panel_label(ax: mpl.axes.Axes, label: str) -> None:
    ax.text(
        -0.055,
        1.035,
        label,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="bottom",
        color=COLORS["ink"],
    )


def rounded_box(
    ax: mpl.axes.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    *,
    fill: str,
    edge: str,
    title: str,
    lines: list[str],
) -> None:
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.1,
        edgecolor=edge,
        facecolor=fill,
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height * 0.76,
        title,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=7.1,
        fontweight="bold",
        color=edge,
    )
    ax.text(
        x + width / 2,
        y + height * 0.39,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=5.8,
        linespacing=1.25,
        color=COLORS["ink"],
    )


def arrow_between(
    ax: mpl.axes.Axes, x0: float, x1: float, y: float, label: str
) -> None:
    arrow = FancyArrowPatch(
        (x0, y),
        (x1, y),
        transform=ax.transAxes,
        arrowstyle="-|>",
        mutation_scale=9,
        linewidth=1.0,
        color=COLORS["muted"],
        shrinkA=1,
        shrinkB=1,
    )
    ax.add_patch(arrow)
    ax.text(
        (x0 + x1) / 2,
        y + 0.065,
        label,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=5.1,
        color=COLORS["muted"],
    )


def draw_architecture(ax: mpl.axes.Axes, evidence: dict) -> None:
    ax.set_axis_off()
    add_panel_label(ax, "a")
    ax.text(
        0.0,
        1.045,
        "Frozen v70 action-selection architecture",
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        color=COLORS["ink"],
        va="bottom",
    )
    ax.text(
        1.0,
        1.045,
        "candidate-conditioned prediction → safety-gated search",
        transform=ax.transAxes,
        fontsize=5.8,
        color=COLORS["muted"],
        ha="right",
        va="bottom",
    )

    y, h, w = 0.31, 0.52, 0.165
    xs = [0.005, 0.211, 0.417, 0.623, 0.829]
    rounded_box(
        ax,
        (xs[0], y),
        w,
        h,
        fill=COLORS["input_fill"],
        edge=COLORS["input_edge"],
        title="Candidate pool",
        lines=["incumbent + NoMaD ×16", "H4 waypoints", "17 actions / state"],
    )
    rounded_box(
        ax,
        (xs[1], y),
        w,
        h,
        fill=COLORS["dino_fill"],
        edge=COLORS["dino_edge"],
        title="Frozen DINOv3-L/16",
        lines=["visual patch tokens", "210-D adapted features", "no candidate ranking alone"],
    )
    rounded_box(
        ax,
        (xs[2], y),
        w,
        h,
        fill=COLORS["ltwm_fill"],
        edge=COLORS["ltwm_edge"],
        title="LTWM ×3",
        lines=["candidate-conditioned", "H1–H4 token rollout", "future-state evidence"],
    )
    rounded_box(
        ax,
        (xs[3], y),
        w,
        h,
        fill=COLORS["arm_fill"],
        edge=COLORS["arm_edge"],
        title="Arm-D ensemble",
        lines=["predicted ATE / objective", "dual-gain probability", "harm probability"],
    )
    rounded_box(
        ax,
        (xs[4], y),
        w,
        h,
        fill=COLORS["cem_fill"],
        edge=COLORS["cem_edge"],
        title="Safe CEM + archive",
        lines=["safety before elite", "best safe cross-round action", "fallback: unchanged"],
    )
    for start, end, label in zip(
        [xs[i] + w for i in range(4)],
        [xs[i + 1] for i in range(4)],
        ["encode", "predict", "score", "optimize"],
    ):
        arrow_between(ax, start + 0.004, end - 0.004, y + h / 2, label)

    cem = evidence["contract"]["cem_contract"]
    badges = [
        f"{cem['iterations']} iterations",
        f"{cem['population_challengers']} challengers",
        f"{cem['elite_count']} elites",
        f"support q={cem['support_quantile']:.3f}",
        "safe archive",
    ]
    x_positions = np.linspace(0.21, 0.90, len(badges))
    for x, text in zip(x_positions, badges):
        ax.text(
            x,
            0.105,
            text,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=5.4,
            color=COLORS["ink"],
            bbox={
                "boxstyle": "round,pad=0.28",
                "facecolor": "white",
                "edgecolor": COLORS["grid"],
                "linewidth": 0.7,
            },
        )
    ax.text(
        0.005,
        0.105,
        "fixed contract",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=5.5,
        fontweight="bold",
        color=COLORS["muted"],
    )


def draw_mean_errors(ax: mpl.axes.Axes, evidence: dict) -> None:
    add_panel_label(ax, "b")
    ax.set_title(
        "Mean trajectory error on frozen SCAND H4-100",
        loc="left",
        fontsize=7.2,
        fontweight="bold",
        color=COLORS["ink"],
        pad=4,
    )
    native = evidence["native_metrics"].mean(axis=0)
    v70 = evidence["v70_metrics"].mean(axis=0)
    oracle = evidence["oracle_metrics"].mean(axis=0)
    x = np.arange(len(METRIC_NAMES))
    width = 0.23
    bar_specs = [
        ("DWM + scorer v2", native, COLORS["baseline"], ""),
        ("v70", v70, COLORS["v70"], "//"),
        ("Union oracle", oracle, COLORS["oracle"], ".."),
    ]
    for offset, (label, values, color, hatch) in zip(
        [-width, 0.0, width], bar_specs
    ):
        ax.bar(
            x + offset,
            values,
            width=width * 0.94,
            label=label,
            color=color,
            edgecolor="white",
            linewidth=0.45,
            hatch=hatch,
            zorder=3,
        )

    reduction = 100.0 * (native - v70) / native
    for xpos, value, percent in zip(x, v70, reduction):
        ax.text(
            xpos,
            value + 0.008,
            f"−{percent:.1f}%",
            ha="center",
            va="bottom",
            fontsize=5.2,
            fontweight="bold",
            color=COLORS["v70"],
        )
    ax.set_xticks(x)
    ax.set_xticklabels(METRIC_NAMES)
    ax.set_ylabel("Mean error (m) ↓")
    ax.set_ylim(0, max(native) * 1.22)
    ax.set_yticks([0.0, 0.1, 0.2])
    ax.yaxis.grid(True, color=COLORS["grid"], linewidth=0.55, zorder=0)
    ax.tick_params(axis="x", length=0)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.00),
        ncol=3,
        fontsize=5.2,
        handlelength=1.5,
        columnspacing=0.8,
    )
    ax.text(
        0.99,
        0.02,
        "n = 100 states",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.2,
        color=COLORS["muted"],
    )


def draw_objective_scatter(ax: mpl.axes.Axes, evidence: dict) -> None:
    add_panel_label(ax, "c")
    ax.set_title(
        "Paired per-state objective",
        loc="left",
        fontsize=7.2,
        fontweight="bold",
        color=COLORS["ink"],
        pad=4,
    )
    x = evidence["native_metrics"][:, 3]
    y = evidence["v70_metrics"][:, 3]
    outcome = evidence["outcome"]
    upper = float(max(np.max(x), np.max(y)) * 1.05)
    ax.plot([0, upper], [0, upper], "--", color=COLORS["muted"], lw=0.8, zorder=1)
    plot_specs = [
        ("unchanged", COLORS["unchanged"], "s", 13, 0.75, "Unchanged (70)"),
        ("improved", COLORS["success"], "o", 18, 0.9, "Improved (28)"),
        ("harm", COLORS["harm"], "X", 28, 1.0, "Worse (2)"),
    ]
    for state, color, marker, size, alpha, label in plot_specs:
        mask = outcome == state
        ax.scatter(
            x[mask],
            y[mask],
            s=size,
            c=color,
            marker=marker,
            alpha=alpha,
            edgecolors="white" if state != "harm" else COLORS["harm"],
            linewidths=0.45,
            label=label,
            zorder=3 if state != "harm" else 4,
        )
    ax.set_xlim(0, upper)
    ax.set_ylim(0, upper)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("DWM + scorer v2 objective (m)")
    ax.set_ylabel("v70 objective (m)")
    ax.text(
        upper * 0.72,
        upper * 0.60,
        "v70 better",
        fontsize=5.2,
        color=COLORS["success"],
        rotation=45,
        ha="center",
        va="center",
    )
    ax.legend(
        loc="upper left",
        fontsize=5.0,
        handletextpad=0.35,
        labelspacing=0.25,
        borderaxespad=0.25,
    )


def draw_selection_outcomes(ax: mpl.axes.Axes, evidence: dict) -> None:
    add_panel_label(ax, "d")
    ax.set_title(
        "Promotion outcomes and safety boundary",
        loc="left",
        fontsize=7.2,
        fontweight="bold",
        color=COLORS["ink"],
        pad=4,
    )
    summary = evidence["summary"]["methods"]
    native = summary["correct_dwm_plus_scorer_v2_native_greedy"]
    v70 = summary["formal_v70_three_round_cem"]
    rows = [
        (
            "DWM + scorer v2",
            [
                native["dual_successes"],
                native["states"] - native["promotions_vs_k17_incumbent"],
                native["harm_count"],
            ],
            native,
        ),
        (
            "v70",
            [
                v70["dual_successes"],
                v70["states"] - v70["promotions_vs_k17_incumbent"],
                v70["harm_count"],
            ],
            v70,
        ),
    ]
    y = np.array([1, 0])
    left = np.zeros(2)
    colors = [COLORS["success"], COLORS["unchanged"], COLORS["harm"]]
    labels = ["Dual success", "Unchanged", "Harm"]
    hatches = ["//", "", ".."]
    for category_index, (color, label, hatch) in enumerate(
        zip(colors, labels, hatches)
    ):
        values = np.array([row[1][category_index] for row in rows])
        bars = ax.barh(
            y,
            values,
            left=left,
            height=0.42,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            hatch=hatch,
            label=label,
        )
        for bar, value in zip(bars, values):
            if value >= 2:
                text_color = "white" if category_index in [0, 2] else COLORS["ink"]
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + bar.get_height() / 2,
                    f"{int(value)}",
                    ha="center",
                    va="center",
                    fontsize=5.6,
                    fontweight="bold",
                    color=text_color,
                )
        left += values

    ax.set_xlim(0, 100)
    ax.set_yticks(y)
    ax.set_yticklabels([row[0] for row in rows])
    ax.set_xlabel("States (n = 100)")
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.xaxis.grid(True, color=COLORS["grid"], linewidth=0.55, zorder=0)
    ax.tick_params(axis="y", length=0)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.37),
        ncol=3,
        fontsize=5.0,
        handlelength=1.5,
        columnspacing=0.8,
    )
    ax.text(
        0.0,
        -0.12,
        "Coverage",
        transform=ax.transAxes,
        fontsize=5.1,
        color=COLORS["muted"],
        ha="left",
        va="top",
    )
    ax.text(
        0.28,
        -0.12,
        f"{native['coverage_vs_k17_incumbent']:.0%} → {v70['coverage_vs_k17_incumbent']:.0%}",
        transform=ax.transAxes,
        fontsize=5.4,
        fontweight="bold",
        color=COLORS["ink"],
        ha="left",
        va="top",
    )
    ax.text(
        0.60,
        -0.12,
        f"v70 precision {v70['dual_precision']:.1%}; harm {v70['harm_rate']:.1%}",
        transform=ax.transAxes,
        fontsize=5.1,
        color=COLORS["muted"],
        ha="left",
        va="top",
    )


def build_figure(evidence: dict) -> mpl.figure.Figure:
    fig = plt.figure(figsize=(7.2, 6.7), facecolor="white")
    grid = fig.add_gridspec(
        3,
        12,
        height_ratios=[1.45, 1.05, 1.02],
        hspace=0.52,
        wspace=0.75,
        left=0.075,
        right=0.985,
        top=0.965,
        bottom=0.105,
    )
    ax_a = fig.add_subplot(grid[0, :])
    ax_b = fig.add_subplot(grid[1:, :6])
    ax_c = fig.add_subplot(grid[1, 7:])
    ax_d = fig.add_subplot(grid[2, 7:])

    draw_architecture(ax_a, evidence)
    draw_mean_errors(ax_b, evidence)
    draw_objective_scatter(ax_c, evidence)
    draw_selection_outcomes(ax_d, evidence)

    fig.text(
        0.075,
        0.027,
        "Frozen SCAND H4-100 descriptive evaluation. Lower error is better. "
        "The release explicitly disallows a production-safety claim.",
        ha="left",
        va="bottom",
        fontsize=5.5,
        color=COLORS["muted"],
    )
    return fig


def write_notes(output_dir: Path, evidence: dict) -> list[Path]:
    contract_note = output_dir / "figure_contract.md"
    contract_note.write_text(
        """# Figure contract

Core conclusion: On the frozen SCAND H4-100 benchmark, the candidate-conditioned
v70 pipeline reduces mean trajectory error relative to the corrected native
DWM + scorer-v2 baseline, but two harmful promotions keep the safety evidence
development-only.

Figure archetype: schematic-led composite.

Target/output: double-column journal figure, 183 mm wide; editable SVG and PDF;
600 dpi TIFF/PNG preview.

Backend: Python (matplotlib) exclusively.

Panel map:

- a: frozen v70 data flow and fixed CEM/safety contract.
- b: mean ATE, RPE, endpoint, and objective errors.
- c: paired per-state objective comparison.
- d: promotion coverage, dual successes, unchanged states, and harm.

Statistics: descriptive means over 100 frozen states. No inferential test or
confidence interval is claimed. The paired state counts are 28 improved,
70 unchanged, and 2 worse for objective.

Reviewer risk: the 100 states come from 43 trajectories and are not presented
as independent biological/statistical replicates; the union oracle is a
non-deployable headroom reference; production safety is not claimed.
""",
        encoding="utf-8",
    )

    legend_note = output_dir / "figure_legend.md"
    legend_note.write_text(
        """# Fig. 1 | Frozen v70 architecture and descriptive SCAND-100 performance

**a,** The v70 action-selection pipeline combines an incumbent and 16 NoMaD
challengers with frozen DINOv3-L/16 features, three candidate-conditioned LTWM
rollouts, an Arm-D ensemble predicting trajectory quality and risk, and a
three-iteration safety-gated cross-entropy search with a cross-round safe
archive. Candidate-level safety is applied before the four-member elite update;
if the safe set is insufficient, the previous proposal is retained.
**b,** Mean unaligned H1–H4 trajectory errors for the corrected native DWM +
scorer-v2 baseline, formal v70 three-round CEM, and a non-deployable union
objective oracle. Labels above v70 bars give the relative reduction from the
native baseline. **c,** Paired objective errors for all 100 frozen SCAND states;
points below the identity line favour v70. **d,** Selection outcomes relative
to the K17 incumbent. Native scoring promotes 3/100 states, all dual successes;
v70 promotes 30/100, comprising 28 dual successes and 2 harmful changes
(93.3% dual precision; 6.7% harm among promotions). Values are descriptive
means/counts from the frozen benchmark; no inferential test is claimed. The
release contract explicitly disallows a production-safety claim. Source data
are provided in the accompanying CSV files.
""",
        encoding="utf-8",
    )

    qa_note = output_dir / "QA.md"
    qa_note.write_text(
        f"""# Figure QA

- Core conclusion and panel hierarchy declared: pass.
- Backend exclusivity: Python/matplotlib only.
- Final width: 7.2 in (182.9 mm), double-column target.
- Editable text: SVG uses `svg.fonttype = none`; PDF uses TrueType font type 42.
- Palette: method colors are consistent across panels; success/harm also use
  position, marker shape, labels, and hatching.
- Source traceability: four CSV files generated from frozen JSON/NPZ artifacts.
- Sample definition: 100 frozen SCAND H4 states; 43 trajectories reported by
  the benchmark summary.
- Statistics: descriptive means/counts only; no unsupported uncertainty bars.
- Integrity boundary: union oracle labelled non-deployable; production-safety
  claim explicitly disallowed.
- Runtime contract version: `{evidence['contract']['version']}`.
""",
        encoding="utf-8",
    )
    return [contract_note, legend_note, qa_note]


def save_outputs(fig: mpl.figure.Figure, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / "v70_structure_scand100_validation"
    paths = [
        base.with_suffix(".svg"),
        base.with_suffix(".pdf"),
        base.with_suffix(".tiff"),
        base.with_suffix(".png"),
    ]
    fig.savefig(paths[0], bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    fig.savefig(paths[2], dpi=600, bbox_inches="tight")
    fig.savefig(paths[3], dpi=300, bbox_inches="tight")
    plt.close(fig)
    return paths


def validate_outputs(paths: list[Path]) -> None:
    for path in paths:
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing or empty output: {path}")
    svg = paths[0].read_text(encoding="utf-8")
    if "<text" not in svg:
        raise RuntimeError("SVG text is not editable")
    if svg.count("<text") < 25:
        raise RuntimeError("Unexpectedly few editable SVG text elements")


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else repo_root / "figures" / "v70_scand100"
    )
    evidence = load_evidence(repo_root)
    source_paths = write_source_data(output_dir, evidence)
    note_paths = write_notes(output_dir, evidence)
    fig = build_figure(evidence)
    figure_paths = save_outputs(fig, output_dir)
    validate_outputs(figure_paths)

    manifest = {
        "schema": "v70_scand100_figure_manifest_v1",
        "backend": "python_matplotlib",
        "figure_files": [str(path) for path in figure_paths],
        "source_data_files": [str(path) for path in source_paths],
        "note_files": [str(path) for path in note_paths],
        "states": 100,
        "trajectories": evidence["summary"]["trajectories"],
        "objective_outcomes": evidence["counts"],
        "production_safe_claim_allowed": evidence["contract"][
            "production_safe_claim_allowed"
        ],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
