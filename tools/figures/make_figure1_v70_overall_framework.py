#!/usr/bin/env python3
"""Build Figure 1: Overall Framework of the v70 Safe Planning System.

The publication render (SVG/PNG), editable draw.io XML, and Mermaid source are
generated from the same code-grounded information model.
"""

from __future__ import annotations

import math
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.path import Path as MplPath


# Nature-style publication defaults: SVG text stays editable.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 6.3
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "figures" / "v70_overall_framework"
BASE = OUT_DIR / "figure1_v70_overall_framework"


COLORS = {
    "ink": "#20242B",
    "muted": "#626A76",
    "line": "#7C8490",
    "very_light": "#F7F8FA",
    "neutral_fill": "#F3F5F8",
    "neutral_stroke": "#C7CDD6",
    "perception_fill": "#E9EFF8",
    "perception_stroke": "#6684AD",
    "candidate_fill": "#E7F3F1",
    "candidate_stroke": "#4E9B94",
    "world_fill": "#EEE9F7",
    "world_stroke": "#7764A8",
    "world_dark": "#5B4A8E",
    "evaluation_fill": "#E8EEF8",
    "evaluation_stroke": "#557AA9",
    "evaluation_dark": "#405F8E",
    "planner_fill": "#FFF0DF",
    "planner_stroke": "#C77635",
    "planner_dark": "#A55722",
    "safe_fill": "#E7F3E9",
    "safe_stroke": "#4E8B5B",
    "danger_fill": "#F8E9E6",
    "danger_stroke": "#B5655C",
    "white": "#FFFFFF",
}


def rounded_box(
    ax,
    x,
    y,
    w,
    h,
    *,
    face,
    edge,
    radius=0.8,
    lw=0.9,
    z=2,
    linestyle="-",
):
    patch = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.02,rounding_size={radius}",
        linewidth=lw,
        edgecolor=edge,
        facecolor=face,
        linestyle=linestyle,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def stage_box(ax, x, y, w, h, title, subtitle, face, edge):
    rounded_box(ax, x, y, w, h, face=face, edge=edge, radius=1.0, lw=1.05, z=0)
    ax.add_patch(
        patches.FancyBboxPatch(
            (x, y + h - 4.6),
            w,
            4.6,
            boxstyle="round,pad=0.02,rounding_size=1.0",
            linewidth=0,
            facecolor=edge,
            zorder=0.5,
        )
    )
    ax.add_patch(
        patches.Rectangle(
            (x, y + h - 4.6),
            w,
            2.3,
            linewidth=0,
            facecolor=edge,
            zorder=0.55,
        )
    )
    ax.text(
        x + 0.8,
        y + h - 2.25,
        title,
        color="white",
        fontsize=5.7,
        fontweight="bold",
        ha="left",
        va="center",
        zorder=4,
    )
    if subtitle:
        ax.text(
            x + 0.8,
            y + h - 5.5,
            subtitle,
            color=COLORS["muted"],
            fontsize=4.7,
            ha="left",
            va="top",
            zorder=4,
        )


def label(
    ax,
    x,
    y,
    text,
    *,
    size=5.6,
    weight="normal",
    color=None,
    ha="center",
    va="center",
    z=5,
    linespacing=1.12,
):
    ax.text(
        x,
        y,
        text,
        fontsize=size,
        fontweight=weight,
        color=color or COLORS["ink"],
        ha=ha,
        va=va,
        zorder=z,
        linespacing=linespacing,
    )


def pill(ax, x, y, w, h, text, face, edge, text_color=None, size=4.7):
    rounded_box(ax, x, y, w, h, face=face, edge=edge, radius=h / 2, lw=0.7, z=4)
    label(
        ax,
        x + w / 2,
        y + h / 2,
        text,
        size=size,
        weight="bold",
        color=text_color or COLORS["ink"],
        z=5,
    )


def arrow(ax, x1, y1, x2, y2, *, color=None, lw=1.15, style="-|>", z=8, dashed=False):
    arr = patches.FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle=style,
        mutation_scale=8,
        linewidth=lw,
        color=color or COLORS["line"],
        linestyle=(0, (3, 2)) if dashed else "-",
        shrinkA=0,
        shrinkB=0,
        zorder=z,
    )
    ax.add_patch(arr)
    return arr


def poly_arrow(ax, points, *, color, lw=1.1, dashed=False, z=7):
    vertices = [points[0]]
    codes = [MplPath.MOVETO]
    for point in points[1:]:
        vertices.append(point)
        codes.append(MplPath.LINETO)
    path = MplPath(vertices, codes)
    arr = patches.FancyArrowPatch(
        path=path,
        arrowstyle="-|>",
        mutation_scale=8,
        linewidth=lw,
        color=color,
        linestyle=(0, (4, 2.5)) if dashed else "-",
        shrinkA=0,
        shrinkB=0,
        zorder=z,
    )
    ax.add_patch(arr)
    return arr


def draw_frame(ax, x, y, w, h, tint, index=None):
    rounded_box(ax, x, y, w, h, face=tint, edge="#8994A3", radius=0.3, lw=0.55, z=3)
    ax.add_patch(
        patches.Polygon(
            [(x + 0.2, y + 0.25), (x + w * 0.43, y + h * 0.57), (x + w * 0.62, y + h * 0.35),
             (x + w - 0.18, y + h * 0.70), (x + w - 0.18, y + 0.25)],
            closed=True,
            facecolor="#9DB5A1",
            edgecolor="none",
            alpha=0.9,
            zorder=3.2,
        )
    )
    ax.plot(
        [x + w * 0.48, x + w * 0.52],
        [y + 0.25, y + h - 0.25],
        color="#F7F1CF",
        lw=0.65,
        zorder=3.4,
    )
    if index is not None:
        label(ax, x + w / 2, y - 0.55, index, size=3.7, color=COLORS["muted"], z=5)


def draw_lock(ax, x, y, scale=1.0, color="white"):
    ax.add_patch(
        patches.Arc(
            (x, y + 0.35 * scale),
            1.05 * scale,
            1.05 * scale,
            theta1=0,
            theta2=180,
            lw=0.85,
            color=color,
            zorder=8,
        )
    )
    ax.add_patch(
        patches.FancyBboxPatch(
            (x - 0.63 * scale, y - 0.45 * scale),
            1.26 * scale,
            0.9 * scale,
            boxstyle="round,pad=0.02,rounding_size=0.16",
            linewidth=0,
            facecolor=color,
            zorder=8,
        )
    )


def draw_grid_tile(ax, x, y, w, h, face, edge, n=4, z=4):
    rounded_box(ax, x, y, w, h, face=face, edge=edge, radius=0.22, lw=0.6, z=z)
    for i in range(1, n):
        ax.plot([x + w * i / n, x + w * i / n], [y + 0.15, y + h - 0.15],
                color=edge, lw=0.25, alpha=0.55, zorder=z + 0.2)
        ax.plot([x + 0.15, x + w - 0.15], [y + h * i / n, y + h * i / n],
                color=edge, lw=0.25, alpha=0.55, zorder=z + 0.2)


def draw_trajectory_bundle(ax, x, y, w, h):
    starts = (x + 0.5, y + 0.5)
    colors = ["#6FA9A4", "#79B4AE", "#8ABDB7", "#5E9C97", "#9AC9C4"]
    for i, color in enumerate(colors):
        verts = [
            starts,
            (x + w * 0.30, y + h * (0.25 + 0.08 * i)),
            (x + w * 0.63, y + h * (0.50 + 0.04 * (i - 2))),
            (x + w - 0.35, y + h * (0.42 + 0.09 * i)),
        ]
        path = MplPath(verts, [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])
        ax.add_patch(
            patches.PathPatch(path, fill=False, edgecolor=color, lw=0.7, alpha=0.9, zorder=5)
        )
    incumbent = [
        starts,
        (x + w * 0.32, y + h * 0.47),
        (x + w * 0.66, y + h * 0.57),
        (x + w - 0.35, y + h * 0.61),
    ]
    ax.add_patch(
        patches.PathPatch(
            MplPath(incumbent, [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4]),
            fill=False,
            edgecolor=COLORS["ink"],
            lw=1.15,
            linestyle=(0, (3, 1.5)),
            zorder=6,
        )
    )


def draw_main_figure():
    fig = plt.figure(figsize=(7.2, 4.35))
    ax = fig.add_axes([0.012, 0.025, 0.976, 0.94])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.axis("off")

    label(
        ax,
        1.0,
        59.0,
        "Overall Framework of the v70 Safe Planning System",
        size=9.4,
        weight="bold",
        ha="left",
        va="top",
    )
    label(
        ax,
        99.0,
        58.7,
        "Frozen visual world model  ·  candidate-wise evaluation  ·  constrained action optimization",
        size=4.8,
        color=COLORS["muted"],
        ha="right",
        va="top",
    )

    # Stage containers.
    stage_box(
        ax, 1.0, 8.0, 19.0, 47.0,
        "1  INPUT & PERCEPTION", "",
        COLORS["neutral_fill"], COLORS["perception_stroke"],
    )
    stage_box(
        ax, 21.2, 31.0, 23.0, 24.0,
        "2  FROZEN WORLD MODEL", "",
        COLORS["world_fill"], COLORS["world_stroke"],
    )
    stage_box(
        ax, 21.2, 8.0, 23.0, 21.5,
        "3  CANDIDATE GENERATION", "",
        COLORS["candidate_fill"], COLORS["candidate_stroke"],
    )
    stage_box(
        ax, 45.4, 8.0, 20.0, 47.0,
        "4  CANDIDATE EVALUATION", "",
        COLORS["evaluation_fill"], COLORS["evaluation_stroke"],
    )
    stage_box(
        ax, 66.6, 8.0, 23.0, 47.0,
        "5  SAFETY-AWARE PLANNING", "",
        COLORS["planner_fill"], COLORS["planner_stroke"],
    )
    stage_box(
        ax, 90.8, 8.0, 8.2, 47.0,
        "6  OUTPUT", "",
        COLORS["safe_fill"], COLORS["safe_stroke"],
    )

    # Stage 1: observations and frozen perception.
    label(ax, 2.2, 47.8, "4-frame history", size=4.8, weight="bold", ha="left")
    frame_x = [2.2, 4.4, 6.6, 8.8]
    for i, x in enumerate(frame_x):
        draw_frame(ax, x, 43.4, 2.7, 3.4, "#D9E3EF", f"t−{3-i}" if i < 3 else "t")
    label(ax, 14.1, 47.8, "goal image", size=4.8, weight="bold")
    draw_frame(ax, 12.0, 43.4, 4.0, 3.4, "#E4E8D9", "goal")

    arrow(ax, 9.1, 42.1, 9.1, 39.8, color=COLORS["perception_stroke"], lw=1.0)
    arrow(ax, 14.0, 42.1, 14.0, 39.8, color=COLORS["perception_stroke"], lw=1.0)
    rounded_box(
        ax, 3.0, 34.1, 14.9, 5.7,
        face=COLORS["perception_fill"], edge=COLORS["perception_stroke"],
        radius=0.7, lw=1.0, z=3,
    )
    label(ax, 10.45, 37.2, "DINOv3 ViT-L/16", size=6.0, weight="bold")
    label(ax, 10.45, 35.4, "patch tokens  [14×14×1024]", size=4.7, color=COLORS["muted"])
    pill(
        ax, 3.7, 38.1, 4.7, 1.35, "FROZEN",
        COLORS["world_dark"], COLORS["world_dark"], "white", size=4.0,
    )
    draw_lock(ax, 4.2, 38.75, scale=0.52)

    arrow(ax, 10.45, 34.1, 10.45, 31.5, color=COLORS["perception_stroke"], lw=1.0)
    rounded_box(
        ax, 3.3, 26.7, 14.3, 4.8,
        face="white", edge=COLORS["perception_stroke"], radius=0.65, lw=0.9, z=3,
    )
    label(ax, 10.45, 29.6, "Adapter", size=5.8, weight="bold")
    label(ax, 10.45, 27.9, "1024→384  ·  14×14→16×16", size=3.85, color=COLORS["muted"])

    arrow(ax, 10.45, 26.7, 10.45, 24.2, color=COLORS["perception_stroke"], lw=1.0)
    rounded_box(
        ax, 3.0, 17.6, 14.9, 6.6,
        face="white", edge=COLORS["perception_stroke"], radius=0.65, lw=0.9, z=3,
    )
    for i in range(4):
        draw_grid_tile(
            ax, 4.0 + i * 1.55, 19.9, 1.25, 1.25,
            "#DEE6F3", COLORS["perception_stroke"], n=4, z=4,
        )
    draw_grid_tile(ax, 11.5, 19.35, 2.2, 2.2, "#E4EAD9", "#87966E", n=4, z=4)
    label(ax, 10.45, 18.5, "history [4×256×384]  ·  goal [256×384]", size=3.75)

    # Stage 3: candidate generation, parallel to perception.
    rounded_box(
        ax, 22.5, 17.0, 8.7, 6.7,
        face="white", edge=COLORS["candidate_stroke"], radius=0.65, lw=0.9, z=3,
    )
    label(ax, 26.85, 22.2, "NoMaD diffusion", size=5.4, weight="bold")
    draw_trajectory_bundle(ax, 23.1, 17.4, 7.5, 3.6)
    pill(
        ax, 22.7, 12.0, 8.3, 3.0, "candidate 0\nincumbent",
        COLORS["neutral_fill"], COLORS["neutral_stroke"], size=3.9,
    )
    rounded_box(
        ax, 33.0, 13.0, 9.8, 8.6,
        face=COLORS["candidate_fill"], edge=COLORS["candidate_stroke"], radius=0.75, lw=1.1, z=3,
    )
    draw_trajectory_bundle(ax, 33.6, 16.8, 8.5, 3.0)
    label(ax, 37.9, 14.9, "K=17\nH4 candidates", size=4.5, weight="bold")
    label(ax, 37.9, 13.6, "[17×4×3] local SE(2)", size=4.2, color=COLORS["muted"])
    arrow(ax, 31.2, 20.3, 33.0, 18.3, color=COLORS["candidate_stroke"], lw=1.0)
    arrow(ax, 31.0, 13.5, 33.0, 15.1, color=COLORS["candidate_stroke"], lw=1.0)

    # Dashed shared-RGB branch into NoMaD.
    poly_arrow(
        ax,
        [(16.0, 42.7), (19.9, 42.7), (19.9, 20.3), (22.5, 20.3)],
        color=COLORS["candidate_stroke"],
        lw=0.95,
        dashed=True,
        z=6,
    )
    label(
        ax, 20.2, 29.5, "same RGB\ncontext", size=3.9,
        color=COLORS["candidate_stroke"], ha="left",
    )

    # Stage 2: frozen LTWM ensemble and H1-H4 rollout.
    for offset in [1.2, 0.6, 0.0]:
        rounded_box(
            ax,
            23.3 + offset,
            38.0 + offset * 0.35,
            9.7,
            8.2,
            face="#FFFFFF" if offset == 0 else "#E6E0F2",
            edge=COLORS["world_stroke"],
            radius=0.65,
            lw=0.9,
            z=3 + (1.2 - offset),
        )
    label(ax, 28.15, 43.2, "LiteTokenFormer", size=5.9, weight="bold", z=7)
    label(ax, 28.15, 41.5, "LTWM  ×3 members", size=5.1, weight="bold", color=COLORS["world_dark"], z=7)
    label(ax, 28.15, 39.7, "frozen parameters", size=4.4, color=COLORS["muted"], z=7)
    pill(
        ax, 23.8, 45.1, 4.2, 1.25, "FROZEN",
        COLORS["world_dark"], COLORS["world_dark"], "white", size=3.8,
    )

    label(ax, 37.9, 47.7, "future latent rollout", size=4.8, weight="bold")
    horizon_colors = ["#DFD8EE", "#D3C9E6", "#C6BADD", "#B9AAD4"]
    for i, (h_label, face) in enumerate(zip(["H1", "H2", "H3", "H4"], horizon_colors)):
        xx = 34.4 + i * 2.05
        draw_grid_tile(ax, xx, 41.1, 1.65, 4.9, face, COLORS["world_stroke"], n=4, z=4)
        label(ax, xx + 0.825, 40.25, h_label, size=4.2, weight="bold", color=COLORS["world_dark"])
    label(
        ax, 37.9, 37.6,
        "action-conditioned · horizon-causal\n[B,K,4,256,384]",
        size=4.25, color=COLORS["muted"],
    )

    # Latent and action inputs into LTWM.
    poly_arrow(
        ax,
        [(17.9, 20.9), (19.6, 20.9), (19.6, 47.5), (23.3, 47.5), (23.3, 44.4)],
        color=COLORS["perception_stroke"],
        lw=1.05,
        z=7,
    )
    label(ax, 20.0, 49.0, "latent", size=4.0, color=COLORS["perception_stroke"], ha="left")
    poly_arrow(
        ax,
        [(37.9, 21.6), (37.9, 33.8), (30.6, 33.8), (30.6, 38.0)],
        color=COLORS["candidate_stroke"],
        lw=1.05,
        z=7,
    )
    label(ax, 38.3, 32.7, "H4 action", size=4.0, color=COLORS["candidate_stroke"], ha="left")
    arrow(ax, 33.0, 42.3, 34.2, 42.3, color=COLORS["world_stroke"], lw=1.15)

    # Stage 4: feature construction and Arm-D evaluation.
    rounded_box(
        ax, 46.8, 37.0, 17.2, 11.6,
        face="white", edge=COLORS["evaluation_stroke"], radius=0.75, lw=0.95, z=3,
    )
    label(ax, 55.4, 46.4, "Candidate feature\nconstruction", size=5.0, weight="bold")
    feature_rows = [
        ("DINO state", "210D"),
        ("LTWM ensemble statistics", "112D"),
        ("action + relative/set + guard", "493D"),
    ]
    for i, (left, right) in enumerate(feature_rows):
        yy = 43.5 - i * 2.05
        rounded_box(
            ax, 47.8, yy - 0.7, 15.2, 1.55,
            face=COLORS["evaluation_fill"], edge="#B6C5DB", radius=0.35, lw=0.45, z=4,
        )
        label(ax, 48.35, yy, left, size=3.75, ha="left")
        label(ax, 62.45, yy, right, size=4.15, weight="bold", ha="right", color=COLORS["evaluation_dark"])
    pill(
        ax, 53.1, 37.0, 9.6, 1.7, "815D per candidate",
        COLORS["evaluation_dark"], COLORS["evaluation_dark"], "white", size=4.4,
    )

    arrow(ax, 55.4, 37.0, 55.4, 34.0, color=COLORS["evaluation_stroke"], lw=1.1)
    rounded_box(
        ax, 47.6, 23.0, 15.6, 11.0,
        face=COLORS["evaluation_dark"], edge=COLORS["evaluation_dark"], radius=0.9, lw=1.1, z=3,
    )
    label(ax, 55.4, 31.0, "Universal Arm-D", size=6.6, weight="bold", color="white")
    label(ax, 55.4, 28.5, "candidate evaluator  ×3 members", size=4.8, color="#E9F0FA")
    label(ax, 55.4, 25.7, "domain-conditioned · setwise\nheteroscedastic", size=3.75, color="#D5E0F0")

    output_specs = [
        ("Δquality", "μ ± σ"),
        ("dual", "p"),
        ("harm", "p"),
        ("rank", "logit"),
    ]
    for i, (name, spec) in enumerate(output_specs):
        xx = 47.0 + i * 4.25
        rounded_box(
            ax, xx, 16.8, 3.7, 3.7,
            face="white", edge=COLORS["evaluation_stroke"], radius=0.5, lw=0.75, z=3,
        )
        label(ax, xx + 1.85, 19.0, name, size=4.3, weight="bold")
        label(ax, xx + 1.85, 17.7, spec, size=3.9, color=COLORS["muted"])
    label(
        ax, 55.4, 13.5,
        "relative to candidate 0\nensemble variance: heteroscedastic + epistemic",
        size=3.75, color=COLORS["muted"],
    )

    # Inputs into feature construction.
    poly_arrow(
        ax,
        [(42.6, 43.2), (44.6, 43.2), (44.6, 44.0), (46.8, 44.0)],
        color=COLORS["world_stroke"],
        lw=1.15,
        z=8,
    )
    poly_arrow(
        ax,
        [(42.8, 17.1), (44.6, 17.1), (44.6, 39.7), (46.8, 39.7)],
        color=COLORS["candidate_stroke"],
        lw=1.0,
        z=7,
    )

    # Stage 5: uncertainty gate, safe CEM, archive.
    rounded_box(
        ax, 68.1, 36.0, 20.0, 13.0,
        face="#FFF8EF", edge=COLORS["planner_stroke"], radius=0.85, lw=1.1, z=3,
    )
    label(ax, 78.1, 46.8, "UNCERTAINTY SAFETY GATE", size=5.8, weight="bold", color=COLORS["planner_dark"])
    gate_lines = [
        "valid = support envelope ∧ physical guard",
        "UCB(ΔATE), UCB(Δobjective) < margins",
        "calibrated p(dual) high · p(harm) low",
    ]
    for i, text in enumerate(gate_lines):
        yy = 44.1 - i * 2.55
        ax.add_patch(
            patches.Circle((69.5, yy), 0.30, facecolor=COLORS["planner_stroke"], edgecolor="none", zorder=5)
        )
        label(ax, 70.2, yy, text, size=3.95, ha="left")
    pill(
        ax, 73.4, 36.7, 9.4, 1.75, "rank proposes · gate admits",
        COLORS["planner_dark"], COLORS["planner_dark"], "white", size=4.1,
    )

    arrow(ax, 78.1, 36.0, 78.1, 32.6, color=COLORS["planner_stroke"], lw=1.2)
    rounded_box(
        ax, 69.3, 23.0, 17.6, 9.6,
        face="#FFE6CD", edge=COLORS["planner_stroke"], radius=0.8, lw=1.15, z=3,
    )
    label(ax, 78.1, 30.1, "SAFE CEM OPTIMIZATION", size=5.9, weight="bold", color=COLORS["planner_dark"])
    label(ax, 78.1, 27.5, "3 rounds · 16 proposals/round", size=4.6)
    label(ax, 78.1, 25.3, "safe elites only\nretain proposal if safe set is insufficient", size=3.7, color=COLORS["muted"])

    arrow(ax, 78.1, 23.0, 78.1, 20.6, color=COLORS["planner_stroke"], lw=1.15)
    rounded_box(
        ax, 69.3, 12.0, 17.6, 8.6,
        face="#FFF8EF", edge=COLORS["planner_stroke"], radius=0.75, lw=1.0, z=3,
    )
    label(ax, 78.1, 18.2, "CROSS-ROUND ARCHIVE", size=5.6, weight="bold", color=COLORS["planner_dark"])
    label(ax, 78.1, 15.7, "lowest-energy safe candidate", size=4.5)
    label(ax, 78.1, 13.7, "fixed pool  ∪  CEM rounds", size=4.1, color=COLORS["muted"])

    # Evaluation output into the gate.
    arrow(ax, 64.1, 19.0, 66.0, 19.0, color=COLORS["evaluation_stroke"], lw=1.1)
    poly_arrow(
        ax,
        [(66.0, 19.0), (66.0, 42.5), (68.1, 42.5)],
        color=COLORS["evaluation_stroke"],
        lw=1.15,
        z=8,
    )

    # Bottom optimization loop: new CEM proposals return to the same evaluation chain.
    poly_arrow(
        ax,
        [(69.3, 27.8), (67.3, 27.8), (67.3, 5.1), (38.0, 5.1), (38.0, 13.0)],
        color=COLORS["planner_stroke"],
        lw=1.15,
        dashed=True,
        z=7,
    )
    pill(
        ax, 45.0, 3.95, 19.5, 2.15,
        "new proposals → same LTWM / Arm-D evaluation",
        "#FFF8EF", COLORS["planner_stroke"], COLORS["planner_dark"], size=3.45,
    )

    # Stage 6: selected safe trajectory or fallback.
    rounded_box(
        ax, 91.7, 35.5, 6.4, 12.1,
        face="white", edge=COLORS["safe_stroke"], radius=0.7, lw=1.0, z=3,
    )
    path = MplPath(
        [(92.6, 37.0), (93.4, 39.7), (95.7, 41.0), (97.1, 45.1)],
        [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4],
    )
    ax.add_patch(patches.PathPatch(path, fill=False, edgecolor=COLORS["safe_stroke"], lw=1.7, zorder=6))
    for t in [0.0, 0.33, 0.66, 1.0]:
        px = 92.6 + (97.1 - 92.6) * t
        py = 37.0 + (45.1 - 37.0) * t
        ax.add_patch(patches.Circle((px, py), 0.28, facecolor=COLORS["safe_stroke"], edgecolor="white", lw=0.4, zorder=7))
    label(ax, 94.9, 33.4, "BEST SAFE\nH4 TRAJECTORY", size=4.6, weight="bold", color=COLORS["safe_stroke"])

    label(ax, 94.9, 27.7, "or", size=4.6, weight="bold", color=COLORS["muted"])
    rounded_box(
        ax, 91.7, 17.1, 6.4, 8.1,
        face=COLORS["neutral_fill"], edge=COLORS["neutral_stroke"], radius=0.7, lw=0.85, z=3,
    )
    label(ax, 94.9, 22.5, "FALLBACK", size=5.0, weight="bold", color=COLORS["muted"])
    label(ax, 94.9, 19.6, "candidate 0\nincumbent", size=4.45)

    # Final decision links.
    poly_arrow(
        ax,
        [(86.9, 16.3), (89.7, 16.3), (89.7, 41.5), (91.7, 41.5)],
        color=COLORS["safe_stroke"],
        lw=1.25,
        z=8,
    )
    poly_arrow(
        ax,
        [(68.1, 39.0), (67.4, 39.0), (67.4, 9.0), (89.8, 9.0), (89.8, 21.0), (91.7, 21.0)],
        color=COLORS["danger_stroke"],
        lw=0.95,
        dashed=True,
        z=6,
    )
    label(ax, 87.8, 9.7, "if no candidate passes", size=3.8, color=COLORS["danger_stroke"], ha="right")

    # Compact semantic footer.
    label(
        ax,
        1.0,
        1.2,
        "Frozen parameters",
        size=4.2,
        weight="bold",
        color=COLORS["world_dark"],
        ha="left",
    )
    label(
        ax,
        12.2,
        1.2,
        "·  learned candidate evaluation",
        size=4.2,
        color=COLORS["evaluation_dark"],
        ha="left",
    )
    label(
        ax,
        31.2,
        1.2,
        "·  outcome-free constrained replay/optimization",
        size=4.2,
        color=COLORS["planner_dark"],
        ha="left",
    )
    label(
        ax,
        99.0,
        1.2,
        "H4 = four-step local SE(2) trajectory",
        size=4.0,
        color=COLORS["muted"],
        ha="right",
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(BASE.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(BASE.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT_DIR / "figure1_v70_overall_framework_preview.png", dpi=250, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


class DrawioBuilder:
    def __init__(self):
        self.mxfile = ET.Element("mxfile", host="drawio", version="26.0.0")
        self.diagram = ET.SubElement(self.mxfile, "diagram", name="Figure 1")
        self.model = ET.SubElement(
            self.diagram,
            "mxGraphModel",
            dx="1920",
            dy="620",
            grid="1",
            gridSize="10",
            guides="1",
            tooltips="1",
            connect="1",
            arrows="1",
            fold="1",
            page="1",
            pageScale="1",
            pageWidth="1920",
            pageHeight="620",
            math="0",
            shadow="0",
        )
        self.root = ET.SubElement(self.model, "root")
        ET.SubElement(self.root, "mxCell", id="0")
        ET.SubElement(self.root, "mxCell", id="1", parent="0")

    def vertex(self, cell_id, value, style, x, y, w, h, parent="1"):
        cell = ET.SubElement(
            self.root,
            "mxCell",
            id=cell_id,
            value=value,
            style=style,
            vertex="1",
            parent=parent,
        )
        ET.SubElement(
            cell,
            "mxGeometry",
            x=str(x),
            y=str(y),
            width=str(w),
            height=str(h),
            **{"as": "geometry"},
        )
        return cell

    def edge(self, cell_id, source, target, value="", style=None, points=None):
        if style is None:
            style = (
                "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;"
                "jettySize=auto;html=1;endArrow=blockThin;endFill=1;"
                "strokeColor=#7C8490;strokeWidth=2;"
            )
        cell = ET.SubElement(
            self.root,
            "mxCell",
            id=cell_id,
            value=value,
            style=style,
            edge="1",
            parent="1",
            source=source,
            target=target,
        )
        geometry = ET.SubElement(cell, "mxGeometry", relative="1", **{"as": "geometry"})
        if points:
            arr = ET.SubElement(geometry, "Array", **{"as": "points"})
            for px, py in points:
                ET.SubElement(arr, "mxPoint", x=str(px), y=str(py))
        return cell

    def write(self, path):
        ET.indent(self.mxfile, space="  ")
        tree = ET.ElementTree(self.mxfile)
        tree.write(path, encoding="utf-8", xml_declaration=True)


def build_drawio():
    b = DrawioBuilder()

    title_style = (
        "text;html=1;align=left;verticalAlign=middle;whiteSpace=wrap;"
        "fontFamily=Arial;fontSize=24;fontStyle=1;fontColor=#20242B;"
    )
    b.vertex(
        "title",
        "Overall Framework of the v70 Safe Planning System",
        title_style,
        20, 10, 1870, 40,
    )

    stage_style = (
        "swimlane;startSize=34;rounded=1;arcSize=10;html=1;whiteSpace=wrap;"
        "container=1;collapsible=0;pointerEvents=0;fontFamily=Arial;"
        "fontSize=13;fontStyle=1;fontColor=#FFFFFF;horizontal=1;"
    )
    stages = {
        "s1": ("1  INPUT &amp; PERCEPTION", 20, 70, 350, 470, "#F3F5F8", "#6684AD"),
        "s2": ("2  FROZEN WORLD MODEL", 390, 70, 420, 225, "#EEE9F7", "#7764A8"),
        "s3": ("3  CANDIDATE GENERATION", 390, 315, 420, 225, "#E7F3F1", "#4E9B94"),
        "s4": ("4  CANDIDATE EVALUATION", 830, 70, 390, 470, "#E8EEF8", "#557AA9"),
        "s5": ("5  SAFETY-AWARE PLANNING", 1240, 70, 430, 470, "#FFF0DF", "#C77635"),
        "s6": ("6  OUTPUT", 1690, 70, 200, 470, "#E7F3E9", "#4E8B5B"),
    }
    for sid, (name, x, y, w, h, fill, stroke) in stages.items():
        b.vertex(
            sid,
            name,
            stage_style + f"fillColor={fill};strokeColor={stroke};swimlaneFillColor={stroke};",
            x, y, w, h,
        )

    node_base = (
        "rounded=1;arcSize=12;whiteSpace=wrap;html=1;fontFamily=Arial;"
        "fontSize=13;fontColor=#20242B;strokeWidth=1.5;"
    )
    small_style = (
        "rounded=1;arcSize=12;whiteSpace=wrap;html=1;fontFamily=Arial;"
        "fontSize=12;fontColor=#20242B;strokeWidth=1.3;"
    )

    # Stage 1 children (relative coordinates).
    b.vertex("history", "<b>4-frame history</b><br>[t−3, t−2, t−1, t]",
             node_base + "fillColor=#FFFFFF;strokeColor=#6684AD;", 20, 55, 145, 70, "s1")
    b.vertex("goal", "<b>goal image</b>",
             node_base + "fillColor=#FFFFFF;strokeColor=#6684AD;", 185, 55, 140, 70, "s1")
    b.vertex("dino", "<b>Frozen DINOv3 ViT-L/16</b><br>14×14×1024 patch tokens",
             node_base + "fillColor=#E9EFF8;strokeColor=#6684AD;", 45, 155, 260, 75, "s1")
    b.vertex("adapter", "<b>Adapter</b><br>1024→384 · 14×14→16×16",
             node_base + "fillColor=#FFFFFF;strokeColor=#6684AD;", 65, 260, 220, 65, "s1")
    b.vertex("latent", "<b>Adapted visual latents</b><br>history [4×256×384]<br>goal [256×384]",
             node_base + "fillColor=#FFFFFF;strokeColor=#6684AD;", 45, 355, 260, 80, "s1")

    # Stage 2 children.
    b.vertex("ltwm", "<b>LiteTokenFormer (LTWM) ×3</b><br>frozen · action-conditioned",
             node_base + "fillColor=#FFFFFF;strokeColor=#7764A8;fontColor=#5B4A8E;",
             20, 65, 190, 85, "s2")
    b.vertex("rollout", "<b>Future latent rollout</b><br>H1 · H2 · H3 · H4<br>[B,K,4,256,384]",
             node_base + "fillColor=#DFD8EE;strokeColor=#7764A8;",
             235, 65, 160, 85, "s2")
    b.vertex("frozen_badge", "FROZEN",
             "rounded=1;arcSize=40;html=1;fontFamily=Arial;fontSize=10;fontStyle=1;"
             "fontColor=#FFFFFF;fillColor=#5B4A8E;strokeColor=#5B4A8E;",
             30, 155, 75, 25, "s2")

    # Stage 3 children.
    b.vertex("nomad", "<b>NoMaD diffusion</b><br>16 waypoint trajectories",
             node_base + "fillColor=#FFFFFF;strokeColor=#4E9B94;",
             20, 65, 165, 70, "s3")
    b.vertex("incumbent", "<b>candidate 0</b><br>trimmed-mean incumbent",
             node_base + "fillColor=#F3F5F8;strokeColor=#C7CDD6;",
             20, 155, 165, 60, "s3")
    b.vertex("k17", "<b>K=17 H4 candidates</b><br>[17×4×3] local SE(2)",
             node_base + "fillColor=#E7F3F1;strokeColor=#4E9B94;",
             220, 105, 175, 85, "s3")

    # Stage 4 children.
    b.vertex(
        "features",
        "<b>Candidate feature construction</b><br>"
        "DINO state 210D<br>LTWM statistics 112D<br>"
        "action/history + relative/set/support 493D<br><b>→ 815D per candidate</b>",
        node_base + "fillColor=#FFFFFF;strokeColor=#557AA9;",
        25, 60, 340, 145, "s4",
    )
    b.vertex(
        "armd",
        "<b>Universal Arm-D ×3</b><br>domain-conditioned · setwise<br>heteroscedastic candidate evaluator",
        node_base + "fillColor=#405F8E;strokeColor=#405F8E;fontColor=#FFFFFF;",
        45, 240, 300, 100, "s4",
    )
    b.vertex(
        "signals",
        "<b>Δquality μ±σ</b>  ·  dual p  ·  harm p  ·  rank logit",
        small_style + "fillColor=#FFFFFF;strokeColor=#557AA9;",
        35, 375, 320, 55, "s4",
    )

    # Stage 5 children.
    b.vertex(
        "gate",
        "<b>UNCERTAINTY SAFETY GATE</b><br>"
        "support ∧ physical guard<br>"
        "UCB(ΔATE), UCB(Δobjective)<br>"
        "calibrated p(dual) / p(harm)",
        node_base + "fillColor=#FFF8EF;strokeColor=#C77635;fontColor=#A55722;",
        25, 55, 380, 125, "s5",
    )
    b.vertex(
        "cem",
        "<b>SAFE CEM OPTIMIZATION</b><br>3 rounds · 16 proposals/round<br>safe elites only",
        node_base + "fillColor=#FFE6CD;strokeColor=#C77635;",
        55, 225, 320, 90, "s5",
    )
    b.vertex(
        "archive",
        "<b>CROSS-ROUND ARCHIVE</b><br>lowest-energy safe candidate<br>fixed pool ∪ CEM rounds",
        node_base + "fillColor=#FFF8EF;strokeColor=#C77635;",
        55, 350, 320, 80, "s5",
    )

    # Stage 6 children.
    b.vertex(
        "best",
        "<b>BEST SAFE</b><br>H4 trajectory",
        node_base + "fillColor=#FFFFFF;strokeColor=#4E8B5B;fontColor=#4E8B5B;",
        20, 85, 160, 100, "s6",
    )
    b.vertex(
        "fallback",
        "<b>FALLBACK</b><br>candidate 0<br>incumbent",
        node_base + "fillColor=#F3F5F8;strokeColor=#C7CDD6;",
        20, 285, 160, 95, "s6",
    )

    edge_base = (
        "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;"
        "jettySize=auto;html=1;endArrow=blockThin;endFill=1;"
        "strokeColor=#7C8490;strokeWidth=2;"
    )
    blue_edge = edge_base + "strokeColor=#6684AD;"
    teal_edge = edge_base + "strokeColor=#4E9B94;"
    purple_edge = edge_base + "strokeColor=#7764A8;"
    eval_edge = edge_base + "strokeColor=#557AA9;"
    orange_edge = edge_base + "strokeColor=#C77635;"
    green_edge = edge_base + "strokeColor=#4E8B5B;"
    dashed_orange = orange_edge + "dashed=1;dashPattern=8 4;"
    dashed_red = edge_base + "strokeColor=#B5655C;dashed=1;dashPattern=8 4;"

    b.edge("e1", "history", "dino", style=blue_edge)
    b.edge("e2", "goal", "dino", style=blue_edge)
    b.edge("e3", "dino", "adapter", style=blue_edge)
    b.edge("e4", "adapter", "latent", style=blue_edge)
    b.edge("e5", "history", "nomad", value="same RGB context", style=teal_edge)
    b.edge("e6", "goal", "nomad", style=teal_edge)
    b.edge("e7", "nomad", "k17", style=teal_edge)
    b.edge("e8", "incumbent", "k17", style=teal_edge)
    b.edge("e9", "latent", "ltwm", value="history + goal latent", style=blue_edge)
    b.edge("e10", "k17", "ltwm", value="H4 actions", style=teal_edge)
    b.edge("e11", "ltwm", "rollout", style=purple_edge)
    b.edge("e12", "rollout", "features", value="H1–H4 latent statistics", style=purple_edge)
    b.edge("e13", "k17", "features", value="action / set features", style=teal_edge)
    b.edge("e14", "features", "armd", style=eval_edge)
    b.edge("e15", "armd", "signals", style=eval_edge)
    b.edge("e16", "signals", "gate", style=eval_edge)
    b.edge("e17", "gate", "cem", value="admitted safe set", style=orange_edge)
    b.edge("e18", "cem", "archive", style=orange_edge)
    b.edge("e19", "archive", "best", style=green_edge)
    b.edge(
        "e20", "gate", "fallback", value="no candidate passes",
        style=dashed_red,
        points=[(1675, 190), (1675, 405)],
    )
    b.edge(
        "e21", "cem", "k17",
        value="new proposals → same LTWM / Arm-D evaluation",
        style=dashed_orange,
        points=[(1225, 340), (1225, 580), (700, 580), (700, 520)],
    )

    b.write(BASE.with_suffix(".drawio"))


MERMAID = r"""%% Figure 1 — Overall Framework of the v70 Safe Planning System
%% Structural companion to the publication SVG/PNG.
%%{init: {"theme":"base","flowchart":{"curve":"basis","htmlLabels":true},"themeVariables":{"fontFamily":"Arial","primaryTextColor":"#20242B","lineColor":"#7C8490","clusterBkg":"#F7F8FA","clusterBorder":"#C7CDD6"}}}%%
flowchart LR
  subgraph S1["1 · Input & Perception"]
    direction TB
    HIST["4-frame history<br/>t−3 … t"]
    GOAL["goal image"]
    DINO["🔒 Frozen DINOv3 ViT-L/16<br/>14×14×1024"]
    ADAPT["Adapter<br/>1024→384 · 14×14→16×16"]
    LAT["adapted visual latents<br/>history [4×256×384]<br/>goal [256×384]"]
    HIST --> DINO
    GOAL --> DINO
    DINO --> ADAPT --> LAT
  end

  subgraph S23["Frozen prediction and action proposals"]
    direction TB
    subgraph S2["2 · Frozen World Model"]
      direction LR
      LTWM["🔒 LTWM ensemble ×3<br/>action-conditioned"]
      ROLL["future latent rollout<br/>H1 · H2 · H3 · H4"]
      LTWM --> ROLL
    end
    subgraph S3["3 · Candidate Generation"]
      direction LR
      NOMAD["NoMaD diffusion<br/>16 trajectories"]
      INC["candidate 0<br/>trimmed-mean incumbent"]
      K17["K=17 H4 candidates<br/>[17×4×3]"]
      NOMAD --> K17
      INC --> K17
    end
  end

  subgraph S4["4 · Candidate Evaluation"]
    direction TB
    FEAT["candidate feature construction<br/>DINO 210D + LTWM 112D<br/>action/history/relative/set/support 493D<br/><b>815D per candidate</b>"]
    ARMD["Universal Arm-D ensemble ×3<br/>domain-conditioned · setwise · heteroscedastic"]
    SIG["Δquality μ±σ · dual p · harm p · rank logit"]
    FEAT --> ARMD --> SIG
  end

  subgraph S5["5 · Safety-aware Planning"]
    direction TB
    GATE{"uncertainty safety gate<br/>support ∧ physical guard<br/>UCB constraints<br/>calibrated dual / harm"}
    CEM["safe CEM optimization<br/>3 rounds · safe elites only"]
    ARCH["cross-round archive<br/>lowest-energy safe candidate"]
    GATE -->|admitted safe set| CEM --> ARCH
  end

  subgraph S6["6 · Final Output"]
    direction TB
    BEST["best safe H4 trajectory"]
    FALL["fallback candidate 0"]
  end

  LAT -->|history + goal latent| LTWM
  HIST -. same RGB context .-> NOMAD
  GOAL -.-> NOMAD
  K17 -->|H4 actions| LTWM
  ROLL -->|ensemble latent statistics| FEAT
  K17 -->|action / set features| FEAT
  SIG --> GATE
  ARCH --> BEST
  GATE -. no candidate passes .-> FALL
  CEM -. new proposals; re-evaluate .-> K17

  classDef perception fill:#E9EFF8,stroke:#6684AD,color:#20242B,stroke-width:1.5px;
  classDef candidate fill:#E7F3F1,stroke:#4E9B94,color:#20242B,stroke-width:1.5px;
  classDef world fill:#EEE9F7,stroke:#7764A8,color:#5B4A8E,stroke-width:2px;
  classDef evaluation fill:#E8EEF8,stroke:#557AA9,color:#20242B,stroke-width:1.5px;
  classDef hero fill:#405F8E,stroke:#405F8E,color:#FFFFFF,stroke-width:2px;
  classDef planner fill:#FFF0DF,stroke:#C77635,color:#A55722,stroke-width:2px;
  classDef output fill:#E7F3E9,stroke:#4E8B5B,color:#245A31,stroke-width:1.5px;
  classDef neutral fill:#F3F5F8,stroke:#C7CDD6,color:#20242B,stroke-width:1.2px;

  class HIST,GOAL,DINO,ADAPT,LAT perception;
  class NOMAD,K17 candidate;
  class INC,FALL neutral;
  class LTWM,ROLL world;
  class FEAT,SIG evaluation;
  class ARMD hero;
  class GATE,CEM,ARCH planner;
  class BEST output;
"""


def build_mermaid():
    BASE.with_suffix(".mmd").write_text(MERMAID, encoding="utf-8")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    draw_main_figure()
    build_drawio()
    build_mermaid()
    print(f"Wrote {BASE.with_suffix('.drawio')}")
    print(f"Wrote {BASE.with_suffix('.svg')}")
    print(f"Wrote {BASE.with_suffix('.mmd')}")
    print(f"Wrote {BASE.with_suffix('.png')}")


if __name__ == "__main__":
    main()
