#!/usr/bin/env python3
"""Replace the split metric outputs with one compact bundled quality bus."""

from __future__ import annotations

import figure1_reference_refined_v25 as v25


v24 = v25.v24
v23 = v24.v23
v21 = v25.v21
v18 = v21.v18
v11 = v18.v11
base = v21.base

raw_box = v18.raw_box
raw_text = v18.raw_text
raw_arrow = v18.raw_arrow
plain_vertex = v18.plain_vertex
raw_edge = v18.raw_edge


def inline_label(ax, x, y, value, *, size=2.35, weight="bold"):
    return ax.text(
        x,
        y,
        value,
        fontsize=size,
        fontweight=weight,
        color=base.MUTED,
        ha="center",
        va="center",
        linespacing=1.08,
        bbox={"facecolor": base.WHITE, "edgecolor": "none", "pad": 0.4},
        zorder=8,
    )


def mlp_card(ax, y, seed):
    raw_box(
        ax,
        120.4,
        y,
        12.6,
        5.2,
        face=base.SUBPANEL,
        edge=base.LIGHT_LINE,
        lw=0.5,
        radius=0.62,
        z=5,
    )
    raw_text(
        ax,
        121.15,
        y + 3.38,
        f"MLP · {seed}",
        size=2.85,
        weight="bold",
        ha="left",
    )
    raw_text(
        ax,
        121.15,
        y + 1.25,
        "1028 → 640 × 3 → 2",
        size=1.95,
        color=base.MUTED,
        ha="left",
    )


def draw_single_bus_d(ax):
    raw_text(
        ax,
        105.0,
        62.8,
        "D  TRAJECTORY-QUALITY SCORING",
        size=5.05,
        weight="bold",
        ha="left",
    )
    ax.plot(
        [105.0, 148.0],
        [61.0, 61.0],
        color="#E1E3E5",
        lw=0.58,
        solid_capstyle="round",
        zorder=2,
    )

    inline_label(ax, 106.6, 57.45, "Route · 192-D")
    inline_label(ax, 106.6, 51.05, "Future · 796-D")
    inline_label(ax, 106.6, 44.65, "Action · 40-D")

    raw_box(
        ax,
        110.4,
        40.2,
        6.6,
        19.0,
        face=base.WHITE,
        edge=base.LIGHT_LINE,
        lw=0.58,
        radius=0.72,
        z=5,
    )
    raw_text(ax, 113.7, 52.5, "Concat", size=3.15, weight="bold")
    raw_text(ax, 113.7, 49.75, "1028-D", size=2.8, weight="bold")
    raw_text(
        ax,
        113.7,
        45.7,
        "frozen\nz-score",
        size=2.15,
        color=base.MUTED,
    )

    # One broadcast rail distributes the same normalized feature vector.
    ax.plot([117.0, 118.35], [49.7, 49.7], color=base.LINE, lw=0.55, zorder=4)
    ax.plot([118.35, 118.35], [42.9, 57.1], color=base.LINE, lw=0.55, zorder=4)

    mlp_specs = ((54.5, 1701), (47.4, 1702), (40.3, 1703))
    for y, seed in mlp_specs:
        center_y = y + 2.6
        raw_arrow(
            ax,
            (118.35, center_y),
            (120.4, center_y),
            color=base.LINE,
            lw=0.52,
        )
        mlp_card(ax, y, seed)

    # Each seed emits one 2-D quality vector. The rail bundles all three vectors.
    output_bus_x = 134.65
    for y, _ in mlp_specs:
        center_y = y + 2.6
        ax.plot(
            [133.0, output_bus_x],
            [center_y, center_y],
            color=base.LINE,
            lw=0.52,
            zorder=4,
        )
    ax.plot(
        [output_bus_x, output_bus_x],
        [42.9, 57.1],
        color=base.LINE,
        lw=0.55,
        zorder=4,
    )

    raw_box(
        ax,
        140.4,
        40.2,
        7.6,
        19.0,
        face=base.WHITE,
        edge=base.LINE,
        lw=0.65,
        radius=0.82,
        z=5,
    )
    raw_text(ax, 144.2, 55.35, "Joint selector", size=2.65, weight="bold")
    raw_text(
        ax,
        144.2,
        52.1,
        "seed mean ± std",
        size=1.95,
        color=base.MUTED,
    )
    raw_text(ax, 144.2, 48.7, "lower error", size=2.05, weight="bold")
    raw_text(ax, 144.2, 46.15, "higher progress", size=2.05, weight="bold")
    raw_text(ax, 144.2, 42.7, "selected H4", size=2.15, weight="bold")
    raw_text(
        ax,
        144.2,
        40.95,
        "optional CEM",
        size=1.55,
        color=base.MUTED,
    )

    raw_arrow(
        ax,
        (output_bus_x, 49.7),
        (140.4, 49.7),
        color=base.LINE,
        lw=0.62,
    )
    inline_label(
        ax,
        137.5,
        50.0,
        "quality predictions\n[3 seeds × 2 metrics]",
        size=1.75,
        weight="normal",
    )


def clean_style(fill="#F5F6F6", stroke="#D7D9DC", size=10):
    return (
        "rounded=1;arcSize=10;whiteSpace=wrap;html=1;"
        f"fillColor={fill};strokeColor={stroke};"
        f"fontColor=#202226;fontFamily=Arial;fontSize={size};"
    )


def text_style(size=10, align="left", color="#202226", bold=False):
    font_style = "fontStyle=1;" if bold else ""
    return (
        "text;html=1;strokeColor=none;fillColor=none;"
        f"align={align};verticalAlign=middle;fontFamily=Arial;"
        f"fontColor={color};fontSize={size};{font_style}"
    )


def edge_style(*, width=1.0, arrow=True, extras=""):
    arrow_style = "endArrow=block;endFill=1;" if arrow else "endArrow=none;"
    return (
        "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;"
        "jettySize=auto;html=1;strokeColor=#7B7F86;"
        f"strokeWidth={width};{arrow_style}fontFamily=Arial;fontSize=9;"
        f"labelBackgroundColor=#FFFFFF;{extras}"
    )


def add_single_bus_d_drawio(self, parent):
    plain_vertex(
        self,
        "d_title",
        "D  TRAJECTORY-QUALITY SCORING",
        0,
        8,
        390,
        32,
        parent=parent,
        style=text_style(size=13, bold=True),
    )
    plain_vertex(
        self,
        "d_rule",
        "",
        0,
        50,
        555,
        1,
        parent=parent,
        style="shape=line;strokeColor=#E1E3E5;strokeWidth=1;html=1;",
    )
    plain_vertex(
        self,
        "concat_norm",
        "<b>Concat</b><br><b>1028-D</b><br>"
        "<font color=\"#666A70\">frozen z-score</font>",
        105,
        80,
        70,
        220,
        parent=parent,
        style=clean_style(fill="#FFFFFF", size=10),
    )
    plain_vertex(
        self,
        "fanout_bus",
        "",
        195,
        100,
        2,
        180,
        parent=parent,
        style="rounded=0;html=1;fillColor=#7B7F86;strokeColor=none;",
    )

    for seed, y in ((1701, 80), (1702, 160), (1703, 240)):
        plain_vertex(
            self,
            f"mlp_{seed}",
            f"<b>MLP · {seed}</b><br>"
            "<font color=\"#666A70\">1028 → 640 × 3 → 2</font>",
            225,
            y,
            150,
            60,
            parent=parent,
            style=clean_style(size=10),
        )

    plain_vertex(
        self,
        "output_bus",
        "",
        400,
        100,
        2,
        180,
        parent=parent,
        style="rounded=0;html=1;fillColor=#7B7F86;strokeColor=none;",
    )
    plain_vertex(
        self,
        "joint_selector",
        "<b>Joint selector</b><br>"
        "<font color=\"#666A70\">seed mean ± std</font><br><br>"
        "<b>lower error</b><br><b>higher progress</b><br><br>"
        "<b>selected H4</b><br>"
        "<font color=\"#666A70\">optional CEM</font>",
        480,
        80,
        75,
        220,
        parent=parent,
        style=clean_style(fill="#FFFFFF", stroke="#7B7F86", size=8),
    )

    raw_edge(
        self,
        "concat_to_fanout",
        "concat_norm",
        "fanout_bus",
        style=edge_style(
            arrow=False,
            extras="exitX=1;exitY=0.5;entryX=0;entryY=0.5;",
        ),
    )
    for index, seed in enumerate((1701, 1702, 1703)):
        port = (0.12, 0.5, 0.88)[index]
        raw_edge(
            self,
            f"fanout_to_mlp_{seed}",
            "fanout_bus",
            f"mlp_{seed}",
            style=edge_style(
                extras=f"exitX=1;exitY={port};entryX=0;entryY=0.5;"
            ),
        )
        raw_edge(
            self,
            f"mlp_{seed}_to_output",
            f"mlp_{seed}",
            "output_bus",
            style=edge_style(
                arrow=False,
                extras=f"exitX=1;exitY=0.5;entryX=0;entryY={port};",
            ),
        )

    raw_edge(
        self,
        "quality_to_selector",
        "output_bus",
        "joint_selector",
        value=(
            "quality predictions<br>"
            "<font color=\"#666A70\">[3 seeds × 2 metrics]</font>"
        ),
        style=edge_style(
            width=1.1,
            extras="exitX=1;exitY=0.5;entryX=0;entryY=0.5;",
        ),
    )


v11.draw_final_d = draw_single_bus_d
v11.add_final_d_drawio = add_single_bus_d_drawio


if __name__ == "__main__":
    v21.figure.main()
