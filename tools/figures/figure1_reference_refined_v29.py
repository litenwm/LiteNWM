#!/usr/bin/env python3
"""Balance panel D typography, module proportions, and output routing."""

from __future__ import annotations

import figure1_reference_refined_v28 as v28


v27 = v28.v27
v26 = v28.v26
v21 = v26.v21
v18 = v26.v18
v11 = v18.v11
base = v26.base

raw_box = v18.raw_box
raw_text = v18.raw_text
raw_arrow = v18.raw_arrow
plain_vertex = v18.plain_vertex
raw_edge = v18.raw_edge


def inline_label(ax, x, y, value, *, size=2.45, weight="bold"):
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
        bbox={"facecolor": base.WHITE, "edgecolor": "none", "pad": 0.3},
        zorder=8,
    )


def mlp_card(ax, y, seed):
    raw_box(
        ax,
        119.5,
        y,
        11.7,
        5.2,
        face=base.SUBPANEL,
        edge=base.LIGHT_LINE,
        lw=0.5,
        radius=0.62,
        z=5,
    )
    raw_text(
        ax,
        120.2,
        y + 3.38,
        f"MLP · {seed}",
        size=2.9,
        weight="bold",
        ha="left",
    )
    raw_text(
        ax,
        120.2,
        y + 1.22,
        "1028 → 640 × 3 → 2",
        size=1.85,
        color=base.MUTED,
        ha="left",
    )


def draw_balanced_d(ax):
    raw_text(
        ax,
        105.0,
        62.8,
        "D  TRAJECTORY-QUALITY SCORING",
        size=5.15,
        weight="bold",
        ha="left",
    )
    ax.plot(
        [105.0, 148.5],
        [61.0, 61.0],
        color="#E1E3E5",
        lw=0.58,
        solid_capstyle="round",
        zorder=2,
    )

    inline_label(ax, 106.55, 57.45, "Route · 192-D", size=2.5)
    inline_label(ax, 106.55, 51.05, "Future · 796-D", size=2.5)
    inline_label(ax, 106.55, 44.65, "Action · 40-D", size=2.5)

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
    raw_text(ax, 113.7, 52.5, "Concat", size=3.2, weight="bold")
    raw_text(ax, 113.7, 49.75, "1028-D", size=2.85, weight="bold")
    raw_text(
        ax,
        113.7,
        45.7,
        "frozen\nz-score",
        size=2.2,
        color=base.MUTED,
    )

    fanout_x = 118.0
    ax.plot([117.0, fanout_x], [49.7, 49.7], color=base.LINE, lw=0.55, zorder=4)
    ax.plot([fanout_x, fanout_x], [42.9, 57.1], color=base.LINE, lw=0.55, zorder=4)

    mlp_specs = ((54.5, 1701), (47.4, 1702), (40.3, 1703))
    for y, seed in mlp_specs:
        center_y = y + 2.6
        raw_arrow(
            ax,
            (fanout_x, center_y),
            (119.5, center_y),
            color=base.LINE,
            lw=0.52,
        )
        mlp_card(ax, y, seed)

    output_bus_x = 132.6
    for y, _ in mlp_specs:
        center_y = y + 2.6
        ax.plot(
            [131.2, output_bus_x],
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

    selector_x = 137.2
    raw_box(
        ax,
        selector_x,
        42.0,
        7.9,
        17.2,
        face=base.WHITE,
        edge=base.LINE,
        lw=0.65,
        radius=0.82,
        z=5,
    )
    raw_text(ax, 141.15, 55.5, "Joint selector", size=2.7, weight="bold")
    raw_text(
        ax,
        141.15,
        51.9,
        "uncertainty-aware gains",
        size=1.78,
        color=base.MUTED,
    )
    raw_text(
        ax,
        141.15,
        48.65,
        "min(Δerror, Δprogress)",
        size=1.72,
        weight="bold",
    )
    raw_text(
        ax,
        141.15,
        45.35,
        "valid-only argmax",
        size=1.85,
        weight="bold",
    )

    raw_arrow(
        ax,
        (output_bus_x, 49.7),
        (selector_x, 49.7),
        color=base.LINE,
        lw=0.62,
    )
    inline_label(
        ax,
        134.9,
        51.25,
        "3-seed estimates\n[error, progress]",
        size=1.85,
        weight="normal",
    )

    raw_arrow(
        ax,
        (145.1, 49.7),
        (148.5, 49.7),
        color=base.INK,
        lw=0.85,
    )
    inline_label(
        ax,
        146.8,
        51.05,
        "selected H4",
        size=1.9,
        weight="bold",
    )
    raw_text(
        ax,
        141.15,
        40.15,
        "optional CEM refinement",
        size=1.55,
        color=base.MUTED,
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


def add_balanced_d_drawio(self, parent):
    plain_vertex(
        self,
        "d_title",
        "D  TRAJECTORY-QUALITY SCORING",
        0,
        8,
        400,
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
        190,
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
            210,
            y,
            140,
            60,
            parent=parent,
            style=clean_style(size=10),
        )

    plain_vertex(
        self,
        "output_bus",
        "",
        370,
        100,
        2,
        180,
        parent=parent,
        style="rounded=0;html=1;fillColor=#7B7F86;strokeColor=none;",
    )
    plain_vertex(
        self,
        "joint_selector",
        "<b>Joint selector</b><br><br>"
        "<font color=\"#666A70\">uncertainty-aware gains</font><br><br>"
        "<b>min(Δerror, Δprogress)</b><br><br>"
        "<b>valid-only argmax</b>",
        425,
        80,
        90,
        220,
        parent=parent,
        style=clean_style(fill="#FFFFFF", stroke="#7B7F86", size=8),
    )
    plain_vertex(
        self,
        "selected_h4",
        "<b>selected H4</b>",
        525,
        165,
        30,
        50,
        parent=parent,
        style=text_style(size=8, align="center", bold=True),
    )
    plain_vertex(
        self,
        "cem_note",
        "optional CEM refinement",
        425,
        310,
        90,
        18,
        parent=parent,
        style=text_style(size=8, align="center", color="#666A70"),
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
            "3-seed estimates<br>"
            "<font color=\"#666A70\">[error, progress]</font>"
        ),
        style=edge_style(
            width=1.1,
            extras=(
                "exitX=1;exitY=0.5;entryX=0;entryY=0.5;"
                "labelPosition=center;verticalLabelPosition=top;"
                "align=center;verticalAlign=bottom;spacingBottom=4;"
            ),
        ),
    )
    raw_edge(
        self,
        "selector_to_selected",
        "joint_selector",
        "selected_h4",
        style=edge_style(
            width=1.2,
            extras="exitX=1;exitY=0.5;entryX=0;entryY=0.5;",
        ),
    )


v11.draw_final_d = draw_balanced_d
v11.add_final_d_drawio = add_balanced_d_drawio


if __name__ == "__main__":
    v21.figure.main()
