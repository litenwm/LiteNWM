#!/usr/bin/env python3
"""Rebuild panel D as a sparse left-to-right scorer pipeline."""

from __future__ import annotations

import figure1_reference_refined_v20 as v20


v19 = v20.v19
v18 = v19.v18
v11 = v18.v11
base = v18.base
figure = v20.figure

raw_box = v18.raw_box
raw_text = v18.raw_text
raw_arrow = v18.raw_arrow
raw_poly_arrow = v18.raw_poly_arrow
plain_vertex = v18.plain_vertex
raw_edge = v18.raw_edge

previous_text = base.text
previous_arrow = base.arrow
previous_poly_arrow = base.poly_arrow
previous_drawio_edge = base.DrawioBuilder.edge


def line_label(ax, x, y, value, *, size=2.35, weight="bold", ha="center"):
    """Place a compact label directly on a connector without hiding its path."""
    return raw_text(
        ax,
        x,
        y,
        value,
        size=size,
        weight=weight,
        color=base.MUTED,
        ha=ha,
        bbox={"facecolor": base.WHITE, "edgecolor": "none", "pad": 0.55},
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


def draw_horizontal_d(ax):
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

    # The three feature definitions live on the input wires rather than in cards.
    input_y = (56.4, 50.0, 43.6)
    line_label(ax, 106.6, 57.45, "Route · 192-D")
    line_label(ax, 106.6, 51.05, "Future · 796-D")
    line_label(ax, 106.6, 44.65, "Action · 40-D")

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

    # A thin broadcast bus keeps the three seed branches ordered and parallel.
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

    # MLP predictions merge on a line, not in a fictitious "ensemble" module.
    output_bus_x = 134.65
    for y, _ in mlp_specs:
        center_y = y + 2.6
        ax.plot([133.0, output_bus_x], [center_y, center_y], color=base.LINE, lw=0.52, zorder=4)
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
        41.8,
        7.6,
        16.2,
        face=base.WHITE,
        edge=base.LINE,
        lw=0.65,
        radius=0.82,
        z=5,
    )
    raw_text(ax, 144.2, 53.6, "Joint selector", size=2.75, weight="bold")
    raw_text(ax, 144.2, 49.6, "selected H4", size=2.45, weight="bold")
    raw_text(
        ax,
        144.2,
        45.3,
        "optional CEM",
        size=2.05,
        color=base.MUTED,
    )

    for y, label in ((53.1, "ATE  μ±σ ↓"), (46.7, "RTP  μ±σ ↑")):
        raw_arrow(
            ax,
            (output_bus_x, y),
            (140.4, y),
            color=base.LINE,
            lw=0.58,
        )
        line_label(ax, 137.45, y + 1.05, label, size=2.05)


def horizontal_text(ax, x, y, value, *args, **kwargs):
    # The future-input line now carries the more informative typed feature label.
    if value == "H1–H4":
        return None
    return previous_text(ax, x, y, value, *args, **kwargs)


def horizontal_arrow(ax, start, end, *args, **kwargs):
    if start == (98, 45.0) and end == (110, 45.0):
        return raw_poly_arrow(
            ax,
            [(98.0, 45.0), (102.6, 45.0), (102.6, 50.0), (110.4, 50.0)],
            color=base.LINE,
            lw=0.75,
            z=6,
        )
    return previous_arrow(ax, start, end, *args, **kwargs)


def horizontal_poly_arrow(ax, points, *args, **kwargs):
    if points and points[-1] == (110, 58.0):
        return raw_poly_arrow(
            ax,
            [(86.0, 74.0), (101.2, 74.0), (101.2, 56.4), (110.4, 56.4)],
            color=base.LINE,
            lw=0.75,
            z=6,
        )
    if points and points[0] == (41.0, 22.0) and points[-1] == (110, 36.5):
        return raw_poly_arrow(
            ax,
            [(44.2, 22.0), (100.0, 22.0), (100.0, 43.6), (110.4, 43.6)],
            color=base.LINE,
            lw=0.75,
            z=6,
        )
    return previous_poly_arrow(ax, points, *args, **kwargs)


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
        f"strokeWidth={width};{arrow_style}fontFamily=Arial;fontSize=10;"
        f"labelBackgroundColor=#FFFFFF;{extras}"
    )


def horizontal_raw_vertex(
    self,
    vertex_id,
    value,
    x,
    y,
    width,
    height,
    *,
    parent="1",
    style,
):
    if vertex_id == "scorer":
        return plain_vertex(
            self,
            vertex_id,
            "",
            990,
            350,
            560,
            360,
            parent=parent,
            style=(
                "group;container=1;pointerEvents=0;html=1;"
                "fillColor=none;strokeColor=none;fontColor=none;"
            ),
        )
    return plain_vertex(
        self,
        vertex_id,
        value,
        x,
        y,
        width,
        height,
        parent=parent,
        style=style,
    )


def add_horizontal_d_drawio(self, parent):
    horizontal_raw_vertex(
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
    horizontal_raw_vertex(
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
    horizontal_raw_vertex(
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
    horizontal_raw_vertex(
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

    mlp_specs = ((1701, 80), (1702, 160), (1703, 240))
    for seed, y in mlp_specs:
        horizontal_raw_vertex(
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

    horizontal_raw_vertex(
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
    horizontal_raw_vertex(
        self,
        "joint_selector",
        "<b>Joint selector</b><br><br><b>selected H4</b><br>"
        "<font color=\"#666A70\">optional CEM</font>",
        480,
        100,
        75,
        180,
        parent=parent,
        style=clean_style(fill="#FFFFFF", stroke="#7B7F86", size=9),
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

    for metric, port in (("ATE  μ±σ ↓", 0.34), ("RTP  μ±σ ↑", 0.66)):
        edge_id = "ate_to_selector" if metric.startswith("ATE") else "rtp_to_selector"
        raw_edge(
            self,
            edge_id,
            "output_bus",
            "joint_selector",
            value=metric,
            style=edge_style(
                width=1.1,
                extras=f"exitX=1;exitY={port};entryX=0;entryY={port};",
            ),
        )


def external_edge_style(entry_y):
    return edge_style(
        width=1.0,
        extras=f"entryX=0;entryY={entry_y};entryDx=0;entryDy=0;",
    )


def horizontal_drawio_edge(
    self,
    edge_id,
    source,
    target,
    value="",
    *,
    style,
    points=(),
):
    specs = {
        "e6": (
            "Route · 192-D",
            0.18,
            ((1000, 230), (1000, 480), (1095, 480)),
        ),
        "e5": (
            "Future · 796-D",
            0.50,
            ((1020, 540), (1095, 540)),
        ),
        "e7": (
            "Action · 40-D",
            0.82,
            ((1010, 730), (1010, 600), (1095, 600)),
        ),
    }
    if edge_id in specs:
        label, entry_y, routed_points = specs[edge_id]
        return raw_edge(
            self,
            edge_id,
            source,
            "concat_norm",
            value=label,
            style=external_edge_style(entry_y),
            points=routed_points,
        )
    return previous_drawio_edge(
        self,
        edge_id,
        source,
        target,
        value=value,
        style=style,
        points=points,
    )


v11.draw_final_d = draw_horizontal_d
v11.add_final_d_drawio = add_horizontal_d_drawio
v11.raw_vertex = horizontal_raw_vertex
v11.raw_text = raw_text

base.text = horizontal_text
base.arrow = horizontal_arrow
base.poly_arrow = horizontal_poly_arrow
base.DrawioBuilder.edge = horizontal_drawio_edge


if __name__ == "__main__":
    figure.main()
