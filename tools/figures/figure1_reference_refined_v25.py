#!/usr/bin/env python3
"""Give the spelled-out scorer outputs enough horizontal reading space."""

from __future__ import annotations

import figure1_reference_refined_v24 as v24


v23 = v24.v23
v21 = v24.v21

previous_box = v21.raw_box
previous_text = v21.raw_text
previous_arrow = v21.raw_arrow
previous_line_label = v21.line_label
previous_horizontal_vertex = v21.horizontal_raw_vertex
previous_raw_edge = v21.raw_edge


def spaced_selector_box(ax, x, y, w, h, *args, **kwargs):
    if (x, y, w, h) == (140.4, 41.8, 7.6, 16.2):
        x, w = 142.0, 6.0
    return previous_box(ax, x, y, w, h, *args, **kwargs)


def spaced_selector_text(ax, x, y, value, *args, **kwargs):
    selector_specs = {
        "Joint selector": (145.0, 2.4),
        "selected H4": (145.0, 2.15),
        "optional CEM": (145.0, 1.85),
    }
    if value in selector_specs:
        x, size = selector_specs[value]
        kwargs["size"] = size
    return previous_text(ax, x, y, value, *args, **kwargs)


def spaced_selector_arrow(ax, start, end, *args, **kwargs):
    if end in {(140.4, 53.1), (140.4, 46.7)}:
        end = (142.0, end[1])
    return previous_arrow(ax, start, end, *args, **kwargs)


def readable_metric_label(
    ax,
    x,
    y,
    value,
    *,
    size=2.35,
    weight="bold",
    ha="center",
):
    replacements = {
        "ATE  μ±σ ↓": "Path error (m) · μ±σ\nlower is better",
        "RTP  μ±σ ↑": "Goal progress · μ±σ\nhigher is better",
    }
    if value in replacements:
        return ax.text(
            138.25,
            y - 0.15,
            replacements[value],
            fontsize=1.95,
            fontweight="normal",
            color=v21.base.MUTED,
            ha="center",
            va="center",
            linespacing=1.08,
            bbox={"facecolor": v21.base.WHITE, "edgecolor": "none", "pad": 0.32},
            zorder=8,
        )
    return previous_line_label(
        ax,
        x,
        y,
        value,
        size=size,
        weight=weight,
        ha=ha,
    )


def wider_drawio_selector(
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
    if vertex_id == "joint_selector":
        x, width = 500, 55
        value = (
            "<b>Joint selector</b><br><br><b>selected H4</b><br>"
            "<font color=\"#666A70\">optional CEM</font>"
        )
        style = style.replace("fontSize=9;", "fontSize=8;")
    return previous_horizontal_vertex(
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


def concise_metric_edge(
    self,
    edge_id,
    source,
    target,
    value="",
    *,
    style,
    points=(),
):
    replacements = {
        "ate_to_selector": (
            "Path error (m) · μ±σ<br>"
            "<font color=\"#666A70\">lower is better</font>"
        ),
        "rtp_to_selector": (
            "Goal progress · μ±σ<br>"
            "<font color=\"#666A70\">higher is better</font>"
        ),
    }
    if edge_id in replacements:
        return v24.previous_raw_edge(
            self,
            edge_id,
            source,
            target,
            value=replacements[edge_id],
            style=style.replace("fontSize=10;", "fontSize=9;align=center;"),
            points=points,
        )
    return previous_raw_edge(
        self,
        edge_id,
        source,
        target,
        value=value,
        style=style,
        points=points,
    )


v21.raw_box = spaced_selector_box
v21.raw_text = spaced_selector_text
v21.raw_arrow = spaced_selector_arrow
v21.line_label = readable_metric_label
v21.horizontal_raw_vertex = wider_drawio_selector
v21.raw_edge = concise_metric_edge


if __name__ == "__main__":
    v21.figure.main()
