#!/usr/bin/env python3
"""Lift and shorten the single quality-bus annotation."""

from __future__ import annotations

import figure1_reference_refined_v27 as v27


v26 = v27.v26
previous_inline_label = v26.inline_label
previous_raw_edge = v26.raw_edge


def lifted_bus_label(ax, x, y, value, *, size=2.35, weight="bold"):
    if value == "quality predictions\n[3 seeds × 2 metrics]":
        return ax.text(
            137.5,
            51.35,
            "3-seed estimates\n[error, progress]",
            fontsize=2.05,
            fontweight="normal",
            color=v26.base.MUTED,
            ha="center",
            va="center",
            linespacing=1.08,
            bbox={"facecolor": v26.base.WHITE, "edgecolor": "none", "pad": 0.25},
            zorder=8,
        )
    return previous_inline_label(
        ax,
        x,
        y,
        value,
        size=size,
        weight=weight,
    )


def lifted_bus_edge(
    self,
    edge_id,
    source,
    target,
    value="",
    *,
    style,
    points=(),
):
    if edge_id == "quality_to_selector":
        value = (
            "3-seed estimates<br>"
            "<font color=\"#666A70\">[error, progress]</font>"
        )
        style += (
            "labelPosition=center;verticalLabelPosition=top;"
            "align=center;verticalAlign=bottom;spacingBottom=4;"
        )
        return v27.previous_raw_edge(
            self,
            edge_id,
            source,
            target,
            value=value,
            style=style,
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


v26.inline_label = lifted_bus_label
v26.raw_edge = lifted_bus_edge


if __name__ == "__main__":
    v26.v21.figure.main()
