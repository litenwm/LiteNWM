#!/usr/bin/env python3
"""Clarify the single quality-bus label without widening panel D."""

from __future__ import annotations

import figure1_reference_refined_v26 as v26


previous_inline_label = v26.inline_label
previous_raw_edge = v26.raw_edge


def clearer_bus_label(ax, x, y, value, *, size=2.35, weight="bold"):
    if value == "quality predictions\n[3 seeds × 2 metrics]":
        value = "3-seed quality estimates\n[error, progress]"
        size = 1.95
        weight = "normal"
    return previous_inline_label(
        ax,
        x,
        y,
        value,
        size=size,
        weight=weight,
    )


def clearer_bus_edge(
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
            "3-seed quality estimates<br>"
            "<font color=\"#666A70\">[error, progress]</font>"
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


v26.inline_label = clearer_bus_label
v26.raw_edge = clearer_bus_edge


if __name__ == "__main__":
    v26.v21.figure.main()
