#!/usr/bin/env python3
"""Spell out the two scorer outputs and their optimization directions."""

from __future__ import annotations

import figure1_reference_refined_v23 as v23


v22 = v23.v22
v21 = v23.v21
previous_line_label = v22.line_label
previous_raw_edge = v21.raw_edge


def explanatory_line_label(
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
        "ATE  μ±σ ↓": "Mean path error (m) · μ±σ\nlower is better",
        "RTP  μ±σ ↑": "Goal progress ratio · μ±σ\nhigher is better",
    }
    if value in replacements:
        return ax.text(
            x,
            y - 0.15,
            replacements[value],
            fontsize=1.72,
            fontweight="normal",
            color=v21.base.MUTED,
            ha="center",
            va="center",
            linespacing=1.08,
            bbox={"facecolor": v21.base.WHITE, "edgecolor": "none", "pad": 0.35},
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


def explanatory_raw_edge(
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
            "Mean path error (m) · μ±σ<br>"
            "<font color=\"#666A70\">lower is better</font>"
        ),
        "rtp_to_selector": (
            "Goal progress ratio · μ±σ<br>"
            "<font color=\"#666A70\">higher is better</font>"
        ),
    }
    if edge_id in replacements:
        value = replacements[edge_id]
        style = style.replace("fontSize=10;", "fontSize=8;align=center;")
    return previous_raw_edge(
        self,
        edge_id,
        source,
        target,
        value=value,
        style=style,
        points=points,
    )


v21.line_label = explanatory_line_label
v21.raw_edge = explanatory_raw_edge


if __name__ == "__main__":
    v21.figure.main()
