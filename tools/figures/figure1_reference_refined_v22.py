#!/usr/bin/env python3
"""Render v21 with connector labels drawn directly by Matplotlib."""

from __future__ import annotations

import figure1_reference_refined_v21 as v21


def line_label(ax, x, y, value, *, size=2.35, weight="bold", ha="center"):
    return ax.text(
        x,
        y,
        value,
        fontsize=size,
        fontweight=weight,
        color=v21.base.MUTED,
        ha=ha,
        va="center",
        bbox={"facecolor": v21.base.WHITE, "edgecolor": "none", "pad": 0.55},
        zorder=8,
    )


v21.line_label = line_label


if __name__ == "__main__":
    v21.figure.main()
