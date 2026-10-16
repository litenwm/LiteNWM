#!/usr/bin/env python3
"""Remove Draw.io corridor crossings by staggering the three turn heights."""

from __future__ import annotations

import figure1_reference_refined_v19 as v19


base = v19.base
figure = v19.figure
raw_edge = v19.raw_edge
previous_edge = base.DrawioBuilder.edge


def corrected_corridor_edge(
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
        "e5": ("future_features", ((1000, 504), (1000, 202), (1290, 202))),
        "e6": ("route_features", ((990, 232), (990, 209), (1095, 209))),
        "e7": ("action_features", ((1010, 776), (1010, 195), (1485, 195))),
    }
    if edge_id in specs:
        mapped_target, mapped_points = specs[edge_id]
        return raw_edge(
            self,
            edge_id,
            source,
            mapped_target,
            value=value,
            style=v19.unified_external_style(),
            points=mapped_points,
        )
    return previous_edge(
        self,
        edge_id,
        source,
        target,
        value=value,
        style=style,
        points=points,
    )


base.DrawioBuilder.edge = corrected_corridor_edge


if __name__ == "__main__":
    figure.main()
