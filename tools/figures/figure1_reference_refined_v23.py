#!/usr/bin/env python3
"""Keep the editable Route edge outside panel D's title band."""

from __future__ import annotations

import figure1_reference_refined_v22 as v22


v21 = v22.v21
previous_edge = v21.base.DrawioBuilder.edge


def clear_title_corridor_edge(
    self,
    edge_id,
    source,
    target,
    value="",
    *,
    style,
    points=(),
):
    if edge_id == "e6":
        return v21.raw_edge(
            self,
            edge_id,
            source,
            "concat_norm",
            value="Route · 192-D",
            style=v21.external_edge_style(0.18),
            points=((970, 230), (970, 480), (1095, 480)),
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


v21.base.DrawioBuilder.edge = clear_title_corridor_edge


if __name__ == "__main__":
    v21.figure.main()
