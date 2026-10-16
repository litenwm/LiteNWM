#!/usr/bin/env python3
"""Route all three D inputs through one ordered C–D wiring corridor."""

from __future__ import annotations

import figure1_reference_refined_v18 as v18


base = v18.base
figure = v18.figure
v11 = v18.v11

raw_text = v18.raw_text
raw_poly_arrow = v18.raw_poly_arrow
plain_vertex = v18.plain_vertex
raw_edge = v18.raw_edge

previous_text = base.text
previous_arrow = base.arrow
previous_poly_arrow = base.poly_arrow
previous_drawio_edge = base.DrawioBuilder.edge


def corridor_text(ax, x, y, value, *args, **kwargs):
    if value == "H1–H4":
        return raw_text(
            ax,
            104.15,
            56.8,
            value,
            size=3.1,
            color=base.MUTED,
            ha="left",
        )
    return previous_text(ax, x, y, value, *args, **kwargs)


def corridor_arrow(ax, start, end, *args, **kwargs):
    if start == (98, 45.0) and end == (110, 45.0):
        return raw_poly_arrow(
            ax,
            [
                (98.0, 45.0),
                (102.5, 45.0),
                (102.5, 73.7),
                (126.25, 73.7),
                (126.25, 72.5),
            ],
            color=base.LINE,
            lw=0.75,
            z=6,
        )
    return previous_arrow(ax, start, end, *args, **kwargs)


def corridor_poly_arrow(ax, points, *args, **kwargs):
    if points and points[-1] == (110, 58.0):
        return raw_poly_arrow(
            ax,
            [
                (86.0, 74.0),
                (101.4, 74.0),
                (101.4, 73.0),
                (111.75, 73.0),
                (111.75, 72.5),
            ],
            color=base.LINE,
            lw=0.75,
            z=6,
        )
    if points and points[0] == (41.0, 22.0) and points[-1] == (110, 36.5):
        return raw_poly_arrow(
            ax,
            [
                (44.2, 22.0),
                (103.6, 22.0),
                (103.6, 74.4),
                (140.75, 74.4),
                (140.75, 72.5),
            ],
            color=base.LINE,
            lw=0.75,
            z=6,
        )
    return previous_poly_arrow(ax, points, *args, **kwargs)


def corridor_group_vertex(
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
            1020,
            130,
            560,
            620,
            parent=parent,
            style=(
                "group;container=1;pointerEvents=0;html=1;"
                "fillColor=none;strokeColor=none;fontColor=none;"
            ),
        )
    return v18.vertical_raw_vertex(
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


def unified_external_style():
    return (
        "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;"
        "jettySize=auto;html=1;strokeColor=#7B7F86;"
        "strokeWidth=1;endArrow=block;endFill=1;"
        "fontFamily=Arial;fontSize=11;labelBackgroundColor=#FFFFFF;"
    )


def corridor_drawio_edge(
    self,
    edge_id,
    source,
    target,
    value="",
    *,
    style,
    points=(),
):
    if edge_id == "e5":
        return raw_edge(
            self,
            edge_id,
            source,
            "future_features",
            value=value,
            style=unified_external_style(),
            points=((1000, 504), (1000, 202), (1290, 202)),
        )
    if edge_id == "e6":
        return raw_edge(
            self,
            edge_id,
            source,
            "route_features",
            value=value,
            style=unified_external_style(),
            points=((990, 232), (990, 195), (1095, 195)),
        )
    if edge_id == "e7":
        return raw_edge(
            self,
            edge_id,
            source,
            "action_features",
            value=value,
            style=unified_external_style(),
            points=((1010, 776), (1010, 209), (1485, 209)),
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


base.text = corridor_text
base.arrow = corridor_arrow
base.poly_arrow = corridor_poly_arrow
v11.raw_vertex = corridor_group_vertex
base.DrawioBuilder.edge = corridor_drawio_edge


if __name__ == "__main__":
    figure.main()
