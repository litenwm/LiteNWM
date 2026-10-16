#!/usr/bin/env python3
"""Add research-figure visual semantics to SAFE CANDIDATE PLANNER only."""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

P = "Lcp2UKG11IvjxIMB6Xdv-"
PLANNER = P + "2164"


def geom(cell, x=None, y=None, w=None, h=None, relative=False):
    old = cell.find("mxGeometry")
    if old is not None:
        cell.remove(old)
    attrs = {"as": "geometry"}
    if relative:
        attrs["relative"] = "1"
    for k, v in (("x", x), ("y", y), ("width", w), ("height", h)):
        if v is not None:
            attrs[k] = str(v)
    return ET.SubElement(cell, "mxGeometry", attrs)


def find(root, cid):
    for c in root.findall("mxCell"):
        if c.get("id") == cid:
            return c
    raise KeyError(cid)


def vertex(root, cid, value, style, x, y, w, h):
    c = ET.SubElement(root, "mxCell", {
        "id": cid, "parent": PLANNER, "vertex": "1",
        "value": value, "style": style
    })
    geom(c, x, y, w, h)
    return c


def edge(root, cid, source, target, style):
    c = ET.SubElement(root, "mxCell", {
        "id": cid, "parent": PLANNER, "edge": "1",
        "source": source, "target": target, "style": style
    })
    geom(c, relative=True)
    return c


def transform(src: Path, dst: Path):
    tree = ET.parse(src)
    root = tree.getroot().find("./diagram/mxGraphModel/root")
    if root is None:
        raise ValueError("mxGraphModel/root not found")

    # Remove only this visual enrichment layer for deterministic reruns.
    remove = {c.get("id") for c in root.findall("mxCell")
              if c.get("id", "").startswith("wm-paper-")}
    for c in list(root.findall("mxCell")):
        if c.get("id") in remove:
            root.remove(c)

    ink = "#1F3147"
    navy = "#31527E"
    slate = "#637387"
    line = "#A8B5C3"
    pale = "#F7F9FB"
    blue = "#D6E4F2"
    violet = "#DDD8EE"
    ochre = "#F0DECB"
    green_tint = "#DCE9E2"
    tan = "#8D6B4F"
    green = "#4B725B"

    # Semantic section markers: small, quiet, publication-like.
    header_style = (
        "text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
        "fontFamily=Arial;fontSize=9;fontColor=#537396;fontStyle=1;"
        "align=left;verticalAlign=middle;"
    )
    for cid, label, x, y, w in (
        ("wm-paper-header-fusion", "FUSION", 76, 301, 80),
        ("wm-paper-header-score", "ENSEMBLE SCORE", 350, 301, 130),
        ("wm-paper-header-gate", "CONSTRAINT GATE", 610, 301, 130),
        ("wm-paper-header-select", "CANDIDATE SET", 820, 301, 130),
        ("wm-paper-header-cem", "ITERATIVE REFINEMENT", 370, 604, 180),
    ):
        vertex(root, cid, label, header_style, x, y, w, 16)

    # 815-D stack receives semantic tinting and tiny feature-map marks.
    fills = (blue, violet, ochre, "#E5E9EF", green_tint)
    strokes = ("#597B9D", "#75699A", "#B07E50", "#8695A6", "#668877")
    heights = (45, 72, 52, 26, 18)
    ys = (400, 445, 517, 569, 595)
    for i, (fill, stroke, y, h) in enumerate(zip(fills, strokes, ys, heights), 1):
        block = find(root, f"wm-focus-815-block-{i}")
        block.set("style",
            "rounded=0;whiteSpace=wrap;html=1;"
            f"fillColor={fill};strokeColor={stroke};strokeWidth=1;"
            "fontFamily=Arial;fontSize=9;fontColor=#223B5D;"
            "align=center;verticalAlign=middle;",
        )
        geom(block, 124, y, 154, h)
        # Discrete token marks show that each band is a feature group.
        n = 3 if h >= 40 else (2 if h >= 24 else 1)
        for j in range(n):
            vertex(
                root,
                f"wm-paper-815-token-{i}-{j}",
                "",
                f"rounded=0;whiteSpace=wrap;html=1;fillColor={stroke};"
                "strokeColor=none;opacity=55;",
                236 + j * 7,
                y + max(3, (h - 12) / 2),
                4,
                min(12, max(8, h - 6)),
            )

    # Ensemble score gets a small mean/uncertainty glyph.
    vertex(
        root,
        "wm-paper-risk-axis",
        "uncertainty",
        f"text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
        f"fontFamily=Arial;fontSize=8;fontColor={slate};align=right;"
        "verticalAlign=middle;",
        478,
        390,
        72,
        14,
    )
    vertex(
        root,
        "wm-paper-risk-band",
        "",
        f"rounded=0;whiteSpace=wrap;html=1;fillColor=#EAF0F5;"
        f"strokeColor={line};strokeWidth=1;",
        385,
        493,
        160,
        7,
    )
    vertex(
        root,
        "wm-paper-risk-mean",
        "",
        f"ellipse;whiteSpace=wrap;html=1;fillColor={navy};"
        "strokeColor=#FFFFFF;strokeWidth=1;",
        459,
        489,
        14,
        14,
    )
    vertex(
        root,
        "wm-paper-risk-label",
        "μ ± σ",
        f"text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
        f"fontFamily=Arial;fontSize=9;fontColor={ink};fontStyle=1;"
        "align=left;verticalAlign=middle;",
        365,
        486,
        34,
        18,
    )

    # CEM sample cloud: a data-driven visual cue instead of a decorative icon.
    sample_points = (
        (438, 689, "#B9C7D5"), (452, 679, "#A9BBD0"),
        (469, 693, "#C7B6D7"), (486, 682, "#D7B28C"),
        (502, 697, "#A9BBD0"), (519, 687, "#B9C7D5"),
        (537, 699, "#D7B28C"), (553, 682, "#9FB3C7"),
    )
    for i, (x, y, color) in enumerate(sample_points, 1):
        vertex(
            root,
            f"wm-paper-sample-{i}",
            "",
            f"ellipse;whiteSpace=wrap;html=1;fillColor={color};"
            "strokeColor=#FFFFFF;strokeWidth=1;",
            x,
            y,
            9,
            9,
        )
    vertex(
        root,
        "wm-paper-sample-caption",
        "candidate samples",
        f"text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
        f"fontFamily=Arial;fontSize=8;fontColor={slate};align=left;"
        "verticalAlign=middle;",
        567,
        685,
        82,
        14,
    )

    # Replace the generic H1-H4 bars with a compact top-down trajectory inset.
    # This is the main visual cue for a paper-style safe planner.
    vertex(
        root,
        "wm-paper-traj-map",
        "",
        "rounded=0;whiteSpace=wrap;html=1;fillColor=#F8FAFD;"
        f"strokeColor={line};strokeWidth=1;",
        835,
        448,
        150,
        80,
    )
    vertex(
        root,
        "wm-paper-obstacle",
        "",
        "ellipse;whiteSpace=wrap;html=1;fillColor=#EBCDB8;"
        "strokeColor=#A8744C;strokeWidth=1.2;",
        904,
        475,
        28,
        28,
    )
    vertex(
        root,
        "wm-paper-start",
        "s₀",
        f"ellipse;whiteSpace=wrap;html=1;fillColor={navy};"
        f"strokeColor=#FFFFFF;strokeWidth=1;fontFamily=Arial;fontSize=8;"
        "fontColor=#FFFFFF;align=center;verticalAlign=middle;",
        842,
        508,
        12,
        12,
    )
    vertex(
        root,
        "wm-paper-goal",
        "g",
        f"ellipse;whiteSpace=wrap;html=1;fillColor={green};"
        f"strokeColor=#FFFFFF;strokeWidth=1;fontFamily=Arial;fontSize=8;"
        "fontColor=#FFFFFF;align=center;verticalAlign=middle;",
        968,
        455,
        12,
        12,
    )

    def path(name, color, points, width=1.5, dashed=False):
        nodes = []
        for j, (x, y) in enumerate(points):
            nid = f"wm-paper-path-{name}-{j}"
            vertex(
                root, nid, "",
                "ellipse;whiteSpace=wrap;html=1;fillColor=none;"
                "strokeColor=none;opacity=0;",
                x, y, 4, 4,
            )
            nodes.append(nid)
        style = (
            "edgeStyle=none;rounded=0;orthogonalLoop=0;html=1;"
            f"strokeColor={color};strokeWidth={width};endArrow=none;"
            + ("dashed=1;dashPattern=4 3;" if dashed else "")
        )
        for j in range(len(nodes) - 1):
            edge(root, f"wm-paper-path-{name}-edge-{j}", nodes[j], nodes[j+1], style)

    # Four hypotheses: one colliding, two valid alternatives, H4 selected.
    path("h1", "#A6AEB8", ((849, 514), (880, 506), (912, 489), (944, 476), (973, 461)), 1.2, True)
    path("h2", "#6B86A9", ((849, 514), (875, 490), (896, 452), (939, 447), (973, 461)), 1.4)
    path("h3", "#9A7FAE", ((849, 514), (878, 530), (909, 540), (943, 522), (973, 461)), 1.4)
    path("h4", navy, ((849, 514), (873, 526), (899, 548), (933, 545), (958, 508), (973, 461)), 2.8)

    vertex(
        root,
        "wm-paper-traj-caption",
        "sampled trajectories  /  obstacle  /  goal",
        f"text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
        f"fontFamily=Arial;fontSize=8;fontColor={slate};align=center;"
        "verticalAlign=middle;",
        835,
        529,
        150,
        14,
    )

    # Reuse the H1-H4 nodes as a compact legend under the trajectory plot.
    legend_pos = ((840, 548), (914, 548), (840, 567), (914, 567))
    legend_fill = ("#E6ECF2", "#E6ECF2", "#E6ECF2", navy)
    legend_font = (ink, ink, ink, "#FFFFFF")
    for i, ((x, y), fill_color, font_color) in enumerate(zip(legend_pos, legend_fill, legend_font), 1):
        c = find(root, f"wm-focus-hyp-{i}")
        c.set("value", f"H{i}" + ("  *" if i == 4 else ""))
        c.set(
            "style",
            "rounded=0;whiteSpace=wrap;html=1;"
            f"fillColor={fill_color};strokeColor={navy};strokeWidth=1;"
            f"fontFamily=Arial;fontSize=8;fontColor={font_color};fontStyle=1;"
            "align=center;verticalAlign=middle;",
        )
        geom(c, x, y, 58, 15)

    action = find(root, "wm-focus-action-pill")
    action.set("value", "H4  →  execute a₀")
    action.set(
        "style",
        f"text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
        f"fontFamily=Arial;fontSize=9;fontColor={navy};fontStyle=1;"
        "align=center;verticalAlign=middle;",
    )
    geom(action, 852, 589, 115, 18)

    # Keep RGB context but use it as an evidence inset, not a decorative card.
    label = find(root, "wm-rich-context-label")
    label.set("value", "RGB context")
    label.set(
        "style",
        f"text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
        f"fontFamily=Arial;fontSize=8;fontColor={slate};fontStyle=1;"
        "align=center;verticalAlign=middle;",
    )

    # Add a tiny explicit semantic legend for the visual encodings.
    vertex(
        root,
        "wm-paper-legend",
        "blue: learned features    ochre: physical/action priors    green: safe set",
        f"text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
        f"fontFamily=Arial;fontSize=8;fontColor={slate};align=left;"
        "verticalAlign=middle;",
        76,
        724,
        520,
        14,
    )

    dst.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(dst, encoding="utf-8", xml_declaration=False)
    print(f"wrote {dst}; mxCells={len(root.findall('mxCell'))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=Path)
    ap.add_argument("destination", type=Path)
    ns = ap.parse_args()
    transform(ns.source, ns.destination)


if __name__ == "__main__":
    main()
