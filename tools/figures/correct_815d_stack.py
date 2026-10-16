#!/usr/bin/env python3
"""Correct the 815-D stack: equal widths, variable vertical heights only."""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

P = "Lcp2UKG11IvjxIMB6Xdv-"
PLANNER = P + "2164"


def geom(cell, x=None, y=None, w=None, h=None):
    old = cell.find("mxGeometry")
    if old is not None:
        cell.remove(old)
    attrs = {"as": "geometry"}
    for k, v in (("x", x), ("y", y), ("width", w), ("height", h)):
        if v is not None:
            attrs[k] = str(v)
    ET.SubElement(cell, "mxGeometry", attrs)


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


def transform(src: Path, dst: Path):
    tree = ET.parse(src)
    root = tree.getroot().find("./diagram/mxGraphModel/root")
    if root is None:
        raise ValueError("mxGraphModel/root not found")

    # Remove the old left-side row labels and the optional height caption.
    remove = {"wm-focus-815-label-" + str(i) for i in range(1, 6)}
    remove.add("wm-focus-815-height-note")
    for c in list(root.findall("mxCell")):
        if c.get("id") in remove:
            root.remove(c)

    fusion = find(root, P + "2078")
    fusion.set(
        "value",
        "<b>815-D</b><br><font style=\"font-size:11px;\">VERTICAL CONCAT</font>",
    )
    fusion.set(
        "style",
        "rounded=1;arcSize=16;whiteSpace=wrap;html=1;"
        "fillColor=#F1FAFA;strokeColor=#0C7976;strokeWidth=3;"
        "fontFamily=Arial;fontColor=#0C7976;fontSize=18;"
        "align=center;verticalAlign=top;spacingTop=10;shadow=0;",
    )
    geom(fusion, 70, 345, 245, 305)

    # Every block has exactly the same width. Only height varies.
    # The heights are proportional to the feature counts while remaining
    # legible at paper-figure scale.
    blocks = (
        (1, "DINOv3 · 210-D", "#45669A", 45, 400, 10),
        (2, "LTWM · 336-D", "#6557A7", 72, 445, 10),
        (3, "traj. · 242-D", "#B27B48", 52, 517, 10),
        (4, "phys. · 24-D", "#C49A37", 26, 569, 9),
        (5, "guard · 3-D", "#8A94A2", 18, 595, 9),
    )
    for i, label, color, height, y, font_size in blocks:
        cell = find(root, f"wm-focus-815-block-{i}")
        cell.set(
            "value", f"<b>{label}</b>"
        )
        cell.set(
            "style",
            "rounded=1;arcSize=8;whiteSpace=wrap;html=1;"
            f"fillColor={color};strokeColor=#FFFFFF;strokeWidth=1;"
            f"fontFamily=Arial;fontSize={font_size};fontColor=#FFFFFF;"
            "fontStyle=1;align=center;verticalAlign=middle;",
        )
        geom(cell, 105, y, 160, height)

    total = find(root, "wm-focus-815-sum")
    total.set("value", "<b>210 + 336 + 242 + 24 + 3 = 815-D</b>")
    total.set(
        "style",
        "rounded=1;arcSize=12;whiteSpace=wrap;html=1;"
        "fillColor=#FFFFFF;strokeColor=#A7CFCB;strokeWidth=1;"
        "fontFamily=Arial;fontSize=10;fontColor=#0C7976;fontStyle=1;"
        "align=center;verticalAlign=middle;",
    )
    geom(total, 84, 620, 215, 23)

    concat = find(root, "wm-focus-815-concat")
    concat.set("value", "⊕")
    geom(concat, 271, 592, 27, 27)

    vertex(
        root,
        "wm-focus-815-height-note",
        "same width  ·  height ∝ feature count",
        "text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
        "fontFamily=Arial;fontSize=10;fontColor=#647081;fontStyle=1;"
        "align=center;verticalAlign=middle;",
        86,
        646,
        213,
        18,
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
