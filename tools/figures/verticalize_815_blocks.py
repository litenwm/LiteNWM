#!/usr/bin/env python3
"""Correct the 815-D visual: equal-width stacked blocks with varying heights."""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

PLANNER = "Lcp2UKG11IvjxIMB6Xdv-2164"


def geom(cell, x=None, y=None, w=None, h=None):
    old = cell.find("mxGeometry")
    if old is not None:
        cell.remove(old)
    attrs = {"as": "geometry"}
    for k, v in (("x", x), ("y", y), ("width", w), ("height", h)):
        if v is not None:
            attrs[k] = str(v)
    ET.SubElement(cell, "mxGeometry", attrs)


def vertex(root, cid, value, style, x, y, w, h):
    cell = ET.SubElement(root, "mxCell", {
        "id": cid, "parent": PLANNER, "vertex": "1",
        "value": value, "style": style,
    })
    geom(cell, x, y, w, h)
    return cell


def transform(src: Path, dst: Path):
    tree = ET.parse(src)
    root = tree.getroot().find("./diagram/mxGraphModel/root")
    if root is None:
        raise ValueError("mxGraphModel/root not found")

    ids = {
        c.get("id") for c in root.findall("mxCell")
        if (c.get("id", "").startswith("wm-focus-815-"))
    }
    for cell in list(root.findall("mxCell")):
        if cell.get("id") in ids:
            root.remove(cell)

    # All blocks have identical x and width. Only heights vary.
    blocks = (
        ("DINOv3  ·  210-D", "#45669A", 390, 34, 11),
        ("LTWM  ·  336-D", "#6557A7", 426, 46, 12),
        ("traj.  ·  242-D", "#B27B48", 474, 40, 11),
        ("phys.  ·  24-D", "#C49A37", 516, 24, 10),
        ("guard  ·  3-D", "#8A94A2", 542, 18, 9),
    )
    for i, (label, color, y, h, fs) in enumerate(blocks, 1):
        vertex(
            root,
            f"wm-focus-815-block-{i}",
            label,
            "rounded=1;arcSize=7;whiteSpace=wrap;html=1;"
            f"fillColor={color};strokeColor=#FFFFFF;strokeWidth=1;"
            f"fontFamily=Arial;fontSize={fs};fontColor=#FFFFFF;fontStyle=1;"
            "align=left;verticalAlign=middle;spacingLeft=10;",
            94,
            y,
            184,
            h,
        )
    vertex(
        root,
        "wm-focus-815-sum",
        "<b>210 + 336 + 242 + 24 + 3 = 815</b>",
        "rounded=1;arcSize=12;whiteSpace=wrap;html=1;"
        "fillColor=#FFFFFF;strokeColor=#A7CFCB;strokeWidth=1;"
        "fontFamily=Arial;fontSize=10;fontColor=#0C7976;fontStyle=1;"
        "align=center;verticalAlign=middle;",
        87,
        570,
        198,
        24,
    )
    vertex(
        root,
        "wm-focus-815-concat",
        "⊕",
        "ellipse;whiteSpace=wrap;html=1;fillColor=#0C7976;"
        "strokeColor=#0C7976;fontFamily=Arial;fontSize=14;"
        "fontColor=#FFFFFF;fontStyle=1;align=center;verticalAlign=middle;",
        274,
        535,
        26,
        26,
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
