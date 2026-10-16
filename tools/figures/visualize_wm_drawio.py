#!/usr/bin/env python3
"""Turn wm_simplified.drawio into a low-text, icon-led paper figure."""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

P = "Lcp2UKG11IvjxIMB6Xdv-"
ROOT = P + "2164"


def cmap(root):
    return {c.get("id", ""): c for c in root.findall("mxCell")}


def geom(cell, x=None, y=None, w=None, h=None, relative=False):
    old = cell.find("mxGeometry")
    if old is not None:
        cell.remove(old)
    attrs = {"as": "geometry"}
    if relative:
        attrs["relative"] = "1"
    if x is not None:
        attrs["x"] = str(x)
    if y is not None:
        attrs["y"] = str(y)
    if w is not None:
        attrs["width"] = str(w)
    if h is not None:
        attrs["height"] = str(h)
    return ET.SubElement(cell, "mxGeometry", attrs)


def drop(root, ids):
    changed = True
    while changed:
        changed = False
        for c in root.findall("mxCell"):
            if c.get("parent") in ids and c.get("id") not in ids:
                ids.add(c.get("id"))
                changed = True
    for c in list(root.findall("mxCell")):
        if c.get("id") in ids or c.get("source") in ids or c.get("target") in ids:
            root.remove(c)


def vertex(root, cid, value, style, x, y, w, h):
    c = ET.SubElement(root, "mxCell", {
        "id": cid, "parent": ROOT, "vertex": "1", "value": value, "style": style
    })
    geom(c, x, y, w, h)
    return c


def edge(root, cid, source, target, style):
    c = ET.SubElement(root, "mxCell", {
        "id": cid, "parent": ROOT, "edge": "1", "source": source,
        "target": target, "value": "", "style": style
    })
    geom(c, relative=True)
    return c


def transform(src, dst):
    tree = ET.parse(src)
    root = tree.getroot().find("./diagram/mxGraphModel/root")
    if root is None:
        raise ValueError("mxGraphModel/root not found")

    visual_ids = {c.get("id") for c in root.findall("mxCell")
                  if (c.get("id") or "").startswith("wm-visual-")}
    remove = {
        P + "4", P + "68", P + "77", P + "78", P + "81", P + "84",
        P + "87", P + "128", P + "1906", P + "2079", P + "2081",
        P + "268", P + "269", P + "270", P + "271", P + "272",
        P + "273", P + "274", P + "275", P + "276", P + "277",
        P + "278", P + "279",
    } | visual_ids
    drop(root, remove)
    cells = cmap(root)

    cells[P + "2"].set("value", "INFERENCE")
    cells[P + "19"].set("value", "H=4")
    cells[P + "46"].set("value", "1024→384")

    title = cells[P + "2080"]
    title.set("value", "SAFE PLANNER")
    title.set(
        "style",
        "text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
        "fontFamily=Arial;fontSize=25;fontColor=#17212B;fontStyle=1;"
        "align=left;verticalAlign=middle;",
    )
    geom(title, 24, 22, 300, 40)

    fusion = cells[P + "2078"]
    fusion.set("value", "<font style=\"font-size:32px;\">⊕</font>")
    fusion.set(
        "style",
        "ellipse;whiteSpace=wrap;html=1;aspect=fixed;"
        "fillColor=#FFFFFF;strokeColor=#0C7976;strokeWidth=4;"
        "fontFamily=Arial;fontColor=#0C7976;align=center;verticalAlign=middle;"
        "shadow=0;",
    )
    geom(fusion, 105, 400, 110, 110)

    scorer = cells[P + "2094"]
    scorer.set(
        "value",
        "<b>SCORE</b><br><font style=\"font-size:20px;color:#D9EEEC;\">×3</font>",
    )
    scorer.set(
        "style",
        "rounded=1;arcSize=18;whiteSpace=wrap;html=1;"
        "fillColor=#173A4A;strokeColor=#0C7976;strokeWidth=3;"
        "fontFamily=Arial;fontColor=#FFFFFF;fontSize=15;fontStyle=0;"
        "align=center;verticalAlign=top;spacingTop=13;shadow=0;",
    )
    geom(scorer, 275, 360, 195, 135)

    safety = cells[P + "2104"]
    safety.set(
        "value",
        "<font style=\"font-size:32px;color:#2C875E;\">✓</font><br>"
        "<font style=\"font-size:13px;color:#214C38;\"><b>SAFE?</b></font>",
    )
    safety.set(
        "style",
        "rhombus;whiteSpace=wrap;html=1;"
        "fillColor=#EDF7F1;strokeColor=#2C875E;strokeWidth=3;"
        "fontFamily=Arial;align=center;verticalAlign=middle;shadow=0;",
    )
    geom(safety, 540, 355, 165, 150)

    search = cells["wm-simplified-safe-search"]
    search.set(
        "value",
        "<font style=\"font-size:34px;color:#B27B48;\">↻</font><br>"
        "<font style=\"font-size:14px;color:#6D4826;\"><b>CEM ×3</b></font>",
    )
    search.set(
        "style",
        "ellipse;whiteSpace=wrap;html=1;"
        "fillColor=#FBF5EF;strokeColor=#B27B48;strokeWidth=3;"
        "fontFamily=Arial;align=center;verticalAlign=middle;shadow=0;",
    )
    geom(search, 410, 545, 245, 120)

    output = cells["wm-simplified-safe-output"]
    output.set(
        "value",
        "<font style=\"font-size:24px;color:#FFFFFF;\"><b>H4</b></font>",
    )
    output.set(
        "style",
        "rounded=1;arcSize=18;whiteSpace=wrap;html=1;"
        "fillColor=#2C875E;strokeColor=#246D4C;strokeWidth=3;"
        "fontFamily=Arial;align=center;verticalAlign=top;spacingTop=14;shadow=1;",
    )
    geom(output, 755, 395, 165, 150)

    for eid in (
        "wm-simplified-edge-fusion-score",
        "wm-simplified-edge-score-safety",
        "wm-simplified-edge-safety-search",
        "wm-simplified-edge-refine-loop",
        "wm-simplified-edge-search-output",
    ):
        cells[eid].set("value", "")

    text_style = (
        "text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
        "fontFamily=Arial;fontSize=12;fontColor=#0C7976;fontStyle=1;"
        "align=center;verticalAlign=middle;"
    )
    vertex(root, "wm-visual-fuse-label", "FUSE", text_style, 115, 515, 90, 22)

    dot_style = "ellipse;html=1;aspect=fixed;strokeColor=#FFFFFF;strokeWidth=1;"
    for cid, color, x, y in (
        ("wm-visual-fuse-blue", "#45669A", 122, 420),
        ("wm-visual-fuse-purple", "#6557A7", 172, 420),
        ("wm-visual-fuse-brown", "#B27B48", 122, 472),
        ("wm-visual-fuse-grey", "#8A94A2", 172, 472),
    ):
        vertex(root, cid, "", dot_style + f"fillColor={color};", x, y, 12, 12)

    bar_style = (
        "rounded=1;arcSize=20;whiteSpace=wrap;html=1;"
        "strokeColor=#D9EEEC;strokeWidth=1;"
    )
    vertex(root, "wm-visual-score-b1", "", bar_style + "fillColor=#55A8A3;", 310, 444, 25, 30)
    vertex(root, "wm-visual-score-b2", "", bar_style + "fillColor=#79C1BE;", 360, 430, 25, 44)
    vertex(root, "wm-visual-score-b3", "", bar_style + "fillColor=#B6DBD8;", 410, 416, 25, 58)

    ring_style = "ellipse;html=1;aspect=fixed;fillColor=#FFFFFF;strokeColor=#B27B48;strokeWidth=2;"
    vertex(root, "wm-visual-cem-r1", "", ring_style, 445, 585, 12, 12)
    vertex(root, "wm-visual-cem-r2", "", ring_style, 605, 585, 12, 12)
    vertex(root, "wm-visual-cem-core", "", dot_style + "fillColor=#2C875E;", 525, 620, 12, 12)

    out_dot_style = "ellipse;html=1;aspect=fixed;fillColor=#FFFFFF;strokeColor=#FFFFFF;strokeWidth=1;"
    coords = ((785, 500), (820, 478), (855, 493), (890, 462))
    out_ids = []
    for i, (x, y) in enumerate(coords, 1):
        cid = f"wm-visual-h4-{i}"
        out_ids.append(cid)
        vertex(root, cid, "", out_dot_style, x, y, 10, 10)

    path_style = (
        "edgeStyle=none;rounded=1;html=1;strokeColor=#FFFFFF;strokeWidth=3;"
        "endArrow=none;startArrow=none;"
    )
    for i in range(3):
        edge(root, f"wm-visual-h4-edge-{i+1}", out_ids[i], out_ids[i+1], path_style)

    vertex(
        root,
        "wm-visual-fallback",
        "fallback",
        "rounded=1;arcSize=50;whiteSpace=wrap;html=1;"
        "fillColor=#F0F2F4;strokeColor=#8A94A2;strokeWidth=1;"
        "fontFamily=Arial;fontSize=10;fontColor=#647081;fontStyle=1;"
        "align=center;verticalAlign=middle;",
        785,
        565,
        105,
        24,
    )
    edge(
        root,
        "wm-visual-fallback-edge",
        "wm-visual-fallback",
        "wm-simplified-safe-output",
        "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;"
        "html=1;strokeColor=#8A94A2;strokeWidth=2;dashed=1;dashPattern=6 5;"
        "endArrow=blockThin;endFill=1;endSize=7;exitX=.5;exitY=0;entryX=.5;entryY=1;",
    )

    dst.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(dst, encoding="utf-8", xml_declaration=False)
    print(f"wrote {dst}")
    print(f"mxCells: {len(root.findall('mxCell'))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=Path)
    ap.add_argument("destination", type=Path)
    ns = ap.parse_args()
    transform(ns.source, ns.destination)


if __name__ == "__main__":
    main()
