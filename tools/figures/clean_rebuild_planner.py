#!/usr/bin/env python3
"""Rebuild SAFE CANDIDATE PLANNER on a strict publication-figure grid."""

from __future__ import annotations

import argparse
import base64
import xml.etree.ElementTree as ET
from pathlib import Path

P = "Lcp2UKG11IvjxIMB6Xdv-"
PLANNER = P + "2164"


def find(root, cid):
    for c in root.findall("mxCell"):
        if c.get("id") == cid:
            return c
    raise KeyError(cid)


def geom(cell, x=None, y=None, w=None, h=None, relative=False, points=()):
    old = cell.find("mxGeometry")
    if old is not None:
        cell.remove(old)
    attrs = {"as": "geometry"}
    if relative:
        attrs["relative"] = "1"
    for k, v in (("x", x), ("y", y), ("width", w), ("height", h)):
        if v is not None:
            attrs[k] = str(v)
    g = ET.SubElement(cell, "mxGeometry", attrs)
    if points:
        arr = ET.SubElement(g, "Array", {"as": "points"})
        for px, py in points:
            ET.SubElement(arr, "mxPoint", {"x": str(px), "y": str(py)})
    return g


def vertex(root, cid, value, style, x, y, w, h, parent=PLANNER):
    c = ET.SubElement(root, "mxCell", {
        "id": cid, "parent": parent, "vertex": "1",
        "value": value, "style": style,
    })
    geom(c, x, y, w, h)
    return c


def transform(src: Path, dst: Path):
    tree = ET.parse(src)
    root = tree.getroot().find("./diagram/mxGraphModel/root")
    if root is None:
        raise ValueError("mxGraphModel/root not found")

    # Remove every incremental decoration previously added inside the planner.
    prefixes = ("wm-focus-", "wm-acad-", "wm-paper-", "wm-clean-")
    remove = set()
    for c in root.findall("mxCell"):
        cid = c.get("id", "")
        parent = c.get("parent", "")
        if cid.startswith(prefixes):
            remove.add(cid)
        elif cid.startswith("wm-rich-") and parent == PLANNER:
            remove.add(cid)
    changed = True
    while changed:
        changed = False
        for c in root.findall("mxCell"):
            if c.get("source") in remove or c.get("target") in remove:
                cid = c.get("id")
                if cid not in remove:
                    remove.add(cid)
                    changed = True
    for c in list(root.findall("mxCell")):
        if c.get("id") in remove:
            root.remove(c)

    ink = "#203247"
    navy = "#355B88"
    blue = "#D8E5F2"
    violet = "#E2DDF0"
    ochre = "#F1DFCB"
    gray = "#E7EBF0"
    green_tint = "#DDE9E2"
    green = "#4F755D"
    tan = "#94704F"
    slate = "#637386"
    border = "#91A1B3"
    panel_fill = "#FBFCFD"

    # Panel shell.
    panel = find(root, P + "2077")
    panel.set(
        "style",
        "rounded=1;arcSize=10;whiteSpace=wrap;html=1;"
        f"fillColor={panel_fill};strokeColor={border};strokeWidth=1.5;"
        "fontFamily=Arial;shadow=0;",
    )
    geom(panel, 52, 245, 1020, 505)

    title = find(root, P + "2080")
    title.set("value", "<b>SAFE CANDIDATE PLANNER</b>")
    title.set(
        "style",
        "text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
        f"fontFamily=Arial;fontSize=19;fontColor={ink};fontStyle=1;"
        "align=left;verticalAlign=middle;",
    )
    geom(title, 24, 18, 520, 30)
    vertex(
        root,
        "wm-clean-subtitle",
        "feature fusion  →  ensemble risk  →  safety screening  →  iterative selection",
        "text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
        f"fontFamily=Arial;fontSize=10;fontColor={slate};"
        "align=left;verticalAlign=middle;",
        24,
        50,
        760,
        18,
        parent=P + "2077",
    )
    vertex(
        root,
        "wm-clean-rule",
        "",
        f"shape=line;html=1;strokeColor=#D5DDE6;strokeWidth=1;",
        24,
        76,
        960,
        1,
        parent=P + "2077",
    )

    card_style = (
        "rounded=1;arcSize=8;whiteSpace=wrap;html=1;"
        "fillColor=#FFFFFF;strokeColor=#355B88;strokeWidth=1.5;"
        f"fontFamily=Arial;fontColor={ink};shadow=0;"
    )

    # Fusion card and equal-width / variable-height 815-D stack.
    fusion = find(root, P + "2078")
    fusion.set(
        "value",
        "<b>FEATURE FUSION</b><br><font style=\"font-size:10px;\">815-D concatenated descriptor</font>",
    )
    fusion.set(
        "style",
        card_style + "fontSize=13;align=center;verticalAlign=top;spacingTop=10;",
    )
    geom(fusion, 76, 345, 230, 260)

    blocks = (
        ("DINOv3 · 210", blue, "#6786A8", 40, 398, 9),
        ("LTWM · 336", violet, "#8074A2", 60, 438, 9),
        ("trajectory · 242", ochre, "#B18458", 45, 498, 9),
        ("physics · 24", gray, "#8795A5", 20, 543, 8),
        ("guard · 3", green_tint, "#6D8A79", 14, 563, 8),
    )
    for i, (label, fill, stroke, h, y, fs) in enumerate(blocks, 1):
        vertex(
            root,
            f"wm-clean-fusion-{i}",
            label,
            "rounded=0;whiteSpace=wrap;html=1;"
            f"fillColor={fill};strokeColor={stroke};strokeWidth=1;"
            f"fontFamily=Arial;fontSize={fs};fontColor={ink};fontStyle=1;"
            "align=center;verticalAlign=middle;",
            116,
            y,
            150,
            h,
        )
    vertex(
        root,
        "wm-clean-fusion-total",
        "210 + 336 + 242 + 24 + 3 = <b>815-D</b>",
        "text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
        f"fontFamily=Arial;fontSize=9;fontColor={slate};"
        "align=center;verticalAlign=middle;",
        88,
        582,
        206,
        16,
    )

    # Risk card: three aligned members and one compact statistic.
    risk = find(root, P + "2094")
    risk.set(
        "value",
        "<b>ENSEMBLE RISK</b><br><font style=\"font-size:10px;\">three independent scorers</font>",
    )
    risk.set(
        "style",
        card_style + "fontSize=13;align=center;verticalAlign=top;spacingTop=10;",
    )
    geom(risk, 345, 370, 205, 185)
    risk_rows = (
        ("MLP-1", "0.78", 116),
        ("MLP-2", "0.74", 106),
        ("MLP-3", "0.82", 124),
    )
    for i, (name, score, width) in enumerate(risk_rows, 1):
        y = 424 + (i - 1) * 33
        vertex(
            root,
            f"wm-clean-risk-name-{i}",
            name,
            "text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
            f"fontFamily=Arial;fontSize=9;fontColor={ink};align=left;"
            "verticalAlign=middle;",
            362,
            y,
            42,
            18,
        )
        vertex(
            root,
            f"wm-clean-risk-bar-{i}",
            score,
            "rounded=0;whiteSpace=wrap;html=1;"
            f"fillColor={blue};strokeColor={navy};strokeWidth=1;"
            f"fontFamily=Arial;fontSize=9;fontColor={ink};fontStyle=1;"
            "align=right;verticalAlign=middle;spacingRight=6;",
            410,
            y,
            width,
            18,
        )
    vertex(
        root,
        "wm-clean-risk-stat",
        "<b>μ = 0.78</b>    σ = 0.03",
        "rounded=0;whiteSpace=wrap;html=1;"
        f"fillColor=#F4F7FA;strokeColor={border};strokeWidth=1;"
        f"fontFamily=Arial;fontSize=9;fontColor={ink};"
        "align=center;verticalAlign=middle;",
        370,
        523,
        155,
        22,
    )

    # Safety diamond plus one compact, aligned check list.
    safety = find(root, P + "2104")
    safety.set(
        "value",
        "<font style=\"font-size:17px;\"><b>SAFE?</b></font><br>"
        "<font style=\"font-size:9px;\">hard constraint gate</font>",
    )
    safety.set(
        "style",
        "rhombus;whiteSpace=wrap;html=1;"
        f"fillColor=#F5FAF7;strokeColor={green};strokeWidth=1.5;"
        f"fontFamily=Arial;fontColor={ink};align=center;"
        "verticalAlign=middle;shadow=0;",
    )
    geom(safety, 590, 370, 160, 130)
    vertex(
        root,
        "wm-clean-safety-checks",
        "✓  kinematics<br>✓  collision<br>✓  support",
        "text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
        f"fontFamily=Arial;fontSize=9;fontColor={green};"
        "align=left;verticalAlign=top;spacingTop=2;",
        620,
        510,
        120,
        56,
    )

    # Clean CEM row: three explicit operations and no overlapping dot cloud.
    cem = find(root, "wm-simplified-safe-search")
    cem.set(
        "value",
        "<b>CEM REFINEMENT</b><br><font style=\"font-size:9px;\">3 rounds · 16 samples per round</font>",
    )
    cem.set(
        "style",
        "rounded=1;arcSize=8;whiteSpace=wrap;html=1;"
        f"fillColor=#FFFFFF;strokeColor={tan};strokeWidth=1.5;"
        f"fontFamily=Arial;fontColor={ink};fontSize=12;"
        "align=center;verticalAlign=top;spacingTop=8;shadow=0;",
    )
    geom(cem, 380, 625, 340, 95)
    step_style = (
        "rounded=0;whiteSpace=wrap;html=1;fillColor=#FBF7F2;"
        f"strokeColor={tan};strokeWidth=1;fontFamily=Arial;"
        f"fontSize=8;fontColor={ink};fontStyle=1;"
        "align=center;verticalAlign=middle;"
    )
    for cid, label, x, w in (
        ("wm-clean-cem-sample", "sample K=16", 398, 78),
        ("wm-clean-cem-elite", "retain elite", 506, 78),
        ("wm-clean-cem-update", "update μ, σ", 614, 78),
    ):
        vertex(root, cid, label, step_style, x, 677, w, 24)
    for i, x in enumerate((480, 588), 1):
        vertex(
            root,
            f"wm-clean-cem-arrow-{i}",
            "→",
            "text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
            f"fontFamily=Arial;fontSize=13;fontColor={tan};fontStyle=1;"
            "align=center;verticalAlign=middle;",
            x,
            678,
            24,
            22,
        )

    # Output card with one self-contained SVG trajectory inset.
    output = find(root, "wm-simplified-safe-output")
    output.set(
        "value",
        "<b>SELECTED HYPOTHESIS</b><br>"
        "<font style=\"font-size:20px;\">H4</font>",
    )
    output.set(
        "style",
        card_style + "fontSize=12;align=center;verticalAlign=top;spacingTop=9;",
    )
    geom(output, 800, 345, 210, 245)

    traj_svg = """<svg xmlns="http://www.w3.org/2000/svg" width="360" height="190" viewBox="0 0 360 190">
<rect x="1" y="1" width="358" height="188" rx="5" fill="#F8FAFC" stroke="#AAB8C7" stroke-width="2"/>
<circle cx="182" cy="96" r="26" fill="#EED1BA" stroke="#A8744C" stroke-width="3"/>
<path d="M30 154 C90 132,128 112,178 90 S270 48,330 28" fill="none" stroke="#AAB2BC" stroke-width="3" stroke-dasharray="8 7"/>
<path d="M30 154 C82 98,104 28,180 25 S275 34,330 28" fill="none" stroke="#6F8BAB" stroke-width="4"/>
<path d="M30 154 C90 176,158 181,230 150 S304 74,330 28" fill="none" stroke="#9A7FAE" stroke-width="4"/>
<path d="M30 154 C92 164,150 160,222 142 S295 70,330 28" fill="none" stroke="#355B88" stroke-width="8"/>
<circle cx="30" cy="154" r="9" fill="#355B88"/>
<circle cx="330" cy="28" r="9" fill="#4F755D"/>
<text x="11" y="178" font-family="Arial" font-size="18" fill="#355B88">s₀</text>
<text x="336" y="24" font-family="Arial" font-size="18" fill="#4F755D">g</text>
<text x="245" y="126" font-family="Arial" font-size="18" font-weight="700" fill="#355B88">H4</text>
</svg>"""
    traj_uri = "data:image/svg+xml;base64," + base64.b64encode(
        traj_svg.encode("utf-8")
    ).decode("ascii")
    vertex(
        root,
        "wm-clean-trajectory",
        "",
        "shape=image;html=1;imageAspect=0;aspect=fixed;"
        f"image={traj_uri};strokeColor=none;fillColor=none;",
        815,
        405,
        180,
        95,
    )
    vertex(
        root,
        "wm-clean-output-caption",
        "candidate trajectories  ·  obstacle  ·  goal",
        "text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
        f"fontFamily=Arial;fontSize=8;fontColor={slate};"
        "align=center;verticalAlign=middle;",
        815,
        504,
        180,
        14,
    )
    vertex(
        root,
        "wm-clean-action",
        "<b>a₀ = [dx, dy, dψ]</b><br><font style=\"font-size:8px;\">execute first action</font>",
        "rounded=0;whiteSpace=wrap;html=1;"
        f"fillColor=#F4F7FA;strokeColor={navy};strokeWidth=1;"
        f"fontFamily=Arial;fontSize=9;fontColor={ink};"
        "align=center;verticalAlign=middle;",
        833,
        530,
        145,
        42,
    )

    # Internal edges: all routes are explicitly placed in whitespace.
    edge_common = (
        "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;"
        "jettySize=auto;html=1;strokeWidth=1.7;endArrow=blockThin;"
        "endFill=1;endSize=7;fontFamily=Arial;fontSize=9;"
        "labelBackgroundColor=#FBFCFD;"
    )
    internal = {
        "wm-simplified-edge-fusion-score": (
            "815-D",
            "strokeColor=#355B88;exitX=1;exitY=0.5;entryX=0;entryY=0.5;",
            (),
        ),
        "wm-simplified-edge-score-safety": (
            "risk",
            "strokeColor=#4F755D;exitX=1;exitY=0.5;entryX=0;entryY=0.5;",
            (),
        ),
        "wm-simplified-edge-safety-search": (
            "safe set",
            "strokeColor=#94704F;exitX=0.5;exitY=1;entryX=0.85;entryY=0;",
            ((670, 588), (669, 588)),
        ),
        "wm-simplified-edge-refine-loop": (
            "3 rounds",
            "strokeColor=#637386;dashed=1;dashPattern=5 4;"
            "exitX=0;exitY=0.5;entryX=0.5;entryY=1;",
            ((330, 672), (330, 584), (447, 584)),
        ),
        "wm-simplified-edge-search-output": (
            "best H4",
            "strokeColor=#355B88;exitX=1;exitY=0.5;entryX=0;entryY=0.72;",
            ((765, 672), (765, 521)),
        ),
    }
    for cid, (value, style, points) in internal.items():
        e = find(root, cid)
        e.set("value", value)
        e.set("style", edge_common + style)
        geom(e, relative=True, points=points)

    # External inputs enter the fusion card on separate ports with moderate weight.
    ext = {
        P + "2072": (
            "#45669A",
            "entryX=0;entryY=0.30;entryDx=0;entryDy=0;entryPerimeter=1;",
        ),
        P + "251": (
            "#6557A7",
            "entryX=0;entryY=0.68;entryDx=0;entryDy=0;entryPerimeter=1;",
        ),
    }
    for cid, (color, port) in ext.items():
        e = find(root, cid)
        e.set(
            "style",
            "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;"
            "jettySize=auto;html=1;strokeWidth=2.3;endArrow=blockThin;"
            "endFill=1;endSize=8;fontFamily=Arial;fontSize=9;"
            f"strokeColor={color};" + port,
        )
        geom(e, relative=True)

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
