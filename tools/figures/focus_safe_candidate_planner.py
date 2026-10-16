#!/usr/bin/env python3
"""Refine only the SAFE CANDIDATE PLANNER panel in wm_rich.drawio."""

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


def vertex(root, cid, value, style, x, y, w, h, parent=PLANNER):
    cell = ET.SubElement(root, "mxCell", {
        "id": cid,
        "parent": parent,
        "vertex": "1",
        "value": value,
        "style": style,
    })
    geom(cell, x, y, w, h)
    return cell


def edge(root, cid, source, target, style, value=""):
    cell = ET.SubElement(root, "mxCell", {
        "id": cid,
        "parent": PLANNER,
        "edge": "1",
        "source": source,
        "target": target,
        "value": value,
        "style": style,
    })
    geom(cell, relative=True)
    return cell


def get_cell(root, cid):
    for cell in root.findall("mxCell"):
        if cell.get("id") == cid:
            return cell
    raise KeyError(cid)


def transform(src: Path, dst: Path):
    tree = ET.parse(src)
    root = tree.getroot().find("./diagram/mxGraphModel/root")
    if root is None:
        raise ValueError("mxGraphModel/root not found")

    # Remove only prior additions inside the planner panel. The visual
    # context image is retained because it is already embedded in wm_rich.
    remove_ids = set()
    keep_context = {
        "wm-rich-context-card",
        "wm-rich-context-image",
        "wm-rich-context-label",
    }
    for cell in list(root.findall("mxCell")):
        cid = cell.get("id", "")
        parent = cell.get("parent", "")
        if cid.startswith("wm-focus-"):
            remove_ids.add(cid)
        elif cid.startswith("wm-rich-") and parent == PLANNER and cid not in keep_context:
            remove_ids.add(cid)
    for cell in list(root.findall("mxCell")):
        if cell.get("source") in remove_ids or cell.get("target") in remove_ids:
            remove_ids.add(cell.get("id"))
    for cell in list(root.findall("mxCell")):
        if cell.get("id") in remove_ids:
            root.remove(cell)

    # Panel shell and title.
    panel = get_cell(root, P + "2077")
    panel.set(
        "style",
        "rounded=1;arcSize=20;whiteSpace=wrap;html=1;"
        "fillColor=#F8FAFC;strokeColor=#2F5597;strokeWidth=3;"
        "fontFamily=Arial;shadow=0;",
    )
    geom(panel, 52, 245, 1020, 505)

    title = get_cell(root, P + "2080")
    title.set(
        "value",
        "<b>SAFE CANDIDATE PLANNER</b>",
    )
    title.set(
        "style",
        "text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
        "fontFamily=Arial;fontSize=22;fontColor=#17212B;fontStyle=1;"
        "align=left;verticalAlign=middle;",
    )
    geom(title, 24, 18, 540, 34)

    subtitle_style = (
        "text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
        "fontFamily=Arial;fontSize=12;fontColor=#647081;align=left;"
        "verticalAlign=middle;"
    )
    vertex(
        root,
        "wm-focus-subtitle",
        "815-D fusion  ·  ensemble risk  ·  hard safety gate  ·  CEM refinement",
        subtitle_style,
        24,
        52,
        780,
        22,
        parent=P + "2077",
    )
    vertex(
        root,
        "wm-focus-input-caption",
        "candidate-conditioned trajectories",
        "text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
        "fontFamily=Arial;fontSize=11;fontColor=#647081;fontStyle=1;"
        "align=left;verticalAlign=middle;",
        78,
        324,
        240,
        18,
    )

    # 815-D: true vertical concatenation, with widths encoding feature size.
    fusion = get_cell(root, P + "2078")
    fusion.set(
        "value",
        "<b>815-D</b><br><font style=\"font-size:11px;\">VERTICAL CONCAT</font>",
    )
    fusion.set(
        "style",
        "rounded=1;arcSize=16;whiteSpace=wrap;html=1;fillColor=#F1FAFA;"
        "strokeColor=#0C7976;strokeWidth=3;fontFamily=Arial;"
        "fontColor=#0C7976;fontSize=18;align=center;verticalAlign=top;"
        "spacingTop=10;shadow=0;",
    )
    geom(fusion, 70, 345, 245, 255)

    bar_rows = (
        ("DINOv3", "210-D", "#45669A", 155, 398),
        ("LTWM", "336-D", "#6557A7", 205, 430),
        ("traj.", "242-D", "#B27B48", 145, 462),
        ("phys.", "24-D", "#C49A37", 75, 494),
        ("guard", "3-D", "#8A94A2", 45, 526),
    )
    for i, (name, dim, color, width, y) in enumerate(bar_rows, 1):
        vertex(
            root,
            f"wm-focus-815-label-{i}",
            name,
            "text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
            "fontFamily=Arial;fontSize=11;fontColor=#4D5968;fontStyle=1;"
            "align=right;verticalAlign=middle;",
            82,
            y,
            70,
            24,
        )
        vertex(
            root,
            f"wm-focus-815-bar-{i}",
            dim,
            "rounded=1;arcSize=8;whiteSpace=wrap;html=1;"
            f"fillColor={color};strokeColor=#FFFFFF;strokeWidth=1;"
            "fontFamily=Arial;fontSize=10;fontColor=#FFFFFF;fontStyle=1;"
            "align=left;verticalAlign=middle;spacingLeft=9;",
            160,
            y,
            width,
            24,
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
        564,
        210,
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
        534,
        26,
        26,
    )

    # Risk ensemble: three explicit members, score bars, and uncertainty.
    scorer = get_cell(root, P + "2094")
    scorer.set(
        "value",
        "<b>RISK SCORE</b><br><font style=\"font-size:11px;color:#D9EEEC;\">ensemble quality + uncertainty</font>",
    )
    scorer.set(
        "style",
        "rounded=1;arcSize=18;whiteSpace=wrap;html=1;"
        "fillColor=#173A4A;strokeColor=#0C7976;strokeWidth=3;"
        "fontFamily=Arial;fontColor=#FFFFFF;fontSize=14;"
        "align=center;verticalAlign=top;spacingTop=10;shadow=0;",
    )
    geom(scorer, 350, 345, 225, 220)

    risk_rows = (
        ("MLP-1", "0.78", "#55A8A3", 128),
        ("MLP-2", "0.74", "#79C1BE", 112),
        ("MLP-3", "0.82", "#B6DBD8", 142),
    )
    for i, (name, score, color, width) in enumerate(risk_rows, 1):
        y = 408 + (i - 1) * 34
        vertex(
            root,
            f"wm-focus-risk-name-{i}",
            name,
            "text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
            "fontFamily=Arial;fontSize=10;fontColor=#D9EEEC;fontStyle=1;"
            "align=left;verticalAlign=middle;",
            365,
            y,
            45,
            20,
        )
        vertex(
            root,
            f"wm-focus-risk-bar-{i}",
            score,
            "rounded=1;arcSize=8;whiteSpace=wrap;html=1;"
            f"fillColor={color};strokeColor=#FFFFFF;strokeWidth=1;"
            "fontFamily=Arial;fontSize=10;fontColor=#173A4A;fontStyle=1;"
            "align=right;verticalAlign=middle;spacingRight=7;",
            418,
            y,
            width,
            20,
        )
    vertex(
        root,
        "wm-focus-risk-summary",
        "<b>μ = 0.78</b>    σ = 0.03",
        "rounded=1;arcSize=10;whiteSpace=wrap;html=1;"
        "fillColor=#0C7976;strokeColor=#0C7976;strokeWidth=1;"
        "fontFamily=Arial;fontSize=11;fontColor=#FFFFFF;fontStyle=1;"
        "align=center;verticalAlign=middle;",
        382,
        510,
        170,
        24,
    )

    # Safety gate: a large visual decision plus concrete checks.
    safety = get_cell(root, P + "2104")
    safety.set(
        "value",
        "<font style=\"font-size:29px;color:#2C875E;\">✓</font><br>"
        "<font style=\"font-size:14px;color:#214C38;\"><b>SAFE?</b></font><br>"
        "<font style=\"font-size:10px;color:#47725C;\">hard gate</font>",
    )
    safety.set(
        "style",
        "rhombus;whiteSpace=wrap;html=1;fillColor=#EDF7F1;"
        "strokeColor=#2C875E;strokeWidth=3;fontFamily=Arial;"
        "align=center;verticalAlign=middle;shadow=0;",
    )
    geom(safety, 610, 345, 175, 160)
    checks = (
        ("✓  kinematics", "#2C875E"),
        ("✓  collision", "#2C875E"),
        ("✓  support", "#2C875E"),
    )
    for i, (label, color) in enumerate(checks, 1):
        vertex(
            root,
            f"wm-focus-check-{i}",
            label,
            "rounded=1;arcSize=50;whiteSpace=wrap;html=1;"
            f"fillColor=#FFFFFF;strokeColor={color};strokeWidth=1;"
            "fontFamily=Arial;fontSize=10;fontColor=#2C875E;fontStyle=1;"
            "align=center;verticalAlign=middle;",
            620,
            520 + (i - 1) * 24,
            155,
            20,
        )
    vertex(
        root,
        "wm-focus-pass",
        "PASS  →  safe set",
        "rounded=1;arcSize=50;whiteSpace=wrap;html=1;"
        "fillColor=#D9F0DF;strokeColor=#2C875E;strokeWidth=1;"
        "fontFamily=Arial;fontSize=10;fontColor=#2C875E;fontStyle=1;"
        "align=center;verticalAlign=middle;",
        637,
        594,
        120,
        22,
    )

    # CEM refinement loop with samples, elite archive and iteration markers.
    search = get_cell(root, "wm-simplified-safe-search")
    search.set(
        "value",
        "<b>CEM REFINER</b><br><font style=\"font-size:11px;\">16 samples / round</font><br>"
        "<font style=\"font-size:10px;\">elite archive  →  update μ, σ</font>",
    )
    search.set(
        "style",
        "rounded=1;arcSize=18;whiteSpace=wrap;html=1;"
        "fillColor=#FFF8EF;strokeColor=#B27B48;strokeWidth=3;"
        "fontFamily=Arial;fontColor=#6D4826;fontSize=14;"
        "align=center;verticalAlign=middle;spacingTop=7;shadow=0;",
    )
    geom(search, 370, 635, 300, 105)
    for i, (label, color) in enumerate(
        (("1", "#D6A373"), ("2", "#C28655"), ("3", "#B27B48")), 1
    ):
        vertex(
            root,
            f"wm-focus-cem-{i}",
            label,
            "ellipse;whiteSpace=wrap;html=1;"
            f"fillColor={color};strokeColor=#FFFFFF;strokeWidth=1;"
            "fontFamily=Arial;fontSize=11;fontColor=#FFFFFF;fontStyle=1;"
            "align=center;verticalAlign=middle;",
            405 + (i - 1) * 38,
            650,
            24,
            24,
        )
    vertex(
        root,
        "wm-focus-cem-loop",
        "↻",
        "ellipse;whiteSpace=wrap;html=1;fillColor=#B27B48;"
        "strokeColor=#B27B48;fontFamily=Arial;fontSize=24;"
        "fontColor=#FFFFFF;fontStyle=1;align=center;verticalAlign=middle;",
        555,
        654,
        34,
        34,
    )
    vertex(
        root,
        "wm-focus-cem-annotation",
        "sample  →  rank  →  keep elite",
        "text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
        "fontFamily=Arial;fontSize=10;fontColor=#795631;align=center;"
        "verticalAlign=middle;",
        423,
        704,
        195,
        18,
    )

    # H1-H4 candidate cards; H4 is selected and its action is explicit.
    output = get_cell(root, "wm-simplified-safe-output")
    output.set(
        "value",
        "<b>SELECTED</b><br><font style=\"font-size:24px;color:#FFFFFF;\">H4</font><br>"
        "<font style=\"font-size:10px;color:#EAF6EF;\">a₀ = (dx, dy, dψ)</font>",
    )
    output.set(
        "style",
        "rounded=1;arcSize=18;whiteSpace=wrap;html=1;"
        "fillColor=#2C875E;strokeColor=#1D6B47;strokeWidth=3;"
        "fontFamily=Arial;fontColor=#FFFFFF;fontSize=13;"
        "align=center;verticalAlign=top;spacingTop=8;shadow=0;",
    )
    geom(output, 820, 390, 180, 220)
    candidate_specs = (
        ("H1", "#B8C4D1", 108),
        ("H2", "#9DB1C4", 118),
        ("H3", "#7F99AF", 132),
        ("H4", "#D9F0DF", 150),
    )
    for i, (label, color, width) in enumerate(candidate_specs, 1):
        y = 470 + (i - 1) * 30
        vertex(
            root,
            f"wm-focus-hyp-{i}",
            label,
            "rounded=1;arcSize=8;whiteSpace=wrap;html=1;"
            f"fillColor={color};strokeColor=#FFFFFF;strokeWidth=1;"
            "fontFamily=Arial;fontSize=10;fontColor=#173A4A;fontStyle=1;"
            "align=left;verticalAlign=middle;spacingLeft=8;",
            840,
            y,
            width,
            20,
        )
    vertex(
        root,
        "wm-focus-action-pill",
        "→ execute a₀",
        "rounded=1;arcSize=50;whiteSpace=wrap;html=1;"
        "fillColor=#FFFFFF;strokeColor=#D9F0DF;strokeWidth=1;"
        "fontFamily=Arial;fontSize=10;fontColor=#2C875E;fontStyle=1;"
        "align=center;verticalAlign=middle;",
        852,
        592,
        115,
        22,
    )

    # Preserve the embedded RGB context inset, but make its role explicit.
    try:
        card = get_cell(root, "wm-rich-context-card")
        image = get_cell(root, "wm-rich-context-image")
        label = get_cell(root, "wm-rich-context-label")
        geom(card, 820, 250, 180, 112)
        geom(image, 828, 258, 164, 84)
        geom(label, 854, 332, 112, 20)
        label.set("value", "RGB / current view")
    except KeyError:
        pass

    # Re-route only planner-internal edges to the new boxes.
    edge_specs = {
        "wm-simplified-edge-fusion-score": ("815-D", "exitX=1;exitY=0.5;entryX=0;entryY=0.5;strokeColor=#0C7976;"),
        "wm-simplified-edge-score-safety": ("risk", "exitX=1;exitY=0.5;entryX=0;entryY=0.5;strokeColor=#2C875E;"),
        "wm-simplified-edge-safety-search": ("safe set", "exitX=0.5;exitY=1;entryX=0.5;entryY=0;strokeColor=#B27B48;"),
        "wm-simplified-edge-refine-loop": ("3 rounds", "exitX=0;exitY=0.5;entryX=0.5;entryY=1;strokeColor=#8A94A2;dashed=1;dashPattern=7 5;"),
        "wm-simplified-edge-search-output": ("best H4", "exitX=1;exitY=0.5;entryX=0;entryY=0.72;strokeColor=#2C875E;"),
    }
    for cid, (value, ports) in edge_specs.items():
        e = get_cell(root, cid)
        e.set("value", value)
        e.set(
            "style",
            "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;"
            "jettySize=auto;html=1;strokeWidth=3;endArrow=blockThin;"
            "endFill=1;endSize=9;fontFamily=Arial;fontSize=11;"
            "labelBackgroundColor=#F8FAFC;" + ports,
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
