#!/usr/bin/env python3
"""Apply a publication-style visual treatment to SAFE CANDIDATE PLANNER only."""

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
    ET.SubElement(cell, "mxGeometry", attrs)


def find(root, cid):
    for c in root.findall("mxCell"):
        if c.get("id") == cid:
            return c
    raise KeyError(cid)


def vertex(root, cid, value, style, x, y, w, h, parent=PLANNER):
    c = ET.SubElement(root, "mxCell", {
        "id": cid, "parent": parent, "vertex": "1",
        "value": value, "style": style
    })
    geom(c, x, y, w, h)
    return c


def transform(src: Path, dst: Path):
    tree = ET.parse(src)
    root = tree.getroot().find("./diagram/mxGraphModel/root")
    if root is None:
        raise ValueError("mxGraphModel/root not found")

    # Idempotent cleanup of this academic-style layer only.
    remove = {c.get("id") for c in root.findall("mxCell")
              if c.get("id", "").startswith("wm-acad-")}
    remove |= {"wm-focus-815-concat", "wm-focus-815-height-note"}
    for c in list(root.findall("mxCell")):
        if c.get("id") in remove:
            root.remove(c)

    # Palette: paper white, navy ink, slate secondary text, muted blue.
    ink = "#1F3147"
    navy = "#31527E"
    slate = "#637387"
    line = "#8797AA"
    pale = "#F7F9FB"
    fill = "#DDE7F1"
    green = "#4B725B"
    tan = "#8D6B4F"

    panel = find(root, P + "2077")
    panel.set(
        "style",
        "rounded=1;arcSize=10;whiteSpace=wrap;html=1;"
        f"fillColor=#FFFFFF;strokeColor={line};strokeWidth=1.5;"
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
    geom(title, 24, 18, 540, 32)

    subtitle = find(root, "wm-focus-subtitle")
    subtitle.set(
        "value",
        "dimension-preserving fusion  ·  ensemble score  ·  constraint gate  ·  CEM refinement",
    )
    subtitle.set(
        "style",
        "text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
        f"fontFamily=Arial;fontSize=11;fontColor={slate};align=left;"
        "verticalAlign=middle;",
    )
    geom(subtitle, 24, 50, 820, 20)

    input_caption = find(root, "wm-focus-input-caption")
    input_caption.set("value", "candidate-conditioned trajectories")
    input_caption.set(
        "style",
        "text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
        f"fontFamily=Arial;fontSize=10;fontColor={slate};fontStyle=1;"
        "align=left;verticalAlign=middle;",
    )
    geom(input_caption, 76, 324, 240, 18)

    # 815-D vertical stack: equal width, dimension encoded only by height.
    fusion = find(root, P + "2078")
    fusion.set(
        "value",
        "<b>815-D</b><br><font style=\"font-size:10px;\">VERTICAL CONCAT</font>",
    )
    fusion.set(
        "style",
        "rounded=1;arcSize=10;whiteSpace=wrap;html=1;"
        f"fillColor={pale};strokeColor={navy};strokeWidth=1.5;"
        f"fontFamily=Arial;fontColor={ink};fontSize=16;"
        "align=center;verticalAlign=top;spacingTop=10;shadow=0;",
    )
    geom(fusion, 70, 345, 245, 305)

    vertex(
        root,
        "wm-acad-815-scale",
        "height  ∝  feature count",
        f"text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
        f"fontFamily=Arial;fontSize=9;fontColor={slate};"
        "align=center;verticalAlign=middle;",
        93,
        375,
        199,
        16,
    )

    rows = (
        (1, "DINOv3-L/16", "210", 45, 400),
        (2, "LTWM", "336", 72, 445),
        (3, "traj.", "242", 52, 517),
        (4, "phys.", "24", 26, 569),
        (5, "guard", "3", 18, 595),
    )
    for i, name, dim, height, y in rows:
        block = find(root, f"wm-focus-815-block-{i}")
        block.set("value", "")
        block.set(
            "style",
            "rounded=0;whiteSpace=wrap;html=1;"
            f"fillColor={fill};strokeColor={navy};strokeWidth=1;"
            "fontFamily=Arial;fontSize=9;fontColor=#223B5D;"
            "align=center;verticalAlign=middle;",
        )
        geom(block, 124, y, 154, height)
        vertex(
            root,
            f"wm-acad-815-name-{i}",
            name,
            f"text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
            f"fontFamily=Arial;fontSize=9;fontColor={ink};"
            "align=right;verticalAlign=middle;",
            76,
            y,
            42,
            height,
        )
        vertex(
            root,
            f"wm-acad-815-dim-{i}",
            dim,
            f"text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
            f"fontFamily=Arial;fontSize=9;fontColor={slate};"
            "align=left;verticalAlign=middle;",
            282,
            y,
            25,
            height,
        )

    total = find(root, "wm-focus-815-sum")
    total.set("value", "<b>concat [210 | 336 | 242 | 24 | 3]  →  815-D</b>")
    total.set(
        "style",
        "rounded=0;whiteSpace=wrap;html=1;"
        f"fillColor=#FFFFFF;strokeColor={line};strokeWidth=1;"
        f"fontFamily=Arial;fontSize=9;fontColor={ink};"
        "align=center;verticalAlign=middle;",
    )
    geom(total, 84, 620, 215, 23)

    # Risk score: clean three-member table with muted bars.
    scorer = find(root, P + "2094")
    scorer.set(
        "value",
        "<b>RISK SCORE</b><br><font style=\"font-size:10px;\">3-member ensemble</font>",
    )
    scorer.set(
        "style",
        "rounded=1;arcSize=10;whiteSpace=wrap;html=1;"
        f"fillColor=#FFFFFF;strokeColor={navy};strokeWidth=1.5;"
        f"fontFamily=Arial;fontColor={ink};fontSize=14;"
        "align=center;verticalAlign=top;spacingTop=10;shadow=0;",
    )
    geom(scorer, 350, 345, 225, 220)

    risk_rows = (
        ("MLP-1", "0.78", 128),
        ("MLP-2", "0.74", 112),
        ("MLP-3", "0.82", 142),
    )
    for i, (name, score, width) in enumerate(risk_rows, 1):
        y = 408 + (i - 1) * 34
        n = find(root, f"wm-focus-risk-name-{i}")
        n.set("value", name)
        n.set(
            "style",
            f"text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
            f"fontFamily=Arial;fontSize=9;fontColor={ink};align=left;"
            "verticalAlign=middle;",
        )
        geom(n, 365, y, 45, 20)
        b = find(root, f"wm-focus-risk-bar-{i}")
        b.set("value", score)
        b.set(
            "style",
            "rounded=0;whiteSpace=wrap;html=1;"
            f"fillColor={fill};strokeColor={navy};strokeWidth=1;"
            f"fontFamily=Arial;fontSize=9;fontColor={ink};fontStyle=1;"
            "align=right;verticalAlign=middle;spacingRight=6;",
        )
        geom(b, 418, y, width, 20)

    summary = find(root, "wm-focus-risk-summary")
    summary.set("value", "μ = 0.78    σ = 0.03")
    summary.set(
        "style",
        "rounded=0;whiteSpace=wrap;html=1;"
        f"fillColor={pale};strokeColor={line};strokeWidth=1;"
        f"fontFamily=Arial;fontSize=10;fontColor={ink};fontStyle=1;"
        "align=center;verticalAlign=middle;",
    )
    geom(summary, 382, 510, 170, 24)

    # Safety gate: restrained diamond and text-only checks.
    safety = find(root, P + "2104")
    safety.set(
        "value",
        "<font style=\"font-size:18px;\"><b>SAFE?</b></font><br>"
        "<font style=\"font-size:9px;\">hard constraints</font>",
    )
    safety.set(
        "style",
        "rhombus;whiteSpace=wrap;html=1;fillColor=#F6FAF7;"
        f"strokeColor={green};strokeWidth=1.5;fontFamily=Arial;"
        f"fontColor={ink};align=center;verticalAlign=middle;shadow=0;",
    )
    geom(safety, 610, 345, 175, 160)

    for i, label in enumerate(("✓  kinematic", "✓  collision", "✓  support"), 1):
        c = find(root, f"wm-focus-check-{i}")
        c.set("value", label)
        c.set(
            "style",
            "text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
            f"fontFamily=Arial;fontSize=9;fontColor={green};"
            "align=left;verticalAlign=middle;",
        )
        geom(c, 622, 520 + (i - 1) * 22, 150, 18)

    passed = find(root, "wm-focus-pass")
    passed.set("value", "safe set")
    passed.set(
        "style",
        "text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
        f"fontFamily=Arial;fontSize=10;fontColor={green};fontStyle=1;"
        "align=center;verticalAlign=middle;",
    )
    geom(passed, 640, 594, 115, 20)

    # CEM: a simple outlined iteration row, not a decorative widget.
    search = find(root, "wm-simplified-safe-search")
    search.set(
        "value",
        "<b>CEM REFINEMENT</b><br><font style=\"font-size:10px;\">K = 16 samples / round</font><br>"
        "<font style=\"font-size:9px;\">rank  →  elite archive  →  update μ, σ</font>",
    )
    search.set(
        "style",
        "rounded=1;arcSize=10;whiteSpace=wrap;html=1;"
        f"fillColor=#FFFFFF;strokeColor={tan};strokeWidth=1.5;"
        f"fontFamily=Arial;fontColor={ink};fontSize=13;"
        "align=center;verticalAlign=middle;spacingTop=8;shadow=0;",
    )
    geom(search, 370, 635, 300, 105)

    for i in (1, 2, 3):
        c = find(root, f"wm-focus-cem-{i}")
        c.set("value", str(i))
        c.set(
            "style",
            "ellipse;whiteSpace=wrap;html=1;fillColor=#FFFFFF;"
            f"strokeColor={tan};strokeWidth=1;fontFamily=Arial;fontSize=9;"
            f"fontColor={ink};fontStyle=1;align=center;verticalAlign=middle;",
        )
        geom(c, 405 + (i - 1) * 38, 650, 24, 24)

    loop = find(root, "wm-focus-cem-loop")
    loop.set("value", "1  →  2  →  3")
    loop.set(
        "style",
        "text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
        f"fontFamily=Arial;fontSize=10;fontColor={tan};fontStyle=1;"
        "align=center;verticalAlign=middle;",
    )
    geom(loop, 510, 653, 115, 20)

    ann = find(root, "wm-focus-cem-annotation")
    ann.set("value", "sample  /  rank  /  elite update")
    ann.set(
        "style",
        "text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
        f"fontFamily=Arial;fontSize=9;fontColor={slate};align=center;"
        "verticalAlign=middle;",
    )
    geom(ann, 423, 704, 195, 18)

    # Candidate output: a restrained mini-table with one selected row.
    output = find(root, "wm-simplified-safe-output")
    output.set(
        "value",
        "<b>SELECTED HYPOTHESIS</b><br><font style=\"font-size:21px;\">H4</font><br>"
        "<font style=\"font-size:9px;\">a₀ = (dx, dy, dψ)</font>",
    )
    output.set(
        "style",
        "rounded=1;arcSize=10;whiteSpace=wrap;html=1;"
        f"fillColor=#FFFFFF;strokeColor={navy};strokeWidth=1.5;"
        f"fontFamily=Arial;fontColor={ink};fontSize=12;"
        "align=center;verticalAlign=top;spacingTop=8;shadow=0;",
    )
    geom(output, 820, 390, 180, 220)

    for i, label in enumerate(("H1", "H2", "H3", "H4"), 1):
        c = find(root, f"wm-focus-hyp-{i}")
        c.set("value", label + ("  *" if i == 4 else ""))
        c.set(
            "style",
            "rounded=0;whiteSpace=wrap;html=1;"
            f"fillColor={'#DDE7F1' if i < 4 else navy};"
            f"strokeColor={navy};strokeWidth=1;fontFamily=Arial;fontSize=9;"
            f"fontColor={'#223B5D' if i < 4 else '#FFFFFF'};fontStyle=1;"
            "align=left;verticalAlign=middle;spacingLeft=8;",
        )
        geom(c, 840, 470 + (i - 1) * 30, 140, 20)

    action = find(root, "wm-focus-action-pill")
    action.set("value", "execute a₀")
    action.set(
        "style",
        "text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
        f"fontFamily=Arial;fontSize=9;fontColor={navy};fontStyle=1;"
        "align=center;verticalAlign=middle;",
    )
    geom(action, 852, 592, 115, 18)

    # RGB context remains, but receives a thin academic frame and plain caption.
    card = find(root, "wm-rich-context-card")
    card.set(
        "style",
        "rounded=0;whiteSpace=wrap;html=1;fillColor=#FFFFFF;"
        f"strokeColor={line};strokeWidth=1;shadow=0;",
    )
    geom(card, 820, 250, 180, 112)
    image = find(root, "wm-rich-context-image")
    geom(image, 828, 258, 164, 84)
    label = find(root, "wm-rich-context-label")
    label.set("value", "RGB context")
    label.set(
        "style",
        "text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
        f"fontFamily=Arial;fontSize=9;fontColor={slate};fontStyle=1;"
        "align=center;verticalAlign=middle;",
    )
    geom(label, 854, 332, 112, 18)

    # Thin structural arrows and quiet edge labels.
    edges = {
        "wm-simplified-edge-fusion-score": ("815-D", "exitX=1;exitY=0.5;entryX=0;entryY=0.5;strokeColor=" + navy + ";"),
        "wm-simplified-edge-score-safety": ("risk", "exitX=1;exitY=0.5;entryX=0;entryY=0.5;strokeColor=" + green + ";"),
        "wm-simplified-edge-safety-search": ("safe set", "exitX=0.5;exitY=1;entryX=0.5;entryY=0;strokeColor=" + tan + ";"),
        "wm-simplified-edge-refine-loop": ("3 rounds", "exitX=0;exitY=0.5;entryX=0.5;entryY=1;strokeColor=" + slate + ";dashed=1;dashPattern=5 4;"),
        "wm-simplified-edge-search-output": ("best H4", "exitX=1;exitY=0.5;entryX=0;entryY=0.72;strokeColor=" + navy + ";"),
    }
    for cid, (value, port) in edges.items():
        e = find(root, cid)
        e.set("value", value)
        e.set(
            "style",
            "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;"
            "jettySize=auto;html=1;strokeWidth=1.5;endArrow=blockThin;"
            "endFill=1;endSize=7;fontFamily=Arial;fontSize=9;"
            "labelBackgroundColor=#FFFFFF;" + port,
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
