#!/usr/bin/env python3
"""Enrich wm_visual.drawio with compact explanatory labels and visual feature bars."""

from __future__ import annotations

import argparse
import base64
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
    for k, v in (("x", x), ("y", y), ("width", w), ("height", h)):
        if v is not None:
            attrs[k] = str(v)
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


def vertex(root, cid, value, style, x, y, w, h, parent=ROOT):
    c = ET.SubElement(root, "mxCell", {
        "id": cid, "parent": parent, "vertex": "1", "value": value, "style": style
    })
    geom(c, x, y, w, h)
    return c


def edge(root, cid, source, target, style, value=""):
    c = ET.SubElement(root, "mxCell", {
        "id": cid, "parent": ROOT, "edge": "1", "source": source,
        "target": target, "value": value, "style": style
    })
    geom(c, relative=True)
    return c


def transform(src: Path, dst: Path):
    tree = ET.parse(src)
    root = tree.getroot().find("./diagram/mxGraphModel/root")
    if root is None:
        raise ValueError("mxGraphModel/root not found")

    ids = {c.get("id") for c in root.findall("mxCell")
           if (c.get("id") or "").startswith("wm-rich-")}
    ids |= {c.get("id") for c in root.findall("mxCell")
            if (c.get("id") or "").startswith("wm-visual-")}
    drop(root, ids)
    cells = cmap(root)

    # Restore a compact amount of context text.
    title = cells[P + "2080"]
    title.set("value", "SAFE CANDIDATE PLANNER")
    title.set(
        "style",
        "text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
        "fontFamily=Arial;fontSize=21;fontColor=#17212B;fontStyle=1;"
        "align=left;verticalAlign=middle;",
    )
    geom(title, 24, 20, 430, 34)
    vertex(
        root, "wm-rich-planner-subtitle",
        "fuse  ·  score  ·  safety gate  ·  CEM refine",
        "text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
        "fontFamily=Arial;fontSize=12;fontColor=#647081;align=left;verticalAlign=middle;",
        24, 55, 430, 22,
    )
    vertex(
        root, "wm-rich-global-subtitle",
        "frozen context  →  K17 latent futures  →  conservative H4",
        "text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
        "fontFamily=Arial;fontSize=17;fontColor=#647081;align=left;verticalAlign=middle;",
        70, 108, 840, 26, parent="1",
    )

    # 815-D fusion: five proportional bars, with essential dimensions.
    fusion = cells[P + "2078"]
    fusion.set("value", "<b>815-D</b><br><font style=\"font-size:11px;\">fusion</font>")
    fusion.set(
        "style",
        "rounded=1;arcSize=16;whiteSpace=wrap;html=1;"
        "fillColor=#F4FBFB;strokeColor=#0C7976;strokeWidth=3;"
        "fontFamily=Arial;fontColor=#0C7976;fontSize=18;"
        "align=center;verticalAlign=top;spacingTop=10;shadow=0;",
    )
    geom(fusion, 80, 365, 205, 180)

    bar_specs = (
        ("DINO  210", "#45669A", 120, 405),
        ("LTWM  336", "#6557A7", 180, 430),
        ("traj.  242", "#B27B48", 145, 455),
        ("phys.  24", "#C49A37", 55, 480),
        ("guard  3", "#8A94A2", 35, 505),
    )
    for i, (label, color, width, y) in enumerate(bar_specs, 1):
        vertex(
            root, f"wm-rich-815-bar-{i}", label,
            "rounded=1;arcSize=12;whiteSpace=wrap;html=1;"
            f"fillColor={color};strokeColor=#FFFFFF;strokeWidth=1;"
            "fontFamily=Arial;fontSize=10;fontColor=#FFFFFF;fontStyle=1;"
            "align=left;verticalAlign=middle;spacingLeft=8;",
            95, y, width, 20,
        )
    vertex(
        root, "wm-rich-815-caption", "concat",
        "text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
        "fontFamily=Arial;fontSize=10;fontColor=#647081;fontStyle=1;"
        "align=left;verticalAlign=middle;",
        230, 515, 45, 16,
    )

    # Ensemble score: retain three members and the uncertainty summary.
    scorer = cells[P + "2094"]
    scorer.set(
        "value",
        "<b>RISK SCORE</b><br><font style=\"font-size:11px;color:#D9EEEC;\">quality + uncertainty</font>",
    )
    scorer.set(
        "style",
        "rounded=1;arcSize=18;whiteSpace=wrap;html=1;"
        "fillColor=#173A4A;strokeColor=#0C7976;strokeWidth=3;"
        "fontFamily=Arial;fontColor=#FFFFFF;fontSize=14;"
        "align=center;verticalAlign=top;spacingTop=10;shadow=0;",
    )
    geom(scorer, 300, 355, 210, 145)
    member_specs = (("M1", "#55A8A3", 320), ("M2", "#79C1BE", 380), ("M3", "#B6DBD8", 440))
    for i, (label, color, x) in enumerate(member_specs, 1):
        vertex(
            root, f"wm-rich-score-{i}", label,
            "rounded=1;arcSize=8;whiteSpace=wrap;html=1;"
            f"fillColor={color};strokeColor=#FFFFFF;strokeWidth=1;"
            "fontFamily=Arial;fontSize=10;fontColor=#173A4A;fontStyle=1;"
            "align=center;verticalAlign=middle;",
            x, 418, 42, 32,
        )
    vertex(
        root, "wm-rich-score-summary", "μ ± σ",
        "text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
        "fontFamily=Arial;fontSize=12;fontColor=#D9EEEC;fontStyle=1;"
        "align=center;verticalAlign=middle;",
        350, 465, 110, 20,
    )

    # Safety decision plus a short row of concrete checks.
    safety = cells[P + "2104"]
    safety.set(
        "value",
        "<font style=\"font-size:27px;color:#2C875E;\">✓</font><br>"
        "<font style=\"font-size:13px;color:#214C38;\"><b>SAFE?</b></font>",
    )
    safety.set(
        "style",
        "rhombus;whiteSpace=wrap;html=1;"
        "fillColor=#EDF7F1;strokeColor=#2C875E;strokeWidth=3;"
        "fontFamily=Arial;align=center;verticalAlign=middle;shadow=0;",
    )
    geom(safety, 540, 355, 165, 145)
    vertex(
        root, "wm-rich-gate-row", "physical   support   harm",
        "rounded=1;arcSize=50;whiteSpace=wrap;html=1;"
        "fillColor=#FFFFFF;strokeColor=#2C875E;strokeWidth=1;"
        "fontFamily=Arial;fontSize=10;fontColor=#2C875E;fontStyle=1;"
        "align=center;verticalAlign=middle;",
        540, 505, 165, 22,
    )

    # CEM search retains the closed-loop semantics and one useful numeric cue.
    search = cells["wm-simplified-safe-search"]
    search.set(
        "value",
        "<font style=\"font-size:31px;color:#B27B48;\">↻</font><br>"
        "<font style=\"font-size:14px;color:#6D4826;\"><b>CEM ×3</b></font><br>"
        "<font style=\"font-size:10px;color:#795631;\">16 / round · safe archive</font>",
    )
    geom(search, 410, 545, 245, 120)

    output = cells["wm-simplified-safe-output"]
    output.set(
        "value",
        "<font style=\"font-size:24px;color:#FFFFFF;\"><b>H4</b></font><br>"
        "<font style=\"font-size:11px;color:#EAF6EF;\">select a₀</font>",
    )
    geom(output, 755, 395, 165, 150)

    # Short labels on the structural edges.
    cells["wm-simplified-edge-fusion-score"].set("value", "815-D")
    cells["wm-simplified-edge-score-safety"].set("value", "risk")
    cells["wm-simplified-edge-safety-search"].set("value", "safe set")
    cells["wm-simplified-edge-refine-loop"].set("value", "3 rounds")
    cells["wm-simplified-edge-search-output"].set("value", "best H4")

    # Small RGB context inset generated with GPT-Image2. Embed it so the
    # drawio remains self-contained when copied or submitted.
    context_png = src.parent / "robot_corridor_context.png"
    context_uri = "data:image/png;base64," + base64.b64encode(
        context_png.read_bytes()
    ).decode("ascii")
    vertex(
        root, "wm-rich-context-card", "",
        "rounded=1;arcSize=12;whiteSpace=wrap;html=1;"
        "fillColor=#FFFFFF;strokeColor=#B8C4D1;strokeWidth=2;shadow=1;",
        735, 265, 190, 112,
    )
    vertex(
        root, "wm-rich-context-image", "",
        "shape=image;html=1;imageAspect=0;aspect=fixed;"
        f"image={context_uri};strokeColor=none;fillColor=none;",
        744, 273, 172, 90,
    )
    vertex(
        root, "wm-rich-context-label", "RGB context",
        "rounded=1;arcSize=50;whiteSpace=wrap;html=1;"
        "fillColor=#173A4A;strokeColor=#173A4A;strokeWidth=1;"
        "fontFamily=Arial;fontSize=10;fontColor=#FFFFFF;fontStyle=1;"
        "align=center;verticalAlign=middle;",
        785, 350, 90, 20,
    )

    # A small, explicit bottom takeaway restores the paper-figure reading cue.
    vertex(
        root, "wm-rich-bottom-takeaway",
        "frozen visual tokens  →  candidate-conditioned futures  →  conservative H4",
        "text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
        "fontFamily=Arial;fontSize=13;fontColor=#647081;fontStyle=1;"
        "align=left;verticalAlign=middle;",
        70, 1142, 940, 20, parent="1",
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
