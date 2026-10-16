
#!/usr/bin/env python3
"""Reduce the planner to a compact, readable submodule for the full figure."""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

P = "Lcp2UKG11IvjxIMB6Xdv-"
PLANNER = P + "2164"


def find(root: ET.Element, cid: str) -> ET.Element:
    for cell in root.findall("mxCell"):
        if cell.get("id") == cid:
            return cell
    raise KeyError(cid)


def geom(cell: ET.Element, x, y, w, h) -> None:
    old = cell.find("mxGeometry")
    if old is not None:
        cell.remove(old)
    ET.SubElement(cell, "mxGeometry", {
        "x": str(x), "y": str(y), "width": str(w), "height": str(h), "as": "geometry"
    })


def remove(root: ET.Element, ids: set[str]) -> None:
    for cell in list(root.findall("mxCell")):
        if cell.get("id") in ids:
            root.remove(cell)


def minimal(src: Path, dst: Path) -> None:
    tree = ET.parse(src)
    root = tree.getroot().find("./diagram/mxGraphModel/root")
    if root is None:
        raise ValueError("missing mxGraphModel/root")

    ink = "#172B4D"
    navy = "#2F5D8A"
    teal = "#2D7A78"
    amber = "#B2733D"
    coral = "#C65D4C"
    slate = "#65758A"

    # Remove decorative cells that read like a standalone subfigure.
    remove(root, {
        "wm-balanced-title-accent",
        "wm-balanced-rail-fusion",
        "wm-balanced-rail-risk",
        "wm-balanced-rail-cem",
        "wm-balanced-rail-output",
        "wm-balanced-safety-number",
        "wm-balanced-fusion-note",
        "wm-balanced-risk-note",
        "wm-balanced-cem-note",
        "wm-clean-output-caption",
    })

    panel = find(root, P + "2077")
    geom(panel, 52, 245, 900, 350)
    panel.set(
        "style",
        "rounded=1;arcSize=7;whiteSpace=wrap;html=1;fillColor=#FBFCFD;"
        "strokeColor=#9AA8B8;strokeWidth=1.2;fontFamily=Arial;shadow=0;",
    )
    title = find(root, P + "2080")
    title.set("value", "<b>SAFE CANDIDATE PLANNER</b>")
    title.set(
        "style",
        "text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
        f"fontFamily=Arial;fontSize=15;fontColor={ink};fontStyle=1;align=left;verticalAlign=middle;",
    )
    geom(title, 24, 16, 460, 24)

    subtitle = find(root, "wm-clean-subtitle")
    subtitle.set("value", "fuse  |  score  |  gate  |  refine  |  select")
    subtitle.set(
        "style",
        "text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
        f"fontFamily=Arial;fontSize=8;fontColor={slate};align=left;verticalAlign=middle;",
    )
    geom(subtitle, 24, 42, 560, 14)
    rule = find(root, "wm-clean-rule")
    geom(rule, 24, 64, 840, 1)

    # Four compact cards and one decision diamond preserve the original topology.
    specs = {
        P + "2078": (76, 325, 205, 190, "#F8FAFC", navy),
        P + "2094": (300, 335, 170, 145, "#FCFAF7", "#9AA8B8"),
        P + "2104": (495, 335, 120, 95, "#EEF7F5", teal),
        "wm-simplified-safe-search": (300, 490, 290, 62, "#FCFAF7", amber),
        "wm-simplified-safe-output": (655, 325, 180, 180, "#F8FAFC", coral),
    }
    for cid, (x, y, w, h, fill, stroke) in specs.items():
        c = find(root, cid)
        geom(c, x, y, w, h)
        shape = "rhombus" if cid == P + "2104" else "rounded=1;arcSize=5"
        c.set(
            "style",
            f"{shape};whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};"
            f"strokeWidth=1.25;fontFamily=Arial;fontColor={ink};shadow=0;",
        )

    # Header text is now a label, not a colored banner.
    header_specs = {
        "wm-balanced-header-fusion": ("FEATURE FUSION", 76, 325, 205, 22),
        "wm-balanced-header-risk": ("ENSEMBLE RISK", 300, 335, 170, 20),
        "wm-balanced-header-cem": ("CEM REFINEMENT", 300, 490, 290, 20),
        "wm-balanced-header-output": ("SELECTED HYPOTHESIS", 655, 325, 180, 22),
    }
    for cid, (label, x, y, w, h) in header_specs.items():
        c = find(root, cid)
        c.set("value", f"<b>{label}</b>")
        c.set(
            "style",
            "text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
            f"fontFamily=Arial;fontSize=9;fontColor={ink};align=left;verticalAlign=middle;"
        )
        geom(c, x + 10, y + 4, w - 18, h)

    # 815-D stack: labels carry the meaning; no duplicate equation.
    blocks = {
        "wm-clean-fusion-1": (100, 353, 157, 24),
        "wm-clean-fusion-2": (100, 377, 157, 30),
        "wm-clean-fusion-3": (100, 407, 157, 24),
        "wm-clean-fusion-4": (100, 431, 157, 12),
        "wm-clean-fusion-5": (100, 443, 157, 10),
    }
    for cid, g in blocks.items():
        geom(find(root, cid), *g)
    total = find(root, "wm-clean-fusion-total")
    total.set("value", "<b>815-D</b>")
    total.set(
        "style",
        "text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
        f"fontFamily=Arial;fontSize=8;fontColor={navy};align=center;verticalAlign=middle;",
    )
    geom(total, 96, 464, 165, 14)

    # Risk: three bars and a single ensemble statistic.
    for i, y, width in ((1, 366, 91), (2, 390, 84), (3, 414, 98)):
        geom(find(root, f"wm-clean-risk-name-{i}"), 311, y, 32, 15)
        geom(find(root, f"wm-clean-risk-bar-{i}"), 347, y, width, 15)
    stat = find(root, "wm-clean-risk-stat")
    geom(stat, 326, 438, 126, 18)
    stat.set("style", "rounded=0;whiteSpace=wrap;html=1;fillColor=#F4F6F8;"
              f"strokeColor=#B4BFCA;strokeWidth=1;fontFamily=Arial;fontSize=8;fontColor={ink};"
              "align=center;verticalAlign=middle;")

    # Safety gate: the diamond carries the decision; checks are one compact line.
    safety = find(root, P + "2104")
    safety.set("value", "<font style=\"font-size:15px;\"><b>SAFE?</b></font><br>"
                "<font style=\"font-size:8px;\">hard constraints</font>")
    checks = find(root, "wm-clean-safety-checks")
    checks.set("value", "✓ kine.   ✓ coll.   ✓ supp.")
    checks.set("style", "text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
                 f"fontFamily=Arial;fontSize=7;fontColor={teal};align=center;verticalAlign=middle;")
    geom(checks, 490, 434, 130, 16)

    # CEM is an unmistakable three-step strip with no repeated explanatory prose.
    cem = find(root, "wm-simplified-safe-search")
    for cid, label, x, w in (
        ("wm-clean-cem-sample", "sample", 314, 60),
        ("wm-clean-cem-elite", "elite", 390, 60),
        ("wm-clean-cem-update", "update", 466, 60),
    ):
        c = find(root, cid)
        c.set("value", label)
        c.set("style", "rounded=0;whiteSpace=wrap;html=1;fillColor=#FBF7F2;"
                  f"strokeColor={amber};strokeWidth=1;fontFamily=Arial;fontSize=8;fontColor={ink};"
                  "align=center;verticalAlign=middle;")
        geom(c, x, 522, w, 18)
    for cid, x in (("wm-clean-cem-arrow-1", 375), ("wm-clean-cem-arrow-2", 451)):
        c = find(root, cid)
        c.set("value", "→")
        geom(c, x, 520, 14, 20)

    # Selected hypothesis: image + H4 + action only.
    h4 = find(root, "wm-balanced-output-h4")
    h4.set("value", "<b>H4</b>")
    h4.set("style", "text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
             f"fontFamily=Arial;fontSize=18;fontColor={coral};align=center;verticalAlign=middle;")
    geom(h4, 700, 351, 90, 22)
    traj = find(root, "wm-clean-trajectory")
    traj.set("style", "shape=image;html=1;imageAspect=0;aspect=fixed;"
             "image=trajectory_planning_inset_small.png;strokeColor=none;fillColor=none;")
    geom(traj, 669, 377, 152, 70)
    action = find(root, "wm-clean-action")
    action.set("value", "<b>a₀</b>  [dx, dy, dψ]")
    action.set("style", "rounded=1;arcSize=4;whiteSpace=wrap;html=1;fillColor=#FCEFEA;"
                 f"strokeColor={coral};strokeWidth=1;fontFamily=Arial;fontSize=8;fontColor={ink};"
                 "align=center;verticalAlign=middle;shadow=0;")
    geom(action, 683, 456, 124, 26)

    # Remove edge labels: the shapes and arrows already make the flow legible.
    edge_ids = (
        "wm-simplified-edge-fusion-score",
        "wm-simplified-edge-score-safety",
        "wm-simplified-edge-safety-search",
        "wm-simplified-edge-refine-loop",
        "wm-simplified-edge-search-output",
    )
    for cid in edge_ids:
        e = find(root, cid)
        e.set("value", "")
    points = {
        "wm-simplified-edge-safety-search": ((555, 450), (555, 475)),
        "wm-simplified-edge-refine-loop": ((278, 521), (278, 482), (385, 482)),
        "wm-simplified-edge-search-output": ((608, 521), (630, 521), (630, 455)),
    }
    for cid, coords in points.items():
        g = find(root, cid).find("mxGeometry")
        if g is None:
            continue
        arr = g.find("Array")
        if arr is not None:
            g.remove(arr)
        arr = ET.SubElement(g, "Array", {"as": "points"})
        for x, y in coords:
            ET.SubElement(arr, "mxPoint", {"x": str(x), "y": str(y)})

    ET.indent(tree, space="  ")
    tree.write(dst, encoding="utf-8", xml_declaration=False)
    print(f"wrote {dst}; cells={len(root.findall('mxCell'))}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=Path)
    ap.add_argument("destination", type=Path)
    ns = ap.parse_args()
    minimal(ns.source, ns.destination)


if __name__ == "__main__":
    main()
