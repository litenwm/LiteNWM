
#!/usr/bin/env python3
"""Compact the planner block for embedding inside the full model figure."""

from __future__ import annotations

import argparse
import base64
import xml.etree.ElementTree as ET
from pathlib import Path

P = "Lcp2UKG11IvjxIMB6Xdv-"


def find(root: ET.Element, cid: str) -> ET.Element:
    for cell in root.findall("mxCell"):
        if cell.get("id") == cid:
            return cell
    raise KeyError(cid)


def geom(cell: ET.Element, x, y, w, h) -> None:
    old = cell.find("mxGeometry")
    if old is not None:
        cell.remove(old)
    ET.SubElement(
        cell,
        "mxGeometry",
        {"x": str(x), "y": str(y), "width": str(w), "height": str(h), "as": "geometry"},
    )


def compact(src: Path, dst: Path, image: Path) -> None:
    tree = ET.parse(src)
    root = tree.getroot().find("./diagram/mxGraphModel/root")
    if root is None:
        raise ValueError("missing mxGraphModel/root")

    ink = "#172B4D"
    navy = "#2F5D8A"
    amber = "#B2733D"
    teal = "#2D7A78"
    coral = "#C65D4C"
    slate = "#65758A"

    panel = find(root, P + "2077")
    geom(panel, 52, 245, 900, 420)
    panel.set(
        "style",
        "rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#FBFCFD;"
        "strokeColor=#8797AA;strokeWidth=1.3;fontFamily=Arial;shadow=0;",
    )
    title = find(root, P + "2080")
    geom(title, 24, 18, 470, 28)
    title.set("style", "text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
              f"fontFamily=Arial;fontSize=17;fontColor={ink};fontStyle=1;align=left;verticalAlign=middle;")
    subtitle = find(root, "wm-clean-subtitle")
    subtitle.set("value", "feature fusion  |  risk ensemble  |  safety gate  |  CEM  |  H4")
    geom(subtitle, 24, 48, 700, 16)
    rule = find(root, "wm-clean-rule")
    geom(rule, 24, 72, 840, 1)

    fusion = find(root, P + "2078")
    geom(fusion, 76, 335, 212, 230)
    fusion.set("style", "rounded=1;arcSize=6;whiteSpace=wrap;html=1;fillColor=#F8FAFC;"
               f"strokeColor={navy};strokeWidth=1.3;fontFamily=Arial;fontColor={ink};shadow=0;")
    risk = find(root, P + "2094")
    geom(risk, 315, 350, 180, 165)
    risk.set("style", "rounded=1;arcSize=6;whiteSpace=wrap;html=1;fillColor=#FCFAF7;"
               f"strokeColor={amber};strokeWidth=1.3;fontFamily=Arial;fontColor={ink};shadow=0;")
    safety = find(root, P + "2104")
    geom(safety, 520, 350, 130, 105)
    safety.set("style", "rhombus;whiteSpace=wrap;html=1;fillColor=#EEF7F5;"
                f"strokeColor={teal};strokeWidth=1.5;fontFamily=Arial;fontColor={ink};"
                "align=center;verticalAlign=middle;shadow=0;")
    cem = find(root, "wm-simplified-safe-search")
    geom(cem, 315, 535, 300, 78)
    cem.set("style", "rounded=1;arcSize=6;whiteSpace=wrap;html=1;fillColor=#FCFAF7;"
               f"strokeColor={amber};strokeWidth=1.3;fontFamily=Arial;fontColor={ink};shadow=0;")
    output = find(root, "wm-simplified-safe-output")
    geom(output, 700, 335, 180, 205)
    output.set("style", "rounded=1;arcSize=6;whiteSpace=wrap;html=1;fillColor=#F8FAFC;"
                  f"strokeColor={coral};strokeWidth=1.3;fontFamily=Arial;fontColor={ink};shadow=0;")

    headers = {
        "fusion": ("01", "FEATURE FUSION", navy, 76, 335, 212, 32),
        "risk": ("02", "ENSEMBLE RISK", amber, 315, 350, 180, 30),
        "cem": ("04", "CEM REFINEMENT", amber, 315, 535, 300, 28),
        "output": ("05", "SELECTED HYPOTHESIS", coral, 700, 335, 180, 32),
    }
    for key, (num, label, color, x, y, w, h) in headers.items():
        head = find(root, f"wm-balanced-header-{key}")
        head.set("value", f'<font color="{color}"><b>{num}</b></font>&nbsp;&nbsp;<font color="{ink}"><b>{label}</b></font>')
        head.set("style", "rounded=0;whiteSpace=wrap;html=1;fillColor=#EEF2F6;strokeColor=none;"
                  "fontFamily=Arial;fontSize=9;align=left;verticalAlign=middle;spacingLeft=11;shadow=0;")
        geom(head, x, y, w, h)
        rail = find(root, f"wm-balanced-rail-{key}")
        rail.set("style", "rounded=1;arcSize=12;html=1;strokeColor=none;" f"fillColor={color};")
        geom(rail, x, y, 4, h)
    nstyle = ("text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
              f"fontFamily=Arial;fontSize=8;fontColor={slate};align=center;verticalAlign=middle;")
    for cid, value, x, y, w, h in (
        ("wm-balanced-fusion-note", "815-D concatenated descriptor", 88, 368, 188, 14),
        ("wm-balanced-risk-note", "three independent scorers", 330, 380, 150, 14),
        ("wm-balanced-cem-note", "3 rounds  |  16 samples / round", 360, 563, 210, 14),
    ):
        c = find(root, cid)
        c.set("value", value)
        c.set("style", nstyle)
        geom(c, x, y, w, h)

    block_geoms = {
        "wm-clean-fusion-1": (110, 386, 144, 30),
        "wm-clean-fusion-2": (110, 416, 144, 43),
        "wm-clean-fusion-3": (110, 459, 144, 34),
        "wm-clean-fusion-4": (110, 493, 144, 16),
        "wm-clean-fusion-5": (110, 509, 144, 12),
    }
    for cid, (x, y, w, h) in block_geoms.items():
        geom(find(root, cid), x, y, w, h)
    total = find(root, "wm-clean-fusion-total")
    geom(total, 88, 528, 188, 14)
    total.set("style", "text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
               f"fontFamily=Arial;fontSize=8;fontColor={slate};align=center;verticalAlign=middle;")

    for i, (name_x, bar_x, y, width) in enumerate(
        ((329, 367, 397, 95), (329, 367, 424, 87), (329, 367, 451, 103)), 1
    ):
        geom(find(root, f"wm-clean-risk-name-{i}"), name_x, y, 34, 16)
        geom(find(root, f"wm-clean-risk-bar-{i}"), bar_x, y, width, 16)
    stat = find(root, "wm-clean-risk-stat")
    geom(stat, 338, 481, 135, 20)

    num = find(root, "wm-balanced-safety-number")
    geom(num, 510, 345, 22, 22)
    checks = find(root, "wm-clean-safety-checks")
    geom(checks, 535, 462, 112, 42)
    checks.set("style", "text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
               f"fontFamily=Arial;fontSize=8;fontColor={teal};align=left;verticalAlign=top;")

    for cid, x, y, w, h in (
        ("wm-clean-cem-sample", 328, 584, 64, 20),
        ("wm-clean-cem-elite", 413, 584, 64, 20),
        ("wm-clean-cem-update", 498, 584, 64, 20),
    ):
        geom(find(root, cid), x, y, w, h)
    for cid, x in (("wm-clean-cem-arrow-1", 393), ("wm-clean-cem-arrow-2", 478)):
        geom(find(root, cid), x, 584, 20, 20)

    h4 = find(root, "wm-balanced-output-h4")
    geom(h4, 744, 369, 92, 22)
    traj = find(root, "wm-clean-trajectory")
    # Keep the raster as a project-local asset so Draw.io renders it reliably
    # without a very large data URI in the parent figure.
    traj.set("style", "shape=image;html=1;imageAspect=0;aspect=fixed;"
             f"image={image.name};strokeColor=none;fillColor=none;")
    geom(traj, 714, 397, 152, 80)
    cap = find(root, "wm-clean-output-caption")
    cap.set("value", "candidate trajectories  |  obstacle  |  goal")
    cap.set("style", "text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
             f"fontFamily=Arial;fontSize=7;fontColor={slate};align=center;verticalAlign=middle;")
    geom(cap, 714, 479, 152, 12)
    action = find(root, "wm-clean-action")
    action.set("value", "<b>a0 = [dx, dy, dpsi]</b><br><font style=\"font-size:7px;\">execute first action</font>")
    action.set("style", "rounded=1;arcSize=5;whiteSpace=wrap;html=1;fillColor=#FCEFEA;"
                f"strokeColor={coral};strokeWidth=1;fontFamily=Arial;fontSize=8;fontColor={ink};"
                "align=center;verticalAlign=middle;shadow=0;")
    geom(action, 728, 498, 124, 34)

    for cid, points in {
        "wm-simplified-edge-safety-search": ((585, 510), (570, 510)),
        "wm-simplified-edge-refine-loop": ((290, 574), (290, 525), (405, 525)),
        "wm-simplified-edge-search-output": ((650, 574), (650, 483)),
    }.items():
        e = find(root, cid)
        g = e.find("mxGeometry")
        if g is not None:
            arr = g.find("Array")
            if arr is not None:
                g.remove(arr)
            arr = ET.SubElement(g, "Array", {"as": "points"})
            for x, y in points:
                ET.SubElement(arr, "mxPoint", {"x": str(x), "y": str(y)})

    ET.indent(tree, space="  ")
    tree.write(dst, encoding="utf-8", xml_declaration=False)
    print(f"wrote {dst}; cells={len(root.findall('mxCell'))}; image_bytes={image.stat().st_size}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=Path)
    ap.add_argument("destination", type=Path)
    ap.add_argument("--image", type=Path, default=Path("docs/diagrams/trajectory_planning_inset_small.png"))
    ns = ap.parse_args()
    compact(ns.source, ns.destination, ns.image)


if __name__ == "__main__":
    main()
