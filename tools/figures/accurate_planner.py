
#!/usr/bin/env python3
"""Make the compact planner semantically explicit and natively drawable."""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

P = "Lcp2UKG11IvjxIMB6Xdv-"
PLANNER = P + "2164"


def find(root, cid):
    for cell in root.findall("mxCell"):
        if cell.get("id") == cid:
            return cell
    raise KeyError(cid)


def geom(cell, x, y, w, h, relative=False, points=()):
    old = cell.find("mxGeometry")
    if old is not None:
        cell.remove(old)
    attrs = {"as": "geometry"}
    if relative:
        attrs["relative"] = "1"
    for key, value in (("x", x), ("y", y), ("width", w), ("height", h)):
        if value is not None:
            attrs[key] = str(value)
    g = ET.SubElement(cell, "mxGeometry", attrs)
    if points:
        arr = ET.SubElement(g, "Array", {"as": "points"})
        for px, py in points:
            ET.SubElement(arr, "mxPoint", {"x": str(px), "y": str(py)})
    return g


def vertex(root, cid, value, style, x, y, w, h):
    c = ET.SubElement(root, "mxCell", {
        "id": cid, "parent": PLANNER, "vertex": "1",
        "value": value, "style": style,
    })
    geom(c, x, y, w, h)
    return c


def edge(root, cid, value, style, source, target, points):
    c = ET.SubElement(root, "mxCell", {
        "id": cid, "parent": PLANNER, "edge": "1",
        "value": value, "source": source, "target": target,
        "style": style,
    })
    geom(c, None, None, None, None, relative=True, points=points)
    return c


def remove_ids(root, ids):
    for c in list(root.findall("mxCell")):
        if c.get("id") in ids:
            root.remove(c)


def transform(src: Path, dst: Path):
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
    light = "#F8FAFC"

    # Remove the unreliable raster inset and any previous native inset.
    remove_ids(root, {
        "wm-clean-trajectory",
        "wm-accurate-frame",
        "wm-accurate-obstacle",
        "wm-accurate-start",
        "wm-accurate-goal",
        "wm-accurate-path-1",
        "wm-accurate-path-2",
        "wm-accurate-path-3",
        "wm-accurate-path-selected",
    })

    # Fusion card: narrow vertical dimension bars with source labels outside.
    fusion = find(root, P + "2078")
    geom(fusion, 76, 325, 205, 190)
    fusion.set("style", "rounded=1;arcSize=5;whiteSpace=wrap;html=1;fillColor=#F8FAFC;"
               f"strokeColor={navy};strokeWidth=1.25;fontFamily=Arial;fontColor={ink};shadow=0;")
    header = find(root, "wm-balanced-header-fusion")
    header.set("value", "<b>FEATURE FUSION</b>")
    header.set("style", "text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
                  f"fontFamily=Arial;fontSize=9;fontColor={ink};align=left;verticalAlign=middle;")
    geom(header, 86, 331, 180, 18)

    block_style = (
        "rounded=0;whiteSpace=wrap;html=1;strokeWidth=1;"
        f"fontFamily=Arial;fontSize=8;fontColor={ink};fontStyle=1;"
        "align=center;verticalAlign=middle;"
    )
    bars = (
        ("wm-clean-fusion-1", "210", "#D9E7F2", "#6A8CAE", 143, 353, 95, 23),
        ("wm-clean-fusion-2", "336", "#D3E1EE", "#6A8CAE", 143, 376, 95, 31),
        ("wm-clean-fusion-3", "242", "#CBDCEB", "#6A8CAE", 143, 407, 95, 24),
        ("wm-clean-fusion-4", "24", "#E5ECF2", "#8798AA", 143, 431, 95, 13),
        ("wm-clean-fusion-5", "3", "#D9ECE8", "#6D9990", 143, 444, 95, 11),
    )
    sources = (
        ("wm-accurate-source-1", "DINOv3-L/16", 79, 355, 59, 19),
        ("wm-accurate-source-2", "LTWM", 79, 382, 59, 17),
        ("wm-accurate-source-3", "trajectory", 79, 412, 59, 17),
        ("wm-accurate-source-4", "physics", 79, 431, 59, 15),
        ("wm-accurate-source-5", "guard", 79, 444, 59, 14),
    )
    source_style = ("text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
                    f"fontFamily=Arial;fontSize=7;fontColor={slate};"
                    "align=right;verticalAlign=middle;")
    for cid, label, x, y, w, h in sources:
        old = next((c for c in root.findall("mxCell") if c.get("id") == cid), None)
        if old is None:
            old = vertex(root, cid, label, source_style, x, y, w, h)
        else:
            old.set("value", label)
            old.set("style", source_style)
            geom(old, x, y, w, h)
    for cid, label, fill, stroke, x, y, w, h in bars:
        c = find(root, cid)
        c.set("value", f"<b>{label}</b>")
        c.set("style", block_style + f"fillColor={fill};strokeColor={stroke};")
        geom(c, x, y, w, h)
    total = find(root, "wm-clean-fusion-total")
    total.set("value", "<b>concat  →  815-D</b>")
    total.set("style", "text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
               f"fontFamily=Arial;fontSize=8;fontColor={navy};align=center;verticalAlign=middle;")
    geom(total, 124, 468, 135, 14)

    # Risk card: compact bars, clear μ/σ summary.
    risk = find(root, P + "2094")
    geom(risk, 300, 335, 170, 145)
    risk.set("style", "rounded=1;arcSize=5;whiteSpace=wrap;html=1;fillColor=#FCFAF7;"
              f"strokeColor=#9AA8B8;strokeWidth=1.25;fontFamily=Arial;fontColor={ink};shadow=0;")
    header = find(root, "wm-balanced-header-risk")
    header.set("value", "<b>ENSEMBLE RISK</b>")
    header.set("style", "text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
                  f"fontFamily=Arial;fontSize=9;fontColor={ink};align=left;verticalAlign=middle;")
    geom(header, 310, 341, 150, 18)
    for i, y, width in ((1, 366, 91), (2, 390, 84), (3, 414, 98)):
        name = find(root, f"wm-clean-risk-name-{i}")
        name.set("style", "text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
                    f"fontFamily=Arial;fontSize=7;fontColor={ink};align=left;verticalAlign=middle;")
        geom(name, 311, y, 32, 15)
        bar = find(root, f"wm-clean-risk-bar-{i}")
        bar.set("style", "rounded=0;whiteSpace=wrap;html=1;fillColor=#D9E7F2;"
                   f"strokeColor={navy};strokeWidth=1;fontFamily=Arial;fontSize=8;"
                   f"fontColor={ink};fontStyle=1;align=right;verticalAlign=middle;spacingRight=5;")
        geom(bar, 347, y, width, 15)
    stat = find(root, "wm-clean-risk-stat")
    stat.set("value", "<b>&mu; = 0.78</b>   &sigma; = 0.03")
    stat.set("style", "rounded=0;whiteSpace=wrap;html=1;fillColor=#F4F6F8;"
               f"strokeColor=#B4BFCA;strokeWidth=1;fontFamily=Arial;fontSize=8;fontColor={ink};"
               "align=center;verticalAlign=middle;")
    geom(stat, 326, 438, 126, 18)

    # Safety gate with plain text constraints and a clean vertical hand-off.
    safety = find(root, P + "2104")
    geom(safety, 495, 335, 120, 95)
    safety.set("value", "<font style=\"font-size:15px;\"><b>SAFE?</b></font><br>"
                "<font style=\"font-size:8px;\">hard constraints</font>")
    safety.set("style", "rhombus;whiteSpace=wrap;html=1;fillColor=#EEF7F5;"
                f"strokeColor={teal};strokeWidth=1.5;fontFamily=Arial;fontColor={ink};"
                "align=center;verticalAlign=middle;shadow=0;")
    checks = find(root, "wm-clean-safety-checks")
    checks.set("value", "kinematics  |  collision  |  support")
    checks.set("style", "text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
                f"fontFamily=Arial;fontSize=7;fontColor={teal};align=center;verticalAlign=middle;")
    geom(checks, 476, 438, 158, 15)

    # CEM: explicit distribution update, not a generic three-box row.
    cem = find(root, "wm-simplified-safe-search")
    geom(cem, 300, 490, 290, 62)
    cem.set("style", "rounded=1;arcSize=5;whiteSpace=wrap;html=1;fillColor=#FCFAF7;"
              f"strokeColor={amber};strokeWidth=1.25;fontFamily=Arial;fontColor={ink};shadow=0;")
    header = find(root, "wm-balanced-header-cem")
    header.set("value", "<b>CEM REFINEMENT</b>")
    header.set("style", "text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
                  f"fontFamily=Arial;fontSize=9;fontColor={ink};align=left;verticalAlign=middle;")
    geom(header, 310, 496, 260, 18)
    for cid, label, x in (
        ("wm-clean-cem-sample", "sample  K=16", 314),
        ("wm-clean-cem-elite", "retain elite", 390),
        ("wm-clean-cem-update", "update &mu;, &sigma;", 466),
    ):
        c = find(root, cid)
        c.set("value", label)
        c.set("style", "rounded=0;whiteSpace=wrap;html=1;fillColor=#FBF7F2;"
                  f"strokeColor={amber};strokeWidth=1;fontFamily=Arial;fontSize=7;fontColor={ink};"
                  "align=center;verticalAlign=middle;")
        geom(c, x, 522, 60, 18)
    for cid, x in (("wm-clean-cem-arrow-1", 375), ("wm-clean-cem-arrow-2", 451)):
        c = find(root, cid)
        c.set("value", "→")
        c.set("style", "text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
                  f"fontFamily=Arial;fontSize=11;fontColor={amber};align=center;verticalAlign=middle;")
        geom(c, x, 520, 14, 20)

    # Output: native vector inset, so the trajectory is always visible.
    output = find(root, "wm-simplified-safe-output")
    geom(output, 655, 325, 180, 180)
    output.set("style", "rounded=1;arcSize=5;whiteSpace=wrap;html=1;fillColor=#F8FAFC;"
                f"strokeColor={coral};strokeWidth=1.25;fontFamily=Arial;fontColor={ink};shadow=0;")
    header = find(root, "wm-balanced-header-output")
    header.set("value", "<b>SELECTED HYPOTHESIS</b>")
    header.set("style", "text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
                  f"fontFamily=Arial;fontSize=9;fontColor={ink};align=left;verticalAlign=middle;")
    geom(header, 665, 331, 160, 18)
    h4 = find(root, "wm-balanced-output-h4")
    h4.set("value", "<b>H4</b>")
    h4.set("style", "text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
             f"fontFamily=Arial;fontSize=17;fontColor={coral};align=center;verticalAlign=middle;")
    geom(h4, 700, 350, 90, 22)
    vertex(root, "wm-accurate-frame", "", "rounded=0;whiteSpace=wrap;html=1;"
           "fillColor=#F7FAFC;strokeColor=#B7C2CC;strokeWidth=1;", 669, 377, 152, 70)

    # Three alternatives plus the selected route, all inside the output inset.
    start = vertex(root, "wm-accurate-start", "", "ellipse;html=1;fillColor=#2F5D8A;strokeColor=none;", 677, 431, 7, 7)
    goal = vertex(root, "wm-accurate-goal", "", "ellipse;html=1;fillColor=#2D7A78;strokeColor=none;", 805, 386, 8, 8)
    path_style = "edgeStyle=none;curved=1;html=1;endArrow=none;startArrow=none;strokeWidth=1.3;strokeColor=#A5B8C7;"
    edge(root, "wm-accurate-path-1", "", path_style, "wm-accurate-start", "wm-accurate-goal",
         ((680, 434), (704, 396), (750, 388), (809, 390)))
    edge(root, "wm-accurate-path-2", "", path_style + "strokeColor=#91A8BA;",
         "wm-accurate-start", "wm-accurate-goal", ((680, 434), (712, 447), (758, 446), (809, 390)))
    edge(root, "wm-accurate-path-3", "", path_style + "strokeColor=#7F99AE;",
         "wm-accurate-start", "wm-accurate-goal", ((680, 434), (713, 416), (758, 413), (809, 390)))
    edge(root, "wm-accurate-path-selected", "", path_style + f"strokeColor={navy};strokeWidth=2.8;",
         "wm-accurate-start", "wm-accurate-goal", ((680, 434), (710, 440), (752, 435), (780, 414), (809, 390)))
    action = find(root, "wm-clean-action")
    action.set("value", "<b>a0 = [dx, dy, dpsi]</b>")
    action.set("style", "rounded=1;arcSize=4;whiteSpace=wrap;html=1;fillColor=#FCEFEA;"
                 f"strokeColor={coral};strokeWidth=1;fontFamily=Arial;fontSize=8;fontColor={ink};"
                 "align=center;verticalAlign=middle;shadow=0;")
    geom(action, 683, 456, 124, 26)

    # Restore only the edge labels needed to decode the topology.
    edge_values = {
        "Lcp2UKG11IvjxIMB6Xdv-2072": "visual / goal",
        "Lcp2UKG11IvjxIMB6Xdv-251": "candidate rollout",
        "wm-simplified-edge-fusion-score": "815-D",
        "wm-simplified-edge-score-safety": "risk",
        "wm-simplified-edge-safety-search": "safe set",
        "wm-simplified-edge-refine-loop": "update &mu;, &sigma;",
        "wm-simplified-edge-search-output": "best safe H4",
    }
    for cid, value in edge_values.items():
        c = find(root, cid)
        c.set("value", value)
        style = c.get("style", "")
        if "labelBackgroundColor=" not in style:
            style += "labelBackgroundColor=#FBFCFD;"
        c.set("style", style + "fontSize=7;")

    # Route the loop to the right of Fusion instead of through its border.
    points = {
        "wm-simplified-edge-safety-search": ((555, 450), (555, 475)),
        "wm-simplified-edge-refine-loop": ((292, 521), (292, 482), (385, 482)),
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=Path)
    ap.add_argument("destination", type=Path)
    ns = ap.parse_args()
    transform(ns.source, ns.destination)


if __name__ == "__main__":
    main()
