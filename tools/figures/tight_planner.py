
#!/usr/bin/env python3
"""Tight, flat paper-style pass for the structurally correct planner."""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

P = "Lcp2UKG11IvjxIMB6Xdv-"
PLANNER = P + "2164"


def find(root, cid):
    for c in root.findall("mxCell"):
        if c.get("id") == cid:
            return c
    raise KeyError(cid)


def geom(cell, x, y, w, h, relative=False, points=()):
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


def vertex(root, cid, value, style, x, y, w, h):
    c = ET.SubElement(root, "mxCell", {
        "id": cid, "parent": PLANNER, "vertex": "1",
        "value": value, "style": style,
    })
    geom(c, x, y, w, h)
    return c


def remove_ids(root, ids):
    for c in list(root.findall("mxCell")):
        if c.get("id") in ids:
            root.remove(c)


def set_points(root, cid, coords):
    g = find(root, cid).find("mxGeometry")
    if g is None:
        return
    arr = g.find("Array")
    if arr is not None:
        g.remove(arr)
    arr = ET.SubElement(g, "Array", {"as": "points"})
    for x, y in coords:
        ET.SubElement(arr, "mxPoint", {"x": str(x), "y": str(y)})


def transform(src: Path, dst: Path):
    tree = ET.parse(src)
    root = tree.getroot().find("./diagram/mxGraphModel/root")
    if root is None:
        raise ValueError("missing mxGraphModel/root")

    ink = "#18283A"
    navy = "#315E86"
    teal = "#347C78"
    amber = "#A96D35"
    coral = "#C55A4A"
    slate = "#627184"
    rule = "#CDD5DE"

    # Remove stale labels and the remaining risk-container title.
    remove_ids(root, {
        "wm-accurate-source-1", "wm-accurate-source-2", "wm-accurate-source-3",
        "wm-accurate-source-4", "wm-accurate-source-5",
        "wm-struct-source-1", "wm-struct-source-2", "wm-struct-source-3",
        "wm-struct-source-4", "wm-struct-source-5",
        "wm-balanced-header-risk",
        "wm-tight-title-accent",
        "wm-tight-risk-label",
        "wm-tight-input-note",
        "wm-tight-obstacle",
    })

    # Flat section shell: much smaller, square corners, nearly no dead margin.
    panel = find(root, P + "2077")
    geom(panel, 52, 245, 650, 260)
    panel.set("style", "rounded=0;whiteSpace=wrap;html=1;fillColor=#FFFFFF;"
                f"strokeColor={rule};strokeWidth=1;fontFamily=Arial;shadow=0;")
    vertex(root, "wm-tight-title-accent", "", "rounded=0;html=1;fillColor=#315E86;strokeColor=none;",
           20, 13, 4, 23)
    title = find(root, P + "2080")
    title.set("value", "<b>SAFE CANDIDATE PLANNER</b>")
    title.set("style", "text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
                 f"fontFamily=Arial;fontSize=14;fontColor={ink};fontStyle=1;align=left;verticalAlign=middle;")
    geom(title, 30, 13, 390, 23)
    subtitle = find(root, "wm-clean-subtitle")
    subtitle.set("value", "815-D fusion  →  3 risk heads  →  safety filter  →  CEM  →  H4")
    subtitle.set("style", "text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
                    f"fontFamily=Arial;fontSize=7;fontColor={slate};align=left;verticalAlign=middle;")
    geom(subtitle, 30, 38, 500, 12)
    divider = find(root, "wm-clean-rule")
    divider.set("style", f"shape=line;html=1;strokeColor={rule};strokeWidth=1;")
    geom(divider, 20, 57, 610, 1)

    # Feature stack, sources on the outside, dimensions inside.
    fheader = find(root, "wm-balanced-header-fusion")
    fheader.set("value", "<b>FEATURE FUSION</b>")
    fheader.set("style", "text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
                   f"fontFamily=Arial;fontSize=8;fontColor={ink};align=left;verticalAlign=middle;")
    geom(fheader, 70, 313, 140, 15)
    vertex(root, "wm-tight-input-note", "visual  ·  rollout  ·  action",
           "text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
           f"fontFamily=Arial;fontSize=6;fontColor={slate};align=left;verticalAlign=middle;",
           70, 327, 140, 10)

    bars = {
        "wm-clean-fusion-1": ("210", "#DCE8F2", "#6F8EAA", 132, 340, 64, 20),
        "wm-clean-fusion-2": ("336", "#D4E2ED", "#6F8EAA", 132, 360, 64, 26),
        "wm-clean-fusion-3": ("242", "#CBDCE9", "#6F8EAA", 132, 386, 64, 20),
        "wm-clean-fusion-4": ("24", "#E6ECF1", "#8796A5", 132, 406, 64, 11),
        "wm-clean-fusion-5": ("3", "#DCECE8", "#73968F", 132, 417, 64, 9),
    }
    for cid, (value, fill, stroke, x, y, w, h) in bars.items():
        c = find(root, cid)
        c.set("value", f"<b>{value}</b>")
        c.set("style", "rounded=0;whiteSpace=wrap;html=1;"
                  f"fillColor={fill};strokeColor={stroke};strokeWidth=1;"
                  f"fontFamily=Arial;fontSize=7;fontColor={ink};fontStyle=1;"
                  "align=center;verticalAlign=middle;")
        geom(c, x, y, w, h)

    source_style = ("text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
                    f"fontFamily=Arial;fontSize=6;fontColor={slate};align=right;verticalAlign=middle;")
    sources = (
        ("wm-struct-source-1", "DINOv3-L/16", 70, 341, 58, 17),
        ("wm-struct-source-2", "LTWM", 70, 366, 58, 15),
        ("wm-struct-source-3", "trajectory", 70, 389, 58, 15),
        ("wm-struct-source-4", "physics", 70, 405, 58, 13),
        ("wm-struct-source-5", "guard", 70, 416, 58, 11),
    )
    for cid, label, x, y, w, h in sources:
        vertex(root, cid, label, source_style, x, y, w, h)
    total = find(root, "wm-clean-fusion-total")
    total.set("value", "<b>815-D</b>")
    total.set("style", "text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
               f"fontFamily=Arial;fontSize=7;fontColor={navy};align=center;verticalAlign=middle;")
    geom(total, 128, 432, 72, 12)
    concat = find(root, "wm-struct-concat")
    concat.set("value", "<b>815-D</b>")
    concat.set("style", "rounded=0;whiteSpace=wrap;html=1;fillColor=#EEF3F7;"
                 f"strokeColor={navy};strokeWidth=1;fontFamily=Arial;fontSize=6;fontColor={ink};"
                 "align=center;verticalAlign=middle;")
    geom(concat, 205, 383, 26, 20)

    # Risk heads without any outer container.
    vertex(root, "wm-tight-risk-label", "<b>RISK HEADS ×3</b>",
           "text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
           f"fontFamily=Arial;fontSize=8;fontColor={ink};align=left;verticalAlign=middle;",
           245, 313, 100, 15)
    mlps = (
        ("wm-struct-mlp-1", "MLP-1<br><font style=\"font-size:6px;\">r<sub>1</sub>=0.78</font>", 245, 333),
        ("wm-struct-mlp-2", "MLP-2<br><font style=\"font-size:6px;\">r<sub>2</sub>=0.74</font>", 245, 365),
        ("wm-struct-mlp-3", "MLP-3<br><font style=\"font-size:6px;\">r<sub>3</sub>=0.82</font>", 245, 397),
    )
    for cid, val, x, y in mlps:
        c = find(root, cid)
        c.set("value", val)
        c.set("style", "rounded=0;whiteSpace=wrap;html=1;fillColor=#F5F8FA;"
                  f"strokeColor={navy};strokeWidth=1;fontFamily=Arial;fontSize=7;fontColor={ink};"
                  "align=center;verticalAlign=middle;")
        geom(c, x, y, 64, 25)
    stats = find(root, "wm-struct-risk-stats")
    stats.set("value", "<b>&mu;, &sigma;</b><br><font style=\"font-size:6px;\">0.78 / 0.03</font>")
    stats.set("style", "rounded=0;whiteSpace=wrap;html=1;fillColor=#F4F6F8;"
                  f"strokeColor=#9AA8B8;strokeWidth=1;fontFamily=Arial;fontSize=7;fontColor={ink};"
                  "align=center;verticalAlign=middle;")
    geom(stats, 330, 365, 63, 25)

    # Fan-out/fan-in routes.
    set_points(root, "wm-struct-input-1", ((237, 393), (237, 345)))
    set_points(root, "wm-struct-input-2", ((237, 393), (237, 377)))
    set_points(root, "wm-struct-input-3", ((237, 393), (237, 409)))
    set_points(root, "wm-struct-output-1", ((320, 345), (320, 377)))
    set_points(root, "wm-struct-output-2", ((320, 377),))
    set_points(root, "wm-struct-output-3", ((320, 409), (320, 377)))
    for cid in (
        "wm-struct-input-1", "wm-struct-input-2", "wm-struct-input-3",
        "wm-struct-output-1", "wm-struct-output-2", "wm-struct-output-3",
    ):
        e = find(root, cid)
        e.set("value", "")
        e.set("style", "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;"
                 f"html=1;strokeColor={navy};strokeWidth=1.1;endArrow=blockThin;endFill=1;endSize=5;")

    # Safety and constraints.
    safety = find(root, P + "2104")
    safety.set("value", "<font style=\"font-size:13px;\"><b>SAFE?</b></font><br>"
                "<font style=\"font-size:6px;\">hard constraints</font>")
    safety.set("style", "rhombus;whiteSpace=wrap;html=1;fillColor=#EFF7F5;"
                f"strokeColor={teal};strokeWidth=1.3;fontFamily=Arial;fontColor={ink};"
                "align=center;verticalAlign=middle;shadow=0;")
    geom(safety, 410, 333, 96, 76)
    risk_edge = find(root, "wm-struct-risk-to-safety")
    risk_edge.set("value", "risk")
    risk_edge.set("style", "edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;"
                  f"strokeColor={teal};strokeWidth=1.2;endArrow=blockThin;endFill=1;endSize=5;"
                  "fontFamily=Arial;fontSize=6;labelBackgroundColor=#FFFFFF;")
    set_points(root, "wm-struct-risk-to-safety", ((399, 377), (410, 371)))
    checks = find(root, "wm-clean-safety-checks")
    checks.set("value", "kinematics  |  collision  |  support")
    checks.set("style", "text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
                f"fontFamily=Arial;fontSize=6;fontColor={teal};align=center;verticalAlign=middle;")
    geom(checks, 397, 411, 122, 12)

    # CEM becomes a compact strip.
    cem = find(root, "wm-simplified-safe-search")
    cem.set("style", "rounded=0;whiteSpace=wrap;html=1;fillColor=#FCFAF7;"
              f"strokeColor={amber};strokeWidth=1;fontFamily=Arial;fontColor={ink};shadow=0;")
    geom(cem, 270, 436, 210, 46)
    chead = find(root, "wm-balanced-header-cem")
    chead.set("value", "<b>CEM REFINEMENT</b>")
    chead.set("style", "text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
                 f"fontFamily=Arial;fontSize=7;fontColor={ink};align=left;verticalAlign=middle;")
    geom(chead, 278, 440, 190, 13)
    for cid, label, x, w in (
        ("wm-clean-cem-sample", "sample K=16", 280, 48),
        ("wm-clean-cem-elite", "retain elite", 342, 48),
        ("wm-clean-cem-update", "update &mu;, &sigma;", 404, 62),
    ):
        c = find(root, cid)
        c.set("value", label)
        c.set("style", "rounded=0;whiteSpace=wrap;html=1;fillColor=#FFFFFF;"
                  f"strokeColor={amber};strokeWidth=0.8;fontFamily=Arial;fontSize=6;fontColor={ink};"
                  "align=center;verticalAlign=middle;")
        geom(c, x, 458, w, 16)
    for cid, x in (("wm-clean-cem-arrow-1", 329), ("wm-clean-cem-arrow-2", 391)):
        c = find(root, cid)
        c.set("value", "→")
        c.set("style", "text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
                  f"fontFamily=Arial;fontSize=8;fontColor={amber};align=center;verticalAlign=middle;")
        geom(c, x, 457, 12, 16)

    s2c = find(root, "wm-struct-safety-to-cem")
    s2c.set("value", "safe set")
    s2c.set("style", "edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;"
              f"strokeColor={amber};strokeWidth=1.1;endArrow=blockThin;endFill=1;endSize=5;"
              "fontFamily=Arial;fontSize=6;labelBackgroundColor=#FFFFFF;"
              "exitX=0.5;exitY=1;entryX=0.85;entryY=0;")
    set_points(root, "wm-struct-safety-to-cem", ((458, 425), (448, 425)))
    loop = find(root, "wm-struct-cem-loop")
    loop.set("value", "update &mu;, &sigma;")
    loop.set("style", "edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;"
               f"strokeColor={slate};strokeWidth=1;dashed=1;dashPattern=4 4;"
               "endArrow=blockThin;endFill=1;endSize=5;fontFamily=Arial;fontSize=6;"
               "labelBackgroundColor=#FFFFFF;exitX=0;exitY=0.5;entryX=0;entryY=0.5;")
    set_points(root, "wm-struct-cem-loop", ((260, 459), (260, 428), (320, 428), (320, 377)))

    # Compact selected-hypothesis card.
    output = find(root, "wm-simplified-safe-output")
    output.set("style", "rounded=0;whiteSpace=wrap;html=1;fillColor=#FFFFFF;"
                f"strokeColor={coral};strokeWidth=1.1;fontFamily=Arial;fontColor={ink};shadow=0;")
    geom(output, 535, 328, 145, 144)
    ohead = find(root, "wm-balanced-header-output")
    ohead.set("value", "<b>SELECTED HYPOTHESIS</b>")
    ohead.set("style", "text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
                 f"fontFamily=Arial;fontSize=7;fontColor={ink};align=left;verticalAlign=middle;")
    geom(ohead, 543, 333, 130, 14)
    h4 = find(root, "wm-balanced-output-h4")
    h4.set("value", "<b>H4</b>")
    h4.set("style", "text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
             f"fontFamily=Arial;fontSize=15;fontColor={coral};align=center;verticalAlign=middle;")
    geom(h4, 564, 348, 86, 18)
    frame = find(root, "wm-accurate-frame")
    frame.set("style", f"rounded=0;whiteSpace=wrap;html=1;fillColor=#F7FAFC;strokeColor={rule};strokeWidth=0.8;")
    geom(frame, 547, 370, 121, 48)
    geom(find(root, "wm-accurate-start"), 554, 406, 6, 6)
    geom(find(root, "wm-accurate-goal"), 655, 373, 7, 7)
    for cid, pts in {
        "wm-accurate-path-1": ((557, 409), (575, 383), (607, 376), (658, 377)),
        "wm-accurate-path-2": ((557, 409), (581, 416), (617, 414), (658, 377)),
        "wm-accurate-path-3": ((557, 409), (582, 394), (617, 392), (658, 377)),
        "wm-accurate-path-selected": ((557, 409), (582, 413), (612, 410), (635, 394), (658, 377)),
    }.items():
        set_points(root, cid, pts)
    vertex(root, "wm-tight-obstacle", "", "ellipse;html=1;fillColor=#F3D7C2;"
           f"strokeColor={amber};strokeWidth=0.8;", 603, 390, 13, 13)
    action = find(root, "wm-clean-action")
    action.set("value", "<b>a</b><sub>0</sub> = [&Delta;x, &Delta;y, &Delta;&psi;]<sup>T</sup>")
    action.set("style", "rounded=0;whiteSpace=wrap;html=1;fillColor=#FCEFEB;"
                 f"strokeColor={coral};strokeWidth=0.9;fontFamily=Arial;fontSize=7;fontColor={ink};"
                 "align=center;verticalAlign=middle;")
    geom(action, 552, 430, 111, 22)

    c2o = find(root, "wm-struct-cem-output")
    c2o.set("value", "best H4")
    c2o.set("style", "edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;"
              f"strokeColor={navy};strokeWidth=1.2;endArrow=blockThin;endFill=1;endSize=5;"
              "fontFamily=Arial;fontSize=6;labelBackgroundColor=#FFFFFF;"
              "exitX=1;exitY=0.5;entryX=0;entryY=0.72;")
    set_points(root, "wm-struct-cem-output", ((500, 459), (520, 459), (520, 432)))

    # External labels are already defined by upstream modules; suppress midpoint labels.
    for cid in (P + "2072", P + "251", P + "252"):
        e = find(root, cid)
        e.set("value", "")
        e.set("target", "wm-struct-concat")
        style = e.get("style", "")
        style = style.replace("fontSize=7;", "fontSize=6;")
        e.set("style", style)
        geom(e, None, None, None, None, relative=True)

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
