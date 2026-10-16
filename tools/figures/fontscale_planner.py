
#!/usr/bin/env python3
"""Keep the planner footprint while shrinking inner boxes and enlarging text."""

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

    remove_ids(root, {
        "wm-accurate-source-1", "wm-accurate-source-2", "wm-accurate-source-3",
        "wm-accurate-source-4", "wm-accurate-source-5",
        "wm-struct-source-1", "wm-struct-source-2", "wm-struct-source-3",
        "wm-struct-source-4", "wm-struct-source-5",
        "wm-balanced-header-risk",
        "wm-font-title-accent",
        "wm-font-risk-label",
        "wm-font-input-note",
    })

    # Restore the larger section footprint.
    panel = find(root, P + "2077")
    geom(panel, 52, 245, 850, 330)
    panel.set("style", "rounded=1;arcSize=6;whiteSpace=wrap;html=1;fillColor=#FFFFFF;"
                f"strokeColor=#A8B4C1;strokeWidth=1.1;fontFamily=Arial;shadow=0;")
    vertex(root, "wm-font-title-accent", "", "rounded=0;html=1;fillColor=#315E86;strokeColor=none;",
           24, 16, 4, 25)
    title = find(root, P + "2080")
    title.set("value", "<b>SAFE CANDIDATE PLANNER</b>")
    title.set("style", "text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
                 f"fontFamily=Arial;fontSize=16;fontColor={ink};fontStyle=1;align=left;verticalAlign=middle;")
    geom(title, 35, 15, 440, 26)
    subtitle = find(root, "wm-clean-subtitle")
    subtitle.set("value", "815-D fusion  →  3 risk heads  →  safety filter  →  CEM  →  H4")
    subtitle.set("style", "text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
                    f"fontFamily=Arial;fontSize=8;fontColor={slate};align=left;verticalAlign=middle;")
    geom(subtitle, 35, 44, 610, 14)
    divider = find(root, "wm-clean-rule")
    divider.set("style", f"shape=line;html=1;strokeColor={rule};strokeWidth=1;")
    geom(divider, 24, 70, 800, 1)

    # Feature fusion: smaller bars, larger typography.
    header = find(root, "wm-balanced-header-fusion")
    header.set("value", "<b>FEATURE FUSION</b>")
    header.set("style", "text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
                  f"fontFamily=Arial;fontSize=10;fontColor={ink};align=left;verticalAlign=middle;")
    geom(header, 76, 318, 170, 18)
    vertex(root, "wm-font-input-note", "visual  ·  rollout  ·  action",
           "text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
           f"fontFamily=Arial;fontSize=7;fontColor={slate};align=left;verticalAlign=middle;",
           76, 336, 170, 12)

    bars = {
        "wm-clean-fusion-1": ("210", "#DCE8F2", "#6F8EAA", 151, 352, 60, 23),
        "wm-clean-fusion-2": ("336", "#D4E2ED", "#6F8EAA", 151, 375, 60, 30),
        "wm-clean-fusion-3": ("242", "#CBDCE9", "#6F8EAA", 151, 405, 60, 23),
        "wm-clean-fusion-4": ("24", "#E6ECF1", "#8796A5", 151, 428, 60, 14),
        "wm-clean-fusion-5": ("3", "#DCECE8", "#73968F", 151, 442, 60, 11),
    }
    for cid, (value, fill, stroke, x, y, w, h) in bars.items():
        c = find(root, cid)
        c.set("value", f"<b>{value}</b>")
        c.set("style", "rounded=0;whiteSpace=wrap;html=1;"
                  f"fillColor={fill};strokeColor={stroke};strokeWidth=1;"
                  f"fontFamily=Arial;fontSize=9;fontColor={ink};fontStyle=1;"
                  "align=center;verticalAlign=middle;")
        geom(c, x, y, w, h)
    source_style = ("text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
                    f"fontFamily=Arial;fontSize=8;fontColor={slate};align=right;verticalAlign=middle;")
    sources = (
        ("wm-struct-source-1", "DINOv3-L/16", 77, 353, 69, 20),
        ("wm-struct-source-2", "LTWM", 77, 382, 69, 17),
        ("wm-struct-source-3", "trajectory", 77, 408, 69, 17),
        ("wm-struct-source-4", "physics", 77, 428, 69, 15),
        ("wm-struct-source-5", "guard", 77, 441, 69, 13),
    )
    for cid, label, x, y, w, h in sources:
        vertex(root, cid, label, source_style, x, y, w, h)
    total = find(root, "wm-clean-fusion-total")
    total.set("value", "<b>815-D</b>")
    total.set("style", "text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
               f"fontFamily=Arial;fontSize=9;fontColor={navy};align=center;verticalAlign=middle;")
    geom(total, 143, 462, 76, 15)
    concat = find(root, "wm-struct-concat")
    concat.set("value", "<b>815-D</b>")
    concat.set("style", "rounded=0;whiteSpace=wrap;html=1;fillColor=#EEF3F7;"
                 f"strokeColor={navy};strokeWidth=1;fontFamily=Arial;fontSize=8;fontColor={ink};"
                 "align=center;verticalAlign=middle;")
    geom(concat, 227, 398, 31, 23)

    # Three smaller MLP boxes with larger readable labels.
    vertex(root, "wm-font-risk-label", "<b>RISK HEADS ×3</b>",
           "text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
           f"fontFamily=Arial;fontSize=10;fontColor={ink};align=left;verticalAlign=middle;",
           300, 318, 120, 18)
    mlps = (
        ("wm-struct-mlp-1", "MLP-1<br><font style=\"font-size:8px;\">r<sub>1</sub>=0.78</font>", 300, 342),
        ("wm-struct-mlp-2", "MLP-2<br><font style=\"font-size:8px;\">r<sub>2</sub>=0.74</font>", 300, 382),
        ("wm-struct-mlp-3", "MLP-3<br><font style=\"font-size:8px;\">r<sub>3</sub>=0.82</font>", 300, 422),
    )
    for cid, value, x, y in mlps:
        c = find(root, cid)
        c.set("value", value)
        c.set("style", "rounded=0;whiteSpace=wrap;html=1;fillColor=#F5F8FA;"
                  f"strokeColor={navy};strokeWidth=1;fontFamily=Arial;fontSize=9;fontColor={ink};"
                  "align=center;verticalAlign=middle;")
        geom(c, x, y, 62, 32)
    stats = find(root, "wm-struct-risk-stats")
    stats.set("value", "<b>&mu;, &sigma;</b><br><font style=\"font-size:8px;\">0.78 / 0.03</font>")
    stats.set("style", "rounded=0;whiteSpace=wrap;html=1;fillColor=#F4F6F8;"
                  f"strokeColor=#9AA8B8;strokeWidth=1;fontFamily=Arial;fontSize=9;fontColor={ink};"
                  "align=center;verticalAlign=middle;")
    geom(stats, 410, 382, 68, 32)

    set_points(root, "wm-struct-input-1", ((278, 409), (278, 358)))
    set_points(root, "wm-struct-input-2", ((278, 409), (278, 398)))
    set_points(root, "wm-struct-input-3", ((278, 409), (278, 438)))
    set_points(root, "wm-struct-output-1", ((388, 358), (390, 358), (390, 398)))
    set_points(root, "wm-struct-output-2", ((388, 398), (410, 398)))
    set_points(root, "wm-struct-output-3", ((388, 438), (390, 438), (390, 398)))
    for cid in (
        "wm-struct-input-1", "wm-struct-input-2", "wm-struct-input-3",
        "wm-struct-output-1", "wm-struct-output-2", "wm-struct-output-3",
    ):
        e = find(root, cid)
        e.set("value", "")
        e.set("style", "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;"
                 f"html=1;strokeColor={navy};strokeWidth=1.2;endArrow=blockThin;endFill=1;endSize=6;")

    # Safety gate.
    safety = find(root, P + "2104")
    safety.set("value", "<font style=\"font-size:16px;\"><b>SAFE?</b></font><br>"
                "<font style=\"font-size:9px;\">hard constraints</font>")
    safety.set("style", "rhombus;whiteSpace=wrap;html=1;fillColor=#EFF7F5;"
                f"strokeColor={teal};strokeWidth=1.4;fontFamily=Arial;fontColor={ink};"
                "align=center;verticalAlign=middle;shadow=0;")
    geom(safety, 510, 342, 120, 95)
    risk_edge = find(root, "wm-struct-risk-to-safety")
    risk_edge.set("value", "risk")
    risk_edge.set("style", "edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;"
                  f"strokeColor={teal};strokeWidth=1.3;endArrow=blockThin;endFill=1;endSize=6;"
                  "fontFamily=Arial;fontSize=8;labelBackgroundColor=#FFFFFF;")
    set_points(root, "wm-struct-risk-to-safety", ((490, 398), (510, 389)))
    checks = find(root, "wm-clean-safety-checks")
    checks.set("value", "kinematics  |  collision  |  support")
    checks.set("style", "text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
                f"fontFamily=Arial;fontSize=8;fontColor={teal};align=center;verticalAlign=middle;")
    geom(checks, 480, 440, 180, 15)

    # CEM: only the inner operation boxes shrink.
    cem = find(root, "wm-simplified-safe-search")
    cem.set("style", "rounded=0;whiteSpace=wrap;html=1;fillColor=#FCFAF7;"
              f"strokeColor={amber};strokeWidth=1.1;fontFamily=Arial;fontColor={ink};shadow=0;")
    geom(cem, 325, 487, 245, 56)
    chead = find(root, "wm-balanced-header-cem")
    chead.set("value", "<b>CEM REFINEMENT</b>")
    chead.set("style", "text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
                 f"fontFamily=Arial;fontSize=10;fontColor={ink};align=left;verticalAlign=middle;")
    geom(chead, 335, 492, 220, 17)
    for cid, label, x, w in (
        ("wm-clean-cem-sample", "sample K=16", 339, 52),
        ("wm-clean-cem-elite", "retain elite", 410, 52),
        ("wm-clean-cem-update", "update &mu;, &sigma;", 481, 68),
    ):
        c = find(root, cid)
        c.set("value", label)
        c.set("style", "rounded=0;whiteSpace=wrap;html=1;fillColor=#FFFFFF;"
                  f"strokeColor={amber};strokeWidth=0.9;fontFamily=Arial;fontSize=8;fontColor={ink};"
                  "align=center;verticalAlign=middle;")
        geom(c, x, 517, w, 19)
    for cid, x in (("wm-clean-cem-arrow-1", 393), ("wm-clean-cem-arrow-2", 464)):
        c = find(root, cid)
        c.set("value", "→")
        c.set("style", "text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
                  f"fontFamily=Arial;fontSize=10;fontColor={amber};align=center;verticalAlign=middle;")
        geom(c, x, 516, 16, 20)

    s2c = find(root, "wm-struct-safety-to-cem")
    s2c.set("value", "safe set")
    s2c.set("style", "edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;"
              f"strokeColor={amber};strokeWidth=1.2;endArrow=blockThin;endFill=1;endSize=6;"
              "fontFamily=Arial;fontSize=8;labelBackgroundColor=#FFFFFF;"
              "exitX=0.5;exitY=1;entryX=0.85;entryY=0;")
    set_points(root, "wm-struct-safety-to-cem", ((570, 464), (540, 464)))
    loop = find(root, "wm-struct-cem-loop")
    loop.set("value", "update &mu;, &sigma;")
    loop.set("style", "edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;"
               f"strokeColor={slate};strokeWidth=1.1;dashed=1;dashPattern=4 4;"
               "endArrow=blockThin;endFill=1;endSize=6;fontFamily=Arial;fontSize=8;"
               "labelBackgroundColor=#FFFFFF;exitX=0;exitY=0.5;entryX=0;entryY=0.5;")
    set_points(root, "wm-struct-cem-loop", ((310, 515), (290, 515), (290, 470), (390, 470), (390, 398)))

    # Output keeps its placement; only internal boxes and padding are reduced.
    output = find(root, "wm-simplified-safe-output")
    output.set("style", "rounded=0;whiteSpace=wrap;html=1;fillColor=#FFFFFF;"
                f"strokeColor={coral};strokeWidth=1.2;fontFamily=Arial;fontColor={ink};shadow=0;")
    geom(output, 690, 342, 170, 163)
    ohead = find(root, "wm-balanced-header-output")
    ohead.set("value", "<b>SELECTED HYPOTHESIS</b>")
    ohead.set("style", "text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
                 f"fontFamily=Arial;fontSize=10;fontColor={ink};align=left;verticalAlign=middle;")
    geom(ohead, 700, 348, 150, 18)
    h4 = find(root, "wm-balanced-output-h4")
    h4.set("value", "<b>H4</b>")
    h4.set("style", "text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
             f"fontFamily=Arial;fontSize=18;fontColor={coral};align=center;verticalAlign=middle;")
    geom(h4, 730, 369, 90, 22)
    frame = find(root, "wm-accurate-frame")
    frame.set("style", f"rounded=0;whiteSpace=wrap;html=1;fillColor=#F7FAFC;strokeColor={rule};strokeWidth=0.9;")
    geom(frame, 704, 394, 142, 55)
    geom(find(root, "wm-accurate-start"), 712, 429, 7, 7)
    geom(find(root, "wm-accurate-goal"), 831, 397, 8, 8)
    for cid, pts in {
        "wm-accurate-path-1": ((715, 432), (735, 405), (770, 399), (835, 401)),
        "wm-accurate-path-2": ((715, 432), (745, 444), (785, 441), (835, 401)),
        "wm-accurate-path-3": ((715, 432), (746, 417), (785, 414), (835, 401)),
        "wm-accurate-path-selected": ((715, 432), (746, 438), (780, 434), (807, 417), (835, 401)),
    }.items():
        set_points(root, cid, pts)
    action = find(root, "wm-clean-action")
    action.set("value", "<b>a</b><sub>0</sub> = [&Delta;x, &Delta;y, &Delta;&psi;]<sup>T</sup>")
    action.set("style", "rounded=0;whiteSpace=wrap;html=1;fillColor=#FCEFEB;"
                 f"strokeColor={coral};strokeWidth=1;fontFamily=Arial;fontSize=9;fontColor={ink};"
                 "align=center;verticalAlign=middle;")
    geom(action, 715, 462, 120, 27)
    c2o = find(root, "wm-struct-cem-output")
    c2o.set("value", "best H4")
    c2o.set("style", "edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;"
              f"strokeColor={navy};strokeWidth=1.3;endArrow=blockThin;endFill=1;endSize=6;"
              "fontFamily=Arial;fontSize=8;labelBackgroundColor=#FFFFFF;"
              "exitX=1;exitY=0.5;entryX=0;entryY=0.72;")
    set_points(root, "wm-struct-cem-output", ((600, 515), (660, 515), (660, 459)))

    # Prevent three external labels from overlapping the section title.
    for cid in (P + "2072", P + "251", P + "252"):
        e = find(root, cid)
        e.set("value", "")
        e.set("target", "wm-struct-concat")
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
