
#!/usr/bin/env python3
"""Rebuild the planner's fusion/risk stage as the actual compact topology."""

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


def edge(root, cid, value, style, source, target, points=()):
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

    # Remove the old feature/risk containers and their obsolete routing.
    remove_ids(root, {
        P + "2078",
        P + "2094",
        "wm-clean-risk-name-1", "wm-clean-risk-name-2", "wm-clean-risk-name-3",
        "wm-clean-risk-bar-1", "wm-clean-risk-bar-2", "wm-clean-risk-bar-3",
        "wm-clean-risk-stat",
        "wm-simplified-edge-fusion-score",
        "wm-simplified-edge-score-safety",
        "wm-simplified-edge-safety-search",
        "wm-simplified-edge-refine-loop",
        "wm-simplified-edge-search-output",
        "wm-struct-concat",
        "wm-struct-source-1", "wm-struct-source-2", "wm-struct-source-3",
        "wm-struct-source-4", "wm-struct-source-5",
        "wm-struct-risk-stats",
        "wm-struct-mlp-1", "wm-struct-mlp-2", "wm-struct-mlp-3",
        "wm-struct-input-1", "wm-struct-input-2", "wm-struct-input-3",
        "wm-struct-output-1", "wm-struct-output-2", "wm-struct-output-3",
        "wm-struct-concat-to-bus",
        "wm-struct-risk-to-safety",
        "wm-struct-safety-to-cem",
        "wm-struct-cem-loop",
        "wm-struct-cem-output",
        "wm-accurate-frame",
        "wm-accurate-obstacle",
        "wm-accurate-start",
        "wm-accurate-goal",
        "wm-accurate-path-1",
        "wm-accurate-path-2",
        "wm-accurate-path-3",
        "wm-accurate-path-selected",
    })

    # Compact planner shell.
    panel = find(root, P + "2077")
    geom(panel, 52, 245, 850, 330)
    panel.set("style", "rounded=1;arcSize=7;whiteSpace=wrap;html=1;fillColor=#FBFCFD;"
                "strokeColor=#9AA8B8;strokeWidth=1.2;fontFamily=Arial;shadow=0;")
    title = find(root, P + "2080")
    title.set("value", "<b>SAFE CANDIDATE PLANNER</b>")
    title.set("style", "text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
                 f"fontFamily=Arial;fontSize=15;fontColor={ink};fontStyle=1;align=left;verticalAlign=middle;")
    geom(title, 24, 16, 460, 24)
    subtitle = find(root, "wm-clean-subtitle")
    subtitle.set("value", "feature fusion  |  3 risk heads  |  safety gate  |  CEM  |  H4")
    subtitle.set("style", "text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
                    f"fontFamily=Arial;fontSize=8;fontColor={slate};align=left;verticalAlign=middle;")
    geom(subtitle, 24, 42, 600, 14)
    geom(find(root, "wm-clean-rule"), 24, 64, 790, 1)

    # Feature fusion is a direct vertical input stack, not a surrounding box.
    header = find(root, "wm-balanced-header-fusion")
    header.set("value", "<b>FEATURE FUSION</b>")
    header.set("style", "text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
                  f"fontFamily=Arial;fontSize=9;fontColor={ink};align=left;verticalAlign=middle;")
    geom(header, 76, 325, 160, 18)

    block_style = ("rounded=0;whiteSpace=wrap;html=1;strokeWidth=1;fontFamily=Arial;"
                   f"fontSize=8;fontColor={ink};fontStyle=1;align=center;verticalAlign=middle;")
    bars = (
        ("wm-clean-fusion-1", "210", "#D9E7F2", "#6A8CAE", 141, 351, 76, 24),
        ("wm-clean-fusion-2", "336", "#D3E1EE", "#6A8CAE", 141, 375, 76, 32),
        ("wm-clean-fusion-3", "242", "#CBDCEB", "#6A8CAE", 141, 407, 76, 25),
        ("wm-clean-fusion-4", "24", "#E5ECF2", "#8798AA", 141, 432, 76, 14),
        ("wm-clean-fusion-5", "3", "#D9ECE8", "#6D9990", 141, 446, 76, 11),
    )
    for cid, val, fill, stroke, x, y, w, h in bars:
        c = find(root, cid)
        c.set("value", f"<b>{val}</b>")
        c.set("style", block_style + f"fillColor={fill};strokeColor={stroke};")
        geom(c, x, y, w, h)
    source_style = ("text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
                    f"fontFamily=Arial;fontSize=7;fontColor={slate};align=right;verticalAlign=middle;")
    sources = (
        ("wm-struct-source-1", "DINOv3-L/16", 78, 352, 59, 19),
        ("wm-struct-source-2", "LTWM", 78, 385, 59, 16),
        ("wm-struct-source-3", "trajectory", 78, 414, 59, 17),
        ("wm-struct-source-4", "physics", 78, 433, 59, 15),
        ("wm-struct-source-5", "guard", 78, 446, 59, 14),
    )
    for cid, label, x, y, w, h in sources:
        vertex(root, cid, label, source_style, x, y, w, h)
    total = find(root, "wm-clean-fusion-total")
    total.set("value", "<b>concat  →  815-D</b>")
    total.set("style", "text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
               f"fontFamily=Arial;fontSize=8;fontColor={navy};align=center;verticalAlign=middle;")
    geom(total, 128, 468, 115, 14)
    concat = vertex(root, "wm-struct-concat", "<b>815-D</b>",
                    "rounded=1;arcSize=4;whiteSpace=wrap;html=1;fillColor=#EEF3F8;"
                    f"strokeColor={navy};strokeWidth=1;fontFamily=Arial;fontSize=7;fontColor={ink};"
                    "align=center;verticalAlign=middle;", 228, 398, 28, 22)

    # Three independent MLP heads, roughly square rather than long bars.
    mlp_style = ("rounded=1;arcSize=4;whiteSpace=wrap;html=1;fillColor=#F4F7FA;"
                 f"strokeColor={navy};strokeWidth=1.1;fontFamily=Arial;fontSize=8;fontColor={ink};"
                 "align=center;verticalAlign=middle;")
    mlps = (
        ("wm-struct-mlp-1", "MLP-1<br><font style=\"font-size:7px;\">r<sub>1</sub>=0.78</font>", 300, 340),
        ("wm-struct-mlp-2", "MLP-2<br><font style=\"font-size:7px;\">r<sub>2</sub>=0.74</font>", 300, 380),
        ("wm-struct-mlp-3", "MLP-3<br><font style=\"font-size:7px;\">r<sub>3</sub>=0.82</font>", 300, 420),
    )
    for cid, val, x, y in mlps:
        vertex(root, cid, val, mlp_style, x, y, 74, 30)

    edge_style = ("edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;"
                  "html=1;endArrow=blockThin;endFill=1;endSize=6;fontFamily=Arial;fontSize=7;"
                  "labelBackgroundColor=#FBFCFD;strokeWidth=1.25;")
    # One concat output fans out to the three heads.
    edge(root, "wm-struct-input-1", "", edge_style + f"strokeColor={navy};",
         "wm-struct-concat", "wm-struct-mlp-1", ((270, 409), (270, 355)))
    edge(root, "wm-struct-input-2", "", edge_style + f"strokeColor={navy};",
         "wm-struct-concat", "wm-struct-mlp-2", ((270, 409), (270, 395)))
    edge(root, "wm-struct-input-3", "", edge_style + f"strokeColor={navy};",
         "wm-struct-concat", "wm-struct-mlp-3", ((270, 409), (270, 435)))

    # Outputs merge into a small μ/σ capsule before the safety gate.
    stats = vertex(root, "wm-struct-risk-stats", "<b>&mu;, &sigma;</b><br><font style=\"font-size:7px;\">0.78 / 0.03</font>",
                   "rounded=1;arcSize=4;whiteSpace=wrap;html=1;fillColor=#F4F6F8;"
                   f"strokeColor=#9AA8B8;strokeWidth=1;fontFamily=Arial;fontSize=8;fontColor={ink};"
                   "align=center;verticalAlign=middle;", 405, 391, 76, 28)
    for cid, source, pts in (
        ("wm-struct-output-1", "wm-struct-mlp-1", ((390, 355), (395, 355), (395, 405))),
        ("wm-struct-output-2", "wm-struct-mlp-2", ((390, 395), (405, 405))),
        ("wm-struct-output-3", "wm-struct-mlp-3", ((390, 435), (395, 435), (395, 405))),
    ):
        edge(root, cid, "", edge_style + f"strokeColor={navy};", source, "wm-struct-risk-stats", pts)
    safety = find(root, P + "2104")
    geom(safety, 510, 340, 120, 95)
    safety.set("value", "<font style=\"font-size:15px;\"><b>SAFE?</b></font><br>"
                "<font style=\"font-size:8px;\">hard constraints</font>")
    safety.set("style", "rhombus;whiteSpace=wrap;html=1;fillColor=#EEF7F5;"
                f"strokeColor={teal};strokeWidth=1.5;fontFamily=Arial;fontColor={ink};"
                "align=center;verticalAlign=middle;shadow=0;")
    edge(root, "wm-struct-risk-to-safety", "risk", edge_style + f"strokeColor={teal};",
         "wm-struct-risk-stats", P + "2104", ((490, 405), (510, 387)))
    checks = find(root, "wm-clean-safety-checks")
    checks.set("value", "kinematics  |  collision  |  support")
    checks.set("style", "text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
                f"fontFamily=Arial;fontSize=7;fontColor={teal};align=center;verticalAlign=middle;")
    geom(checks, 480, 439, 180, 15)

    # CEM and output.
    cem = find(root, "wm-simplified-safe-search")
    geom(cem, 325, 485, 245, 58)
    cem.set("style", "rounded=1;arcSize=5;whiteSpace=wrap;html=1;fillColor=#FCFAF7;"
              f"strokeColor={amber};strokeWidth=1.25;fontFamily=Arial;fontColor={ink};shadow=0;")
    header = find(root, "wm-balanced-header-cem")
    header.set("value", "<b>CEM REFINEMENT</b>")
    header.set("style", "text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
                  f"fontFamily=Arial;fontSize=9;fontColor={ink};align=left;verticalAlign=middle;")
    geom(header, 335, 491, 215, 17)
    for cid, label, x, w in (
        ("wm-clean-cem-sample", "sample K=16", 338, 56),
        ("wm-clean-cem-elite", "retain elite", 408, 56),
        ("wm-clean-cem-update", "update &mu;, &sigma;", 478, 70),
    ):
        c = find(root, cid)
        c.set("value", label)
        c.set("style", "rounded=0;whiteSpace=wrap;html=1;fillColor=#FBF7F2;"
                  f"strokeColor={amber};strokeWidth=1;fontFamily=Arial;fontSize=7;fontColor={ink};"
                  "align=center;verticalAlign=middle;")
        geom(c, x, 516, w, 17)
    for cid, x in (("wm-clean-cem-arrow-1", 395), ("wm-clean-cem-arrow-2", 465)):
        c = find(root, cid)
        c.set("value", "→")
        c.set("style", "text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
                  f"fontFamily=Arial;fontSize=10;fontColor={amber};align=center;verticalAlign=middle;")
        geom(c, x, 514, 12, 18)
    edge(root, "wm-struct-safety-to-cem", "safe set", edge_style + f"strokeColor={amber};",
         P + "2104", "wm-simplified-safe-search", ((570, 460), (550, 460)))
    edge(root, "wm-struct-cem-loop", "update &mu;, &sigma;", edge_style + f"strokeColor={slate};dashed=1;dashPattern=5 4;",
         "wm-simplified-safe-search", "wm-struct-risk-stats", ((310, 514), (290, 514), (290, 465), (390, 465), (390, 405)))

    output = find(root, "wm-simplified-safe-output")
    geom(output, 690, 340, 170, 165)
    output.set("style", "rounded=1;arcSize=5;whiteSpace=wrap;html=1;fillColor=#F8FAFC;"
                f"strokeColor={coral};strokeWidth=1.25;fontFamily=Arial;fontColor={ink};shadow=0;")
    header = find(root, "wm-balanced-header-output")
    header.set("value", "<b>SELECTED HYPOTHESIS</b>")
    header.set("style", "text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
                  f"fontFamily=Arial;fontSize=9;fontColor={ink};align=left;verticalAlign=middle;")
    geom(header, 700, 346, 150, 18)
    h4 = find(root, "wm-balanced-output-h4")
    h4.set("value", "<b>H4</b>")
    h4.set("style", "text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
             f"fontFamily=Arial;fontSize=17;fontColor={coral};align=center;verticalAlign=middle;")
    geom(h4, 730, 366, 90, 20)
    frame = vertex(root, "wm-accurate-frame", "", "rounded=0;whiteSpace=wrap;html=1;fillColor=#F7FAFC;strokeColor=#B7C2CC;strokeWidth=1;", 704, 389, 142, 58)
    start = vertex(root, "wm-accurate-start", "", "ellipse;html=1;fillColor=#2F5D8A;strokeColor=none;", 712, 427, 7, 7)
    goal = vertex(root, "wm-accurate-goal", "", "ellipse;html=1;fillColor=#2D7A78;strokeColor=none;", 831, 392, 8, 8)
    path_style = "edgeStyle=none;curved=1;html=1;endArrow=none;startArrow=none;strokeWidth=1.2;strokeColor=#A5B8C7;"
    edge(root, "wm-accurate-path-1", "", path_style, "wm-accurate-start", "wm-accurate-goal", ((715, 430), (735, 402), (770, 395), (835, 396)))
    edge(root, "wm-accurate-path-2", "", path_style + "strokeColor=#91A8BA;", "wm-accurate-start", "wm-accurate-goal", ((715, 430), (745, 441), (785, 438), (835, 396)))
    edge(root, "wm-accurate-path-3", "", path_style + "strokeColor=#7F99AE;", "wm-accurate-start", "wm-accurate-goal", ((715, 430), (746, 415), (785, 412), (835, 396)))
    edge(root, "wm-accurate-path-selected", "", path_style + f"strokeColor={navy};strokeWidth=2.8;", "wm-accurate-start", "wm-accurate-goal", ((715, 430), (746, 436), (780, 432), (807, 414), (835, 396)))
    action = find(root, "wm-clean-action")
    action.set("value", "<b>a</b><sub>0</sub> = [&Delta;x, &Delta;y, &Delta;&psi;]<sup>T</sup>")
    action.set("style", "rounded=1;arcSize=4;whiteSpace=wrap;html=1;fillColor=#FCEFEA;"
                 f"strokeColor={coral};strokeWidth=1;fontFamily=Arial;fontSize=8;fontColor={ink};"
                 "align=center;verticalAlign=middle;shadow=0;")
    geom(action, 715, 465, 120, 24)

    # Inputs enter the actual concat node; output arrow leaves CEM and terminates at H4.
    ext = {
        P + "2072": ("visual / goal", "#45669A"),
        P + "251": ("candidate rollout", "#6557A7"),
        P + "252": ("candidate action", "#B27B48"),
    }
    for cid, (label, color) in ext.items():
        e = find(root, cid)
        e.set("target", "wm-struct-concat")
        e.set("value", label)
        e.set("style", "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;"
                 "html=1;strokeWidth=1.8;endArrow=blockThin;endFill=1;endSize=7;fontFamily=Arial;fontSize=7;"
                 f"labelBackgroundColor=#FBFCFD;entryX=0;entryY=0.5;strokeColor={color};")
        geom(e, None, None, None, None, relative=True)

    edge(root, "wm-struct-cem-output", "best safe H4",
         edge_style + f"strokeColor={navy};", "wm-simplified-safe-search", "wm-simplified-safe-output",
         ((610, 514), (640, 514), (640, 460)))

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
