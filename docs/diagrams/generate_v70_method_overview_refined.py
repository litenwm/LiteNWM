#!/usr/bin/env python3
"""Generate a publication-style, editable v70 method overview."""

from __future__ import annotations

import base64
import math
import xml.etree.ElementTree as ET
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUT = HERE / "v70_method_overview_refined.drawio"
ASSETS = HERE / "assets"
WIDTH, HEIGHT = 2600, 1250

C = {
    "ink": "#17212B",
    "muted": "#647081",
    "line": "#C8D0DA",
    "blue": "#45669A",
    "blue2": "#83A4CC",
    "blue_fill": "#EEF3FA",
    "purple": "#6557A7",
    "purple2": "#A99DD2",
    "purple_fill": "#F3F0FA",
    "orange": "#B27B48",
    "orange2": "#D9AD83",
    "orange_fill": "#FBF5EF",
    "teal": "#0C7976",
    "teal2": "#55A8A3",
    "teal_fill": "#EAF6F4",
    "red": "#984E72",
    "red2": "#D8A6BD",
    "red_fill": "#FBF1F6",
    "green": "#2C875E",
    "green2": "#72B08C",
    "green_fill": "#EDF7F1",
    "gold": "#C49A37",
    "gold_fill": "#FFF8E5",
    "gray": "#8A94A2",
    "gray_fill": "#F4F6F8",
    "navy": "#173A4A",
    "white": "#FFFFFF",
}


def text_style(size=20, color=None, bold=False, align="center"):
    return (
        "text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
        f"fontFamily=Arial;fontSize={size};fontColor={color or C['ink']};"
        f"fontStyle={1 if bold else 0};align={align};verticalAlign=middle;"
    )


def box_style(fill, stroke, width=2, radius=14, shadow=False, extra=""):
    return (
        f"rounded=1;arcSize={radius};whiteSpace=wrap;html=1;fillColor={fill};"
        f"strokeColor={stroke};strokeWidth={width};fontFamily=Arial;"
        f"fontColor={C['ink']};fontSize=20;shadow={1 if shadow else 0};{extra}"
    )


def edge_style(color, width=4, dashed=False, arrow=True, curved=False, extra=""):
    base = "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;"
    if curved:
        base = "curved=1;html=1;"
    return (
        base
        + f"strokeColor={color};strokeWidth={width};dashed={1 if dashed else 0};"
        + ("dashPattern=8 6;" if dashed else "")
        + ("endArrow=blockThin;endFill=1;endSize=12;" if arrow else "endArrow=none;")
        + extra
    )


class Diagram:
    def __init__(self):
        self.mx = ET.Element(
            "mxfile",
            {"host": "drawio", "version": "30.2.4", "compressed": "false", "agent": "Codex drawio-skill"},
        )
        page = ET.SubElement(self.mx, "diagram", {"id": "v70-topconf", "name": "v70 Top-Conference Overview"})
        self.model = ET.SubElement(
            page,
            "mxGraphModel",
            {
                "dx": str(WIDTH),
                "dy": str(HEIGHT),
                "grid": "1",
                "gridSize": "10",
                "guides": "1",
                "page": "1",
                "pageScale": "1",
                "pageWidth": str(WIDTH),
                "pageHeight": str(HEIGHT),
                "background": "#FFFFFF",
            },
        )
        self.root = ET.SubElement(self.model, "root")
        ET.SubElement(self.root, "mxCell", {"id": "0"})
        ET.SubElement(self.root, "mxCell", {"id": "1", "parent": "0"})
        self.seq = 2

    def next_id(self, prefix="n"):
        value = f"{prefix}{self.seq}"
        self.seq += 1
        return value

    def vertex(self, value, x, y, w, h, style, cid=None):
        cid = cid or self.next_id()
        cell = ET.SubElement(
            self.root,
            "mxCell",
            {"id": cid, "value": value, "style": style, "vertex": "1", "parent": "1"},
        )
        ET.SubElement(
            cell,
            "mxGeometry",
            {"x": str(x), "y": str(y), "width": str(w), "height": str(h), "as": "geometry"},
        )
        return cid

    def label(self, value, x, y, w, h, size=20, color=None, bold=False, align="center", cid=None):
        return self.vertex(value, x, y, w, h, text_style(size, color, bold, align), cid)

    def box(self, value, x, y, w, h, fill, stroke, width=2, radius=14, shadow=False, extra="", cid=None):
        return self.vertex(value, x, y, w, h, box_style(fill, stroke, width, radius, shadow, extra), cid)

    def pill(self, value, x, y, w, h, fill, stroke, size=17, color=None, bold=True, cid=None):
        return self.box(
            value,
            x,
            y,
            w,
            h,
            fill,
            stroke,
            1,
            50,
            False,
            f"fontSize={size};fontColor={color or C['ink']};fontStyle={1 if bold else 0};",
            cid,
        )

    def dot(self, x, y, r, fill, stroke=None, width=1, value="", cid=None):
        return self.vertex(
            value,
            x,
            y,
            r,
            r,
            f"ellipse;html=1;aspect=fixed;fillColor={fill};strokeColor={stroke or fill};strokeWidth={width};"
            f"fontFamily=Arial;fontColor={C['white']};fontSize={max(12, int(r * .48))};fontStyle=1;",
            cid,
        )

    def edge(self, source, target, color, width=4, dashed=False, points=None, extra="", arrow=True, curved=False, cid=None):
        cid = cid or self.next_id("e")
        cell = ET.SubElement(
            self.root,
            "mxCell",
            {
                "id": cid,
                "value": "",
                "style": edge_style(color, width, dashed, arrow, curved, extra),
                "edge": "1",
                "parent": "1",
                "source": source,
                "target": target,
            },
        )
        geom = ET.SubElement(cell, "mxGeometry", {"relative": "1", "as": "geometry"})
        if points:
            arr = ET.SubElement(geom, "Array", {"as": "points"})
            for px, py in points:
                ET.SubElement(arr, "mxPoint", {"x": str(px), "y": str(py)})
        return cid

    def free_edge(self, source_xy, target_xy, color, width=3, points=None, dashed=False, arrow=False, curved=True):
        cid = self.next_id("f")
        cell = ET.SubElement(
            self.root,
            "mxCell",
            {
                "id": cid,
                "value": "",
                "style": edge_style(color, width, dashed, arrow, curved),
                "edge": "1",
                "parent": "1",
            },
        )
        geom = ET.SubElement(cell, "mxGeometry", {"relative": "1", "as": "geometry"})
        ET.SubElement(geom, "mxPoint", {"x": str(source_xy[0]), "y": str(source_xy[1]), "as": "sourcePoint"})
        ET.SubElement(geom, "mxPoint", {"x": str(target_xy[0]), "y": str(target_xy[1]), "as": "targetPoint"})
        if points:
            arr = ET.SubElement(geom, "Array", {"as": "points"})
            for px, py in points:
                ET.SubElement(arr, "mxPoint", {"x": str(px), "y": str(py)})
        return cid

    def image(self, path, x, y, w, h, cid=None):
        uri = "data:image/jpeg," + base64.b64encode(path.read_bytes()).decode("ascii")
        return self.vertex(
            "",
            x,
            y,
            w,
            h,
            f"shape=image;html=1;imageAspect=0;aspect=fixed;image={uri};strokeColor=none;fillColor=none;",
            cid,
        )


def section(g, letter, title, x, y, color, width=620):
    g.box("", x, y + 2, 8, 42, color, color, 0, 50)
    g.label(f"{letter}  {title}", x + 18, y, width, 46, 27, color, True, "left")


def token_grid(g, x, y, rows, cols, cell, gap, palette, prefix):
    ids = []
    for rr in range(rows):
        for cc in range(cols):
            color = palette[(rr + cc) % len(palette)]
            ids.append(g.box("", x + cc * (cell + gap), y + rr * (cell + gap), cell, cell, color, color, 0, 2, cid=f"{prefix}_{rr}_{cc}"))
    return ids


def build():
    g = Diagram()

    # Header, deliberately compact to reserve space for the method itself.
    g.label("DINOv3-LTWM safe action selection", 70, 24, 1350, 54, 40, C["ink"], True, "left")
    g.pill("V70  INFERENCE TIME", 2180, 34, 330, 34, C["blue_fill"], C["blue"], 16, C["blue"])
    g.box("", 70, 98, 2440, 2, "#DCE2E8", "#DCE2E8", 0, 0)
    g.label("frozen visual context + K17 candidates -> latent outcomes -> conservative H4", 70, 108, 1500, 26, 18, C["muted"], False, "left")

    # A. Visual context is a short local chain.
    section(g, "A", "VISUAL CONTEXT", 70, 154, C["blue"], 440)
    frame_paths = [
        ASSETS / "v70_sacson_t-3.jpg",
        ASSETS / "v70_sacson_t-2.jpg",
        ASSETS / "v70_sacson_t-1.jpg",
        ASSETS / "v70_sacson_t.jpg",
    ]
    for i, (path, stamp) in enumerate(zip(frame_paths, ["t-3", "t-2", "t-1", "t"])):
        x = 90 + i * 105
        g.box("", x, 205, 96, 84, C["white"], C["blue2"], 2, 8, True)
        g.image(path, x + 4, 209, 88, 61, cid=f"frame_{i}")
        g.pill(stamp, x + 25, 268, 46, 17, "#17384D", C["white"], 11, C["white"])
    g.pill("4 history", 90, 306, 122, 26, C["blue_fill"], C["blue"], 14, C["blue"])

    g.box("", 250, 342, 140, 108, C["white"], C["green2"], 3, 10, True, cid="goal")
    g.image(ASSETS / "v70_sacson_goal.jpg", 256, 348, 128, 80, cid="goal_image")
    g.pill("GOAL", 281, 423, 78, 22, C["green"], C["white"], 13, C["white"])

    g.box("", 560, 185, 230, 180, C["blue_fill"], C["blue"], 3, 16, True, cid="dino")
    g.label("DINOv3-L/16", 580, 206, 190, 34, 25, C["ink"], True)
    g.pill("FROZEN", 610, 252, 130, 26, C["white"], C["blue"], 14, C["blue"])
    for i in range(4):
        g.box("", 590 + i * 12, 304 - i * 3, 90, 40, ["#DDEAF4", "#D0E1EE", "#BED6E7", "#A8C8DF"][i], C["blue"], 1, 5)
    token_grid(g, 680, 300, 3, 5, 8, 4, ["#7AA5C8", "#4F83B0", "#A5C2D8"], "dino_patch")

    g.vertex("Adapter", 830, 230, 140, 72, f"shape=trapezoid;direction=south;whiteSpace=wrap;html=1;fillColor=#F3ECF3;strokeColor={C['red2']};strokeWidth=3;fontFamily=Arial;fontSize=21;fontStyle=1;fontColor={C['ink']};", "adapter")
    g.pill("1024 to 384", 820, 318, 160, 25, C["white"], C["red2"], 13, C["red"])

    g.box("", 1005, 205, 92, 110, C["white"], C["blue2"], 2, 12, True, cid="visual_tokens")
    token_grid(g, 1022, 221, 5, 4, 9, 3, ["#BBD3E4", "#8DB6D2", "#5E91B8"], "visual_token")
    g.label("TOKENS", 1014, 288, 74, 18, 11, C["blue"], True)

    g.edge("frame_3", "dino", C["blue"], 4, points=[(530, 247)], extra="exitX=1;exitY=.5;entryX=0;entryY=.35;")
    g.edge("goal", "dino", C["green2"], 4, points=[(470, 396), (520, 396)], extra="exitX=1;exitY=.5;entryX=0;entryY=.85;")
    g.edge("dino", "adapter", C["blue"], 4, extra="exitX=1;exitY=.5;entryX=0;entryY=.5;")
    g.edge("adapter", "visual_tokens", C["blue"], 4, extra="exitX=1;exitY=.5;entryX=0;entryY=.5;")

    # B. The world model stays above the central planner; its candidate condition is an internal badge.
    section(g, "B", "LATENT WORLD MODEL", 1130, 154, C["purple"], 520)
    g.box("", 1130, 190, 560, 200, C["purple_fill"], C["purple"], 4, 18, True, cid="ltwm")
    g.pill("LiteTokenFormer x3", 1160, 208, 215, 28, C["purple"], C["purple"], 15, C["white"])
    g.pill("K17-conditioned", 1452, 208, 190, 28, C["white"], C["orange"], 13, C["orange"])
    for i in range(3):
        y = 255 + i * 34
        g.pill(f"M{i+1}", 1162, y, 46, 22, C["purple"], C["purple"], 12, C["white"])
        g.box("", 1220, y, 210, 22, C["white"], C["purple2"], 1, 40)
        g.free_edge((1234, y + 11), (1414, y + 11), C["purple2"], 2, points=[(1280, y + 4), (1330, y + 19), (1380, y + 5)], arrow=False, curved=True)
    g.label("PREDICTED H1-H4", 1470, 250, 170, 20, 14, C["teal"], True)
    for h in range(4):
        x = 1470 + h * 43
        token_grid(g, x, 280, 3, 3, 8, 2, ["#B6DBD8", "#79C1BE", "#459F9D"], f"future_{h}")
        g.label(f"H{h+1}", x - 2, 310, 34, 16, 11, C["muted"], True)
    g.pill("K x 4 x 256 x 384", 1340, 350, 190, 24, C["white"], C["purple"], 13, C["purple"])
    g.edge("visual_tokens", "ltwm", C["blue"], 5, extra="exitX=1;exitY=.5;entryX=0;entryY=.5;")

    # C. Candidate proposals remain local and visual, without a long round-trip connector.
    section(g, "C", "CANDIDATE ACTIONS", 70, 500, C["orange"], 520)
    g.box("", 90, 585, 170, 105, C["orange_fill"], C["orange"], 3, 15, True, cid="nomad")
    g.label("NoMaD", 110, 605, 130, 32, 24, C["ink"], True)
    g.label("16 + incumbent", 105, 642, 140, 20, 15, C["muted"])
    g.pill("H4  local SE(2)", 99, 661, 152, 21, C["white"], C["orange"], 12, C["orange"])

    g.box("", 320, 618, 80, 54, C["ink"], C["white"], 3, 40, True, cid="robot")
    g.dot(346, 631, 28, C["white"], C["ink"], 3)
    for ax, ay, bx, by in [(326, 624, 306, 604), (394, 624, 414, 604), (326, 666, 306, 686), (394, 666, 414, 686)]:
        g.free_edge((ax, ay), (bx, by), C["ink"], 5, arrow=False, curved=False)
        g.dot(bx - 5, by - 5, 10, C["ink"])

    endpoints = [(620, 548), (640, 576), (652, 604), (660, 632), (656, 660), (645, 688), (630, 716)]
    colors = [C["line"], C["orange2"], C["orange"], C["red"], C["orange"], C["orange2"], C["line"]]
    for i, ((tx, ty), color) in enumerate(zip(endpoints, colors)):
        g.free_edge((400, 645), (tx, ty), color, 5 if color in (C["orange"], C["red"]) else 3, points=[(475, 645 + (i - 3) * 4), (545, 645 + (i - 3) * 18)], arrow=False, curved=True)
        g.dot(tx - 5, ty - 5, 10, color, C["white"], 1)
    g.pill("K17", 690, 590, 82, 62, C["orange"], C["orange"], 22, C["white"], cid="candidate_set")
    g.edge("nomad", "robot", C["orange"], 4, extra="exitX=1;exitY=.5;entryX=0;entryY=.5;")

    # D. Central core: a compact, ordered main chain with no panel-spanning return paths.
    g.box("", 820, 480, 1000, 460, "#FCFDFD", C["green"], 5, 20, True, "pointerEvents=0;", cid="planner_bg")
    g.pill("V70  CENTRAL CORE", 850, 502, 205, 30, C["green"], C["green"], 15, C["white"])
    g.label("SAFE CANDIDATE PLANNER", 850, 540, 540, 34, 29, C["ink"], True, "left")
    g.label("feature fusion  ->  risk gates  ->  CEM selection", 850, 575, 500, 20, 16, C["muted"], False, "left")

    g.box("", 860, 630, 160, 175, C["white"], C["teal"], 3, 16, True, cid="feature")
    g.label("815-D", 880, 647, 120, 31, 25, C["teal"], True)
    g.label("candidate fusion", 876, 680, 128, 18, 14, C["muted"], True)
    feature_rows = [("242", "traj.", C["orange"]), ("24", "phys.", C["gold"]), ("210", "DINO", C["blue"]), ("336", "LTWM", C["purple"]), ("3", "guard", C["gray"])]
    for i, (dim, name, color) in enumerate(feature_rows):
        y = 710 + i * 17
        g.pill(dim, 876, y, 42, 15, color, color, 10, C["white"])
        g.label(name, 925, y - 1, 68, 17, 12, color, True, "left")

    g.box("", 1060, 610, 250, 205, C["navy"], C["teal"], 5, 18, True, cid="armd")
    g.pill("Arm-D x3", 1084, 628, 120, 25, C["teal"], C["teal"], 14, C["white"])
    g.label("risk ensemble", 1080, 660, 210, 26, 21, C["white"], True)
    for i in range(3):
        y = 702 + i * 30
        g.pill(f"member {i+1}", 1082, y, 90, 21, C["white"], C["teal2"], 11, C["teal"])
        g.box("", 1184, y, 96, 21, C["teal"], C["teal2"], 1, 40)
    g.pill("mean", 1085, 782, 70, 18, C["white"], C["teal2"], 11, C["teal"])
    g.pill("std", 1164, 782, 62, 18, C["white"], C["teal2"], 11, C["teal"])

    g.box("", 1360, 610, 380, 78, C["red_fill"], C["red2"], 3, 16, False, cid="risk")
    g.label("risk mean / std", 1380, 623, 175, 26, 20, C["red"], True, "left")
    g.free_edge((1560, 655), (1645, 655), C["red"], 3, points=[(1585, 638), (1615, 630), (1635, 648)], arrow=False, curved=True)
    g.pill("ATE  RPE  endpoint", 1572, 623, 145, 21, C["white"], C["red2"], 11, C["red"])
    g.pill("calibrated", 1590, 652, 108, 18, C["white"], C["red2"], 11, C["red"])

    gate_defs = [("physical", "UCB"), ("support", "in-support"), ("dual", "p-dual"), ("harm", "p-harm")]
    gate_ids = []
    for i, (name, sub) in enumerate(gate_defs):
        x = 1360 + i * 95
        gate_ids.append(g.box("", x, 720, 80, 74, C["green_fill"], C["green"], 3, 18, False, cid=name))
        g.label(name, x + 4, 733, 72, 18, 12, C["green"], True)
        g.label(sub, x + 4, 757, 72, 14, 10, C["muted"], False)
    g.edge("risk", "physical", C["red"], 4, extra="exitX=.11;exitY=1;entryX=.5;entryY=0;")
    for a, b in zip(gate_ids, gate_ids[1:]):
        g.edge(a, b, C["green"], 3, extra="exitX=1;exitY=.5;entryX=0;entryY=.5;")

    g.box("", 1050, 842, 250, 70, C["orange_fill"], C["orange"], 3, 18, False, cid="cem")
    g.label("CEM x3", 1070, 852, 105, 24, 22, C["orange"], True, "left")
    g.pill("16 / round", 1172, 852, 105, 19, C["white"], C["orange"], 11, C["orange"])
    for i, (cx, rx) in enumerate([(1155, 23), (1212, 18), (1260, 13)], 1):
        g.vertex("", cx - rx, 886 - rx // 2, rx * 2, rx, f"ellipse;html=1;fillColor=#FFFFFF;strokeColor={C['orange']};strokeWidth={3 if i == 3 else 2};dashed={1 if i < 3 else 0};dashPattern=5 4;", f"cem_r{i}")
        for j, (dx, dy) in enumerate([(-8, -5), (7, -6), (8, 4), (-7, 5)]):
            g.dot(cx + dx - 3, 886 + dy - 3, 6, C["green"] if j in (2, 3) else C["line"], C["white"], 1)

    g.box("", 1360, 838, 155, 76, C["green_fill"], C["green"], 4, 18, False, cid="archive")
    g.label("union65", 1374, 849, 126, 25, 21, C["green"], True)
    g.label("K17 + R1 + R2 + R3", 1368, 876, 140, 15, 11, C["muted"], True)
    for group, color in enumerate([C["orange"], C["orange2"], C["teal2"], C["green"]]):
        for j in range(3):
            g.dot(1376 + group * 29 + j * 7, 896, 7, color, C["white"], 1)

    # The selector is intentionally a small terminal chip, not a competing panel.
    g.box("", 1570, 838, 180, 76, C["green"], C["green"], 3, 18, True, cid="h4")
    g.label("SELECTED H4", 1584, 849, 152, 20, 16, C["white"], True)
    g.label("a_t ... a_t+3", 1584, 870, 152, 15, 11, C["white"], False)
    for i in range(4):
        g.dot(1600 + i * 32, 892, 9, C["white"], C["white"], 1)

    # Four short main-chain connectors.  The only routed link stays in the open corridor above CEM.
    g.edge("ltwm", "feature", C["purple"], 5, points=[(790, 430), (790, 680)], extra="exitX=.1;exitY=1;entryX=0;entryY=.3;")
    g.edge("candidate_set", "feature", C["orange"], 5, points=[(731, 740), (830, 740)], extra="exitX=.5;exitY=1;entryX=0;entryY=.63;")
    g.edge("feature", "armd", C["teal"], 5, extra="exitX=1;exitY=.5;entryX=0;entryY=.5;")
    g.edge("armd", "risk", C["red"], 5, extra="exitX=1;exitY=.35;entryX=0;entryY=.5;")
    g.edge("harm", "cem", C["orange"], 5, points=[(1685, 815), (1175, 815)], extra="exitX=.5;exitY=1;entryX=.5;entryY=0;")
    g.edge("cem", "archive", C["green"], 5, extra="exitX=1;exitY=.5;entryX=0;entryY=.5;")
    g.edge("archive", "h4", C["green"], 5, extra="exitX=1;exitY=.5;entryX=0;entryY=.5;")

    # 6B is explicitly an external, local controller loop; no long return edge crosses the figure.
    g.box("", 1870, 760, 610, 140, C["gray_fill"], C["gray"], 2, 16, False, "dashed=1;dashPattern=8 6;pointerEvents=0;", cid="loop_bg")
    g.label("6B  EXTERNAL CONTROL LOOP", 1895, 777, 300, 23, 16, C["muted"], True, "left")
    g.pill("execute a_t", 1895, 820, 140, 34, C["white"], C["gray"], 14, C["muted"], cid="execute")
    g.pill("observe o_t+1", 2085, 820, 170, 34, C["white"], C["gray"], 14, C["muted"], cid="observe")
    g.pill("replan", 2310, 820, 130, 34, C["white"], C["gray"], 14, C["muted"], cid="replan")
    g.label("outside formal v70", 1895, 862, 210, 18, 12, C["muted"], False, "left")
    g.edge("h4", "execute", C["gray"], 3, dashed=True, points=[(1840, 876), (1840, 837)], extra="exitX=1;exitY=.5;entryX=0;entryY=.5;")
    g.edge("execute", "observe", C["gray"], 3, dashed=True, extra="exitX=1;exitY=.5;entryX=0;entryY=.5;")
    g.edge("observe", "replan", C["gray"], 3, dashed=True, extra="exitX=1;exitY=.5;entryX=0;entryY=.5;")
    g.free_edge((2375, 866), (1985, 866), C["gray"], 2, points=[(2320, 888), (2050, 888)], dashed=True, arrow=True, curved=True)

    # Compact legend and provenance.
    g.box("", 70, 1155, 2440, 2, "#DCE2E8", "#DCE2E8", 0, 0)
    lx = 80
    for color, label in [(C["blue"], "visual"), (C["orange"], "candidates"), (C["purple"], "future"), (C["teal"], "scoring"), (C["green"], "safe")]:
        g.dot(lx, 1175, 12, color)
        g.label(label, lx + 18, 1168, 105, 20, 13, C["muted"], True, "left")
        lx += 135
    g.label("embedded SACSoN RGB frames", 2070, 1164, 420, 24, 13, C["muted"], False, "right")

    ET.indent(g.mx, space="  ")
    OUT.write_text(ET.tostring(g.mx, encoding="unicode", xml_declaration=True), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    build()
