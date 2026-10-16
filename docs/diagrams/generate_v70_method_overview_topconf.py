#!/usr/bin/env python3
"""Generate a publication-style, editable v70 method overview."""

from __future__ import annotations

import base64
import math
import xml.etree.ElementTree as ET
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUT = HERE / "v70_method_overview_topconf.drawio"
ASSETS = HERE / "assets"
WIDTH, HEIGHT = 2880, 1450

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

    # Header: compact and typography-led.
    g.label("DINOv3-LTWM safe trajectory planning", 80, 28, 1600, 60, 42, C["ink"], True, "left")
    g.pill("v70  INFERENCE-TIME", 2320, 38, 470, 36, C["blue_fill"], C["blue"], 18, C["blue"])
    g.box("", 80, 116, 2710, 2, "#DCE2E8", "#DCE2E8", 0, 0)
    g.label("visual context + candidate actions -> latent futures -> calibrated safe H4", 80, 122, 1900, 30, 19, C["muted"], False, "left")

    # A. Compact visual conditioning in the upper-left.
    section(g, "A", "VISUAL CONDITIONING", 80, 165, C["blue"], 560)
    frame_paths = [
        ASSETS / "v70_sacson_t-3.jpg",
        ASSETS / "v70_sacson_t-2.jpg",
        ASSETS / "v70_sacson_t-1.jpg",
        ASSETS / "v70_sacson_t.jpg",
    ]
    for i, (path, stamp) in enumerate(zip(frame_paths, ["t-3", "t-2", "t-1", "t"])):
        x = 100 + i * 110
        g.box("", x, 220, 100, 90, C["white"], C["blue2"], 2, 8, True)
        g.image(path, x + 4, 224, 92, 66, cid=f"frame_{i}")
        g.pill(stamp, x + 26, 288, 48, 18, "#17384D", C["white"], 12, C["white"])
    g.pill("4 history", 100, 328, 135, 28, C["blue_fill"], C["blue"], 15, C["blue"])

    g.box("", 250, 350, 150, 112, C["white"], C["green2"], 3, 10, True, cid="goal")
    g.image(ASSETS / "v70_sacson_goal.jpg", 256, 356, 138, 84, cid="goal_image")
    g.pill("GOAL", 286, 430, 78, 25, C["green"], C["white"], 14, C["white"])

    g.box("", 560, 205, 240, 190, C["blue_fill"], C["blue"], 3, 16, True, cid="dino")
    g.label("DINOv3-L/16", 585, 225, 190, 38, 26, C["ink"], True)
    g.pill("FROZEN", 610, 270, 135, 28, C["white"], C["blue"], 15, C["blue"])
    for i in range(5):
        g.box("", 590 + i * 11, 328 - i * 4, 95, 48, ["#DDEAF4", "#D0E1EE", "#BED6E7", "#A8C8DF", "#8DB8D4"][i], C["blue"], 1, 5)
    token_grid(g, 686, 320, 3, 5, 9, 4, ["#7AA5C8", "#4F83B0", "#A5C2D8"], "dino_patch")

    g.vertex("Adapter", 835, 250, 145, 80, f"shape=trapezoid;direction=south;whiteSpace=wrap;html=1;fillColor=#F3ECF3;strokeColor={C['red2']};strokeWidth=3;fontFamily=Arial;fontSize=22;fontStyle=1;fontColor={C['ink']};", "adapter")
    g.pill("1024 to 384", 827, 345, 160, 27, C["white"], C["red2"], 14, C["red"])

    g.box("", 1000, 220, 100, 120, C["white"], C["blue2"], 2, 12, True, cid="visual_tokens")
    token_grid(g, 1018, 238, 5, 4, 10, 4, ["#BBD3E4", "#8DB6D2", "#5E91B8"], "visual_token")
    g.label("TOKENS", 1008, 310, 84, 18, 12, C["blue"], True)

    g.edge("frame_3", "dino", C["blue"], 4, points=[(520, 264)], extra="exitX=1;exitY=.5;entryX=0;entryY=.3;")
    g.edge("goal", "dino", C["green2"], 4, points=[(450, 404), (510, 404)], extra="exitX=1;exitY=.5;entryX=0;entryY=.85;")
    g.edge("dino", "adapter", C["blue"], 4, extra="exitX=1;exitY=.5;entryX=0;entryY=.5;")
    g.edge("adapter", "visual_tokens", C["blue"], 4, extra="exitX=1;exitY=.5;entryX=0;entryY=.5;")

    # LTWM is a compact upper-band predictor, not a tall column.
    g.box("", 1160, 185, 600, 255, C["purple_fill"], C["purple"], 4, 18, True, cid="ltwm")
    g.pill("C  LATENT WORLD MODEL", 1190, 205, 245, 32, C["purple"], C["purple"], 16, C["white"])
    g.pill("K17-conditioned", 1520, 207, 195, 30, C["white"], C["orange"], 14, C["orange"])
    g.label("LiteTokenFormer x3", 1190, 248, 300, 36, 28, C["purple"], True, "left")
    for i in range(3):
        y = 300 + i * 40
        g.pill(f"M{i+1}", 1190, y, 55, 26, C["purple"], C["purple"], 14, C["white"])
        g.box("", 1260, y, 210, 26, C["white"], C["purple2"], 1, 40)
        g.free_edge((1280, y + 13), (1450, y + 13), C["purple2"], 2, points=[(1320, y + 4), (1370, y + 22), (1420, y + 6)], arrow=False, curved=True)
    g.label("PREDICTED H1-H4", 1510, 254, 200, 25, 16, C["teal"], True)
    for h in range(4):
        x = 1510 + h * 52
        token_grid(g, x, 300, 3, 3, 9, 3, ["#B6DBD8", "#79C1BE", "#459F9D"], f"future_{h}")
        g.label(f"H{h+1}", x, 340, 40, 18, 13, C["muted"], True)
    g.pill("K x 4 x 256 x 384", 1380, 392, 220, 26, C["white"], C["purple"], 14, C["purple"])
    g.edge("visual_tokens", "ltwm", C["blue"], 5, extra="exitX=1;exitY=.5;entryX=0;entryY=.5;")

    # B. Candidate proposals: local, visual, and immediately below the visual stream.
    section(g, "B", "CANDIDATE ACTIONS", 80, 575, C["orange"], 530)
    g.box("", 100, 660, 190, 120, C["orange_fill"], C["orange"], 3, 16, True, cid="nomad")
    g.label("NoMaD", 125, 678, 140, 35, 27, C["ink"], True)
    g.label("16 proposals", 125, 718, 140, 24, 17, C["muted"])
    g.pill("H4  local SE(2)", 116, 748, 158, 24, C["white"], C["orange"], 14, C["orange"])
    g.pill("plus incumbent", 105, 805, 180, 34, C["white"], C["gray"], 16, C["muted"], cid="incumbent")

    g.box("", 350, 700, 90, 60, C["ink"], C["white"], 3, 40, True, cid="robot")
    g.dot(382, 716, 28, C["white"], C["ink"], 3)
    for ax, ay, bx, by in [(356, 706, 332, 682), (434, 706, 458, 682), (356, 754, 332, 778), (434, 754, 458, 778)]:
        g.free_edge((ax, ay), (bx, by), C["ink"], 6, arrow=False, curved=False)
        g.dot(bx - 6, by - 6, 12, C["ink"])

    endpoints = [(640, 625), (660, 660), (680, 695), (692, 730), (695, 765), (690, 800), (678, 835), (660, 870), (640, 905)]
    colors = [C["line"], C["orange2"], C["orange"], C["line"], C["red"], C["orange"], C["orange2"], C["line"], C["orange"]]
    for i, ((tx, ty), color) in enumerate(zip(endpoints, colors)):
        g.free_edge((440, 730), (tx, ty), color, 5 if color in (C["orange"], C["red"]) else 3, points=[(505, 730 + (i - 4) * 3), (570, 730 + (i - 4) * 20)], arrow=False, curved=True)
        g.dot(tx - 6, ty - 6, 12, color, C["white"], 2)
    g.pill("K17", 770, 700, 110, 70, C["orange"], C["orange"], 25, C["white"], cid="candidate_set")
    g.edge("nomad", "robot", C["orange"], 4, extra="exitX=1;exitY=.5;entryX=0;entryY=.5;")
    g.edge("incumbent", "robot", C["gray"], 3, dashed=True, points=[(320, 822), (320, 752)], extra="exitX=1;exitY=.5;entryX=0;entryY=.85;")
    g.edge("candidate_set", "ltwm", C["orange"], 3, dashed=True, points=[(900, 470), (1460, 470)], extra="exitX=1;exitY=.35;entryX=.5;entryY=1;")

    # Central selection core: all v70 action-selection logic lives in one panel.
    g.box("", 920, 510, 1200, 550, "#FCFDFD", C["green"], 5, 20, True, "pointerEvents=0;", cid="planner_bg")
    g.pill("V70  CENTRAL CORE", 960, 535, 210, 34, C["green"], C["green"], 16, C["white"])
    g.label("SAFE CANDIDATE PLANNER", 960, 578, 500, 42, 31, C["ink"], True, "left")
    g.label("future-aware, risk-calibrated action selection", 960, 618, 520, 24, 17, C["muted"], False, "left")

    g.box("", 970, 680, 180, 210, C["white"], C["teal"], 3, 16, True, cid="feature")
    g.label("815-D", 990, 698, 140, 35, 28, C["teal"], True)
    g.label("fusion", 990, 732, 140, 22, 17, C["muted"], True)
    feature_rows = [("242", "traj.", C["orange"]), ("24", "phys.", C["gold"]), ("210", "DINO", C["blue"]), ("336", "LTWM", C["purple"]), ("3", "guard", C["gray"])]
    for i, (dim, name, color) in enumerate(feature_rows):
        y = 770 + i * 23
        g.pill(dim, 988, y, 52, 20, color, color, 12, C["white"])
        g.label(name, 1050, y, 72, 20, 14, color, True, "left")

    g.box("", 1210, 650, 280, 250, C["navy"], C["teal"], 5, 18, True, cid="armd")
    g.pill("x3", 1408, 670, 52, 28, C["teal"], C["teal"], 15, C["white"])
    g.label("Universal Arm-D", 1235, 704, 230, 38, 27, C["white"], True)
    for i in range(3):
        y = 760 + i * 42
        g.pill(f"seed {i+1}", 1238, y, 86, 28, C["white"], C["teal2"], 14, C["teal"])
        g.box("", 1338, y, 112, 28, C["teal"], C["teal2"], 1, 40)
    g.pill("mean", 1240, 860, 86, 24, C["white"], C["teal2"], 13, C["teal"])
    g.pill("std", 1340, 860, 86, 24, C["white"], C["teal2"], 13, C["teal"])

    g.box("", 1550, 650, 500, 90, C["red_fill"], C["red2"], 3, 16, False, cid="risk")
    g.label("risk mean / std", 1580, 666, 190, 32, 23, C["red"], True, "left")
    g.free_edge((1770, 710), (1875, 710), C["red"], 3, points=[(1800, 685), (1830, 665), (1860, 700)], arrow=False, curved=True)
    g.pill("ATE  RPE  endpoint", 1850, 670, 165, 26, C["white"], C["red2"], 13, C["red"])
    g.pill("dual  harm  rank", 1850, 704, 165, 24, C["white"], C["red2"], 13, C["red"])

    gate_defs = [("physical", "physical", "UCB"), ("support", "support", "in-support"), ("dual", "dual", "p-dual"), ("harm", "harm", "p-harm")]
    gate_ids = []
    for i, (cid, label, sub) in enumerate(gate_defs):
        x = 1550 + i * 115
        gate_ids.append(g.box("", x, 770, 100, 90, C["green_fill"], C["green"], 3, 24, False, cid=cid))
        g.label(label, x + 8, 784, 84, 25, 15, C["green"], True)
        g.label(sub, x + 8, 815, 84, 18, 12, C["muted"], False)
    g.edge("risk", "physical", C["red"], 4, points=[(1580, 755)], extra="exitX=.06;exitY=1;entryX=.3;entryY=0;")
    for a, b in zip(gate_ids, gate_ids[1:]):
        g.edge(a, b, C["green"], 3, extra="exitX=1;exitY=.5;entryX=0;entryY=.5;")

    g.box("", 1210, 920, 310, 100, C["orange_fill"], C["orange"], 3, 18, False, cid="cem")
    g.label("CEM x3", 1230, 932, 120, 30, 25, C["orange"], True, "left")
    g.pill("16 / round", 1370, 934, 118, 23, C["white"], C["orange"], 13, C["orange"])
    for i, (cx, rx, ry) in enumerate([(1300, 33, 35), (1380, 27, 29), (1450, 22, 23)], 1):
        g.vertex("", cx - rx, 985 - ry, rx * 2, ry * 2, f"ellipse;html=1;fillColor=#FFFFFF;strokeColor={C['orange']};strokeWidth={4 if i == 3 else 2};dashed={1 if i < 3 else 0};dashPattern=6 4;", f"cem_r{i}")
        g.label(f"R{i}", cx - 15, 962, 30, 16, 12, C["orange"], True)
        for j, (dx, dy) in enumerate([(-13, -14), (8, -16), (14, -4), (-9, 5), (6, 10), (-10, 18)]):
            color = C["green"] if j in (3, 4) else C["line"]
            scale = [1.0, 0.75, 0.55][i - 1]
            g.dot(cx + dx * scale - 4, 985 + dy * scale - 4, 8, color, C["white"], 1)

    g.box("", 1570, 900, 220, 120, C["green_fill"], C["green"], 4, 18, False, cid="archive")
    g.label("union65", 1595, 915, 170, 30, 25, C["green"], True)
    g.label("17 + 16 + 16 + 16", 1595, 947, 170, 20, 14, C["green"], True)
    g.label("K17   R1   R2   R3", 1595, 970, 170, 18, 12, C["muted"], True)
    for group, color in enumerate([C["orange"], C["orange2"], C["teal2"], C["green"]]):
        x = 1600 + group * 40
        for j in range(4):
            g.dot(x + j * 8, 994, 8, color, C["white"], 1)

    g.edge("ltwm", "feature", C["purple"], 6, extra="exitX=.32;exitY=1;entryX=.5;entryY=0;")
    g.edge("candidate_set", "feature", C["orange"], 5, extra="exitX=1;exitY=.5;entryX=0;entryY=.35;")
    g.edge("feature", "armd", C["teal"], 6, extra="exitX=1;exitY=.5;entryX=0;entryY=.5;")
    g.edge("armd", "risk", C["red"], 6, extra="exitX=1;exitY=.35;entryX=0;entryY=.5;")
    g.edge("harm", "cem", C["orange"], 5, points=[(1945, 880), (1510, 880)], extra="exitX=.5;exitY=1;entryX=.85;entryY=0;")
    g.edge("cem", "archive", C["green"], 5, extra="exitX=1;exitY=.6;entryX=0;entryY=.6;")

    # Selected plan and external controller stay visually separate.
    g.box("", 2180, 700, 270, 170, C["green"], C["green"], 4, 32, True, cid="h4")
    g.label("SELECTED H4", 2205, 718, 220, 34, 26, C["white"], True)
    g.label("a_t -> a_t+3", 2205, 753, 220, 22, 15, C["white"], False)
    for i in range(4):
        g.dot(2230 + i * 48, 800 + (i % 2) * 12, 16, C["white"], C["white"], 1)
    g.edge("archive", "h4", C["green"], 6, points=[(2050, 960), (2050, 785)], extra="exitX=1;exitY=.5;entryX=0;entryY=.5;")

    g.box("", 2180, 1010, 610, 150, C["gray_fill"], C["gray"], 2, 16, False, "dashed=1;dashPattern=8 6;pointerEvents=0;", cid="loop_bg")
    g.label("6B  EXTERNAL CONTROL LOOP", 2210, 1025, 290, 26, 17, C["muted"], True, "left")
    g.pill("execute a_t", 2210, 1072, 135, 38, C["white"], C["gray"], 16, C["muted"], cid="execute")
    g.pill("observe o_t+1", 2390, 1072, 170, 38, C["white"], C["gray"], 16, C["muted"], cid="observe")
    g.pill("replan", 2610, 1072, 130, 38, C["white"], C["gray"], 16, C["muted"], cid="replan")
    g.edge("h4", "execute", C["gray"], 3, dashed=True, extra="exitX=.35;exitY=1;entryX=.5;entryY=0;")
    g.edge("execute", "observe", C["gray"], 3, dashed=True, extra="exitX=1;exitY=.5;entryX=0;entryY=.5;")
    g.edge("observe", "replan", C["gray"], 3, dashed=True, extra="exitX=1;exitY=.5;entryX=0;entryY=.5;")
    g.edge("replan", "frame_0", C["gray"], 2, dashed=True, points=[(2810, 1240), (50, 1240), (50, 265)], extra="exitX=1;exitY=.5;entryX=0;entryY=.5;")

    # Minimal legend and provenance.
    g.box("", 80, 1330, 2710, 2, "#DCE2E8", "#DCE2E8", 0, 0)
    lx = 90
    for color, label in [(C["blue"], "visual"), (C["orange"], "candidates"), (C["purple"], "future"), (C["teal"], "scoring"), (C["green"], "safe")]:
        g.dot(lx, 1350, 13, color)
        g.label(label, lx + 20, 1343, 120, 24, 14, C["muted"], True, "left")
        lx += 145
    g.label("SACSoN frames  Jan-20-2023-bww8_00000001_3", 1950, 1340, 820, 28, 14, C["muted"], False, "right")

    ET.indent(g.mx, space="  ")
    OUT.write_text(ET.tostring(g.mx, encoding="unicode", xml_declaration=True), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    build()
