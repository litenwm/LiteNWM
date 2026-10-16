#!/usr/bin/env python3
"""Generate a publication-style, editable v70 method overview."""

from __future__ import annotations

import base64
import math
import xml.etree.ElementTree as ET
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUT = HERE / "v70_method_overview_refined_v2.drawio"
ASSETS = HERE / "assets"
WIDTH, HEIGHT = 2600, 1100

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

    g.label("DINOv3-LTWM safe action selection", 70, 24, 1350, 54, 40, C["ink"], True, "left")
    g.pill("V70  INFERENCE TIME", 2180, 34, 330, 34, C["blue_fill"], C["blue"], 16, C["blue"])
    g.box("", 70, 98, 2440, 2, "#DCE2E8", "#DCE2E8", 0, 0)
    g.label("frozen visual context + K17 candidates -> latent outcomes -> conservative H4", 70, 108, 1500, 26, 18, C["muted"], False, "left")

    # A. A short, uncluttered visual encoder chain.
    section(g, "A", "VISUAL CONTEXT", 70, 154, C["blue"], 420)
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

    g.box("", 500, 185, 240, 170, C["blue_fill"], C["blue"], 3, 16, True, cid="dino")
    g.label("DINOv3-L/16", 525, 205, 190, 33, 25, C["ink"], True)
    g.pill("FROZEN", 555, 246, 130, 25, C["white"], C["blue"], 14, C["blue"])
    token_grid(g, 585, 286, 4, 5, 13, 4, ["#C5D9E9", "#96B9D2", "#6095BD"], "dino_patch")
    g.label("patch features", 550, 336, 140, 15, 12, C["blue"], True)

    g.box("", 790, 220, 150, 82, "#F6EEF5", C["red2"], 3, 14, True, cid="adapter")
    g.label("Adapter", 808, 234, 114, 26, 21, C["ink"], True)
    g.label("1024 -> 384", 808, 264, 114, 16, 12, C["red"], True)

    g.box("", 970, 205, 92, 112, C["white"], C["blue2"], 2, 12, True, cid="visual_tokens")
    token_grid(g, 987, 221, 5, 4, 9, 3, ["#BBD3E4", "#8DB6D2", "#5E91B8"], "visual_token")
    g.label("TOKENS", 979, 289, 74, 17, 11, C["blue"], True)

    g.edge("frame_3", "dino", C["blue"], 4, points=[(470, 247)], extra="exitX=1;exitY=.5;entryX=0;entryY=.35;")
    g.edge("goal", "dino", C["green2"], 4, points=[(450, 396), (470, 396)], extra="exitX=1;exitY=.5;entryX=0;entryY=.85;")
    g.edge("dino", "adapter", C["blue"], 4, extra="exitX=1;exitY=.5;entryX=0;entryY=.5;")
    g.edge("adapter", "visual_tokens", C["blue"], 4, extra="exitX=1;exitY=.5;entryX=0;entryY=.5;")

    # B. Latent prediction remains a compact upper band.
    section(g, "B", "LATENT WORLD MODEL", 1080, 154, C["purple"], 500)
    g.box("", 1080, 185, 580, 190, C["purple_fill"], C["purple"], 4, 18, True, cid="ltwm")
    g.pill("LiteTokenFormer x3", 1110, 202, 220, 27, C["purple"], C["purple"], 15, C["white"])
    g.pill("K17-conditioned", 1440, 202, 170, 27, C["white"], C["orange"], 13, C["orange"])
    for i in range(3):
        y = 246 + i * 34
        g.pill(f"M{i+1}", 1112, y, 46, 22, C["purple"], C["purple"], 12, C["white"])
        g.box("", 1170, y, 210, 22, C["white"], C["purple2"], 1, 40)
        g.free_edge((1190, y + 11), (1360, y + 11), C["purple2"], 2, points=[(1235, y + 4), (1285, y + 19), (1335, y + 5)], arrow=False, curved=True)
    g.label("PREDICTED H1-H4", 1435, 245, 160, 19, 14, C["teal"], True)
    for h in range(4):
        x = 1435 + h * 43
        token_grid(g, x, 278, 3, 3, 8, 2, ["#B6DBD8", "#79C1BE", "#459F9D"], f"future_{h}")
        g.label(f"H{h+1}", x - 2, 309, 34, 16, 11, C["muted"], True)
    g.pill("K x 4 x 256 x 384", 1320, 335, 200, 24, C["white"], C["purple"], 13, C["purple"])
    g.edge("visual_tokens", "ltwm", C["blue"], 5, extra="exitX=1;exitY=.5;entryX=0;entryY=.5;")

    # C. Candidate trajectories stay local and never share the LTWM input corridor.
    section(g, "C", "CANDIDATE ACTIONS", 70, 500, C["orange"], 520)
    g.box("", 90, 580, 170, 105, C["orange_fill"], C["orange"], 3, 15, True, cid="nomad")
    g.label("NoMaD", 110, 600, 130, 31, 24, C["ink"], True)
    g.label("16 + incumbent", 105, 636, 140, 19, 15, C["muted"])
    g.pill("H4  local SE(2)", 99, 656, 152, 21, C["white"], C["orange"], 12, C["orange"])

    g.box("", 320, 615, 80, 54, C["ink"], C["white"], 3, 40, True, cid="robot")
    g.dot(346, 628, 28, C["white"], C["ink"], 3)
    for ax, ay, bx, by in [(326, 621, 306, 601), (394, 621, 414, 601), (326, 663, 306, 683), (394, 663, 414, 683)]:
        g.free_edge((ax, ay), (bx, by), C["ink"], 5, arrow=False, curved=False)
        g.dot(bx - 5, by - 5, 10, C["ink"])
    endpoints = [(535, 555), (550, 582), (560, 610), (565, 638), (560, 666), (550, 694), (535, 721)]
    colors = [C["line"], C["orange2"], C["orange"], C["red"], C["orange"], C["orange2"], C["line"]]
    for i, ((tx, ty), color) in enumerate(zip(endpoints, colors)):
        g.free_edge((400, 642), (tx, ty), color, 5 if color in (C["orange"], C["red"]) else 3, points=[(445, 642 + (i - 3) * 4), (495, 642 + (i - 3) * 18)], arrow=False, curved=True)
        g.dot(tx - 5, ty - 5, 10, color, C["white"], 1)
    g.pill("K17", 570, 590, 70, 60, C["orange"], C["orange"], 20, C["white"], cid="candidate_set")
    g.edge("nomad", "robot", C["orange"], 4, extra="exitX=1;exitY=.5;entryX=0;entryY=.5;")

    # D. Safety planner: intentionally secondary in scale, but still a clean causal chain.
    g.box("", 760, 465, 1060, 430, "#FCFDFD", C["green"], 4, 18, True, "pointerEvents=0;", cid="planner_bg")
    g.pill("V70  CENTRAL CORE", 790, 485, 170, 26, C["green"], C["green"], 13, C["white"])
    g.label("SAFE CANDIDATE PLANNER", 790, 518, 450, 28, 23, C["ink"], True, "left")
    g.label("fuse -> score -> gate -> refine", 790, 548, 390, 16, 14, C["muted"], False, "left")

    g.box("", 800, 620, 120, 135, C["white"], C["teal"], 3, 15, True, cid="feature")
    g.label("815-D", 816, 634, 88, 25, 21, C["teal"], True)
    g.label("fusion", 808, 662, 104, 15, 12, C["muted"], True)
    for i, (dim, name, color) in enumerate([("242", "traj.", C["orange"]), ("24", "phys.", C["gold"]), ("210", "DINO", C["blue"]), ("336", "LTWM", C["purple"]), ("3", "guard", C["gray"])]):
        y = 685 + i * 13
        g.pill(dim, 812, y, 36, 12, color, color, 10, C["white"])
        g.label(name, 854, y - 1, 52, 14, 10, color, True, "left")

    g.box("", 950, 600, 170, 155, C["navy"], C["teal"], 4, 16, True, cid="armd")
    g.pill("Arm-D x3", 968, 615, 104, 21, C["teal"], C["teal"], 12, C["white"])
    g.label("risk ensemble", 964, 642, 142, 20, 17, C["white"], True)
    for i in range(3):
        y = 667 + i * 23
        g.pill(f"M{i+1}", 968, y, 68, 17, C["white"], C["teal2"], 10, C["teal"])
        g.box("", 1045, y, 55, 17, C["teal"], C["teal2"], 1, 40)
    g.label("mean + std", 968, 737, 130, 14, 10, C["white"], True, "left")

    g.box("", 1150, 600, 180, 65, C["red_fill"], C["red2"], 3, 15, False, cid="risk")
    g.label("RISK", 1166, 611, 48, 20, 17, C["red"], True, "left")
    g.label("mean + uncertainty", 1166, 635, 130, 14, 11, C["red"], True, "left")
    g.free_edge((1250, 642), (1308, 642), C["red"], 2, points=[(1265, 628), (1285, 625), (1300, 640)], arrow=False, curved=True)

    g.box("", 1370, 585, 130, 145, C["green_fill"], C["green"], 3, 15, False, cid="gate_stack")
    g.pill("SAFETY GATES", 1380, 598, 110, 19, C["green"], C["green"], 11, C["white"])
    for i, name in enumerate(["physical", "support", "dual", "harm"]):
        g.pill(name, 1388, 625 + i * 24, 95, 18, C["white"], C["green"], 10, C["green"], cid=name)
    g.edge("risk", "gate_stack", C["red"], 4, points=[(1350, 632)], extra="exitX=1;exitY=.5;entryX=0;entryY=.33;")

    g.box("", 1360, 770, 150, 70, C["orange_fill"], C["orange"], 3, 16, False, cid="cem")
    g.label("CEM x3", 1372, 781, 80, 20, 18, C["orange"], True, "left")
    g.pill("16 / r", 1448, 782, 48, 15, C["white"], C["orange"], 10, C["orange"])
    for i, (cx, rx) in enumerate([(1408, 14), (1447, 11), (1480, 8)], 1):
        g.vertex("", cx-rx, 820-rx//2, rx*2, rx, f"ellipse;html=1;fillColor=#FFFFFF;strokeColor={C['orange']};strokeWidth={3 if i==3 else 2};dashed={1 if i<3 else 0};dashPattern=5 4;", f"cem_r{i}")
        for j, (dx,dy) in enumerate([(-5,-3),(4,-3),(5,3),(-5,3)]):
            g.dot(cx+dx-3, 820+dy-3, 6, C["green"] if j in (2,3) else C["line"], C["white"], 1)

    g.box("", 1530, 770, 100, 70, C["green_fill"], C["green"], 4, 16, False, cid="archive")
    g.label("union65", 1538, 781, 84, 19, 16, C["green"], True)
    g.label("safe archive", 1538, 803, 84, 13, 10, C["muted"], True)
    for group, color in enumerate([C["orange"], C["orange2"], C["teal2"], C["green"]]):
        g.dot(1542 + group * 17, 822, 7, color, C["white"], 1)

    g.box("", 1650, 770, 130, 70, C["green"], C["green"], 3, 16, True, cid="h4")
    g.label("SELECTED H4", 1658, 781, 114, 17, 13, C["white"], True)
    g.label("a_t...a_t+3", 1658, 800, 114, 13, 10, C["white"], False)
    for i in range(4):
        g.dot(1670 + i * 22, 822, 8, C["white"], C["white"], 1)

    # Separate input corridors enter the compact planner from the left.
    g.edge("ltwm", "feature", C["purple"], 5, points=[(720, 410), (720, 660)], extra="exitX=0;exitY=1;entryX=0;entryY=.3;")
    g.edge("candidate_set", "feature", C["orange"], 5, points=[(740, 690), (780, 700)], extra="exitX=.5;exitY=1;entryX=0;entryY=.6;")
    g.edge("feature", "armd", C["teal"], 5, extra="exitX=1;exitY=.5;entryX=0;entryY=.5;")
    g.edge("armd", "risk", C["red"], 5, extra="exitX=1;exitY=.35;entryX=0;entryY=.8;")
    g.edge("gate_stack", "cem", C["orange"], 5, extra="exitX=.5;exitY=1;entryX=.5;entryY=0;")
    g.edge("cem", "archive", C["green"], 5, extra="exitX=1;exitY=.5;entryX=0;entryY=.5;")
    g.edge("archive", "h4", C["green"], 5, extra="exitX=1;exitY=.5;entryX=0;entryY=.5;")

    # 6B remains outside the formal method, but moves with the smaller H4 endpoint.
    g.box("", 1900, 740, 550, 130, C["gray_fill"], C["gray"], 2, 16, False, "dashed=1;dashPattern=8 6;pointerEvents=0;", cid="loop_bg")
    g.label("6B  EXTERNAL CONTROL LOOP", 1920, 754, 280, 22, 15, C["muted"], True, "left")
    g.pill("execute a_t", 1920, 790, 120, 30, C["white"], C["gray"], 12, C["muted"], cid="execute")
    g.pill("observe o_t+1", 2080, 790, 150, 30, C["white"], C["gray"], 12, C["muted"], cid="observe")
    g.pill("replan", 2280, 790, 110, 30, C["white"], C["gray"], 12, C["muted"], cid="replan")
    g.label("external feedback only", 1920, 835, 220, 16, 12, C["muted"], False, "left")
    g.edge("h4", "execute", C["gray"], 3, dashed=True, extra="exitX=1;exitY=.5;entryX=0;entryY=.5;")
    g.edge("execute", "observe", C["gray"], 3, dashed=True, extra="exitX=1;exitY=.5;entryX=0;entryY=.5;")
    g.edge("observe", "replan", C["gray"], 3, dashed=True, extra="exitX=1;exitY=.5;entryX=0;entryY=.5;")

    g.box("", 70, 1010, 2440, 2, "#DCE2E8", "#DCE2E8", 0, 0)
    lx=80
    for color,label in [(C["blue"],"visual"),(C["orange"],"candidates"),(C["purple"],"future"),(C["teal"],"scoring"),(C["green"],"safe")]:
        g.dot(lx, 1030, 12, color)
        g.label(label, lx+18, 1023, 105, 20, 13, C["muted"], True, "left")
        lx += 135
    g.label("embedded SACSoN RGB frames", 2070, 1020, 420, 24, 13, C["muted"], False, "right")

    ET.indent(g.mx, space="  ")
    OUT.write_text(ET.tostring(g.mx, encoding="unicode", xml_declaration=True), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    build()
