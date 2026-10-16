#!/usr/bin/env python3
"""Generate the paper-level v70 method overview in draw.io, SVG, Mermaid, and PNG."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path
import math
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw, ImageFont


OUT_DIR = Path(__file__).resolve().parent
STEM = "v70_method_overview"
WIDTH = 3600
HEIGHT = 1420

COLORS = {
    "ink": "#17324A",
    "muted": "#607487",
    "line": "#9AA9B5",
    "white": "#FFFFFF",
    "blue": "#2C78A5",
    "blue_mid": "#75A9C7",
    "blue_fill": "#EAF4FA",
    "purple": "#6856A3",
    "purple_mid": "#9A8BC3",
    "purple_fill": "#F1EEFA",
    "orange": "#D97728",
    "orange_mid": "#E8A05D",
    "orange_fill": "#FFF1E4",
    "teal": "#00878A",
    "teal_dark": "#075D63",
    "teal_mid": "#61B7B4",
    "teal_fill": "#E7F6F4",
    "red": "#C75458",
    "red_fill": "#FCEBEC",
    "green": "#3F8F68",
    "green_mid": "#78B28F",
    "green_fill": "#E8F4EC",
    "gold": "#D6A126",
    "gold_fill": "#FFF6D8",
    "gray": "#8896A3",
    "gray_fill": "#F3F5F7",
    "navy": "#073F47",
}

FONT_REGULAR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


@dataclass(frozen=True)
class Stage:
    cell_id: str
    x: int
    y: int
    w: int
    h: int
    color: str
    fill: str
    label: str


class Figure:
    def __init__(self) -> None:
        self.svg_background: list[str] = []
        self.svg_edges: list[str] = []
        self.svg_nodes: list[str] = []
        self.svg_text: list[str] = []
        self.image = Image.new("RGB", (WIDTH, HEIGHT), COLORS["white"])
        self.draw = ImageDraw.Draw(self.image)
        self.mxfile = ET.Element(
            "mxfile",
            {
                "host": "drawio",
                "version": "30.2.4",
                "compressed": "false",
                "agent": "Codex drawio-skill",
            },
        )
        diagram = ET.SubElement(
            self.mxfile,
            "diagram",
            {"id": "v70-method-overview", "name": "v70 Method Overview"},
        )
        graph = ET.SubElement(
            diagram,
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
                "background": COLORS["white"],
            },
        )
        self.root = ET.SubElement(graph, "root")
        ET.SubElement(self.root, "mxCell", {"id": "0"})
        ET.SubElement(self.root, "mxCell", {"id": "1", "parent": "0"})
        self.cells: dict[str, tuple[int, int, int, int]] = {}
        self.counter = 2

    def font(self, size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
        return ImageFont.truetype(FONT_BOLD if bold else FONT_REGULAR, size)

    def next_id(self, prefix: str = "n") -> str:
        value = f"{prefix}{self.counter}"
        self.counter += 1
        return value

    def stage(self, stage: Stage) -> None:
        style = (
            "swimlane;horizontal=1;startSize=62;rounded=1;arcSize=12;"
            "container=1;collapsible=0;pointerEvents=0;whiteSpace=wrap;html=1;"
            f"fillColor={stage.fill};swimlaneFillColor={COLORS['white']};"
            f"strokeColor={stage.color};strokeWidth=2;fontColor={stage.color};"
            "fontFamily=Arial;fontSize=24;fontStyle=1;align=left;spacingLeft=20;"
        )
        cell = ET.SubElement(
            self.root,
            "mxCell",
            {
                "id": stage.cell_id,
                "value": stage.label,
                "style": style,
                "vertex": "1",
                "parent": "1",
            },
        )
        ET.SubElement(
            cell,
            "mxGeometry",
            {
                "x": str(stage.x),
                "y": str(stage.y),
                "width": str(stage.w),
                "height": str(stage.h),
                "as": "geometry",
            },
        )
        self.cells[stage.cell_id] = (stage.x, stage.y, stage.w, stage.h)
        self.svg_background.append(
            f'<rect x="{stage.x}" y="{stage.y}" width="{stage.w}" height="{stage.h}" '
            f'rx="22" fill="{stage.fill}" stroke="{stage.color}" stroke-width="2"/>'
        )
        self.svg_background.append(
            f'<path d="M {stage.x + 22} {stage.y} H {stage.x + stage.w - 22} '
            f'Q {stage.x + stage.w} {stage.y} {stage.x + stage.w} {stage.y + 22} '
            f'V {stage.y + 62} H {stage.x} V {stage.y + 22} '
            f'Q {stage.x} {stage.y} {stage.x + 22} {stage.y} Z" '
            f'fill="{COLORS["white"]}"/>'
        )
        self.svg_background.append(
            f'<line x1="{stage.x}" y1="{stage.y + 62}" x2="{stage.x + stage.w}" '
            f'y2="{stage.y + 62}" stroke="{stage.color}" stroke-width="2"/>'
        )
        self._svg_text(
            stage.label,
            stage.x + 22,
            stage.y + 38,
            24,
            stage.color,
            bold=True,
            anchor="start",
        )
        self.draw.rounded_rectangle(
            (stage.x, stage.y, stage.x + stage.w, stage.y + stage.h),
            radius=22,
            fill=stage.fill,
            outline=stage.color,
            width=2,
        )
        self.draw.rounded_rectangle(
            (stage.x, stage.y, stage.x + stage.w, stage.y + 84),
            radius=22,
            fill=COLORS["white"],
            outline=stage.color,
            width=2,
        )
        self.draw.rectangle(
            (stage.x, stage.y + 42, stage.x + stage.w, stage.y + 84),
            fill=COLORS["white"],
        )
        self.draw.line(
            (stage.x, stage.y + 62, stage.x + stage.w, stage.y + 62),
            fill=stage.color,
            width=2,
        )
        self.draw.text(
            (stage.x + 22, stage.y + 31),
            stage.label,
            font=self.font(24, True),
            fill=stage.color,
            anchor="lm",
        )

    def _absolute(
        self, parent: Stage | None, x: int, y: int
    ) -> tuple[int, int]:
        if parent is None:
            return x, y
        return parent.x + x, parent.y + y

    def box(
        self,
        cell_id: str,
        x: int,
        y: int,
        w: int,
        h: int,
        *,
        parent: Stage | None,
        fill: str,
        stroke: str,
        radius: int = 16,
        width: int = 2,
        dashed: bool = False,
        shadow: bool = False,
        title: str = "",
        title_size: int = 24,
        title_color: str | None = None,
        lines: tuple[str, ...] = (),
        line_size: int = 18,
        align: str = "center",
    ) -> None:
        px, py = self._absolute(parent, x, y)
        self.cells[cell_id] = (px, py, w, h)
        style = (
            f"rounded=1;arcSize={max(8, radius)};whiteSpace=wrap;html=1;"
            f"fillColor={fill};strokeColor={stroke};strokeWidth={width};"
            f"dashed={1 if dashed else 0};dashPattern=8 6;shadow={1 if shadow else 0};"
            f"fontFamily=Arial;fontColor={title_color or COLORS['ink']};"
            f"fontSize={title_size};fontStyle=1;align={align};verticalAlign=top;"
            "spacingTop=14;spacingLeft=12;spacingRight=12;"
        )
        value = title
        if lines:
            value += "&#xa;" + "&#xa;".join(lines)
        cell = ET.SubElement(
            self.root,
            "mxCell",
            {
                "id": cell_id,
                "value": value,
                "style": style,
                "vertex": "1",
                "parent": parent.cell_id if parent else "1",
            },
        )
        ET.SubElement(
            cell,
            "mxGeometry",
            {
                "x": str(x),
                "y": str(y),
                "width": str(w),
                "height": str(h),
                "as": "geometry",
            },
        )
        dash = ' stroke-dasharray="10 7"' if dashed else ""
        filter_value = ' filter="url(#shadow)"' if shadow else ""
        self.svg_nodes.append(
            f'<rect x="{px}" y="{py}" width="{w}" height="{h}" rx="{radius}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{width}"{dash}{filter_value}/>'
        )
        self.draw.rounded_rectangle(
            (px, py, px + w, py + h),
            radius=radius,
            fill=fill,
            outline=stroke,
            width=width,
        )
        title_y = py + 32
        text_anchor = "middle" if align == "center" else "start"
        text_x = px + w / 2 if align == "center" else px + 18
        if title:
            self._svg_text(
                title,
                text_x,
                title_y,
                title_size,
                title_color or COLORS["ink"],
                bold=True,
                anchor=text_anchor,
            )
            self.draw.text(
                (text_x, title_y),
                title,
                font=self.font(title_size, True),
                fill=title_color or COLORS["ink"],
                anchor="mm" if align == "center" else "lm",
            )
        if lines:
            line_y = py + 66
            for line in lines:
                self._svg_text(
                    line,
                    text_x,
                    line_y,
                    line_size,
                    COLORS["muted"],
                    anchor=text_anchor,
                )
                self.draw.text(
                    (text_x, line_y),
                    line,
                    font=self.font(line_size),
                    fill=COLORS["muted"],
                    anchor="mm" if align == "center" else "lm",
                )
                line_y += line_size + 12

    def pill(
        self,
        cell_id: str,
        x: int,
        y: int,
        w: int,
        h: int,
        text: str,
        *,
        parent: Stage | None,
        fill: str,
        stroke: str,
        color: str,
        size: int = 16,
        bold: bool = True,
    ) -> None:
        px, py = self._absolute(parent, x, y)
        self.cells[cell_id] = (px, py, w, h)
        style = (
            "rounded=1;arcSize=50;whiteSpace=wrap;html=1;"
            f"fillColor={fill};strokeColor={stroke};strokeWidth=1;"
            f"fontFamily=Arial;fontColor={color};fontSize={size};"
            f"fontStyle={1 if bold else 0};align=center;verticalAlign=middle;"
        )
        cell = ET.SubElement(
            self.root,
            "mxCell",
            {
                "id": cell_id,
                "value": text,
                "style": style,
                "vertex": "1",
                "parent": parent.cell_id if parent else "1",
            },
        )
        ET.SubElement(
            cell,
            "mxGeometry",
            {
                "x": str(x),
                "y": str(y),
                "width": str(w),
                "height": str(h),
                "as": "geometry",
            },
        )
        self.svg_nodes.append(
            f'<rect x="{px}" y="{py}" width="{w}" height="{h}" rx="{h / 2}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="1"/>'
        )
        self._svg_text(text, px + w / 2, py + h / 2 + 1, size, color, bold, "middle")
        self.draw.rounded_rectangle(
            (px, py, px + w, py + h),
            radius=h // 2,
            fill=fill,
            outline=stroke,
            width=1,
        )
        self.draw.text(
            (px + w / 2, py + h / 2),
            text,
            font=self.font(size, bold),
            fill=color,
            anchor="mm",
        )

    def label(
        self,
        cell_id: str,
        x: int,
        y: int,
        w: int,
        h: int,
        text: str,
        *,
        parent: Stage | None = None,
        size: int = 18,
        color: str | None = None,
        bold: bool = False,
        align: str = "left",
    ) -> None:
        px, py = self._absolute(parent, x, y)
        self.cells[cell_id] = (px, py, w, h)
        style = (
            "text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
            f"fontFamily=Arial;fontSize={size};fontColor={color or COLORS['ink']};"
            f"fontStyle={1 if bold else 0};align={align};verticalAlign=middle;"
        )
        cell = ET.SubElement(
            self.root,
            "mxCell",
            {
                "id": cell_id,
                "value": text,
                "style": style,
                "vertex": "1",
                "parent": parent.cell_id if parent else "1",
            },
        )
        ET.SubElement(
            cell,
            "mxGeometry",
            {
                "x": str(x),
                "y": str(y),
                "width": str(w),
                "height": str(h),
                "as": "geometry",
            },
        )
        anchor = "start" if align == "left" else "middle"
        tx = px if align == "left" else px + w / 2
        ty = py + h / 2
        self._svg_text(text, tx, ty, size, color or COLORS["ink"], bold, anchor)
        self.draw.text(
            (tx, ty),
            text,
            font=self.font(size, bold),
            fill=color or COLORS["ink"],
            anchor="lm" if align == "left" else "mm",
        )

    def token_grid(
        self,
        x: int,
        y: int,
        *,
        parent: Stage,
        rows: int,
        cols: int,
        cell: int,
        gap: int,
        color: str,
    ) -> None:
        px, py = self._absolute(parent, x, y)
        for row in range(rows):
            for col in range(cols):
                cx = px + col * (cell + gap)
                cy = py + row * (cell + gap)
                opacity = 0.50 + 0.45 * (row + col + 2) / (rows + cols)
                self.svg_nodes.append(
                    f'<rect x="{cx}" y="{cy}" width="{cell}" height="{cell}" rx="2" '
                    f'fill="{color}" opacity="{opacity:.2f}"/>'
                )
                self.draw.rounded_rectangle(
                    (cx, cy, cx + cell, cy + cell),
                    radius=2,
                    fill=color,
                )

    def edge(
        self,
        cell_id: str,
        source: str,
        target: str,
        *,
        color: str,
        width: int = 4,
        dashed: bool = False,
        points: tuple[tuple[int, int], ...] = (),
        label: str = "",
        label_x: int | None = None,
        label_y: int | None = None,
    ) -> None:
        sx, sy, sw, sh = self.cells[source]
        tx, ty, tw, th = self.cells[target]
        start = (sx + sw, sy + sh / 2)
        end = (tx, ty + th / 2)
        if abs((ty + th / 2) - (sy + sh / 2)) > abs(tx - (sx + sw)):
            start = (sx + sw / 2, sy + sh)
            end = (tx + tw / 2, ty)
        path_points = (start, *points, end)
        style = (
            "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;"
            "jettySize=auto;html=1;endArrow=blockThin;endFill=1;endSize=11;"
            f"strokeColor={color};strokeWidth={width};"
            f"dashed={1 if dashed else 0};dashPattern=9 7;"
            "labelBackgroundColor=#FFFFFF;fontFamily=Arial;fontSize=15;"
            f"fontColor={color};"
        )
        edge = ET.SubElement(
            self.root,
            "mxCell",
            {
                "id": cell_id,
                "value": label,
                "style": style,
                "edge": "1",
                "parent": "1",
                "source": source,
                "target": target,
            },
        )
        geometry = ET.SubElement(
            edge,
            "mxGeometry",
            {"relative": "1", "as": "geometry"},
        )
        if points:
            array = ET.SubElement(geometry, "Array", {"as": "points"})
            for point_x, point_y in points:
                ET.SubElement(
                    array,
                    "mxPoint",
                    {"x": str(point_x), "y": str(point_y)},
                )
        dash = ' stroke-dasharray="11 8"' if dashed else ""
        point_string = " ".join(f"{px},{py}" for px, py in path_points)
        self.svg_edges.append(
            f'<polyline points="{point_string}" fill="none" stroke="{color}" '
            f'stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round"'
            f'{dash} marker-end="url(#arrow-{self._color_key(color)})"/>'
        )
        self.draw.line(path_points, fill=color, width=width, joint="curve")
        self._pil_arrow(path_points[-2], path_points[-1], color, 14)
        if label and label_x is not None and label_y is not None:
            self.svg_text.append(
                f'<rect x="{label_x - 8}" y="{label_y - 16}" '
                f'width="{max(80, len(label) * 9 + 16)}" height="28" rx="8" '
                f'fill="{COLORS["white"]}" opacity="0.94"/>'
            )
            self._svg_text(label, label_x, label_y, 15, color, True, "start")
            self.draw.rounded_rectangle(
                (
                    label_x - 8,
                    label_y - 16,
                    label_x + max(72, len(label) * 9 + 8),
                    label_y + 12,
                ),
                radius=8,
                fill=COLORS["white"],
            )
            self.draw.text(
                (label_x, label_y),
                label,
                font=self.font(15, True),
                fill=color,
                anchor="lm",
            )

    def _color_key(self, color: str) -> str:
        return color.replace("#", "").lower()

    def _pil_arrow(
        self,
        previous: tuple[float, float],
        end: tuple[float, float],
        color: str,
        size: int,
    ) -> None:
        angle = math.atan2(end[1] - previous[1], end[0] - previous[0])
        left = (
            end[0] - size * math.cos(angle - math.pi / 6),
            end[1] - size * math.sin(angle - math.pi / 6),
        )
        right = (
            end[0] - size * math.cos(angle + math.pi / 6),
            end[1] - size * math.sin(angle + math.pi / 6),
        )
        self.draw.polygon((end, left, right), fill=color)

    def _svg_text(
        self,
        text: str,
        x: float,
        y: float,
        size: int,
        color: str,
        bold: bool = False,
        anchor: str = "middle",
    ) -> None:
        self.svg_text.append(
            f'<text x="{x}" y="{y}" text-anchor="{anchor}" '
            f'dominant-baseline="middle" font-family="Arial, Helvetica, sans-serif" '
            f'font-size="{size}" font-weight="{700 if bold else 400}" '
            f'fill="{color}">{escape(text)}</text>'
        )

    def write(self) -> None:
        ET.indent(self.mxfile, space="  ")
        drawio_path = OUT_DIR / f"{STEM}.drawio"
        drawio_path.write_text(
            ET.tostring(
                self.mxfile,
                encoding="unicode",
                xml_declaration=True,
            ),
            encoding="utf-8",
        )
        marker_colors = sorted(
            {
                COLORS["blue"],
                COLORS["purple"],
                COLORS["orange"],
                COLORS["teal"],
                COLORS["red"],
                COLORS["green"],
                COLORS["gray"],
            }
        )
        markers = "\n".join(
            f'<marker id="arrow-{self._color_key(color)}" markerWidth="10" '
            f'markerHeight="10" refX="9" refY="3" orient="auto" '
            f'markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" '
            f'fill="{color}"/></marker>'
            for color in marker_colors
        )
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" '
            f'height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">\n'
            "<defs>\n"
            '<filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">'
            '<feDropShadow dx="0" dy="5" stdDeviation="6" '
            'flood-color="#17324A" flood-opacity="0.12"/></filter>\n'
            f"{markers}\n"
            "</defs>\n"
            f"{''.join(self.svg_background)}\n"
            f"{''.join(self.svg_edges)}\n"
            f"{''.join(self.svg_nodes)}\n"
            f"{''.join(self.svg_text)}\n"
            "</svg>\n"
        )
        (OUT_DIR / f"{STEM}.svg").write_text(svg, encoding="utf-8")
        self.image.save(OUT_DIR / f"{STEM}.png", optimize=True)


def build() -> None:
    fig = Figure()

    stage1 = Stage(
        "stage1",
        40,
        150,
        1020,
        700,
        COLORS["blue"],
        "#F7FBFD",
        "STAGE 1  ·  VISUAL REPRESENTATION & FROZEN WORLD MODEL",
    )
    stage2 = Stage(
        "stage2",
        390,
        900,
        670,
        360,
        COLORS["orange"],
        "#FFFAF5",
        "STAGE 2  ·  CANDIDATE ACTION GENERATION",
    )
    stage3 = Stage(
        "stage3",
        1100,
        150,
        990,
        1110,
        COLORS["teal"],
        "#F7FCFB",
        "STAGE 3  ·  UNIVERSAL ARM-D CANDIDATE EVALUATION",
    )
    stage4 = Stage(
        "stage4",
        2130,
        150,
        800,
        1110,
        COLORS["red"],
        "#FFF9F9",
        "STAGE 4  ·  UNCERTAINTY-AWARE SAFETY PLANNING",
    )
    stage5 = Stage(
        "stage5",
        2970,
        150,
        590,
        1110,
        COLORS["green"],
        "#F8FCF9",
        "STAGE 5  ·  ARCHIVE REPLAY",
    )

    fig.label(
        "title",
        40,
        24,
        2250,
        52,
        "v70 Method Overview: Universal Candidate Scoring for Safe Visual Planning",
        size=35,
        color=COLORS["ink"],
        bold=True,
    )
    fig.label(
        "subtitle",
        40,
        78,
        2400,
        34,
        "Frozen visual world model  ·  single universal Arm-D  ·  uncertainty-aware CEM  ·  cross-round safe archive",
        size=19,
        color=COLORS["muted"],
    )
    legend = [
        ("Perception / world model", COLORS["blue"]),
        ("Candidate generation", COLORS["orange"]),
        ("Universal scoring", COLORS["teal"]),
        ("Safety planning", COLORS["red"]),
        ("Archive replay", COLORS["green"]),
    ]
    lx = 2530
    for index, (name, color) in enumerate(legend):
        x = lx + (index % 3) * 330
        y = 40 + (index // 3) * 40
        fig.pill(
            f"legend_{index}",
            x,
            y,
            26,
            26,
            "",
            parent=None,
            fill=color,
            stroke=color,
            color=color,
        )
        fig.label(
            f"legend_label_{index}",
            x + 36,
            y - 2,
            280,
            30,
            name,
            size=15,
            color=COLORS["muted"],
        )

    for stage in (stage1, stage2, stage3, stage4, stage5):
        fig.stage(stage)

    # Stage 1: observations, frozen DINOv3, adapter, and three-member LTWM.
    fig.box(
        "observations",
        30,
        105,
        220,
        455,
        parent=stage1,
        fill=COLORS["white"],
        stroke=COLORS["blue_mid"],
        shadow=True,
        title="Visual inputs",
        title_size=24,
        lines=("4-frame history", "t−3, t−2, t−1, t", "+ goal image"),
        line_size=18,
    )
    for frame in range(4):
        fx = stage1.x + 66 + (frame % 2) * 82
        fy = stage1.y + 298 + (frame // 2) * 92
        fig.svg_nodes.append(
            f'<rect x="{fx}" y="{fy}" width="68" height="68" rx="8" '
            f'fill="{COLORS["blue_fill"]}" stroke="{COLORS["blue"]}" stroke-width="2"/>'
        )
        fig.draw.rounded_rectangle(
            (fx, fy, fx + 68, fy + 68),
            radius=8,
            fill=COLORS["blue_fill"],
            outline=COLORS["blue"],
            width=2,
        )
        fig._svg_text(f"t−{3 - frame}", fx + 34, fy + 35, 15, COLORS["blue"], True)
        fig.draw.text(
            (fx + 34, fy + 34),
            f"t−{3 - frame}",
            font=fig.font(15, True),
            fill=COLORS["blue"],
            anchor="mm",
        )
    gx = stage1.x + 100
    gy = stage1.y + 505
    fig.svg_nodes.append(
        f'<rect x="{gx}" y="{gy}" width="90" height="68" rx="8" '
        f'fill="{COLORS["gold_fill"]}" stroke="{COLORS["gold"]}" stroke-width="2"/>'
    )
    fig.draw.rounded_rectangle(
        (gx, gy, gx + 90, gy + 68),
        radius=8,
        fill=COLORS["gold_fill"],
        outline=COLORS["gold"],
        width=2,
    )
    fig._svg_text("goal", gx + 45, gy + 35, 16, COLORS["gold"], True)
    fig.draw.text(
        (gx + 45, gy + 34),
        "goal",
        font=fig.font(16, True),
        fill=COLORS["gold"],
        anchor="mm",
    )

    fig.box(
        "dino",
        300,
        150,
        220,
        245,
        parent=stage1,
        fill=COLORS["blue_fill"],
        stroke=COLORS["blue"],
        shadow=True,
        title="DINOv3 ViT-L/16",
        title_size=24,
        lines=("patch representation", "frozen backbone"),
        line_size=17,
    )
    fig.pill(
        "frozen_badge",
        345,
        335,
        130,
        36,
        "FROZEN",
        parent=stage1,
        fill=COLORS["white"],
        stroke=COLORS["blue"],
        color=COLORS["blue"],
        size=15,
    )
    fig.token_grid(
        350,
        270,
        parent=stage1,
        rows=4,
        cols=5,
        cell=14,
        gap=7,
        color=COLORS["blue"],
    )
    fig.box(
        "adapter",
        570,
        190,
        180,
        165,
        parent=stage1,
        fill="#F0F7FB",
        stroke=COLORS["blue"],
        title="Adapter",
        title_size=24,
        lines=("1024 → 384", "14×14 → 16×16"),
        line_size=18,
    )
    fig.box(
        "ltwm",
        790,
        115,
        200,
        330,
        parent=stage1,
        fill=COLORS["purple_fill"],
        stroke=COLORS["purple"],
        width=3,
        shadow=True,
        title="LiteTokenFormer",
        title_size=23,
        title_color=COLORS["purple"],
        lines=("World Model ×3", "action-conditioned", "future latent rollout"),
        line_size=17,
    )
    for member in range(3):
        fig.pill(
            f"wm_member_{member}",
            822,
            300 + member * 40,
            136,
            28,
            f"member {member + 1}",
            parent=stage1,
            fill=COLORS["white"],
            stroke=COLORS["purple_mid"],
            color=COLORS["purple"],
            size=14,
        )
    fig.box(
        "rollout",
        745,
        490,
        245,
        150,
        parent=stage1,
        fill=COLORS["white"],
        stroke=COLORS["purple_mid"],
        title="Future latent statistics",
        title_size=20,
        lines=("H1  ·  H2  ·  H3  ·  H4", "mean / std / min / max"),
        line_size=16,
    )
    fig.edge("e_obs_dino", "observations", "dino", color=COLORS["blue"], width=4)
    fig.edge("e_dino_adapter", "dino", "adapter", color=COLORS["blue"], width=4)
    fig.edge("e_adapter_wm", "adapter", "ltwm", color=COLORS["purple"], width=4)
    fig.edge(
        "e_wm_rollout",
        "ltwm",
        "rollout",
        color=COLORS["purple"],
        width=4,
    )

    # Stage 2: incumbent plus deterministic NoMaD proposals.
    fig.box(
        "incumbent",
        35,
        105,
        190,
        80,
        parent=stage2,
        fill=COLORS["white"],
        stroke=COLORS["gray"],
        title="Incumbent",
        title_size=21,
        lines=("trajectory 0",),
        line_size=15,
    )
    fig.box(
        "nomad",
        35,
        215,
        190,
        85,
        parent=stage2,
        fill=COLORS["orange_fill"],
        stroke=COLORS["orange"],
        title="NoMaD",
        title_size=22,
        lines=("16 proposals",),
        line_size=16,
    )
    fig.box(
        "k17",
        290,
        115,
        335,
        175,
        parent=stage2,
        fill=COLORS["white"],
        stroke=COLORS["orange"],
        width=3,
        shadow=True,
        title="K = 17 candidate trajectories",
        title_size=22,
        lines=("H = 4 waypoints per candidate", "(dx, dy, dψ) local SE(2)"),
        line_size=17,
    )
    fig.pill(
        "k17_shape",
        370,
        250,
        180,
        30,
        "[17 × 4 × 3]",
        parent=stage2,
        fill=COLORS["orange"],
        stroke=COLORS["orange"],
        color=COLORS["white"],
        size=15,
    )
    fig.edge("e_inc_k17", "incumbent", "k17", color=COLORS["orange"], width=3)
    fig.edge("e_nomad_k17", "nomad", "k17", color=COLORS["orange"], width=3)
    fig.edge(
        "e_k17_wm",
        "k17",
        "ltwm",
        color=COLORS["purple"],
        width=4,
        dashed=True,
        points=((1030, 930), (1030, 715), (960, 715)),
        label="condition each aₖ",
        label_x=870,
        label_y=885,
    )

    # Stage 3: exact 815-D contract and one universal scorer.
    fig.box(
        "features",
        35,
        105,
        420,
        840,
        parent=stage3,
        fill=COLORS["white"],
        stroke=COLORS["teal_mid"],
        shadow=True,
        title="Candidate feature construction",
        title_size=24,
        title_color=COLORS["teal_dark"],
    )
    fig.pill(
        "dim815",
        150,
        165,
        190,
        46,
        "xₖ ∈ ℝ⁸¹⁵",
        parent=stage3,
        fill=COLORS["teal"],
        stroke=COLORS["teal"],
        color=COLORS["white"],
        size=22,
    )
    feature_rows = [
        ("242", "trajectory + action / incumbent Δ", COLORS["orange_mid"]),
        ("12", "physical H4 actions", "#D7A64F"),
        ("12", "spacing-normalized H4", "#E0BC77"),
        ("210", "DINOv3 history / goal", COLORS["blue_mid"]),
        ("112", "raw LTWM rollout statistics", COLORS["purple_mid"]),
        ("112", "candidate − incumbent LTWM", "#6AACA9"),
        ("112", "set-relative LTWM statistics", COLORS["green_mid"]),
        ("3", "spacing / support / physical guard", COLORS["gray"]),
    ]
    row_y = 235
    for index, (dim, name, color) in enumerate(feature_rows):
        fig.pill(
            f"feature_dim_{index}",
            55,
            row_y,
            64,
            43,
            dim,
            parent=stage3,
            fill=color,
            stroke=color,
            color=COLORS["white"],
            size=16,
        )
        fig.label(
            f"feature_name_{index}",
            132,
            row_y,
            275,
            43,
            name,
            parent=stage3,
            size=15,
            color=COLORS["ink"],
        )
        row_y += 67
    fig.pill(
        "feature_contract",
        58,
        795,
        345,
        42,
        "deployment-safe inputs only",
        parent=stage3,
        fill=COLORS["teal_fill"],
        stroke=COLORS["teal"],
        color=COLORS["teal_dark"],
        size=15,
    )
    fig.label(
        "feature_exclusion",
        62,
        850,
        340,
        58,
        "Scorer-v2 auxiliary predictions are excluded from the 815-D contract.",
        parent=stage3,
        size=14,
        color=COLORS["muted"],
    )

    fig.box(
        "armd",
        510,
        120,
        430,
        340,
        parent=stage3,
        fill=COLORS["navy"],
        stroke=COLORS["teal"],
        width=4,
        shadow=True,
        title="Universal Arm-D",
        title_size=31,
        title_color=COLORS["white"],
    )
    fig.pill(
        "single_scorer",
        590,
        180,
        270,
        38,
        "ONE UNIVERSAL SCORER",
        parent=stage3,
        fill=COLORS["teal"],
        stroke=COLORS["teal"],
        color=COLORS["white"],
        size=15,
    )
    armd_layers = [
        ("shared_backbone", 555, 245, 340, 52, "Shared scoring backbone"),
        ("set_context", 555, 315, 340, 52, "Setwise candidate context"),
        ("quality_unc", 555, 385, 340, 52, "Quality + uncertainty prediction"),
    ]
    for layer_id, x, y, w, h, name in armd_layers:
        fig.pill(
            layer_id,
            x,
            y,
            w,
            h,
            name,
            parent=stage3,
            fill=COLORS["white"],
            stroke=COLORS["teal_mid"],
            color=COLORS["teal_dark"],
            size=16,
        )
    fig.pill(
        "scorer_ensemble",
        755,
        475,
        170,
        34,
        "3-seed ensemble",
        parent=stage3,
        fill=COLORS["teal_fill"],
        stroke=COLORS["teal"],
        color=COLORS["teal_dark"],
        size=14,
    )
    fig.edge("e_features_armd", "features", "armd", color=COLORS["teal"], width=5)

    fig.box(
        "metric_outputs",
        510,
        520,
        430,
        235,
        parent=stage3,
        fill=COLORS["white"],
        stroke=COLORS["teal_mid"],
        title="Candidate outcome prediction",
        title_size=22,
        lines=(
            "ΔATE  ·  ΔRPE  ·  ΔEndpoint  ·  ΔObjective",
            "mean μ + total uncertainty σ",
            "p(dual improvement)  ·  p(harm)  ·  rank",
        ),
        line_size=16,
    )
    fig.pill(
        "unc_type",
        585,
        700,
        280,
        34,
        "aleatoric + epistemic",
        parent=stage3,
        fill=COLORS["purple_fill"],
        stroke=COLORS["purple"],
        color=COLORS["purple"],
        size=14,
    )
    fig.edge(
        "e_armd_outputs",
        "armd",
        "metric_outputs",
        color=COLORS["teal"],
        width=4,
    )
    fig.box(
        "calibration",
        510,
        800,
        430,
        185,
        parent=stage3,
        fill=COLORS["gold_fill"],
        stroke=COLORS["gold"],
        width=2,
        title="Safety-aware calibration layer",
        title_size=22,
        lines=(
            "temperature / uncertainty calibration",
            "false-safe reduction  ·  risk-aware decision",
        ),
        line_size=15,
    )
    fig.edge(
        "e_outputs_cal",
        "metric_outputs",
        "calibration",
        color=COLORS["gold"],
        width=3,
    )
    fig.edge(
        "e_k17_features",
        "k17",
        "features",
        color=COLORS["orange"],
        width=4,
        points=((1080, 1080), (1080, 610)),
        label="trajectory + rollout features",
        label_x=1070,
        label_y=880,
    )
    fig.edge(
        "e_rollout_features",
        "rollout",
        "features",
        color=COLORS["purple"],
        width=4,
        points=((1080, 715), (1080, 560)),
    )

    # Stage 4: frozen UCB gate followed by safe-elite CEM.
    fig.box(
        "safety_gate",
        40,
        105,
        720,
        390,
        parent=stage4,
        fill=COLORS["red_fill"],
        stroke=COLORS["red"],
        width=3,
        shadow=True,
        title="Uncertainty safety gate",
        title_size=28,
        title_color=COLORS["red"],
    )
    fig.pill(
        "ucb_formula",
        165,
        175,
        470,
        70,
        "UCB = μ + z · σ",
        parent=stage4,
        fill=COLORS["white"],
        stroke=COLORS["red"],
        color=COLORS["red"],
        size=27,
    )
    fig.label(
        "gate_rule_1",
        85,
        275,
        630,
        38,
        "support-valid  ∧  UCBATE < −mATE  ∧  UCBObjective < −mObjective",
        parent=stage4,
        size=16,
        color=COLORS["ink"],
        bold=True,
        align="center",
    )
    fig.label(
        "gate_rule_2",
        90,
        325,
        620,
        38,
        "pdual ≥ τdual   ∧   pharm ≤ τharm",
        parent=stage4,
        size=19,
        color=COLORS["ink"],
        bold=True,
        align="center",
    )
    fig.pill(
        "gate_result",
        225,
        405,
        350,
        44,
        "safe candidate set only",
        parent=stage4,
        fill=COLORS["red"],
        stroke=COLORS["red"],
        color=COLORS["white"],
        size=17,
    )
    fig.edge(
        "e_cal_gate",
        "calibration",
        "safety_gate",
        color=COLORS["red"],
        width=5,
        label="calibrated μ, σ, probabilities",
        label_x=2040,
        label_y=990,
    )

    fig.box(
        "cem",
        40,
        555,
        720,
        485,
        parent=stage4,
        fill=COLORS["orange_fill"],
        stroke=COLORS["orange"],
        width=3,
        shadow=True,
        title="Safety-aware CEM optimization",
        title_size=27,
        title_color=COLORS["orange"],
    )
    fig.pill(
        "cem_contract",
        165,
        615,
        470,
        42,
        "3 iterations  ·  16 challengers / iteration",
        parent=stage4,
        fill=COLORS["white"],
        stroke=COLORS["orange"],
        color=COLORS["orange"],
        size=16,
    )
    round_positions = [(160, 790, 128), (400, 790, 96), (610, 790, 68)]
    for index, (center_x, center_y, radius) in enumerate(round_positions, start=1):
        px = stage4.x + center_x
        py = stage4.y + center_y
        fig.svg_nodes.append(
            f'<ellipse cx="{px}" cy="{py}" rx="{radius / 2}" ry="{radius}" '
            f'fill="{COLORS["white"]}" stroke="{COLORS["orange"]}" '
            f'stroke-width="{3 if index == 3 else 2}" '
            f'stroke-dasharray="{"0" if index == 3 else "9 6"}"/>'
        )
        fig.draw.ellipse(
            (
                px - radius / 2,
                py - radius,
                px + radius / 2,
                py + radius,
            ),
            fill=COLORS["white"],
            outline=COLORS["orange"],
            width=3 if index == 3 else 2,
        )
        fig._svg_text(f"R{index}", px, py - radius - 24, 17, COLORS["orange"], True)
        fig.draw.text(
            (px, py - radius - 24),
            f"R{index}",
            font=fig.font(17, True),
            fill=COLORS["orange"],
            anchor="mm",
        )
        cloud_scale = [1.0, 0.72, 0.48][index - 1]
        for point_index, (dx, dy) in enumerate(
            ((-35, -70), (18, -86), (35, -38), (-28, -15), (15, 8), (-34, 42), (12, 76))
        ):
            cx = px + dx * cloud_scale
            cy = py + dy * cloud_scale
            dot_color = COLORS["green"] if point_index in (3, 4, 6) else COLORS["line"]
            fig.svg_nodes.append(
                f'<circle cx="{cx}" cy="{cy}" r="8" fill="{dot_color}" '
                f'stroke="{COLORS["white"]}" stroke-width="2"/>'
            )
            fig.draw.ellipse(
                (cx - 8, cy - 8, cx + 8, cy + 8),
                fill=dot_color,
                outline=COLORS["white"],
                width=2,
            )
    fig.label(
        "cem_steps",
        75,
        940,
        650,
        34,
        "safe elite selection (4 / 16)  →  mean & covariance update",
        parent=stage4,
        size=17,
        color=COLORS["orange"],
        bold=True,
        align="center",
    )
    fig.pill(
        "cem_shortfall",
        190,
        990,
        420,
        34,
        "safe shortfall → retain previous proposal",
        parent=stage4,
        fill=COLORS["white"],
        stroke=COLORS["gray"],
        color=COLORS["gray"],
        size=14,
    )
    fig.edge(
        "e_gate_cem",
        "safety_gate",
        "cem",
        color=COLORS["orange"],
        width=5,
    )

    # Stage 5: retain the best safe trajectory across the fixed pool and all CEM rounds.
    fig.pill(
        "v70_badge",
        390,
        82,
        145,
        40,
        "v70 NEW",
        parent=stage5,
        fill=COLORS["gold"],
        stroke=COLORS["gold"],
        color=COLORS["white"],
        size=16,
    )
    fig.box(
        "archive",
        35,
        130,
        520,
        670,
        parent=stage5,
        fill=COLORS["green_fill"],
        stroke=COLORS["green"],
        width=4,
        shadow=True,
        title="Cross-round safe archive",
        title_size=27,
        title_color=COLORS["green"],
    )
    fig.pill(
        "archive_rule",
        105,
        195,
        380,
        42,
        "running best safe candidate",
        parent=stage5,
        fill=COLORS["white"],
        stroke=COLORS["green"],
        color=COLORS["green"],
        size=16,
    )
    archive_rows = [
        ("archive_fixed", 85, 285, "fixed K17", "best safe"),
        ("archive_r1", 85, 390, "CEM round 1", "safe candidate"),
        ("archive_r2", 85, 495, "CEM round 2", "safe candidate"),
        ("archive_r3", 85, 600, "CEM round 3", "safe candidate"),
    ]
    for row_id, x, y, name, note in archive_rows:
        fig.box(
            row_id,
            x,
            y,
            350,
            78,
            parent=stage5,
            fill=COLORS["white"],
            stroke=COLORS["green_mid"],
            title=name,
            title_size=18,
            lines=(note,),
            line_size=14,
            align="left",
        )
        fig.pill(
            f"{row_id}_check",
            455,
            y + 17,
            42,
            42,
            "✓",
            parent=stage5,
            fill=COLORS["green"],
            stroke=COLORS["green"],
            color=COLORS["white"],
            size=20,
        )
    fig.pill(
        "archive_trace",
        105,
        715,
        380,
        38,
        "store source round + candidate index",
        parent=stage5,
        fill=COLORS["white"],
        stroke=COLORS["gray"],
        color=COLORS["muted"],
        size=14,
    )
    fig.edge(
        "e_cem_archive",
        "cem",
        "archive",
        color=COLORS["green"],
        width=6,
        label="safe elites across rounds",
        label_x=2870,
        label_y=870,
    )

    fig.box(
        "decision",
        35,
        850,
        520,
        205,
        parent=stage5,
        fill=COLORS["white"],
        stroke=COLORS["green"],
        width=3,
        title="Final action",
        title_size=25,
        title_color=COLORS["green"],
    )
    fig.pill(
        "safe_h4",
        80,
        920,
        410,
        48,
        "best safe H4 trajectory",
        parent=stage5,
        fill=COLORS["green"],
        stroke=COLORS["green"],
        color=COLORS["white"],
        size=18,
    )
    fig.pill(
        "fallback",
        80,
        990,
        410,
        42,
        "no archived safe candidate → incumbent",
        parent=stage5,
        fill=COLORS["gray_fill"],
        stroke=COLORS["gray"],
        color=COLORS["gray"],
        size=14,
    )
    fig.edge(
        "e_archive_decision",
        "archive",
        "decision",
        color=COLORS["green"],
        width=5,
    )

    fig.box(
        "method_note",
        40,
        1300,
        3520,
        72,
        parent=None,
        fill=COLORS["gray_fill"],
        stroke="#D7DEE4",
        radius=14,
        title="Single shared inference path",
        title_size=18,
        title_color=COLORS["ink"],
        lines=(
            "Frozen DINOv3 + LTWM predicts candidate futures; one universal Arm-D scores every candidate; calibration changes risk decisions, not model identity.",
        ),
        line_size=15,
    )

    fig.write()
    write_mermaid()


def write_mermaid() -> None:
    mermaid = """%% v70 paper method overview
flowchart LR
  subgraph S1["Stage 1 · Visual representation & frozen world model"]
    O["4-frame history + goal image"] --> D["Frozen DINOv3 ViT-L/16"]
    D --> A["Adapter<br/>1024→384 · 14×14→16×16"]
    A --> W["LiteTokenFormer World Model ×3<br/>action-conditioned H1–H4 latent rollout"]
    W --> WS["Future latent statistics<br/>mean · std · min · max"]
  end

  subgraph S2["Stage 2 · Candidate action generation"]
    I["Incumbent trajectory"] --> K["K=17 candidate trajectories<br/>H=4 · (dx,dy,dψ)"]
    N["NoMaD ×16"] --> K
  end

  K -. "condition each candidate" .-> W

  subgraph S3["Stage 3 · Universal Arm-D candidate evaluation"]
    F["815-D candidate feature<br/>242 trajectory/action + 24 physical/normalized<br/>210 DINOv3 + 112×3 LTWM blocks + 3 support scalars"]
    U["Universal Arm-D<br/>shared backbone → setwise context → quality + uncertainty"]
    P["ΔATE · ΔRPE · ΔEndpoint · ΔObjective<br/>μ, σ, p(dual), p(harm), rank"]
    C["Safety-aware calibration layer<br/>uncertainty calibration · false-safe reduction"]
    F --> U --> P --> C
  end

  K --> F
  WS --> F

  subgraph S4["Stage 4 · Uncertainty-aware safety planning"]
    G["Safety gate<br/>UCB = μ + z·σ<br/>support + UCB + dual/harm constraints"]
    E["Safety-aware CEM<br/>3 iterations · 16 challengers · elite=4<br/>mean/covariance update"]
    G --> E
  end
  C --> G

  subgraph S5["Stage 5 · Archive replay"]
    R["Cross-round safe archive<br/>fixed K17 + CEM rounds 1–3<br/>running best safe candidate"]
    B{"Archived safe candidate?"}
    H["Best safe H4 trajectory"]
    X["Fallback incumbent"]
    R --> B
    B -->|yes| H
    B -.->|no| X
  end
  E -->|"safe elites across rounds"| R

  classDef perception fill:#EAF4FA,stroke:#2C78A5,color:#17324A,stroke-width:2px;
  classDef world fill:#F1EEFA,stroke:#6856A3,color:#17324A,stroke-width:2px;
  classDef candidate fill:#FFF1E4,stroke:#D97728,color:#17324A,stroke-width:2px;
  classDef scoring fill:#E7F6F4,stroke:#00878A,color:#17324A,stroke-width:2px;
  classDef planning fill:#FCEBEC,stroke:#C75458,color:#17324A,stroke-width:2px;
  classDef archive fill:#E8F4EC,stroke:#3F8F68,color:#17324A,stroke-width:3px;
  class O,D,A perception;
  class W,WS world;
  class I,N,K candidate;
  class F,U,P,C scoring;
  class G,E planning;
  class R,B,H,X archive;
"""
    (OUT_DIR / f"{STEM}.mmd").write_text(mermaid, encoding="utf-8")


if __name__ == "__main__":
    build()
