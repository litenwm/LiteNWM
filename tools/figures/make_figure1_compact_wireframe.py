#!/usr/bin/env python3
"""Build the compact black-and-white Figure 1 wireframe.

The figure is deliberately limited to inference-time information flow:

    visual context -> frozen encoder -> action-conditioned LTWM rollout
    -> candidate-wise scoring -> selection / optional CEM refinement

It writes an editable draw.io source plus SVG, PDF, and PNG previews.
"""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.path import Path as MplPath


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "figures" / "figure1_compact"
BASE = OUT_DIR / "figure1_compact_wireframe"

INK = "#202226"
MUTED = "#666A70"
LINE = "#7B7F86"
LIGHT_LINE = "#B9BCC1"
PANEL = "#F7F7F6"
SUBPANEL = "#ECEDEC"
WHITE = "#FFFFFF"


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "font.size": 6.2,
        "figure.facecolor": WHITE,
        "savefig.facecolor": WHITE,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)


def rounded_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    face: str = WHITE,
    edge: str = LINE,
    lw: float = 0.9,
    radius: float = 1.2,
    z: float = 2,
    linestyle: str = "-",
):
    box = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.02,rounding_size={radius}",
        facecolor=face,
        edgecolor=edge,
        linewidth=lw,
        linestyle=linestyle,
        zorder=z,
    )
    ax.add_patch(box)
    return box


def text(
    ax,
    x: float,
    y: float,
    value: str,
    *,
    size: float = 5.6,
    weight: str = "normal",
    color: str = INK,
    ha: str = "center",
    va: str = "center",
    z: float = 7,
):
    ax.text(
        x,
        y,
        value,
        fontsize=size,
        fontweight=weight,
        color=color,
        ha=ha,
        va=va,
        linespacing=1.12,
        zorder=z,
    )


def arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = INK,
    lw: float = 1.15,
    dashed: bool = False,
    z: float = 8,
):
    patch = patches.FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=8,
        linewidth=lw,
        color=color,
        linestyle=(0, (3, 2)) if dashed else "-",
        shrinkA=0,
        shrinkB=0,
        connectionstyle="arc3,rad=0",
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def poly_arrow(
    ax,
    points: list[tuple[float, float]],
    *,
    color: str = LINE,
    lw: float = 0.85,
    dashed: bool = False,
    z: float = 5,
):
    path = MplPath(
        points,
        [MplPath.MOVETO] + [MplPath.LINETO] * (len(points) - 1),
    )
    patch = patches.FancyArrowPatch(
        path=path,
        arrowstyle="-|>",
        mutation_scale=7,
        linewidth=lw,
        color=color,
        linestyle=(0, (3, 2)) if dashed else "-",
        shrinkA=0,
        shrinkB=0,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def draw_frame(ax, x: float, y: float, w: float, h: float, label: str):
    rounded_box(ax, x, y, w, h, face=WHITE, edge=LINE, lw=0.55, radius=0.35, z=4)
    ax.add_patch(
        patches.Polygon(
            [
                (x + 0.25, y + 0.25),
                (x + 0.40 * w, y + 0.55 * h),
                (x + 0.58 * w, y + 0.34 * h),
                (x + w - 0.25, y + 0.67 * h),
                (x + w - 0.25, y + 0.25),
            ],
            closed=True,
            facecolor=SUBPANEL,
            edgecolor="none",
            zorder=4.2,
        )
    )
    ax.plot(
        [x + 0.48 * w, x + 0.52 * w],
        [y + 0.25, y + h - 0.25],
        color=LIGHT_LINE,
        linewidth=0.5,
        zorder=4.4,
    )
    text(ax, x + w / 2, y - 0.75, label, size=3.8, color=MUTED)


def draw_token_grid(ax, x: float, y: float, w: float, h: float, label: str):
    rounded_box(ax, x, y, w, h, face=WHITE, edge=LINE, lw=0.55, radius=0.35, z=4)
    for index in range(1, 4):
        ax.plot(
            [x + w * index / 4, x + w * index / 4],
            [y + 0.18, y + h - 0.18],
            color=LIGHT_LINE,
            linewidth=0.25,
            zorder=4.2,
        )
        ax.plot(
            [x + 0.18, x + w - 0.18],
            [y + h * index / 4, y + h * index / 4],
            color=LIGHT_LINE,
            linewidth=0.25,
            zorder=4.2,
        )
    text(ax, x + w / 2, y - 0.7, label, size=3.7, color=MUTED)


def draw_candidate_scene(ax, x: float, y: float, w: float, h: float):
    rounded_box(ax, x, y, w, h, face=WHITE, edge=LIGHT_LINE, lw=0.6, radius=0.8, z=3)
    ax.plot(
        [x + 0.46 * w, x + 0.55 * w],
        [y + 1.0, y + h - 1.0],
        color=LIGHT_LINE,
        linewidth=0.6,
        linestyle=(0, (3, 2)),
        zorder=3.2,
    )
    start = (x + 0.50 * w, y + 1.2)
    endpoints = (0.23, 0.37, 0.50, 0.63, 0.77)
    for index, fraction in enumerate(endpoints):
        vertices = [
            start,
            (x + w * (0.48 + 0.025 * (index - 2)), y + 0.32 * h),
            (x + w * (0.44 + 0.07 * (index - 2)), y + 0.66 * h),
            (x + w * fraction, y + h - 1.2),
        ]
        path = MplPath(
            vertices,
            [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4],
        )
        ax.add_patch(
            patches.PathPatch(
                path,
                facecolor="none",
                edgecolor=INK if index == 2 else LINE,
                linewidth=1.15 if index == 2 else 0.55,
                alpha=1.0 if index == 2 else 0.72,
                zorder=5,
            )
        )
    ax.add_patch(patches.Circle(start, radius=0.6, facecolor=INK, edgecolor=INK, zorder=6))


def draw_wireframe() -> None:
    fig = plt.figure(figsize=(7.4, 4.8))
    ax = fig.add_axes([0.02, 0.025, 0.96, 0.94])
    ax.set_xlim(0, 150)
    ax.set_ylim(0, 100)
    ax.axis("off")

    text(
        ax,
        4,
        97,
        "Compact overall framework — structural wireframe",
        size=9.0,
        weight="bold",
        ha="left",
        va="top",
    )
    text(
        ax,
        146,
        96.3,
        "inference-time information flow only",
        size=4.7,
        color=MUTED,
        ha="right",
        va="top",
    )

    # A. Visual context.
    rounded_box(ax, 4, 62, 43, 29, face=PANEL, edge=LINE, lw=0.9, radius=1.4)
    text(ax, 7, 88.2, "A  VISUAL CONTEXT", size=5.4, weight="bold", ha="left")
    for index, x in enumerate((7.5, 15.5, 23.5, 31.5)):
        draw_frame(ax, x, 72.5, 9.5, 9.5, f"t{index - 3:+d}" if index < 3 else "t")
    draw_frame(ax, 30.5, 64.2, 11.0, 6.5, "goal")
    text(ax, 8, 66.8, "history", size=4.3, color=MUTED, ha="left")

    # B. Frozen perception.
    rounded_box(ax, 54, 68, 34, 18, face=WHITE, edge=INK, lw=1.0, radius=1.2)
    text(ax, 71, 81.6, "FROZEN VISUAL ENCODER", size=5.3, weight="bold")
    rounded_box(ax, 57, 72.5, 15, 6.2, face=SUBPANEL, edge=LIGHT_LINE, lw=0.55, radius=0.7)
    rounded_box(ax, 74, 72.5, 11, 6.2, face=SUBPANEL, edge=LIGHT_LINE, lw=0.55, radius=0.7)
    text(ax, 64.5, 75.6, "DINOv3-L/16", size=4.6, weight="bold")
    text(ax, 79.5, 75.6, "Adapter", size=4.6, weight="bold")
    text(ax, 71, 69.9, "1024 → 384  ·  256 patch tokens", size=4.0, color=MUTED)
    text(ax, 84.5, 83.6, "LOCKED", size=3.7, weight="bold", color=MUTED)

    # C. Candidate action pool.
    rounded_box(ax, 4, 20, 37, 31, face=PANEL, edge=LINE, lw=0.9, radius=1.4)
    text(ax, 7, 48.1, "B  CANDIDATE ACTIONS", size=5.4, weight="bold", ha="left")
    draw_candidate_scene(ax, 8, 25.2, 29, 18.0)
    text(ax, 22.5, 22.6, "K trajectories  ·  H4 actions [dx, dy, dψ]", size=4.0, color=MUTED)

    # D. Central LTWM hero module.
    rounded_box(ax, 49, 30, 55, 31, face=WHITE, edge=INK, lw=1.55, radius=1.7)
    text(ax, 52, 57.8, "C  LITETOKENFORMER WORLD MODEL", size=6.1, weight="bold", ha="left")
    text(ax, 101, 57.8, "action-conditioned", size=4.1, color=MUTED, ha="right")
    rounded_box(ax, 53, 39.5, 14, 12, face=PANEL, edge=LIGHT_LINE, lw=0.6, radius=0.7)
    rounded_box(ax, 70, 39.5, 15, 12, face=PANEL, edge=LIGHT_LINE, lw=0.6, radius=0.7)
    text(ax, 60, 45.5, "History\nencoder", size=4.6, weight="bold")
    text(ax, 77.5, 45.5, "Factorized\nrollout", size=4.6, weight="bold")
    arrow(ax, (67, 45.5), (70, 45.5), color=LINE, lw=0.75)
    for index, (x, y) in enumerate(((88, 45.5), (94, 45.5), (88, 37), (94, 37)), start=1):
        draw_token_grid(ax, x, y, 4.5, 4.5, f"H{index}")
    text(ax, 76.5, 33.2, "predicted future latents  [K × 4 × 256 × 384]", size=4.2, color=MUTED)

    # E. Candidate-wise quality estimation and selection.
    rounded_box(ax, 111, 23, 35, 43, face=PANEL, edge=INK, lw=1.0, radius=1.4)
    text(ax, 114, 62.8, "D  SCORING & SELECTION", size=5.4, weight="bold", ha="left")
    text(ax, 128.5, 58.3, "Trajectory-quality Scorer v2", size=5.0, weight="bold")
    rounded_box(ax, 114, 46.0, 14, 8.5, face=WHITE, edge=LIGHT_LINE, lw=0.6, radius=0.7)
    rounded_box(ax, 131, 46.0, 11.5, 8.5, face=WHITE, edge=LIGHT_LINE, lw=0.6, radius=0.7)
    text(ax, 121, 50.3, "per-candidate\nfeatures", size=4.2)
    text(ax, 136.75, 50.3, "shared\nMLP", size=4.2, weight="bold")
    arrow(ax, (128, 50.3), (131, 50.3), color=LINE, lw=0.7)
    text(ax, 117, 42.7, "ATE ↓", size=4.6, weight="bold", ha="left")
    text(ax, 130, 42.7, "RTP ↑", size=4.6, weight="bold", ha="left")
    for index, width in enumerate((20, 14, 9)):
        y = 38.5 - 3.2 * index
        ax.add_patch(
            patches.Rectangle(
                (116, y),
                width,
                1.2,
                facecolor=INK if index == 0 else LIGHT_LINE,
                edgecolor="none",
                zorder=5,
            )
        )
    text(ax, 128.5, 27.3, "selected trajectory", size=4.5, weight="bold")
    arrow(ax, (120.5, 29.8), (136.5, 29.8), color=INK, lw=1.5)
    rounded_box(
        ax,
        119,
        24.3,
        19,
        4.7,
        face=WHITE,
        edge=LINE,
        lw=0.6,
        radius=2.3,
        linestyle="--",
    )
    text(ax, 128.5, 26.65, "optional CEM", size=3.9, color=MUTED)

    # Main information flow.
    arrow(ax, (47, 77), (54, 77), color=INK, lw=1.2)
    poly_arrow(ax, [(71, 68), (71, 64.5), (61, 64.5), (61, 61)], color=INK, lw=1.1)
    text(ax, 64, 66.1, "history tokens", size=3.8, color=MUTED)
    arrow(ax, (41, 40), (49, 40), color=INK, lw=1.1)
    text(ax, 45, 42.3, "actions", size=3.8, color=MUTED)
    arrow(ax, (104, 45.5), (111, 45.5), color=INK, lw=1.25)
    text(ax, 107.5, 48.2, "H1–H4", size=3.8, color=MUTED)

    # Auxiliary scorer inputs. They are thin and grey so the LTWM path remains dominant.
    poly_arrow(
        ax,
        [(88, 75), (106.5, 75), (106.5, 57), (111, 57)],
        color=LINE,
        lw=0.75,
    )
    text(ax, 101.5, 77.0, "current + goal latents", size=3.8, color=MUTED)
    poly_arrow(
        ax,
        [(41, 28), (106.5, 28), (106.5, 34), (111, 34)],
        color=LINE,
        lw=0.75,
    )
    text(ax, 74, 26.4, "candidate actions", size=3.8, color=MUTED)

    text(
        ax,
        4,
        8.5,
        "Frozen perception  ·  candidate-conditioned latent rollout  ·  independent absolute ATE/RTP prediction",
        size=4.4,
        color=MUTED,
        ha="left",
    )
    text(
        ax,
        146,
        8.5,
        "wireframe — visual styling intentionally deferred",
        size=4.2,
        color=MUTED,
        ha="right",
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(BASE.with_suffix(".png"), dpi=260, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(BASE.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.05)
    fig.savefig(BASE.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


class DrawioBuilder:
    def __init__(self) -> None:
        self.mxfile = ET.Element("mxfile", host="drawio", version="26.0.0")
        self.diagram = ET.SubElement(self.mxfile, "diagram", name="Compact Wireframe")
        self.model = ET.SubElement(
            self.diagram,
            "mxGraphModel",
            dx="1460",
            dy="960",
            grid="1",
            gridSize="10",
            guides="1",
            tooltips="1",
            connect="1",
            arrows="1",
            fold="1",
            page="1",
            pageScale="1",
            pageWidth="1500",
            pageHeight="1000",
            background="#FFFFFF",
        )
        self.root = ET.SubElement(self.model, "root")
        ET.SubElement(self.root, "mxCell", id="0")
        ET.SubElement(self.root, "mxCell", id="1", parent="0")

    def vertex(
        self,
        cell_id: str,
        value: str,
        x: int,
        y: int,
        width: int,
        height: int,
        *,
        parent: str = "1",
        style: str,
    ) -> None:
        cell = ET.SubElement(
            self.root,
            "mxCell",
            id=cell_id,
            value=value,
            style=style,
            vertex="1",
            parent=parent,
        )
        ET.SubElement(
            cell,
            "mxGeometry",
            x=str(x),
            y=str(y),
            width=str(width),
            height=str(height),
            **{"as": "geometry"},
        )

    def edge(
        self,
        cell_id: str,
        source: str,
        target: str,
        *,
        value: str = "",
        style: str,
        points: tuple[tuple[int, int], ...] = (),
    ) -> None:
        cell = ET.SubElement(
            self.root,
            "mxCell",
            id=cell_id,
            value=value,
            style=style,
            edge="1",
            parent="1",
            source=source,
            target=target,
        )
        geometry = ET.SubElement(cell, "mxGeometry", relative="1", **{"as": "geometry"})
        if points:
            array = ET.SubElement(geometry, "Array", **{"as": "points"})
            for x, y in points:
                ET.SubElement(array, "mxPoint", x=str(x), y=str(y))

    def write(self, path: Path) -> None:
        ET.indent(self.mxfile, space="  ")
        ET.ElementTree(self.mxfile).write(path, encoding="UTF-8", xml_declaration=True)


def build_drawio() -> None:
    builder = DrawioBuilder()
    panel = (
        "rounded=1;arcSize=10;whiteSpace=wrap;html=1;container=1;"
        "pointerEvents=0;fillColor=#F7F7F6;strokeColor=#7B7F86;"
        "fontColor=#202226;fontFamily=Arial;fontSize=14;fontStyle=1;"
        "verticalAlign=top;align=left;spacingTop=10;spacingLeft=12;"
    )
    hero = (
        "rounded=1;arcSize=10;whiteSpace=wrap;html=1;container=1;"
        "pointerEvents=0;fillColor=#FFFFFF;strokeColor=#202226;strokeWidth=2;"
        "fontColor=#202226;fontFamily=Arial;fontSize=16;fontStyle=1;"
        "verticalAlign=top;align=left;spacingTop=10;spacingLeft=12;"
    )
    module = (
        "rounded=1;arcSize=12;whiteSpace=wrap;html=1;fillColor=#ECEDEC;"
        "strokeColor=#B9BCC1;fontColor=#202226;fontFamily=Arial;fontSize=13;"
    )
    white_module = module.replace("fillColor=#ECEDEC", "fillColor=#FFFFFF")
    small = (
        "rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#FFFFFF;"
        "strokeColor=#B9BCC1;fontColor=#202226;fontFamily=Arial;fontSize=11;"
    )
    text_style = (
        "text;html=1;strokeColor=none;fillColor=none;align=center;"
        "verticalAlign=middle;fontFamily=Arial;fontColor=#666A70;fontSize=11;"
    )
    edge_main = (
        "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;"
        "jettySize=auto;html=1;strokeColor=#202226;strokeWidth=2;"
        "endArrow=block;endFill=1;fontFamily=Arial;fontSize=11;"
        "labelBackgroundColor=#FFFFFF;"
    )
    edge_aux = edge_main.replace("strokeColor=#202226;strokeWidth=2", "strokeColor=#7B7F86;strokeWidth=1")

    builder.vertex(
        "title",
        "<b>Compact overall framework</b><br><font color=\"#666A70\">structural wireframe · inference-time flow only</font>",
        40,
        20,
        700,
        60,
        style="text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;fontFamily=Arial;fontColor=#202226;fontSize=18;",
    )
    builder.vertex("context", "A  VISUAL CONTEXT", 40, 100, 410, 280, style=panel)
    for index, x in enumerate((25, 110, 195, 280)):
        label = ("t−3", "t−2", "t−1", "t")[index]
        builder.vertex(f"frame{index}", label, x, 80, 70, 90, parent="context", style=small)
    builder.vertex("goal", "goal image", 260, 190, 100, 60, parent="context", style=small)

    builder.vertex("encoder", "FROZEN VISUAL ENCODER", 520, 140, 320, 180, style=hero)
    builder.vertex("dino", "<b>DINOv3-L/16</b>", 30, 70, 140, 60, parent="encoder", style=module)
    builder.vertex("adapter", "<b>Adapter</b><br>1024 → 384", 190, 70, 100, 60, parent="encoder", style=module)

    builder.vertex("candidates", "B  CANDIDATE ACTIONS", 40, 520, 360, 280, style=panel)
    builder.vertex(
        "actionpool",
        "<b>K candidate trajectories</b><br>H4 actions [dx, dy, dψ]",
        30,
        70,
        300,
        150,
        parent="candidates",
        style=white_module,
    )

    builder.vertex("ltwm", "C  LITETOKENFORMER WORLD MODEL", 480, 420, 520, 290, style=hero)
    builder.vertex("history", "<b>History encoder</b>", 35, 100, 135, 80, parent="ltwm", style=module)
    builder.vertex("rollout", "<b>Factorized rollout</b><br>action-conditioned", 200, 100, 145, 80, parent="ltwm", style=module)
    for index, (x, y) in enumerate(((380, 90), (440, 90), (380, 160), (440, 160)), start=1):
        builder.vertex(f"h{index}", f"H{index}", x, y, 45, 45, parent="ltwm", style=small)
    builder.vertex(
        "futurelabel",
        "future latent tokens  [K × 4 × 256 × 384]",
        170,
        225,
        310,
        30,
        parent="ltwm",
        style=text_style,
    )

    builder.vertex("scorer", "D  SCORING &amp; SELECTION", 1080, 360, 350, 410, style=panel)
    builder.vertex(
        "features",
        "<b>Per-candidate features</b><br>future + route + action",
        25,
        85,
        150,
        80,
        parent="scorer",
        style=white_module,
    )
    builder.vertex(
        "mlp",
        "<b>Shared MLP</b><br>Scorer v2",
        200,
        85,
        120,
        80,
        parent="scorer",
        style=white_module,
    )
    builder.vertex(
        "metrics",
        "<b>ATE ↓    RTP ↑</b><br><br>candidate ranking",
        50,
        195,
        250,
        90,
        parent="scorer",
        style=white_module,
    )
    builder.vertex(
        "selected",
        "<b>Selected trajectory</b><br><font color=\"#666A70\">optional CEM refinement</font>",
        65,
        315,
        220,
        60,
        parent="scorer",
        style=small,
    )

    builder.edge("e1", "context", "encoder", style=edge_main)
    builder.edge(
        "e2",
        "encoder",
        "history",
        value="history tokens",
        style=edge_main,
        points=((680, 370), (600, 370)),
    )
    builder.edge("e3", "actionpool", "rollout", value="candidate actions", style=edge_main)
    builder.edge("e4", "rollout", "h1", style=edge_main)
    builder.edge("e5", "ltwm", "features", value="H1–H4", style=edge_main)
    builder.edge(
        "e6",
        "encoder",
        "features",
        value="current + goal latents",
        style=edge_aux,
        points=((1030, 230), (1030, 460)),
    )
    builder.edge(
        "e7",
        "actionpool",
        "features",
        value="actions",
        style=edge_aux,
        points=((1030, 730), (1030, 540)),
    )
    builder.edge("e8", "features", "mlp", style=edge_main)
    builder.edge("e9", "mlp", "metrics", style=edge_main)
    builder.edge("e10", "metrics", "selected", style=edge_main)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    builder.write(BASE.with_suffix(".drawio"))


def main() -> None:
    build_drawio()
    draw_wireframe()
    print(BASE.with_suffix(".drawio"))
    print(BASE.with_suffix(".png"))
    print(BASE.with_suffix(".svg"))
    print(BASE.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
