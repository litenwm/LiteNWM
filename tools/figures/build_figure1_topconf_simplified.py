#!/usr/bin/env python3
"""Build a compact top-conference overview of the DINOv3-LTWM planner.

The figure intentionally abstracts implementation detail while preserving the
runtime contract implemented in this repository:

* four history frames and one goal frame are encoded by a frozen DINOv3-L/16
  backbone plus a frozen 1024->384 adapter;
* LiteTokenFormer consumes history tokens and K candidate H4 action sequences;
* each candidate produces four dense future latent maps;
* scorer-v2 forms 192-D route, 796-D rollout, and 40-D action features;
* a shared candidate-independent MLP is ensembled over seeds 1701/1702/1703;
* conservative joint ATE/RTP gains determine promotion, with incumbent fallback.

Outputs are self-contained SVG, PNG, PDF, and editable Draw.io XML.
"""

from __future__ import annotations

import base64
from html import escape
from pathlib import Path
import xml.etree.ElementTree as ET

import cairosvg


ROOT = Path(__file__).resolve().parents[2]
ASSET_DIR = ROOT / "figures" / "figure1_compact" / "assets"
OUT_DIR = ROOT / "figures" / "figure1_topconf_simplified"
BASE = OUT_DIR / "figure1_topconf_simplified"

OUT_SVG = BASE.with_suffix(".svg")
OUT_PNG = BASE.with_suffix(".png")
OUT_PDF = BASE.with_suffix(".pdf")
OUT_DRAWIO = BASE.with_suffix(".drawio")

W, H = 1200.0, 570.0
FONT = "Arial, Helvetica, DejaVu Sans, sans-serif"

INK = "#18212B"
MUTED = "#667180"
HAIR = "#DDE3EA"
PANEL = "#F9FAFC"
WHITE = "#FFFFFF"

BLUE = "#315E9A"
BLUE_MID = "#7698C4"
BLUE_SOFT = "#EDF3FA"

ORANGE = "#A96A35"
ORANGE_MID = "#D3A477"
ORANGE_SOFT = "#FCF3E9"

PURPLE = "#745075"
PURPLE_MID = "#B68EB1"
PURPLE_SOFT = "#F7F0F6"

GREEN = "#2F7D60"
GREEN_MID = "#81B59F"
GREEN_SOFT = "#EAF5F0"

ASSETS = {
    "t-3": ASSET_DIR / "history_t-3.jpg",
    "t-2": ASSET_DIR / "history_t-2.jpg",
    "t-1": ASSET_DIR / "history_t-1.jpg",
    "t": ASSET_DIR / "history_t.jpg",
    "goal": ASSET_DIR / "goal.jpg",
}


def data_uri(path: Path) -> str:
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def rect(
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    fill: str = WHITE,
    stroke: str = HAIR,
    sw: float = 1.0,
    rx: float = 8.0,
    dash: str | None = None,
) -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<rect x="{x:g}" y="{y:g}" width="{width:g}" height="{height:g}" '
        f'rx="{rx:g}" fill="{fill}" stroke="{stroke}" stroke-width="{sw:g}"'
        f"{dash_attr} />"
    )


def line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    color: str = HAIR,
    sw: float = 1.0,
    dash: str | None = None,
) -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1:g}" y1="{y1:g}" x2="{x2:g}" y2="{y2:g}" '
        f'stroke="{color}" stroke-width="{sw:g}" stroke-linecap="round"'
        f"{dash_attr} />"
    )


def path(
    d: str,
    *,
    color: str = HAIR,
    sw: float = 1.0,
    fill: str = "none",
    dash: str | None = None,
    marker: str | None = None,
) -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    marker_attr = f' marker-end="url(#{marker})"' if marker else ""
    return (
        f'<path d="{d}" fill="{fill}" stroke="{color}" stroke-width="{sw:g}" '
        'stroke-linecap="round" stroke-linejoin="round"'
        f"{dash_attr}{marker_attr} />"
    )


def text(
    x: float,
    y: float,
    value: str,
    *,
    size: float = 12.0,
    color: str = INK,
    weight: int = 400,
    anchor: str = "start",
    letter: float | None = None,
    italic: bool = False,
) -> str:
    spacing = f"letter-spacing:{letter:g}px;" if letter is not None else ""
    style = "font-style:italic;" if italic else ""
    return (
        f'<text x="{x:g}" y="{y:g}" text-anchor="{anchor}" '
        f'style="font-family:{FONT};font-size:{size:g}px;font-weight:{weight};'
        f'fill:{color};{spacing}{style}">{escape(value)}</text>'
    )


def arrow_path(
    d: str,
    *,
    color: str,
    sw: float = 2.2,
    dash: str | None = None,
) -> str:
    marker = {
        BLUE: "arrow_blue",
        ORANGE: "arrow_orange",
        PURPLE: "arrow_purple",
        GREEN: "arrow_green",
        MUTED: "arrow_muted",
    }.get(color, "arrow_ink")
    return path(d, color=color, sw=sw, dash=dash, marker=marker)


def image_tile(
    tile_id: str,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    stroke: str,
    sw: float = 1.0,
) -> str:
    clip_id = f"clip_{tile_id.replace('-', '_')}"
    uri = data_uri(ASSETS[tile_id])
    return "\n".join(
        (
            f'<clipPath id="{clip_id}"><rect x="{x:g}" y="{y:g}" '
            f'width="{width:g}" height="{height:g}" rx="5" /></clipPath>',
            f'<image x="{x:g}" y="{y:g}" width="{width:g}" height="{height:g}" '
            f'preserveAspectRatio="xMidYMid slice" clip-path="url(#{clip_id})" '
            f'href="{uri}" />',
            rect(x, y, width, height, fill="none", stroke=stroke, sw=sw, rx=5),
        )
    )


def token_grid(
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    label: str,
    stroke: str = BLUE_MID,
    emphasized: bool = False,
) -> str:
    parts = [
        rect(
            x,
            y,
            width,
            height,
            fill=WHITE,
            stroke=BLUE if emphasized else stroke,
            sw=1.6 if emphasized else 1.0,
            rx=4,
        )
    ]
    for index in range(1, 4):
        parts.append(
            line(
                x + width * index / 4,
                y + 4,
                x + width * index / 4,
                y + height - 4,
                color="#C9D6E5",
                sw=0.65,
            )
        )
        parts.append(
            line(
                x + 4,
                y + height * index / 4,
                x + width - 4,
                y + height * index / 4,
                color="#C9D6E5",
                sw=0.65,
            )
        )
    parts.append(
        text(
            x + width / 2,
            y + height + 17,
            label,
            size=10,
            color=BLUE if emphasized else MUTED,
            weight=700 if emphasized else 500,
            anchor="middle",
        )
    )
    return "\n".join(parts)


def trajectory_fan(
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    selected: bool = False,
) -> str:
    start_x = x + width * 0.5
    start_y = y + height - 4
    parts = [
        line(
            start_x,
            start_y,
            start_x,
            y + 6,
            color="#BBC3CD",
            sw=1.0,
            dash="4 4",
        )
    ]
    offsets = (-0.34, -0.18, 0.0, 0.18, 0.34)
    for index, offset in enumerate(offsets):
        end_x = start_x + width * offset
        mid_x = start_x + width * offset * 0.42
        is_selected = selected and index == 2
        color = GREEN if is_selected else ORANGE_MID
        sw = 4.0 if is_selected else (2.4 if index == 2 else 1.5)
        parts.append(
            path(
                f"M {start_x:g} {start_y:g} "
                f"C {start_x + width * offset * 0.08:g} {y + height * 0.66:g}, "
                f"{mid_x:g} {y + height * 0.34:g}, {end_x:g} {y + 6:g}",
                color=color,
                sw=sw,
            )
        )
    parts.append(
        f'<circle cx="{start_x:g}" cy="{start_y:g}" r="4.5" '
        f'fill="{GREEN if selected else ORANGE}" />'
    )
    return "\n".join(parts)


def stage_header(
    x: float,
    y: float,
    number: str,
    title: str,
    *,
    color: str,
) -> str:
    return "\n".join(
        (
            f'<circle cx="{x + 10:g}" cy="{y - 5:g}" r="10" fill="{color}" />',
            text(
                x + 10,
                y - 1.2,
                number,
                size=10,
                color=WHITE,
                weight=700,
                anchor="middle",
            ),
            text(x + 29, y, title, size=15, color=color, weight=700),
        )
    )


def build_svg() -> str:
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W:g} {H:g}" '
        f'width="{W:g}pt" height="{H:g}pt">',
        "<defs>",
    ]
    for name, color in (
        ("arrow_ink", INK),
        ("arrow_muted", MUTED),
        ("arrow_blue", BLUE),
        ("arrow_orange", ORANGE),
        ("arrow_purple", PURPLE),
        ("arrow_green", GREEN),
    ):
        parts.extend(
            (
                f'<marker id="{name}" markerWidth="8" markerHeight="8" '
                'refX="7" refY="4" orient="auto" markerUnits="strokeWidth">',
                f'<path d="M0,0 L8,4 L0,8 z" fill="{color}" />',
                "</marker>",
            )
        )
    parts.extend(
        (
            "</defs>",
            rect(0, 0, W, H, fill=WHITE, stroke="none", rx=0),
            text(
                24,
                30,
                "DINOv3–LTWM: candidate-conditioned latent trajectory planning",
                size=22,
                color=INK,
                weight=700,
            ),
            text(
                24,
                53,
                "Frozen visual tokens  →  action-conditioned latent futures  →  uncertainty-aware H4 selection",
                size=11,
                color=MUTED,
                weight=500,
            ),
            text(
                1176,
                31,
                "INFERENCE-TIME OVERVIEW",
                size=9,
                color=BLUE,
                weight=700,
                anchor="end",
                letter=0.8,
            ),
            line(24, 67, 1176, 67, color=HAIR, sw=1.0),
        )
    )

    # Stage containers.
    parts.extend(
        (
            rect(24, 94, 292, 402, fill=PANEL, stroke="#CCD5E0", sw=1.1, rx=12),
            rect(340, 94, 316, 402, fill=BLUE_SOFT, stroke=BLUE_MID, sw=1.25, rx=12),
            rect(680, 94, 272, 402, fill=PURPLE_SOFT, stroke=PURPLE_MID, sw=1.25, rx=12),
            rect(976, 94, 200, 402, fill=GREEN_SOFT, stroke=GREEN_MID, sw=1.25, rx=12),
            stage_header(40, 118, "1", "Condition", color=BLUE),
            stage_header(356, 118, "2", "Predict latent futures", color=BLUE),
            stage_header(696, 118, "3", "Score each candidate", color=PURPLE),
            stage_header(992, 118, "4", "Select", color=GREEN),
        )
    )

    # Stage 1: observations + tokenizer.
    parts.extend(
        (
            text(48, 151, "VISUAL CONTEXT", size=9.5, color=BLUE, weight=700, letter=0.5),
            text(48, 170, "history", size=10, color=MUTED),
        )
    )
    frame_x = (92, 138, 184, 230)
    for tile_id, x in zip(("t-3", "t-2", "t-1", "t"), frame_x):
        parts.append(
            image_tile(
                tile_id,
                x,
                146,
                44,
                39,
                stroke=BLUE if tile_id == "t" else "#AEBBCB",
                sw=1.5 if tile_id == "t" else 0.9,
            )
        )
        parts.append(
            text(
                x + 22,
                197,
                tile_id.replace("-", "−"),
                size=8.5,
                color=BLUE if tile_id == "t" else MUTED,
                weight=700 if tile_id == "t" else 400,
                anchor="middle",
            )
        )
    parts.extend(
        (
            text(48, 230, "goal", size=10, color=MUTED),
            image_tile("goal", 92, 210, 70, 43, stroke=GREEN, sw=1.4),
            text(174, 236, "t + 4", size=8.5, color=GREEN, weight=700),
            rect(48, 270, 244, 66, fill=WHITE, stroke=BLUE_MID, sw=1.0, rx=8),
            text(62, 293, "FROZEN TOKENIZER", size=10.5, color=BLUE, weight=700),
            text(62, 316, "DINOv3-L/16", size=12.0, color=INK, weight=700),
            text(158, 316, "+", size=12, color=MUTED, weight=700, anchor="middle"),
            text(176, 316, "Adapter", size=12.0, color=INK, weight=700),
            text(278, 316, "1024→384", size=8.5, color=MUTED, weight=600, anchor="end"),
            arrow_path("M 170 253 V 270", color=BLUE, sw=1.8),
            text(
                170,
                354,
                "zₜ₋₃:ₜ , zgoal  ∈  ℝ²⁵⁶ˣ³⁸⁴",
                size=10.5,
                color=BLUE,
                weight=600,
                anchor="middle",
            ),
            line(48, 372, 292, 372, color=HAIR, sw=0.9),
            text(48, 395, "CANDIDATE ACTIONS", size=9.5, color=ORANGE, weight=700, letter=0.5),
            trajectory_fan(54, 404, 102, 69),
            text(180, 424, "K trajectories", size=12, color=INK, weight=700),
            text(180, 444, "H4 actions  [dx, dy, dψ]", size=9.5, color=MUTED),
            text(180, 463, "external proposal policy", size=8.5, color=ORANGE, weight=600),
        )
    )

    # Stage 2: LiteTokenFormer.
    parts.extend(
        (
            text(364, 155, "LiteTokenFormer world model", size=18, color=INK, weight=700),
            text(
                632,
                155,
                "candidate-conditioned",
                size=9,
                color=BLUE,
                weight=700,
                anchor="end",
            ),
            rect(364, 179, 268, 126, fill=WHITE, stroke=BLUE, sw=1.4, rx=10),
            rect(382, 202, 92, 52, fill=BLUE_SOFT, stroke=BLUE_MID, sw=1.0, rx=7),
            text(428, 224, "History", size=11.5, color=INK, weight=700, anchor="middle"),
            text(428, 241, "encoder", size=10, color=MUTED, anchor="middle"),
            rect(486, 202, 92, 52, fill=ORANGE_SOFT, stroke=ORANGE_MID, sw=1.0, rx=7),
            text(532, 224, "Action", size=11.5, color=INK, weight=700, anchor="middle"),
            text(532, 241, "prefix", size=10, color=MUTED, anchor="middle"),
            arrow_path("M 474 228 H 486", color=MUTED, sw=1.5),
            rect(482, 270, 100, 27, fill=BLUE, stroke=BLUE, sw=1.0, rx=13),
            text(532, 288, "factorized rollout", size=10, color=WHITE, weight=700, anchor="middle"),
            arrow_path("M 428 254 V 283 H 482", color=BLUE, sw=1.6),
            arrow_path("M 532 254 V 270", color=ORANGE, sw=1.6),
            text(364, 337, "one-pass H1–H4 latent rollout for every candidate", size=10.5, color=MUTED),
        )
    )
    grid_specs = (
        (378, "H1", False),
        (438, "H2", False),
        (498, "H3", False),
        (558, "H4", True),
    )
    for x, label, emphasized in grid_specs:
        parts.append(token_grid(x, 362, 46, 46, label=label, emphasized=emphasized))
    parts.extend(
        (
            text(
                498,
                447,
                "ẑ₁:₄⁽ᵏ⁾  ∈  ℝᴷˣ⁴ˣ²⁵⁶ˣ³⁸⁴",
                size=12,
                color=BLUE,
                weight=700,
                anchor="middle",
            ),
            text(
                498,
                472,
                "history encoder · action prefix · factorized token rollout",
                size=8.8,
                color=MUTED,
                anchor="middle",
            ),
        )
    )

    # Stage 3: compact candidate representation and one shared scorer ensemble.
    parts.extend(
        (
            text(704, 153, "Candidate representation", size=14.5, color=INK, weight=700),
            rect(704, 176, 224, 92, fill=WHITE, stroke=PURPLE_MID, sw=1.0, rx=9),
            text(816, 198, "φₖ  =  [ route  |  rollout  |  action ]", size=12, color=INK, weight=700, anchor="middle"),
            rect(720, 215, 58, 30, fill=BLUE_SOFT, stroke=BLUE_MID, sw=0.8, rx=6),
            rect(787, 215, 64, 30, fill="#EEF1FA", stroke=BLUE, sw=0.8, rx=6),
            rect(860, 215, 52, 30, fill=ORANGE_SOFT, stroke=ORANGE_MID, sw=0.8, rx=6),
            text(749, 234, "192-D", size=9.5, color=BLUE, weight=700, anchor="middle"),
            text(819, 234, "796-D", size=9.5, color=BLUE, weight=700, anchor="middle"),
            text(886, 234, "40-D", size=9.5, color=ORANGE, weight=700, anchor="middle"),
            text(816, 259, "1028-D · frozen z-score", size=8.8, color=MUTED, weight=600, anchor="middle"),
            arrow_path("M 816 268 V 294", color=PURPLE, sw=1.8),
            rect(717, 304, 224, 122, fill=WHITE, stroke=PURPLE, sw=1.4, rx=10),
            rect(730, 292, 196, 111, fill="#FCF8FB", stroke=PURPLE_MID, sw=0.8, rx=9),
            rect(742, 280, 168, 101, fill=PURPLE_SOFT, stroke=PURPLE_MID, sw=0.9, rx=8),
            text(826, 313, "shared per-candidate MLP", size=13.5, color=PURPLE, weight=700, anchor="middle"),
            text(826, 339, "1028 → 640 × 3 → 2", size=11.5, color=INK, weight=700, anchor="middle"),
            text(826, 362, "independent seeds 1701 · 1702 · 1703", size=9.3, color=MUTED, anchor="middle"),
            text(826, 390, "ATEₖ ↓     RTPₖ ↑", size=13, color=PURPLE, weight=700, anchor="middle"),
            text(826, 414, "ensemble mean ± std", size=9.3, color=MUTED, weight=600, anchor="middle"),
            text(816, 462, "candidate-independent · no cross-K operations", size=8.8, color=MUTED, anchor="middle"),
        )
    )

    # Stage 4: conservative selection and output.
    parts.extend(
        (
            rect(996, 154, 160, 176, fill=WHITE, stroke=GREEN, sw=1.4, rx=10),
            text(1076, 182, "Joint lower-confidence gain", size=11.5, color=GREEN, weight=700, anchor="middle"),
            text(1076, 212, "uₖ = min( LCB ΔATE / sA ,", size=10, color=INK, weight=600, anchor="middle"),
            text(1076, 232, "LCB ΔRTP / sR )", size=10, color=INK, weight=600, anchor="middle"),
            line(1015, 249, 1137, 249, color=HAIR, sw=0.9),
            text(1076, 273, "promote only if both", size=9.8, color=MUTED, anchor="middle"),
            text(1076, 293, "margins > 0", size=11.5, color=GREEN, weight=700, anchor="middle"),
            text(1076, 317, "otherwise → incumbent", size=9.5, color=MUTED, weight=600, anchor="middle"),
            arrow_path("M 1076 330 V 354", color=GREEN, sw=2.0),
            trajectory_fan(1012, 365, 128, 88, selected=True),
            text(1076, 478, "selected H4 trajectory", size=12, color=GREEN, weight=700, anchor="middle"),
        )
    )

    # Cross-stage information flow: three clean semantic streams.
    parts.extend(
        (
            arrow_path("M 316 333 H 330 V 218 H 364", color=BLUE, sw=2.2),
            text(331, 209, "history tokens", size=8.3, color=BLUE, weight=600),
            arrow_path("M 316 438 H 330 V 246 H 364", color=ORANGE, sw=2.2),
            text(331, 264, "actions", size=8.3, color=ORANGE, weight=600),
            arrow_path("M 656 387 H 672 V 224 H 704", color=BLUE, sw=2.3),
            text(669, 376, "future latents", size=8.4, color=BLUE, weight=600, anchor="end"),
            arrow_path("M 316 350 H 670 V 196 H 704", color=BLUE, sw=1.5),
            text(641, 188, "current + goal", size=8.2, color=BLUE, weight=600),
            arrow_path("M 316 454 H 666 V 244 H 704", color=ORANGE, sw=1.5),
            text(641, 463, "candidate actions", size=8.2, color=ORANGE, weight=600),
            arrow_path("M 952 356 H 976", color=PURPLE, sw=2.4),
            text(964, 346, "3-seed estimates", size=8.2, color=PURPLE, weight=600, anchor="middle"),
        )
    )

    parts.extend(
        (
            line(24, 525, 1176, 525, color=HAIR, sw=1.0),
            text(
                24,
                550,
                "Exact runtime contract: 4-frame context · H4 candidate actions · dense latent rollout · 1028-D candidate-wise scoring",
                size=9.5,
                color=MUTED,
                weight=600,
            ),
            text(
                1176,
                550,
                "fallback-safe fixed-pool readout",
                size=9.5,
                color=GREEN,
                weight=700,
                anchor="end",
            ),
            "</svg>",
        )
    )
    return "\n".join(parts)


def cell_style(
    *,
    fill: str = WHITE,
    stroke: str = HAIR,
    size: int = 11,
    color: str = INK,
    bold: bool = False,
    rounded: bool = True,
    extra: str = "",
) -> str:
    return (
        f"rounded={1 if rounded else 0};arcSize=12;whiteSpace=wrap;html=1;"
        f"fillColor={fill};strokeColor={stroke};fontColor={color};"
        f"fontFamily=Arial;fontSize={size};fontStyle={1 if bold else 0};"
        f"{extra}"
    )


def text_style(
    *,
    size: int = 11,
    color: str = INK,
    bold: bool = False,
    align: str = "left",
) -> str:
    return (
        "text;html=1;strokeColor=none;fillColor=none;whiteSpace=wrap;"
        f"fontFamily=Arial;fontSize={size};fontColor={color};"
        f"fontStyle={1 if bold else 0};align={align};verticalAlign=middle;"
    )


def image_style(path_value: Path, *, stroke: str) -> str:
    uri = data_uri(path_value).replace(";", "%3B", 1)
    return (
        "shape=image;html=1;imageAspect=0;aspect=fixed;"
        f"image={uri};strokeColor={stroke};strokeWidth=1;"
    )


def add_vertex(
    root: ET.Element,
    cell_id: str,
    value: str,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    style_value: str,
    parent: str = "1",
) -> None:
    cell = ET.SubElement(
        root,
        "mxCell",
        id=cell_id,
        value=value,
        style=style_value,
        vertex="1",
        parent=parent,
    )
    ET.SubElement(
        cell,
        "mxGeometry",
        x=f"{x:g}",
        y=f"{y:g}",
        width=f"{width:g}",
        height=f"{height:g}",
        **{"as": "geometry"},
    )


def add_edge(
    root: ET.Element,
    edge_id: str,
    source: str,
    target: str,
    *,
    color: str,
    label: str = "",
    exit_xy: tuple[float, float] | None = None,
    entry_xy: tuple[float, float] | None = None,
    points: tuple[tuple[float, float], ...] = (),
) -> None:
    slots = [
        "edgeStyle=orthogonalEdgeStyle",
        "rounded=1",
        "orthogonalLoop=1",
        "jettySize=auto",
        "html=1",
        f"strokeColor={color}",
        "strokeWidth=2",
        "endArrow=block",
        "endFill=1",
        "fontFamily=Arial",
        "fontSize=9",
        f"fontColor={color}",
        "labelBackgroundColor=#FFFFFF",
    ]
    if exit_xy:
        slots.extend(
            (
                f"exitX={exit_xy[0]}",
                f"exitY={exit_xy[1]}",
                "exitDx=0",
                "exitDy=0",
            )
        )
    if entry_xy:
        slots.extend(
            (
                f"entryX={entry_xy[0]}",
                f"entryY={entry_xy[1]}",
                "entryDx=0",
                "entryDy=0",
            )
        )
    cell = ET.SubElement(
        root,
        "mxCell",
        id=edge_id,
        value=label,
        style=";".join(slots) + ";",
        edge="1",
        parent="1",
        source=source,
        target=target,
    )
    geometry = ET.SubElement(
        cell,
        "mxGeometry",
        relative="1",
        **{"as": "geometry"},
    )
    if points:
        array = ET.SubElement(geometry, "Array", **{"as": "points"})
        for x, y in points:
            ET.SubElement(array, "mxPoint", x=f"{x:g}", y=f"{y:g}")


def build_drawio() -> None:
    mxfile = ET.Element(
        "mxfile",
        host="drawio",
        version="26.0.0",
        compressed="false",
        agent="Codex drawio-skill",
    )
    diagram = ET.SubElement(
        mxfile,
        "diagram",
        id="figure1-topconf-simple",
        name="Simplified Main Figure",
    )
    graph = ET.SubElement(
        diagram,
        "mxGraphModel",
        dx="1200",
        dy="570",
        grid="1",
        gridSize="10",
        guides="1",
        page="1",
        pageScale="1",
        pageWidth="1200",
        pageHeight="570",
        background="#FFFFFF",
    )
    root = ET.SubElement(graph, "root")
    ET.SubElement(root, "mxCell", id="0")
    ET.SubElement(root, "mxCell", id="1", parent="0")

    add_vertex(
        root,
        "title",
        "<b>DINOv3–LTWM: candidate-conditioned latent trajectory planning</b>",
        24,
        8,
        760,
        28,
        style_value=text_style(size=20, bold=True),
    )
    add_vertex(
        root,
        "subtitle",
        "Frozen visual tokens → action-conditioned latent futures → uncertainty-aware H4 selection",
        24,
        40,
        800,
        22,
        style_value=text_style(size=10, color=MUTED),
    )

    stages = (
        ("condition", "1  CONDITION", 24, 94, 292, 402, PANEL, "#CCD5E0", BLUE),
        ("predict", "2  PREDICT LATENT FUTURES", 340, 94, 316, 402, BLUE_SOFT, BLUE_MID, BLUE),
        ("score", "3  SCORE EACH CANDIDATE", 680, 94, 272, 402, PURPLE_SOFT, PURPLE_MID, PURPLE),
        ("select", "4  SELECT", 976, 94, 200, 402, GREEN_SOFT, GREEN_MID, GREEN),
    )
    for stage_id, label, x, y, width, height, fill, stroke, color in stages:
        add_vertex(
            root,
            stage_id,
            f"<b>{label}</b>",
            x,
            y,
            width,
            height,
            style_value=cell_style(
                fill=fill,
                stroke=stroke,
                size=13,
                color=color,
                bold=True,
                extra="container=1;pointerEvents=0;verticalAlign=top;align=left;spacingTop=10;spacingLeft=14;",
            ),
        )

    # Stage 1 children use coordinates relative to their stage container.
    for index, (tile_id, x) in enumerate(zip(("t-3", "t-2", "t-1", "t"), (68, 114, 160, 206))):
        add_vertex(
            root,
            f"img_{tile_id}",
            "",
            x,
            52,
            44,
            39,
            parent="condition",
            style_value=image_style(
                ASSETS[tile_id],
                stroke=BLUE if index == 3 else "#AEBBCB",
            ),
        )
    add_vertex(
        root,
        "goal_img",
        "",
        68,
        116,
        70,
        43,
        parent="condition",
        style_value=image_style(ASSETS["goal"], stroke=GREEN),
    )
    add_vertex(
        root,
        "tokenizer",
        "<b>FROZEN TOKENIZER</b><br><b>DINOv3-L/16 + Adapter</b><br>"
        "<font color=\"#667180\">1024→384 · 256 tokens</font>",
        24,
        176,
        244,
        66,
        parent="condition",
        style_value=cell_style(fill=WHITE, stroke=BLUE_MID, size=10, color=BLUE),
    )
    add_vertex(
        root,
        "token_output",
        "<b>z<sub>t−3:t</sub>, z<sub>goal</sub></b> ∈ ℝ<sup>256×384</sup>",
        44,
        248,
        204,
        26,
        parent="condition",
        style_value=text_style(size=10, color=BLUE, bold=True, align="center"),
    )
    add_vertex(
        root,
        "proposals",
        "<b>K candidate trajectories</b><br>H4 actions [dx, dy, dψ]<br>"
        "<font color=\"#A96A35\">external proposal policy</font>",
        24,
        304,
        244,
        72,
        parent="condition",
        style_value=cell_style(fill=WHITE, stroke=ORANGE_MID, size=10),
    )

    # Stage 2.
    add_vertex(
        root,
        "ltwm",
        "<b>LiteTokenFormer world model</b><br>"
        "<font color=\"#667180\">candidate-conditioned · one-pass H1–H4</font>",
        24,
        55,
        268,
        62,
        parent="predict",
        style_value=cell_style(fill=WHITE, stroke=BLUE, size=12),
    )
    add_vertex(
        root,
        "history_encoder",
        "<b>History encoder</b>",
        40,
        140,
        100,
        44,
        parent="predict",
        style_value=cell_style(fill=BLUE_SOFT, stroke=BLUE_MID, size=10),
    )
    add_vertex(
        root,
        "action_prefix",
        "<b>Action prefix</b>",
        154,
        140,
        100,
        44,
        parent="predict",
        style_value=cell_style(fill=ORANGE_SOFT, stroke=ORANGE_MID, size=10),
    )
    add_vertex(
        root,
        "rollout",
        "<b>Factorized token rollout</b>",
        82,
        205,
        152,
        40,
        parent="predict",
        style_value=cell_style(fill=BLUE, stroke=BLUE, size=10, color=WHITE),
    )
    for index, label in enumerate(("H1", "H2", "H3", "H4")):
        add_vertex(
            root,
            f"latent_{label.lower()}",
            f"<b>{label}</b>",
            44 + index * 58,
            274,
            46,
            46,
            parent="predict",
            style_value=cell_style(
                fill=WHITE,
                stroke=BLUE if label == "H4" else BLUE_MID,
                size=10,
                color=BLUE if label == "H4" else MUTED,
            ),
        )
    add_vertex(
        root,
        "future_output",
        "<b>ẑ<sub>1:4</sub><sup>(k)</sup></b> ∈ ℝ<sup>K×4×256×384</sup>",
        48,
        346,
        220,
        30,
        parent="predict",
        style_value=text_style(size=11, color=BLUE, bold=True, align="center"),
    )

    # Stage 3.
    add_vertex(
        root,
        "phi",
        "<b>φ<sub>k</sub> = [route | rollout | action]</b><br>"
        "<font color=\"#315E9A\">192-D | 796-D</font> | "
        "<font color=\"#A96A35\">40-D</font><br>"
        "<font color=\"#667180\">1028-D · frozen z-score</font>",
        24,
        58,
        224,
        92,
        parent="score",
        style_value=cell_style(fill=WHITE, stroke=PURPLE_MID, size=10),
    )
    add_vertex(
        root,
        "scorer_stack",
        "<b>shared per-candidate MLP ×3</b><br>"
        "1028 → 640 × 3 → 2<br>"
        "<font color=\"#667180\">seeds 1701 · 1702 · 1703</font><br><br>"
        "<b><font color=\"#745075\">ATE<sub>k</sub> ↓ &nbsp;&nbsp; RTP<sub>k</sub> ↑</font></b><br>"
        "<font color=\"#667180\">ensemble mean ± std</font>",
        36,
        186,
        200,
        148,
        parent="score",
        style_value=cell_style(fill=WHITE, stroke=PURPLE, size=11),
    )
    add_vertex(
        root,
        "candidate_independent",
        "candidate-independent · no cross-K operations",
        26,
        352,
        220,
        28,
        parent="score",
        style_value=text_style(size=9, color=MUTED, align="center"),
    )

    # Stage 4.
    add_vertex(
        root,
        "selector",
        "<b>Joint lower-confidence gain</b><br><br>"
        "u<sub>k</sub> = min(LCB ΔATE/s<sub>A</sub>,<br>"
        "LCB ΔRTP/s<sub>R</sub>)<br><br>"
        "<b><font color=\"#2F7D60\">both margins &gt; 0</font></b><br>"
        "<font color=\"#667180\">else → incumbent</font>",
        20,
        60,
        160,
        176,
        parent="select",
        style_value=cell_style(fill=WHITE, stroke=GREEN, size=10),
    )
    add_vertex(
        root,
        "selected",
        "<b>SELECTED H4</b><br><font color=\"#667180\">execute first action</font>",
        38,
        292,
        124,
        62,
        parent="select",
        style_value=cell_style(fill=WHITE, stroke=GREEN, size=11, color=GREEN),
    )

    # Draw.io flow.
    add_edge(
        root,
        "token_to_ltwm",
        "token_output",
        "history_encoder",
        color=BLUE,
        label="history",
        exit_xy=(1, 0.4),
        entry_xy=(0, 0.5),
        points=((330, 350), (330, 256)),
    )
    add_edge(
        root,
        "proposal_to_ltwm",
        "proposals",
        "action_prefix",
        color=ORANGE,
        label="actions",
        exit_xy=(1, 0.5),
        points=((328, 434), (640, 434), (640, 220), (544, 220)),
        entry_xy=(0.5, 0),
    )
    add_edge(
        root,
        "history_to_rollout",
        "history_encoder",
        "rollout",
        color=BLUE,
        exit_xy=(0.5, 1),
        entry_xy=(0.25, 0),
    )
    add_edge(
        root,
        "action_to_rollout",
        "action_prefix",
        "rollout",
        color=ORANGE,
        exit_xy=(0.5, 1),
        entry_xy=(0.75, 0),
    )
    add_edge(
        root,
        "rollout_to_h4",
        "rollout",
        "latent_h4",
        color=BLUE,
        exit_xy=(0.75, 1),
        entry_xy=(0.5, 0),
    )
    add_edge(
        root,
        "future_to_phi",
        "future_output",
        "phi",
        color=BLUE,
        label="future",
        exit_xy=(1, 0.5),
        entry_xy=(0, 0.55),
        points=((660, 455), (660, 222)),
    )
    add_edge(
        root,
        "tokens_to_phi",
        "token_output",
        "phi",
        color=BLUE,
        label="current + goal",
        exit_xy=(1, 0.2),
        entry_xy=(0, 0.2),
        points=((320, 342), (320, 78), (668, 78), (668, 190)),
    )
    add_edge(
        root,
        "actions_to_phi",
        "proposals",
        "phi",
        color=ORANGE,
        label="actions",
        exit_xy=(1, 0.8),
        entry_xy=(0, 0.82),
        points=((330, 508), (676, 508), (676, 240)),
    )
    add_edge(
        root,
        "phi_to_scorer",
        "phi",
        "scorer_stack",
        color=PURPLE,
        exit_xy=(0.5, 1),
        entry_xy=(0.5, 0),
    )
    add_edge(
        root,
        "scorer_to_selector",
        "scorer_stack",
        "selector",
        color=PURPLE,
        label="mean ± std",
        exit_xy=(1, 0.5),
        entry_xy=(0, 0.5),
    )
    add_edge(
        root,
        "selector_to_selected",
        "selector",
        "selected",
        color=GREEN,
        exit_xy=(0.5, 1),
        entry_xy=(0.5, 0),
    )

    ET.indent(mxfile, space="  ")
    OUT_DRAWIO.write_bytes(
        ET.tostring(mxfile, encoding="UTF-8", xml_declaration=True)
    )


def build_readme() -> None:
    content = """# Figure 1 — simplified top-conference overview

## Main claim

Frozen DINOv3 visual tokens and candidate H4 actions drive
candidate-conditioned LiteTokenFormer latent rollouts. A candidate-independent
three-seed scorer estimates absolute ATE/RTP, and a conservative joint-gain
readout selects an H4 plan or falls back to the incumbent.

## Abstraction boundary

- Preserved: 4-frame context, goal frame, frozen DINOv3-L/16 + adapter,
  256 × 384 stored tokens, K × 4 × 3 actions, H1–H4 dense latent rollout,
  192 + 796 + 40 = 1028-D scorer input, 3 × 640 MLP, three seeds,
  ATE/RTP outputs, uncertainty-aware promotion, incumbent fallback.
- Abstracted: individual Transformer sublayers, token-warp internals, feature
  names, training losses, dataset/split details, and optional proposal-search
  refinements.

## Code-grounding

- `pipeline_contract.json`
- `src/recon_latents/models_v2.py`
- `src/scoring/features.py`
- `src/scoring/trajectory_quality_scorer_v2.py`
- `src/scoring/safe_promotion_v1.py`

The five RGB tiles reuse the provenance-tracked assets in
`figures/figure1_compact/assets/`. All exported files are self-contained.
"""
    (OUT_DIR / "README.md").write_text(content, encoding="utf-8")


def main() -> None:
    for path_value in ASSETS.values():
        if not path_value.exists():
            raise FileNotFoundError(path_value)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    svg = build_svg()
    OUT_SVG.write_text(svg, encoding="utf-8")
    cairosvg.svg2png(
        bytestring=svg.encode("utf-8"),
        write_to=str(OUT_PNG),
        output_width=2400,
        output_height=1140,
    )
    cairosvg.svg2pdf(
        bytestring=svg.encode("utf-8"),
        write_to=str(OUT_PDF),
    )
    build_drawio()
    build_readme()
    print(OUT_DRAWIO)
    print(OUT_SVG)
    print(OUT_PNG)
    print(OUT_PDF)


if __name__ == "__main__":
    main()
