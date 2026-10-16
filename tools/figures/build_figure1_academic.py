#!/usr/bin/env python3
"""Build a camera-ready, schematic-led overview for the DINOv3–LTWM planner.

This is deliberately vector-first: SVG/PDF stay editable and the Draw.io file
contains the same semantic layout.  Raster imagery is used only as evidence in
the observation strip; all model structure remains vector geometry.
"""

from __future__ import annotations

import base64
from html import escape
from pathlib import Path
import xml.etree.ElementTree as ET

import cairosvg


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "figures" / "figure1_compact"
ASSET_DIR = OUT_DIR / "assets"
OUT_SVG = OUT_DIR / "figure1_compact_wireframe.svg"
OUT_PNG = OUT_DIR / "figure1_compact_wireframe.png"
OUT_PDF = OUT_DIR / "figure1_compact_wireframe.pdf"
OUT_DRAWIO = OUT_DIR / "figure1_compact_wireframe.drawio"

W, H = 760.0, 360.0
INK = "#20252B"
MUTED = "#68717C"
HAIR = "#D8DEE6"
PALE = "#F7F9FB"
BLUE = "#416EAA"
BLUE_DARK = "#2F568A"
BLUE_SOFT = "#EAF1FA"
BLUE_MID = "#C8D9EC"
ORANGE = "#AF754A"
ORANGE_SOFT = "#F8EEE4"
PURPLE = "#8C5D82"
PURPLE_SOFT = "#F7EDF4"
GREEN = "#2E8165"
GREEN_SOFT = "#E8F3EE"
FONT = "Arial, Helvetica, DejaVu Sans, sans-serif"

ASSETS = {
    "frame0": ASSET_DIR / "history_t-3.jpg",
    "frame1": ASSET_DIR / "history_t-2.jpg",
    "frame2": ASSET_DIR / "history_t-1.jpg",
    "frame3": ASSET_DIR / "history_t.jpg",
    "goal": ASSET_DIR / "goal.jpg",
}


def data_uri(path: Path) -> str:
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64,{payload}"


def text(
    x: float,
    y: float,
    value: str,
    *,
    size: float = 7.0,
    color: str = INK,
    anchor: str = "start",
    weight: int = 400,
    letter: float | None = None,
) -> str:
    extra = f"letter-spacing:{letter}px;" if letter is not None else ""
    return (
        f'<text x="{x:g}" y="{y:g}" text-anchor="{anchor}" '
        f'style="font-family:{FONT};font-size:{size:g}px;'
        f'font-weight:{weight};fill:{color};{extra}">{escape(value)}</text>'
    )


def rect(
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    fill: str = "none",
    stroke: str = "none",
    sw: float = 0.8,
    rx: float = 0.0,
    opacity: float = 1.0,
) -> str:
    return (
        f'<rect x="{x:g}" y="{y:g}" width="{width:g}" height="{height:g}" '
        f'rx="{rx:g}" fill="{fill}" stroke="{stroke}" stroke-width="{sw:g}" '
        f'opacity="{opacity:g}" />'
    )


def line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    color: str = HAIR,
    sw: float = 0.8,
    dash: str | None = None,
) -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1:g}" y1="{y1:g}" x2="{x2:g}" y2="{y2:g}" '
        f'style="stroke:{color};stroke-width:{sw:g};stroke-linecap:round"'
        f"{dash_attr} />"
    )


def path(
    d: str,
    *,
    color: str = HAIR,
    sw: float = 0.8,
    marker: str | None = None,
    dash: str | None = None,
    fill: str = "none",
) -> str:
    marker_attr = f' marker-end="url(#{marker})"' if marker else ""
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<path d="{d}" fill="{fill}" '
        f'style="stroke:{color};stroke-width:{sw:g};stroke-linecap:round;'
        f'stroke-linejoin:round"{dash_attr}{marker_attr} />'
    )


def marker_name(color: str) -> str:
    return {
        BLUE: "arrow_blue",
        BLUE_DARK: "arrow_blue_dark",
        ORANGE: "arrow_orange",
        PURPLE: "arrow_purple",
        GREEN: "arrow_green",
        MUTED: "arrow_muted",
    }.get(color, "arrow_ink")


def arrow(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    color: str = MUTED,
    sw: float = 1.0,
) -> str:
    return line(x1, y1, x2, y2, color=color, sw=sw) + (
        f'<polygon points="{x2:g},{y2:g} {x2 - 5:g},{y2 - 2.7:g} '
        f'{x2 - 5:g},{y2 + 2.7:g}" fill="{color}" />'
    )


def section_tag(x: float, y: float, letter: str, title: str, color: str) -> str:
    return "\n".join(
        (
            rect(x, y - 11, 2.3, 14, fill=color, rx=1.1),
            text(x + 8, y, f"{letter}  {title}", size=9.1, color=color, weight=700),
        )
    )


def image_tile(
    tile_id: str,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    stroke: str,
    sw: float,
) -> str:
    clip = f"{tile_id}_clip"
    uri = data_uri(ASSETS[tile_id])
    return "\n".join(
        (
            f'<clipPath id="{clip}"><rect x="{x:g}" y="{y:g}" '
            f'width="{width:g}" height="{height:g}" rx="1.7" /></clipPath>',
            f'<image x="{x:g}" y="{y:g}" width="{width:g}" height="{height:g}" '
            f'preserveAspectRatio="xMidYMid slice" clip-path="url(#{clip})" '
            f'xlink:href="{uri}" />',
            rect(x, y, width, height, fill="none", stroke=stroke, sw=sw, rx=1.7),
        )
    )


def latent_grid(
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    label: str,
    color: str = BLUE,
    fill: str = "#FFFFFF",
) -> str:
    parts = [rect(x, y, width, height, fill=fill, stroke=color, sw=0.65, rx=2.0)]
    for i in range(1, 4):
        parts.append(line(x + width * i / 4, y + 2, x + width * i / 4, y + height - 2, color="#B8C6D7", sw=0.35))
        parts.append(line(x + 2, y + height * i / 4, x + width - 2, y + height * i / 4, color="#B8C6D7", sw=0.35))
    parts.append(text(x + width / 2, y + height + 8, label, size=5.5, color=MUTED, anchor="middle"))
    return "\n".join(parts)


def build_svg() -> str:
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'viewBox="0 0 {W:g} {H:g}" width="{W:g}pt" height="{H:g}pt">',
        "<defs>",
        '<linearGradient id="hero_wash" x1="0" x2="1" y1="0" y2="1">',
        f'<stop offset="0%" stop-color="{BLUE_SOFT}" />',
        '<stop offset="100%" stop-color="#FFFFFF" />',
        "</linearGradient>",
        '<linearGradient id="encoder_wash" x1="0" x2="0" y1="0" y2="1">',
        '<stop offset="0%" stop-color="#F8FBFE" />',
        '<stop offset="100%" stop-color="#FFFFFF" />',
        "</linearGradient>",
    ]
    for name, color in (
        ("arrow_ink", INK),
        ("arrow_muted", MUTED),
        ("arrow_blue", BLUE),
        ("arrow_blue_dark", BLUE_DARK),
        ("arrow_orange", ORANGE),
        ("arrow_purple", PURPLE),
        ("arrow_green", GREEN),
    ):
        parts.extend(
            (
                f'<marker id="{name}" markerWidth="7" markerHeight="7" refX="6.2" refY="3.5" '
                'orient="auto" markerUnits="strokeWidth">',
                f'<path d="M0,0 L7,3.5 L0,7 z" fill="{color}" />',
                "</marker>",
            )
        )
    parts.extend(("</defs>", rect(0, 0, W, H, fill="#FFFFFF")))

    # Header: one claim line, one quiet metadata line.
    parts.extend(
        (
            text(24, 25, "DINOv3–LTWM trajectory planning", size=13.2, weight=700),
            text(
                24,
                42,
                "Fixed visual tokens  →  candidate-conditioned latent futures  →  conservative H4 selection",
                size=6.2,
                color=MUTED,
                weight=600,
            ),
            text(736, 25, "INFERENCE-TIME METHOD", size=5.2, color=BLUE, anchor="end", weight=700, letter=0.7),
            line(24, 51, 736, 51, color=HAIR, sw=0.7),
            line(193, 68, 193, 363, color="#EEF1F4", sw=0.8),
            line(519, 68, 519, 363, color="#EEF1F4", sw=0.8),
        )
    )

    # A — real RGB context, with no enclosing box.
    parts.append(section_tag(24, 76, "A", "VISUAL OBSERVATIONS", BLUE))
    parts.extend(
        (
            text(58, 92, "RGB OBSERVATIONS", size=5.2, color=BLUE, weight=700, letter=0.45),
            text(24, 115, "history", size=6.2, color=MUTED),
        )
    )
    frame_specs = (
        ("frame0", 58, "#9BAFC3", 0.65, "t−3"),
        ("frame1", 87, "#9BAFC3", 0.65, "t−2"),
        ("frame2", 116, "#9BAFC3", 0.65, "t−1"),
        ("frame3", 145, BLUE_DARK, 1.05, "t"),
    )
    for tile_id, x, stroke, sw, label in frame_specs:
        parts.append(image_tile(tile_id, x, 99, 31, 28, stroke=stroke, sw=sw))
        parts.append(text(x + 15.5, 137, label, size=5.8, color=MUTED, anchor="middle"))
    parts.extend(
        (
            text(24, 171, "goal", size=6.2, color=MUTED),
            image_tile("goal", 112, 148, 52, 30, stroke=GREEN, sw=0.95),
            text(138, 187, "goal frame", size=5.5, color=GREEN, anchor="middle", weight=600),
            path("M176 112 H194", color=BLUE, sw=1.1, marker=marker_name(BLUE)),
            text(178, 106, "history + goal", size=5.3, color=BLUE, anchor="end"),
            path("M164 163 H185 V141 H204", color=GREEN, sw=0.9, marker=marker_name(GREEN)),
        )
    )

    # Frozen encoder / tokenization bridge.
    parts.extend(
        (
            rect(204, 68, 148, 96, fill="url(#encoder_wash)", stroke=BLUE_DARK, sw=1.05, rx=7),
            text(215, 82, "FROZEN VISUAL TOKENIZER", size=7.1, color=BLUE_DARK, weight=700),
            text(342, 82, "LOCKED", size=4.4, color=MUTED, anchor="end", weight=700, letter=0.45),
            rect(215, 98, 78, 31, fill=BLUE_SOFT, stroke=BLUE_MID, sw=0.7, rx=3),
            text(254, 112, "DINOv3-L/16", size=6.4, anchor="middle", weight=700),
            text(254, 121, "patch features", size=4.7, anchor="middle", color=MUTED),
            rect(302, 98, 36, 31, fill="#F2EAF2", stroke="#CDAEC7", sw=0.7, rx=3),
            text(320, 116, "Adapter", size=5.7, anchor="middle", weight=700),
            text(215, 146, "14×14 patches  ·  384-D tokens", size=5.3, color=MUTED),
            path("M278 164 V184", color=BLUE_DARK, sw=1.0, marker=marker_name(BLUE_DARK)),
        )
    )

    # B — action proposals: a small NoMaD source plus an uncluttered fan.
    parts.append(section_tag(24, 218, "B", "CANDIDATE ACTIONS", ORANGE))
    parts.extend(
        (
            rect(24, 237, 66, 53, fill=ORANGE_SOFT, stroke=ORANGE, sw=0.95, rx=5),
            text(57, 252, "NoMaD", size=8.0, color=ORANGE, anchor="middle", weight=700),
            text(57, 266, "K candidates", size=5.2, color=MUTED, anchor="middle"),
            text(57, 277, "H = 4 · aₕ = [dx, dy, dψ]", size=4.6, color=MUTED, anchor="middle"),
            text(106, 232, "candidate proposals", size=5.2, color=ORANGE, weight=700),
        )
    )
    # The fan is the action trajectory itself; no second enclosing box.
    curves = (
        ("M90 282 C112 273 133 246 181 240", "#B99578", 0.75),
        ("M90 282 C116 280 140 260 183 257", "#C9B29E", 0.7),
        ("M90 282 C119 286 145 278 183 276", ORANGE, 1.6),
        ("M90 282 C118 291 147 298 183 300", "#C9B29E", 0.7),
        ("M90 282 C114 302 143 316 181 322", "#B99578", 0.75),
    )
    for d, color, sw in curves:
        parts.append(path(d, color=color, sw=sw))
    parts.extend(
        (
            rect(84, 277, 10, 10, fill=ORANGE, stroke="#FFFFFF", sw=1.0, rx=5),
            image_tile("frame3", 24, 301, 44, 28, stroke=ORANGE, sw=0.8),
            text(46, 338, "current view", size=4.5, color=MUTED, anchor="middle"),
            path("M68 315 H78 V282 H84", color=ORANGE, sw=0.7, marker=marker_name(ORANGE)),
            text(107, 331, "K × H4 action sequences", size=5.4, color=MUTED),
            path("M181 300 H204", color=ORANGE, sw=1.05, marker=marker_name(ORANGE)),
            text(190, 294, "action", size=5.1, color=ORANGE, anchor="end"),
            path("M181 322 H197 V371 H507 V198 H530", color=ORANGE, sw=0.9, marker=marker_name(ORANGE)),
        )
    )

    # C — hero latent rollout.
    parts.extend(
        (
            section_tag(204, 218, "C", "ACTION-CONDITIONED LATENT ROLLOUT", BLUE_DARK),
            rect(204, 230, 302, 126, fill="url(#hero_wash)", stroke=BLUE, sw=1.15, rx=8),
            text(218, 246, "LiteTokenFormer", size=8.0, color=BLUE_DARK, weight=700),
            text(494, 246, "candidate-conditioned", size=5.0, color=MUTED, anchor="end"),
            rect(218, 260, 92, 27, fill="#FFFFFF", stroke=BLUE_MID, sw=0.7, rx=3),
            text(264, 271, "history tokens", size=5.7, anchor="middle", weight=700),
            text(264, 280, "T = 4 × 196 × 384", size=4.7, anchor="middle", color=MUTED),
            rect(322, 260, 78, 27, fill=ORANGE_SOFT, stroke="#D7B99F", sw=0.7, rx=3),
            text(361, 271, "action h₁:₄", size=5.7, anchor="middle", color=ORANGE, weight=700),
            text(361, 280, "K candidates", size=4.7, anchor="middle", color=MUTED),
            path("M310 273 H322", color=BLUE_DARK, sw=0.9, marker=marker_name(BLUE_DARK)),
            path("M361 287 V297 H322", color=ORANGE, sw=0.85, marker=marker_name(ORANGE)),
            # One compact core keeps the transformer abstract without
            # turning LTWM into a training-detail figure.
            rect(258, 305, 112, 34, fill="#FFFFFF", stroke=BLUE, sw=0.85, rx=3),
            text(314, 318, "token mixing  ·  temporal mixing", size=5.2, anchor="middle", color=BLUE_DARK, weight=700),
            text(314, 329, "factorized rollout head", size=4.7, anchor="middle", color=MUTED),
            path("M378 322 H402", color=BLUE_DARK, sw=0.95, marker=marker_name(BLUE_DARK)),
            latent_grid(407, 260, 25, 24, label="H1", color=BLUE),
            latent_grid(439, 260, 25, 24, label="H2", color=BLUE),
            latent_grid(407, 298, 25, 24, label="H3", color=BLUE),
            latent_grid(439, 298, 25, 24, label="H4", color=BLUE_DARK),
            text(494, 340, "predicted future tokens  [K × 4 × 256 × 384]", size=5.3, color=MUTED, anchor="end"),
            path("M506 322 H517 V168 H530", color=BLUE_DARK, sw=0.95, marker=marker_name(BLUE_DARK)),
        )
    )


    # D — scorer and selector, organized as rails rather than a stack of boxes.
    parts.extend(
        (
            section_tag(530, 76, "D", "CONSERVATIVE TRAJECTORY RANKING", PURPLE),
            rect(530, 92, 206, 264, fill="#FCF8FB", stroke="none", rx=7),
            rect(530, 92, 2.4, 22, fill=PURPLE, rx=1.2),
            text(544, 108, "3-seed absolute quality model", size=5.8, color=PURPLE, weight=700),
            line(544, 115, 724, 115, color="#E6D7E1", sw=0.65),
            # Input rails: every label is detached from its wire.
            text(536, 132, "route  ·  192-D", size=5.0, color=BLUE, weight=700),
            text(536, 162, "future  ·  796-D", size=5.0, color=BLUE_DARK, weight=700),
            text(536, 192, "action  ·  40-D", size=5.0, color=ORANGE, weight=700),
            path("M530 138 H552", color=BLUE, sw=0.85, marker=marker_name(BLUE)),
            path("M530 168 H552", color=BLUE_DARK, sw=0.85, marker=marker_name(BLUE_DARK)),
            path("M530 198 H552", color=ORANGE, sw=0.85, marker=marker_name(ORANGE)),
            rect(552, 126, 34, 90, fill="#FFFFFF", stroke="#BFC7D0", sw=0.8, rx=3),
            text(569, 161, "concat", size=6.6, anchor="middle", weight=700),
            text(569, 174, "1028-D", size=5.6, anchor="middle", weight=700),
            text(569, 188, "frozen", size=4.6, anchor="middle", color=MUTED),
            text(569, 198, "z-score", size=4.6, anchor="middle", color=MUTED),
            line(586, 171, 592, 171, color=PURPLE, sw=0.7),
            line(592, 135, 592, 207, color=PURPLE, sw=0.7),
            text(628, 126, "shared MLP geometry", size=4.6, color=MUTED, anchor="middle"),
        )
    )
    parts.extend((rect(600, 143, 72, 42, fill=PURPLE_SOFT, stroke="#D6B7CC", sw=0.7, rx=4), text(636, 158, "MLP × 3", size=6.0, color=PURPLE, anchor="middle", weight=700), text(636, 171, "1028 → 640 → 2", size=4.2, color=MUTED, anchor="middle"), path("M672 164 H683", color=PURPLE, sw=0.85, marker=marker_name(PURPLE)), rect(683, 132, 42, 80, fill="#FFFFFF", stroke=PURPLE, sw=0.9, rx=4), text(704, 147, "joint", size=5.2, color=PURPLE, anchor="middle", weight=700), text(704, 161, "uₖ = min(LCBᴱ/sₑ,", size=3.8, color=MUTED, anchor="middle"), text(704, 170, "LCBᴾ/sₚ)", size=3.8, color=MUTED, anchor="middle"), text(704, 185, "argmax valid k", size=4.0, color=INK, anchor="middle", weight=700), text(704, 197, "gate → incumbent", size=3.8, color=MUTED, anchor="middle"), rect(600, 234, 72, 18, fill="#FFF8FB", stroke="#D6B7CC", sw=0.65, rx=3), text(636, 246, "ATE ↓   RTP ↑", size=4.8, color=PURPLE, anchor="middle", weight=700), path("M672 243 H683 V264", color=GREEN, sw=0.9, marker=marker_name(GREEN)), rect(683, 264, 42, 22, fill=GREEN_SOFT, stroke="#A9CDBE", sw=0.7, rx=4), text(704, 278, "selected H4", size=4.6, color=GREEN, anchor="middle", weight=700), text(544, 319, "fallback: incumbent", size=4.6, color=MUTED), text(724, 340, "absolute metrics · lower ATE / higher RTP", size=4.1, color=MUTED, anchor="end")))
    # A quiet baseline and footer keep the composition anchored.
    parts.extend(
        (
            line(24, 385, 736, 385, color=HAIR, sw=0.7),
            text(24, 403, "RGB evidence", size=5.6, color=BLUE, weight=700),
            text(94, 403, "→", size=6.0, color=MUTED, weight=700),
            text(108, 403, "frozen tokens", size=5.6, color=BLUE_DARK, weight=700),
            text(184, 403, "→", size=6.0, color=MUTED, weight=700),
            text(198, 403, "latent rollout", size=5.6, color=BLUE_DARK, weight=700),
            text(275, 403, "→", size=6.0, color=MUTED, weight=700),
            text(289, 403, "safe trajectory selection", size=5.6, color=GREEN, weight=700),
            text(736, 403, "No training details shown", size=5.0, color=MUTED, anchor="end"),
            text(24, 433, "Overview figure · inference-time information flow only", size=4.8, color=MUTED),
            text(736, 433, "DINOv3 → LTWM → scorer", size=4.8, color=MUTED, anchor="end"),
        )
    )
    parts.extend((rect(0, 0, W, H, fill="#FFFFFF"), text(24, 26, "DINOv3–LTWM trajectory planning", size=13.0, weight=700), text(24, 44, "visual context + candidate actions → latent rollout → safe H4 selection", size=5.8, color=MUTED, weight=600), text(736, 26, "INFERENCE ONLY", size=4.8, color=BLUE, anchor="end", weight=700, letter=0.6), line(24, 57, 736, 57, color=HAIR, sw=0.7), line(236, 74, 236, 334, color="#EEF1F4", sw=0.8), line(520, 74, 520, 334, color="#EEF1F4", sw=0.8), text(24, 347, "real observations", size=4.5, color=BLUE, weight=700), text(116, 347, "candidate actions", size=4.5, color=ORANGE, weight=700), text(300, 347, "predicted latent futures", size=4.5, color=BLUE_DARK, weight=700), text(650, 347, "selected trajectory", size=4.5, color=GREEN, anchor="middle", weight=700)))
    parts.extend((section_tag(24, 82, "A", "VISUAL CONTEXT", BLUE), text(32, 105, "history", size=4.7, color=MUTED), image_tile("frame0", 66, 94, 31, 27, stroke="#9BAFC3", sw=0.55), image_tile("frame1", 98, 94, 31, 27, stroke="#9BAFC3", sw=0.55), image_tile("frame2", 130, 94, 31, 27, stroke="#9BAFC3", sw=0.55), image_tile("frame3", 162, 94, 31, 27, stroke=BLUE_DARK, sw=0.9), text(81.5, 131, "t−3", size=4.4, color=MUTED, anchor="middle"), text(113.5, 131, "t−2", size=4.4, color=MUTED, anchor="middle"), text(145.5, 131, "t−1", size=4.4, color=MUTED, anchor="middle"), text(177.5, 131, "t", size=4.4, color=MUTED, anchor="middle"), text(32, 164, "goal", size=4.7, color=MUTED), image_tile("goal", 142, 143, 51, 31, stroke=GREEN, sw=0.8), path("M193 109 H236", color=BLUE, sw=0.9, marker=marker_name(BLUE)), path("M193 158 H220 V129 H252", color=GREEN, sw=0.8, marker=marker_name(GREEN)), section_tag(24, 207, "B", "CANDIDATE ACTIONS", ORANGE), text(32, 235, "NoMaD", size=7.3, color=ORANGE, weight=700), text(32, 249, "K trajectories · H = 4", size=4.8, color=MUTED), path("M80 276 C112 257 147 237 218 232", color="#CDB9A8", sw=0.7), path("M80 276 C118 267 156 259 220 256", color="#B99578", sw=0.75), path("M80 276 C120 278 160 279 220 280", color=ORANGE, sw=1.7), path("M80 276 C116 289 158 299 220 306", color="#CDB9A8", sw=0.7), path("M80 276 C110 301 152 318 218 326", color="#D8CABC", sw=0.7), rect(75, 271, 10, 10, fill=ORANGE, stroke="#FFFFFF", sw=0.8, rx=5), text(32, 306, "a₁:₄⁽ᵏ⁾ = [dx, dy, dψ]", size=4.8, color=MUTED), path("M220 280 H252", color=ORANGE, sw=0.95, marker=marker_name(ORANGE))))
    parts.extend((section_tag(252, 82, "C", "FROZEN ENCODER + LATENT ROLLOUT", BLUE_DARK), text(260, 112, "DINOv3-L/16", size=6.2, color=BLUE_DARK, weight=700), path("M324 109 H346", color=BLUE_DARK, sw=0.8, marker=marker_name(BLUE_DARK)), text(354, 112, "Adapter", size=6.0, color=PURPLE, weight=700), text(436, 112, "frozen visual tokens", size=4.7, color=MUTED, anchor="end"), path("M348 122 V156", color=BLUE_DARK, sw=0.9, marker=marker_name(BLUE_DARK)), text(260, 173, "z≤t", size=5.3, color=BLUE_DARK, weight=700), text(260, 194, "a₁:₄⁽ᵏ⁾", size=5.3, color=ORANGE, weight=700), path("M278 171 H300", color=BLUE_DARK, sw=0.8, marker=marker_name(BLUE_DARK)), path("M278 192 H300", color=ORANGE, sw=0.8, marker=marker_name(ORANGE)), rect(300, 154, 140, 54, fill=BLUE_SOFT, stroke=BLUE, sw=0.9, rx=7), text(370, 176, "LiteTokenFormer", size=8.0, color=BLUE_DARK, anchor="middle", weight=700), text(370, 193, "action-conditioned rollout", size=5.0, color=MUTED, anchor="middle"), path("M370 208 V231", color=BLUE_DARK, sw=0.9, marker=marker_name(BLUE_DARK)), latent_grid(300, 239, 25, 24, label="H1", color=BLUE), latent_grid(338, 239, 25, 24, label="H2", color=BLUE), latent_grid(376, 239, 25, 24, label="H3", color=BLUE), latent_grid(414, 239, 25, 24, label="H4", color=BLUE_DARK), text(370, 300, "ẑₜ₊₁:ₜ₊₄⁽ᵏ⁾ = Fθ(z≤t, a₁:₄⁽ᵏ⁾)", size=5.6, color=BLUE_DARK, anchor="middle", weight=600), text(370, 320, "K candidate-conditioned latent futures", size=4.7, color=MUTED, anchor="middle")))
    parts.extend((section_tag(540, 82, "D", "TRAJECTORY RANKING", PURPLE), text(548, 116, "route", size=5.0, color=BLUE, weight=700), text(548, 138, "future", size=5.0, color=BLUE_DARK, weight=700), text(548, 160, "action", size=5.0, color=ORANGE, weight=700), path("M580 114 H608", color=BLUE, sw=0.75), path("M580 136 H608", color=BLUE_DARK, sw=0.75), path("M580 158 H608", color=ORANGE, sw=0.75), rect(608, 106, 76, 60, fill=PURPLE_SOFT, stroke="#D1B1C8", sw=0.75, rx=6), text(646, 128, "MLP × 3", size=7.0, color=PURPLE, anchor="middle", weight=700), text(646, 143, "ATE ↓ · RTP ↑", size=5.1, color=MUTED, anchor="middle"), text(646, 156, "1028 → 2", size=4.4, color=MUTED, anchor="middle"), text(548, 202, "uₖ = min(LCBₖᴬᵀᴱ/sₑ, LCBₖᴿᵀᴾ/sₚ)", size=5.3, color=INK, weight=600), text(548, 218, "k* = arg maxₖ∈valid uₖ", size=5.3, color=PURPLE, weight=700), path("M550 274 C585 260 625 252 716 248", color="#C7CCD2", sw=0.7), path("M550 274 C590 274 635 276 716 282", color="#BFC5CC", sw=0.7), path("M550 274 C590 288 640 300 716 314", color="#D4D8DD", sw=0.7), path("M550 274 C592 268 642 262 716 260", color=GREEN, sw=1.8, marker=marker_name(GREEN)), rect(546, 270, 8, 8, fill=INK, stroke="#FFFFFF", sw=0.8, rx=4), text(716, 244, "candidate set", size=4.5, color=MUTED, anchor="end"), text(716, 331, "selected H4", size=5.4, color=GREEN, anchor="end", weight=700)))
    parts.extend((rect(0, 0, W, H, fill="#FFFFFF"), text(24, 26, "DINOv3–LTWM trajectory planning", size=13.0, weight=700), text(24, 44, "context + candidates → latent futures → safe selection", size=5.8, color=MUTED, weight=600), text(736, 26, "INFERENCE ONLY", size=4.8, color=BLUE, anchor="end", weight=700, letter=0.6), line(24, 57, 736, 57, color=HAIR, sw=0.7), section_tag(24, 82, "A", "VISUAL CONTEXT", BLUE), text(32, 105, "history", size=4.6, color=MUTED), image_tile("frame0", 66, 94, 31, 27, stroke="#9BAFC3", sw=0.55), image_tile("frame1", 98, 94, 31, 27, stroke="#9BAFC3", sw=0.55), image_tile("frame2", 130, 94, 31, 27, stroke="#9BAFC3", sw=0.55), image_tile("frame3", 162, 94, 31, 27, stroke=BLUE_DARK, sw=0.9), text(177.5, 131, "t", size=4.4, color=MUTED, anchor="middle"), text(32, 164, "goal", size=4.6, color=MUTED), image_tile("goal", 142, 143, 51, 31, stroke=GREEN, sw=0.8), path("M193 108 C220 92 245 88 270 104", color=BLUE, sw=0.85, marker=marker_name(BLUE)), path("M193 158 C230 150 255 142 300 176", color=GREEN, sw=0.75, marker=marker_name(GREEN)), section_tag(24, 216, "B", "CANDIDATE ACTIONS", ORANGE), text(32, 244, "NoMaD", size=7.2, color=ORANGE, weight=700), text(32, 258, "K trajectories · H = 4", size=4.7, color=MUTED), path("M80 290 C120 264 172 246 218 238", color="#CDB9A8", sw=0.7), path("M80 290 C124 281 172 270 220 266", color="#B99578", sw=0.75), path("M80 290 C128 291 178 292 222 292", color=ORANGE, sw=1.7), path("M80 290 C124 302 174 312 220 320", color="#D8CABC", sw=0.7), rect(75, 285, 10, 10, fill=ORANGE, stroke="#FFFFFF", sw=0.8, rx=5), text(32, 324, "a₁:₄⁽ᵏ⁾ = [dx, dy, dψ]", size=4.7, color=MUTED), path("M222 292 C255 292 274 223 300 188", color=ORANGE, sw=0.85, marker=marker_name(ORANGE)), line(236, 74, 236, 334, color="#EEF1F4", sw=0.8), line(520, 74, 520, 334, color="#EEF1F4", sw=0.8), text(24, 347, "real observations", size=4.5, color=BLUE, weight=700), text(130, 347, "candidate actions", size=4.5, color=ORANGE, weight=700), text(330, 347, "latent futures", size=4.5, color=BLUE_DARK, weight=700), text(650, 347, "selected trajectory", size=4.5, color=GREEN, anchor="middle", weight=700)))
    parts.extend((section_tag(270, 82, "C", "FROZEN ENCODER", BLUE_DARK), text(270, 111, "DINOv3-L/16", size=6.1, color=BLUE_DARK, weight=700), path("M334 108 H350", color=BLUE_DARK, sw=0.8, marker=marker_name(BLUE_DARK)), text(357, 111, "Adapter", size=6.0, color=PURPLE, weight=700), text(436, 111, "14×14 patches · 384-D", size=4.6, color=MUTED, anchor="end"), path("M360 122 C350 132 334 145 330 160", color=BLUE_DARK, sw=0.8, marker=marker_name(BLUE_DARK)), section_tag(300, 147, "D", "LATENT ROLLOUT", BLUE_DARK), rect(300, 161, 160, 48, fill=BLUE_SOFT, stroke=BLUE, sw=0.9, rx=7), text(380, 182, "LiteTokenFormer", size=8.0, color=BLUE_DARK, anchor="middle", weight=700), text(380, 197, "action-conditioned", size=4.8, color=MUTED, anchor="middle"), text(270, 167, "z≤t", size=5.0, color=BLUE_DARK, weight=700), text(270, 194, "a₁:₄⁽ᵏ⁾", size=5.0, color=ORANGE, weight=700), path("M286 166 C292 166 296 170 300 174", color=BLUE_DARK, sw=0.75, marker=marker_name(BLUE_DARK)), path("M286 193 C292 193 296 189 300 188", color=ORANGE, sw=0.75, marker=marker_name(ORANGE)), path("M380 209 C376 223 362 232 348 241", color=BLUE_DARK, sw=0.8, marker=marker_name(BLUE_DARK)), latent_grid(338, 239, 23, 22, label="H1", color=BLUE), latent_grid(373, 250, 23, 22, label="H2", color=BLUE), latent_grid(408, 261, 23, 22, label="H3", color=BLUE), latent_grid(443, 272, 23, 22, label="H4", color=BLUE_DARK), text(380, 323, "ẑₜ₊₁:ₜ₊₄⁽ᵏ⁾ = Fθ(z≤t, a₁:₄⁽ᵏ⁾)", size=5.4, color=BLUE_DARK, anchor="middle", weight=600)))
    parts.extend((section_tag(536, 82, "E", "TRAJECTORY RANKING", PURPLE), text(540, 114, "route", size=4.9, color=BLUE, weight=700), text(540, 136, "future", size=4.9, color=BLUE_DARK, weight=700), text(540, 158, "action", size=4.9, color=ORANGE, weight=700), path("M570 112 C590 94 600 100 620 112", color=BLUE, sw=0.8, marker=marker_name(BLUE)), path("M470 286 C520 250 540 170 620 136", color=BLUE_DARK, sw=0.8, marker=marker_name(BLUE_DARK)), path("M220 300 C370 342 500 304 620 158", color=ORANGE, sw=0.8, marker=marker_name(ORANGE)), rect(620, 104, 84, 58, fill=PURPLE_SOFT, stroke="#D1B1C8", sw=0.75, rx=6), text(662, 126, "MLP × 3", size=7.1, color=PURPLE, anchor="middle", weight=700), text(662, 141, "ATE ↓ · RTP ↑", size=5.0, color=MUTED, anchor="middle"), text(662, 154, "1028 → 2", size=4.3, color=MUTED, anchor="middle"), text(540, 194, "uₖ = min(LCBₖᴬᵀᴱ/sₑ, LCBₖᴿᵀᴾ/sₚ)", size=5.0, color=INK, weight=600), text(540, 210, "k* = argmax valid uₖ", size=5.0, color=PURPLE, weight=700), path("M546 264 C580 252 630 240 716 236", color="#C8CDD3", sw=0.7), path("M546 264 C590 264 648 276 716 292", color="#BFC5CC", sw=0.7), path("M546 264 C594 276 650 302 716 318", color="#D4D8DD", sw=0.7), path("M546 264 C592 260 648 260 716 268", color=GREEN, sw=1.8, marker=marker_name(GREEN)), rect(542, 260, 8, 8, fill=INK, stroke="#FFFFFF", sw=0.8, rx=4), text(716, 232, "candidate set", size=4.4, color=MUTED, anchor="end"), text(716, 334, "selected H4", size=5.4, color=GREEN, anchor="end", weight=700)))
    parts.append("</svg>")
    return "\n".join(parts)


def style(
    *,
    fill: str = "#FFFFFF",
    stroke: str = "#C7CED7",
    font: str = INK,
    size: int = 11,
    rounded: bool = True,
    extra: str = "",
) -> str:
    base = (
        f"rounded={1 if rounded else 0};arcSize=12;whiteSpace=wrap;html=1;"
        f"fillColor={fill};strokeColor={stroke};fontColor={font};"
        f"fontFamily=Arial;fontSize={size};"
    )
    return base + extra


def text_style(*, color: str = INK, size: int = 11, bold: bool = False) -> str:
    return (
        "text;html=1;strokeColor=none;fillColor=none;align=left;"
        "verticalAlign=middle;fontFamily=Arial;"
        f"fontColor={color};fontSize={size};fontStyle={1 if bold else 0};"
    )


def drawio_vertex(
    root: ET.Element,
    cells: dict[str, ET.Element],
    cell_id: str,
    value: str,
    x: int,
    y: int,
    width: int,
    height: int,
    *,
    parent: str = "1",
    cell_style: str,
) -> ET.Element:
    cell = cells.get(cell_id)
    if cell is None:
        cell = ET.SubElement(root, "mxCell", id=cell_id, vertex="1", parent=parent)
        cells[cell_id] = cell
    cell.set("value", value)
    cell.set("style", cell_style)
    cell.set("vertex", "1")
    cell.set("parent", parent)
    geom = cell.find("mxGeometry")
    if geom is None:
        geom = ET.SubElement(cell, "mxGeometry", **{"as": "geometry"})
    geom.set("x", str(x))
    geom.set("y", str(y))
    geom.set("width", str(width))
    geom.set("height", str(height))
    return cell


def drawio_edge(
    root: ET.Element,
    cells: dict[str, ET.Element],
    edge_id: str,
    source: str,
    target: str,
    *,
    color: str,
    points: tuple[tuple[int, int], ...] = (),
    exit_xy: tuple[float, float] | None = None,
    entry_xy: tuple[float, float] | None = None,
) -> None:
    cell = cells.get(edge_id)
    if cell is None:
        cell = ET.SubElement(root, "mxCell", id=edge_id, edge="1", parent="1")
        cells[edge_id] = cell
    slots = [
        "edgeStyle=orthogonalEdgeStyle",
        "rounded=1",
        "orthogonalLoop=1",
        "jettySize=auto",
        "html=1",
        f"strokeColor={color}",
        "endArrow=block",
        "endFill=1",
    ]
    if exit_xy:
        slots.extend((f"exitX={exit_xy[0]}", f"exitY={exit_xy[1]}", "exitDx=0", "exitDy=0"))
    if entry_xy:
        slots.extend((f"entryX={entry_xy[0]}", f"entryY={entry_xy[1]}", "entryDx=0", "entryDy=0"))
    cell.set("style", ";".join(slots) + ";")
    cell.set("source", source)
    cell.set("target", target)
    geom = cell.find("mxGeometry")
    if geom is None:
        geom = ET.SubElement(cell, "mxGeometry", relative="1", **{"as": "geometry"})
    else:
        geom.set("relative", "1")
    arr = geom.find("Array")
    if points:
        if arr is None:
            arr = ET.SubElement(geom, "Array", **{"as": "points"})
        for child in list(arr):
            arr.remove(child)
        for x, y in points:
            ET.SubElement(arr, "mxPoint", x=str(x), y=str(y))
    elif arr is not None:
        geom.remove(arr)


def image_style(path: Path, *, stroke: str, font_color: str) -> str:
    uri = data_uri(path).replace(";", "%3B", 1)
    return (
        "shape=image;html=1;imageAspect=0;aspect=fixed;"
        "verticalLabelPosition=bottom;verticalAlign=top;spacingTop=4;"
        f"image={uri};strokeColor={stroke};strokeWidth=1;"
        f"fontColor={font_color};fontFamily=Arial;fontSize=11;"
    )


def build_drawio() -> None:
    root = ET.Element("root")
    ET.SubElement(root, "mxCell", id="0")
    ET.SubElement(root, "mxCell", id="1", parent="0")
    cells: dict[str, ET.Element] = {}
    drawio_vertex(root, cells, "title", "<b>DINOv3–LTWM trajectory planning</b>", 24, 12, 500, 30, cell_style=text_style(size=18, bold=True))
    drawio_vertex(root, cells, "claim", "Fixed visual tokens → candidate-conditioned latent futures → conservative H4 selection", 24, 38, 650, 22, cell_style=text_style(color=MUTED, size=9, bold=True))
    drawio_vertex(root, cells, "meta", "<b>INFERENCE-TIME METHOD</b>", 600, 16, 136, 22, cell_style=text_style(color=BLUE, size=8, bold=True))

    for cell_id, value, x, y, color in (
        ("tag_a", "<b>A  VISUAL OBSERVATIONS</b>", 32, 64, BLUE),
        ("tag_b", "<b>B  CANDIDATE ACTIONS</b>", 32, 206, ORANGE),
        ("tag_c", "<b>C  ACTION-CONDITIONED LATENT ROLLOUT</b>", 212, 206, BLUE_DARK),
        ("tag_d", "<b>D  CONSERVATIVE TRAJECTORY RANKING</b>", 538, 64, PURPLE),
    ):
        drawio_vertex(root, cells, cell_id, value, x, y, 250, 22, cell_style=text_style(color=color, size=9, bold=True))

    for idx, (cell_id, label, x) in enumerate(
        (
            ("frame0", "t−3", 58),
            ("frame1", "t−2", 87),
            ("frame2", "t−1", 116),
            ("frame3", "t", 145),
        )
    ):
        drawio_vertex(
            root,
            cells,
            cell_id,
            label,
            x,
            94,
            31,
            38,
            cell_style=image_style(ASSETS[cell_id], stroke=BLUE_DARK if idx == 3 else "#9BAFC3", font_color=MUTED),
        )
    drawio_vertex(root, cells, "goal_img", "", 112, 148, 52, 34, cell_style=image_style(ASSETS["goal"], stroke=GREEN, font_color=GREEN))
    drawio_vertex(root, cells, "nomad", "<b>NoMaD</b><br>K candidates · H = 4<br>aₕ = [dx, dy, dψ]", 24, 232, 66, 54, cell_style=style(fill=ORANGE_SOFT, stroke=ORANGE, size=9))
    drawio_vertex(root, cells, "fan", "candidate proposals<br>K × H4 action sequences", 104, 250, 84, 76, cell_style=style(fill="#FFFFFF", stroke="none", size=8, rounded=False, extra="dashed=1;"))

    drawio_vertex(root, cells, "encoder", "<b>FROZEN VISUAL TOKENIZER</b><br>DINOv3-L/16 + Adapter<br><font color='#68717C'>14×14 patches · 384-D tokens</font>", 204, 68, 148, 96, cell_style=style(fill=BLUE_SOFT, stroke=BLUE_DARK, size=9, extra="container=1;pointerEvents=0;"))
    drawio_vertex(root, cells, "dino", "<b>DINOv3-L/16</b><br>patch features", 216, 98, 78, 31, parent="encoder", cell_style=style(fill="#FFFFFF", stroke=BLUE_MID, size=8))
    drawio_vertex(root, cells, "adapter", "<b>Adapter</b>", 302, 98, 36, 31, parent="encoder", cell_style=style(fill="#F2EAF2", stroke="#CDAEC7", size=8))

    drawio_vertex(root, cells, "ltwm", "<b>LiteTokenFormer</b><br>token mixing · temporal mixing<br>factorized rollout head<br><font color='#68717C'>K × 4 × 256 × 384</font>", 204, 230, 302, 126, cell_style=style(fill=BLUE_SOFT, stroke=BLUE, size=10, extra="container=1;pointerEvents=0;"))
    drawio_vertex(root, cells, "history_band", "<b>history tokens</b><br>T = 4 × 196 × 384", 218, 260, 92, 27, parent="ltwm", cell_style=style(fill="#FFFFFF", stroke=BLUE_MID, size=8))
    drawio_vertex(root, cells, "action_band", "<b>action h₁:₄</b><br>K candidates", 322, 260, 78, 27, parent="ltwm", cell_style=style(fill=ORANGE_SOFT, stroke="#D7B99F", size=8))
    drawio_vertex(root, cells, "transformer", "<b>factorized rollout</b><br>LiteTokenFormer block", 258, 300, 112, 39, parent="ltwm", cell_style=style(fill="#FFFFFF", stroke=BLUE, size=8))
    for idx, (cell_id, label, x, y) in enumerate(
        (("h1", "H1", 407, 260), ("h2", "H2", 439, 260), ("h3", "H3", 407, 298), ("h4", "H4", 439, 298))
    ):
        drawio_vertex(root, cells, cell_id, label, x, y, 25, 30, parent="ltwm", cell_style=style(fill="#FFFFFF", stroke=BLUE_DARK if idx == 3 else BLUE, size=8))

    drawio_vertex(root, cells, "d_bg", "", 530, 92, 206, 264, cell_style=style(fill="#FCF8FB", stroke="none", size=8, extra="container=1;pointerEvents=0;"))
    drawio_vertex(root, cells, "concat", "<b>concat</b><br><b>1028-D</b><br>frozen z-score", 552, 126, 34, 90, parent="d_bg", cell_style=style(fill="#FFFFFF", stroke="#BFC7D0", size=8))
    drawio_vertex(root, cells, "mlp", "<b>MLP × 3</b><br>1028 → 640 → 2", 600, 143, 72, 42, parent="d_bg", cell_style=style(fill=PURPLE_SOFT, stroke="#D6B7CC", size=8))
    drawio_vertex(root, cells, "selector", "<b>JOINT</b><br>uₖ = min(LCBᴱ/sₑ, LCBᴾ/sₚ)<br><b>argmax valid k</b><br>gate → incumbent", 683, 132, 42, 80, parent="d_bg", cell_style=style(fill="#FFFFFF", stroke=PURPLE, size=7))
    drawio_vertex(root, cells, "score_output", "<b>ATE ↓ · RTP ↑</b>", 600, 234, 72, 18, parent="d_bg", cell_style=style(fill="#FFF8FB", stroke="#D6B7CC", size=7))
    drawio_vertex(root, cells, "selected_h4", "<b>selected H4</b>", 683, 264, 42, 22, parent="d_bg", cell_style=style(fill=GREEN_SOFT, stroke="#A9CDBE", size=8))
    drawio_vertex(root, cells, "footer_note", "No training details shown", 610, 395, 126, 24, cell_style=text_style(color=MUTED, size=8))

    drawio_edge(root, cells, "a_to_encoder", "frame3", "encoder", color=BLUE, exit_xy=(1, 0.5), entry_xy=(0, 0.45))
    drawio_edge(root, cells, "goal_to_encoder", "goal_img", "encoder", color=GREEN, points=((185, 163), (185, 141)), exit_xy=(1, 0.5), entry_xy=(0, 0.76))
    drawio_edge(root, cells, "encoder_to_ltwm", "encoder", "ltwm", color=BLUE_DARK, exit_xy=(0.5, 1), entry_xy=(0.25, 0))
    drawio_edge(root, cells, "nomad_to_ltwm", "nomad", "ltwm", color=ORANGE, exit_xy=(1, 0.5), entry_xy=(0, 0.58))
    drawio_edge(root, cells, "nomad_to_d", "fan", "action_band", color=ORANGE, points=((197, 371), (507, 371), (507, 198)), exit_xy=(1, 0.8), entry_xy=(0, 0.5))
    drawio_edge(root, cells, "ltwm_to_d", "h4", "future_in", color=BLUE_DARK, points=((517, 322), (517, 168)), exit_xy=(1, 0.5), entry_xy=(0, 0.5))
    drawio_vertex(root, cells, "route_in", "route · 192-D", 530, 126, 22, 20, parent="d_bg", cell_style=text_style(color=BLUE, size=7, bold=True))
    drawio_vertex(root, cells, "future_in", "future · 796-D", 530, 156, 22, 20, parent="d_bg", cell_style=text_style(color=BLUE_DARK, size=7, bold=True))
    drawio_vertex(root, cells, "action_in", "action · 40-D", 530, 186, 22, 20, parent="d_bg", cell_style=text_style(color=ORANGE, size=7, bold=True))
    drawio_vertex(root, cells, "route_source", "", 518, 138, 2, 2, parent="1", cell_style="ellipse;opacity=0;")
    drawio_vertex(root, cells, "action_source", "", 518, 198, 2, 2, parent="1", cell_style="ellipse;opacity=0;")
    drawio_edge(root, cells, "route_to_concat", "route_source", "concat", color=BLUE, entry_xy=(0, 0.15))
    drawio_edge(root, cells, "future_to_concat", "future_in", "concat", color=BLUE_DARK, exit_xy=(1, 0.5), entry_xy=(0, 0.47))
    drawio_edge(root, cells, "action_to_concat", "action_source", "concat", color=ORANGE, entry_xy=(0, 0.82))
    drawio_edge(root, cells, "concat_to_mlp", "concat", "mlp", color=PURPLE, exit_xy=(1, 0.5), entry_xy=(0, 0.5))
    drawio_edge(root, cells, "mlp_to_selector", "mlp", "selector", color=PURPLE, exit_xy=(1, 0.5), entry_xy=(0, 0.5))
    drawio_edge(root, cells, "selector_to_score", "selector", "score_output", color=PURPLE, exit_xy=(0.5, 1), entry_xy=(0.5, 0))
    drawio_edge(root, cells, "selector_to_h4", "selector", "selected_h4", color=GREEN, exit_xy=(0.5, 1), entry_xy=(0.5, 0))

    drawio_vertex(root, cells, "m_title", "<b>DINOv3–LTWM trajectory planning</b>", 24, 12, 480, 24, cell_style=text_style(size=18, bold=True))
    drawio_vertex(root, cells, "m_claim", "visual context + candidate actions → latent rollout → safe H4 selection", 24, 39, 580, 16, cell_style=text_style(color=MUTED, size=9, bold=True))
    drawio_vertex(root, cells, "m_tag_a", "<b>A  VISUAL CONTEXT</b>", 24, 72, 180, 20, cell_style=text_style(color=BLUE, size=10, bold=True))
    drawio_vertex(root, cells, "m_tag_b", "<b>B  CANDIDATE ACTIONS</b>", 24, 197, 190, 20, cell_style=text_style(color=ORANGE, size=10, bold=True))
    drawio_vertex(root, cells, "m_tag_c", "<b>C  FROZEN ENCODER + LATENT ROLLOUT</b>", 252, 72, 255, 20, cell_style=text_style(color=BLUE_DARK, size=10, bold=True))
    drawio_vertex(root, cells, "m_tag_d", "<b>D  TRAJECTORY RANKING</b>", 540, 72, 190, 20, cell_style=text_style(color=PURPLE, size=10, bold=True))
    drawio_vertex(root, cells, "m_frame0", "t−3", 66, 94, 31, 27, cell_style=image_style(ASSETS["frame0"], stroke="#9BAFC3", font_color=MUTED))
    drawio_vertex(root, cells, "m_frame1", "t−2", 98, 94, 31, 27, cell_style=image_style(ASSETS["frame1"], stroke="#9BAFC3", font_color=MUTED))
    drawio_vertex(root, cells, "m_frame2", "t−1", 130, 94, 31, 27, cell_style=image_style(ASSETS["frame2"], stroke="#9BAFC3", font_color=MUTED))
    drawio_vertex(root, cells, "m_frame_t", "t", 162, 94, 31, 27, cell_style=image_style(ASSETS["frame3"], stroke=BLUE_DARK, font_color=MUTED))
    drawio_vertex(root, cells, "m_goal", "goal", 142, 143, 51, 31, cell_style=image_style(ASSETS["goal"], stroke=GREEN, font_color=GREEN))
    drawio_vertex(root, cells, "m_nomad", "<b>NoMaD</b><br>K trajectories · H=4", 32, 222, 68, 34, cell_style=text_style(color=ORANGE, size=9, bold=True))
    drawio_vertex(root, cells, "m_pose", "", 75, 271, 10, 10, cell_style="ellipse;html=1;fillColor=#AF754A;strokeColor=#FFFFFF;")
    drawio_vertex(root, cells, "m_end1", "", 216, 232, 2, 2, cell_style="ellipse;opacity=0;")
    drawio_vertex(root, cells, "m_end2", "", 218, 280, 2, 2, cell_style="ellipse;opacity=0;")
    drawio_vertex(root, cells, "m_end3", "", 216, 326, 2, 2, cell_style="ellipse;opacity=0;")
    drawio_vertex(root, cells, "m_encoder", "<b>DINOv3-L/16 → Adapter</b><br>frozen visual tokens", 280, 98, 180, 38, cell_style=style(fill=BLUE_SOFT, stroke=BLUE_DARK, size=9))
    drawio_vertex(root, cells, "m_ltwm", "<b>LiteTokenFormer</b><br>action-conditioned rollout", 300, 154, 140, 54, cell_style=style(fill=BLUE_SOFT, stroke=BLUE, size=10))
    drawio_vertex(root, cells, "m_h1", "H1", 300, 239, 25, 24, cell_style=style(fill="#FFFFFF", stroke=BLUE, size=8))
    drawio_vertex(root, cells, "m_h2", "H2", 338, 239, 25, 24, cell_style=style(fill="#FFFFFF", stroke=BLUE, size=8))
    drawio_vertex(root, cells, "m_h3", "H3", 376, 239, 25, 24, cell_style=style(fill="#FFFFFF", stroke=BLUE, size=8))
    drawio_vertex(root, cells, "m_h4", "H4", 414, 239, 25, 24, cell_style=style(fill="#FFFFFF", stroke=BLUE_DARK, size=8))
    drawio_vertex(root, cells, "m_formula", "ẑₜ₊₁:ₜ₊₄⁽ᵏ⁾ = Fθ(z≤t, a₁:₄⁽ᵏ⁾)", 270, 282, 200, 26, cell_style=text_style(color=BLUE_DARK, size=9, bold=True))
    drawio_vertex(root, cells, "m_mlp", "<b>MLP × 3</b><br>ATE ↓ · RTP ↑", 608, 106, 76, 60, cell_style=style(fill=PURPLE_SOFT, stroke="#D1B1C8", size=9))
    drawio_vertex(root, cells, "m_joint", "uₖ = min(LCBᴬᵀᴱ/sₑ, LCBᴿᵀᴾ/sₚ)<br><b>k* = argmax valid uₖ</b>", 548, 188, 178, 42, cell_style=text_style(color=PURPLE, size=8, bold=True))
    drawio_vertex(root, cells, "m_selected", "<b>selected H4</b>", 646, 294, 80, 28, cell_style=style(fill=GREEN_SOFT, stroke="#A9CDBE", size=9))
    drawio_vertex(root, cells, "m_sep1", "", 236, 74, 1, 260, cell_style=style(fill="#EEF1F4", stroke="#EEF1F4", rounded=False))
    drawio_vertex(root, cells, "m_sep2", "", 520, 74, 1, 260, cell_style=style(fill="#EEF1F4", stroke="#EEF1F4", rounded=False))
    drawio_edge(root, cells, "m_context_to_encoder", "m_frame_t", "m_encoder", color=BLUE, exit_xy=(1, 0.5), entry_xy=(0, 0.5))
    drawio_edge(root, cells, "m_encoder_to_ltwm", "m_encoder", "m_ltwm", color=BLUE_DARK, exit_xy=(0.5, 1), entry_xy=(0.5, 0))
    drawio_edge(root, cells, "m_nomad_to_ltwm", "m_nomad", "m_ltwm", color=ORANGE, exit_xy=(1, 0.5), entry_xy=(0, 0.7))
    drawio_edge(root, cells, "m_ltwm_to_mlp", "m_h4", "m_mlp", color=BLUE_DARK, exit_xy=(1, 0.5), entry_xy=(0, 0.5))
    drawio_edge(root, cells, "m_mlp_to_selected", "m_mlp", "m_selected", color=GREEN, exit_xy=(0.5, 1), entry_xy=(0.5, 0))
    drawio_edge(root, cells, "m_fan1", "m_pose", "m_end1", color="#CDB9A8")
    drawio_edge(root, cells, "m_fan2", "m_pose", "m_end2", color=ORANGE)
    drawio_edge(root, cells, "m_fan3", "m_pose", "m_end3", color="#CDB9A8")
    for cell in list(root):
        cell_id = cell.get("id", "")
        if cell_id not in {"0", "1"} and not cell_id.startswith("m_"):
            root.remove(cell)
    model = ET.Element(
        "mxfile",
        host="drawio",
        version="26.0",
    )
    diagram = ET.SubElement(model, "diagram", name="Academic Overview")
    graph = ET.SubElement(
        diagram,
        "mxGraphModel",
        dx="760",
        dy="460",
        grid="1",
        gridSize="10",
        guides="1",
        tooltips="1",
        connect="1",
        arrows="1",
        fold="1",
        page="1",
        pageScale="1",
        pageWidth="760",
        pageHeight="460",
        background="#FFFFFF",
    )
    graph.append(root)
    ET.indent(model, space="  ")
    OUT_DRAWIO.write_bytes(
        ET.tostring(model, encoding="UTF-8", xml_declaration=True)
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    svg = build_svg()
    cairosvg.svg2png(
        bytestring=svg.encode("utf-8"),
        write_to=str(OUT_PNG),
        output_width=2280,
        output_height=1080,
    )
    build_drawio()


if __name__ == "__main__":
    main()
