#!/usr/bin/env python3
"""Upgrade the approved framework composition to publication-grade styling."""

from __future__ import annotations

import base64
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import cairosvg


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "figures" / "figure1_compact"
BASE_SVG = OUT_DIR / "figure1_compact_wireframe_base_v29.svg"
OUT_SVG = OUT_DIR / "figure1_compact_wireframe.svg"
OUT_PNG = OUT_DIR / "figure1_compact_wireframe.png"
OUT_PDF = OUT_DIR / "figure1_compact_wireframe.pdf"
DRAWIO = OUT_DIR / "figure1_compact_wireframe.drawio"
ASSET_DIR = OUT_DIR / "assets"
VISUAL_ASSETS = {
    "frame0": ASSET_DIR / "history_t-3.jpg",
    "frame1": ASSET_DIR / "history_t-2.jpg",
    "frame2": ASSET_DIR / "history_t-1.jpg",
    "frame3": ASSET_DIR / "history_t.jpg",
    "goal": ASSET_DIR / "goal.jpg",
}

INK = "#202226"
MUTED = "#626875"
LINE = "#7A808B"
LIGHT = "#C7CCD4"
WHITE = "#FFFFFF"

# Restrained publication palette: one cool method family, one warm input
# family, one scorer family, and one success accent.
PERCEPTION = "#526F9C"
PERCEPTION_SOFT = "#EFF3F9"
PERCEPTION_MID = "#DCE6F1"
WORLD = "#536DA8"
WORLD_DARK = "#405786"
WORLD_SOFT = "#F3F6FB"
WORLD_MID = "#E5ECF7"
ACTION = "#A7784D"
ACTION_SOFT = "#FAF2E9"
SCORER = "#955A7C"
SCORER_DARK = "#75445F"
SCORER_SOFT = "#F8EEF4"
SCORER_MID = "#E9D3DF"
SUCCESS = "#2F7D62"
SUCCESS_SOFT = "#EAF5F0"
FONT = "'Arial', 'DejaVu Sans', 'Liberation Sans', sans-serif"


def image_data_uri(path: Path) -> str:
    """Return a self-contained data URI for a local raster asset."""
    if not path.is_file():
        raise FileNotFoundError(f"Figure asset not found: {path}")
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def svg_text(
    x: float,
    y: float,
    value: str,
    *,
    size: float,
    anchor: str = "middle",
    color: str = INK,
    bold: bool = False,
) -> str:
    weight = "font-weight:700;" if bold else ""
    return (
        f'<text x="{x}" y="{y}" '
        f'style="{weight}font-size:{size}px;font-family:{FONT};'
        f'text-anchor:{anchor};fill:{color}">{value}</text>'
    )


def svg_line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    color: str = LINE,
    width: float = 0.62,
) -> str:
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
        f'style="stroke:{color};stroke-width:{width};stroke-linecap:round" />'
    )


def svg_arrow(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    color: str = LINE,
    width: float = 0.68,
    head: float = 3.0,
) -> str:
    half = head * 0.52
    return (
        svg_line(x1, y1, x2, y2, color=color, width=width)
        + f'<polygon points="{x2},{y2} {x2-head},{y2-half} '
        f'{x2-head},{y2+half}" fill="{color}" />'
    )


def svg_box(
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    fill: str,
    stroke: str,
    stroke_width: float,
    radius: float,
) -> str:
    return (
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" '
        f'rx="{radius}" ry="{radius}" fill="{fill}" stroke="{stroke}" '
        f'stroke-width="{stroke_width}" />'
    )


def _set_css(style: str, key: str, value: str) -> str:
    pattern = re.compile(rf"(?<![-\w]){re.escape(key)}:\s*[^;]+")
    if pattern.search(style):
        return pattern.sub(f"{key}: {value}", style, count=1)
    suffix = "" if style.endswith(";") else ";"
    return f"{style}{suffix} {key}: {value}"


def restyle_svg_group(
    source: str,
    group_id: str,
    *,
    fill: str | None = None,
    stroke: str | None = None,
    stroke_width: float | None = None,
) -> str:
    pattern = re.compile(rf'(<g id="{re.escape(group_id)}">.*?</g>)', re.DOTALL)
    match = pattern.search(source)
    if match is None:
        raise ValueError(f"SVG group not found: {group_id}")
    block = match.group(1)
    style_match = re.search(r'style="([^"]+)"', block)
    if style_match is None:
        raise ValueError(f"SVG group has no style: {group_id}")
    style = style_match.group(1)
    if fill is not None:
        style = _set_css(style, "fill", fill)
    if stroke is not None:
        style = _set_css(style, "stroke", stroke)
    if stroke_width is not None:
        style = _set_css(style, "stroke-width", str(stroke_width))
    block = (
        block[: style_match.start(1)]
        + style
        + block[style_match.end(1) :]
    )
    return source[: match.start(1)] + block + source[match.end(1) :]


def recolor_svg_text(source: str, label: str, color: str) -> str:
    pattern = re.compile(rf"<text\b[^>]*>{re.escape(label)}</text>")
    match = pattern.search(source)
    if match is None:
        raise ValueError(f"SVG text not found: {label}")
    block = re.sub(
        r"fill:\s*#[0-9a-fA-F]{6}",
        f"fill: {color}",
        match.group(0),
        count=1,
    )
    return source[: match.start()] + block + source[match.end() :]


def restyle_base_svg(source: str) -> str:
    source = source.replace("#666a70", MUTED)

    groups = {
        # Visual context: quiet blue history, green goal.
        "patch_4": (WHITE, "#8DA4BA", 0.58),
        "patch_5": (WHITE, "#8DA4BA", 0.58),
        "patch_6": (WHITE, "#8DA4BA", 0.58),
        "patch_7": (WHITE, "#8DA4BA", 0.58),
        "patch_8": ("#F5FAF7", "#67947F", 0.65),
        "patch_9": (PERCEPTION_MID, None, None),
        "patch_10": (PERCEPTION_MID, None, None),
        "patch_11": (PERCEPTION_MID, None, None),
        "patch_12": (PERCEPTION_MID, None, None),
        "patch_13": ("#DCEDE5", None, None),
        # Frozen visual encoder.
        "patch_2": (PERCEPTION_SOFT, WORLD_DARK, 1.12),
        "patch_14": ("#E4EAF5", "#A4B1CE", 0.6),
        "patch_15": ("#F2EBF4", "#C2AEC6", 0.6),
        # NoMaD proposals.
        "patch_16": (ACTION_SOFT, ACTION, 0.72),
        "patch_17": (None, "#A98A70", 0.68),
        "patch_18": (None, "#CBB9A7", 0.58),
        "patch_19": (None, ACTION, 1.48),
        "patch_20": (None, "#CBB9A7", 0.58),
        "patch_21": (None, "#A98A70", 0.68),
        "patch_34": (ACTION, ACTION, None),
        # LTWM is the visual hero.
        "patch_3": (WORLD_SOFT, WORLD, 1.4),
        "patch_22": (WORLD_MID, "#9AABCB", 0.58),
        "patch_23": (WHITE, "#8295BC", 0.58),
        "patch_24": (WHITE, "#8295BC", 0.58),
        "patch_25": (WHITE, "#8295BC", 0.58),
        "patch_26": (WHITE, "#8295BC", 0.58),
        # Main semantic connectors.
        "patch_32": (None, "#7792AB", 0.78),
        "patch_33": (None, WORLD_DARK, 1.08),
        "patch_35": (None, WORLD, 0.82),
        "patch_36": (None, PERCEPTION, 0.82),
        "patch_37": (None, ACTION, 0.82),
        "patch_38": (None, ACTION, 0.82),
        "patch_39": (None, WORLD, 0.82),
        "patch_50": (None, PERCEPTION, 1.12),
        "patch_51": (None, ACTION, 1.08),
    }
    for group_id, (fill, stroke, width) in groups.items():
        source = restyle_svg_group(
            source,
            group_id,
            fill=fill,
            stroke=stroke,
            stroke_width=width,
        )

    for index in range(9, 14):
        source = restyle_svg_group(
            source,
            f"line2d_{index}",
            stroke="#AFC0CF" if index < 13 else "#9DBBAE",
            stroke_width=0.52,
        )
    for index in range(14, 38):
        source = restyle_svg_group(
            source,
            f"line2d_{index}",
            stroke="#BDC8DC",
            stroke_width=0.23,
        )

    for label, color in (
        ("A  VISUAL CONTEXT", PERCEPTION),
        ("FROZEN VISUAL ENCODER", WORLD_DARK),
        ("B  CANDIDATE ACTION PROPOSALS", ACTION),
        ("C  LITETOKENFORMER WORLD MODEL", WORLD_DARK),
    ):
        source = recolor_svg_text(source, label, color)
    return source


def svg_image_tile(
    tile_id: str,
    path: Path,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    stroke: str,
    stroke_width: float,
    radius: float = 1.2,
) -> str:
    clip_id = f"{tile_id}_clip"
    uri = image_data_uri(path)
    return "\n".join(
        [
            f'<clipPath id="{clip_id}">',
            f'<rect x="{x}" y="{y}" width="{width}" height="{height}" '
            f'rx="{radius}" ry="{radius}" />',
            "</clipPath>",
            f'<image x="{x}" y="{y}" width="{width}" height="{height}" '
            f'preserveAspectRatio="xMidYMid slice" '
            f'clip-path="url(#{clip_id})" xlink:href="{uri}" />',
            f'<rect x="{x}" y="{y}" width="{width}" height="{height}" '
            f'rx="{radius}" ry="{radius}" fill="{PERCEPTION}" '
            'fill-opacity="0.025" stroke="none" />',
            f'<rect x="{x}" y="{y}" width="{width}" height="{height}" '
            f'rx="{radius}" ry="{radius}" fill="none" stroke="{stroke}" '
            f'stroke-width="{stroke_width}" />',
        ]
    )


def build_visual_context_overlay() -> str:
    """Replace schematic placeholders with a coherent real RGB sequence."""
    parts = [
        '<g id="visual_context_raster_overlay">',
        svg_text(
            44.45,
            58.55,
            "RGB OBSERVATIONS",
            size=2.7,
            anchor="start",
            color=PERCEPTION,
            bold=True,
        ),
    ]
    frames = (
        ("frame0", 44.450842, 63.3816, 32.530636, 30.9168, "#9AAEC1", 0.56),
        ("frame1", 72.753178, 63.3816, 32.530636, 30.9168, "#9AAEC1", 0.56),
        ("frame2", 101.055514, 63.3816, 32.530636, 30.9168, "#9AAEC1", 0.56),
        ("frame3", 129.35785, 63.3816, 32.530636, 30.9168, WORLD_DARK, 0.92),
    )
    for tile_id, x, y, width, height, stroke, stroke_width in frames:
        parts.append(
            svg_image_tile(
                tile_id,
                VISUAL_ASSETS[tile_id],
                x,
                y,
                width,
                height,
                stroke=stroke,
                stroke_width=stroke_width,
            )
        )
    parts.append(
        svg_image_tile(
            "goal",
            VISUAL_ASSETS["goal"],
            102.419482,
            112.6296,
            44.465356,
            27.4968,
            stroke=SUCCESS,
            stroke_width=0.82,
        )
    )
    parts.append("</g>")
    return "\n".join(parts)


def build_publication_overlay() -> str:
    return "\n".join(
        [
            '<g id="publication_grade_overlay">',
            '<rect x="0" y="0" width="518.688" height="31.0" fill="#ffffff" />',
            svg_text(
                17.24,
                19.5,
                "DINOv3–LTWM trajectory planning",
                size=8.4,
                anchor="start",
                color=INK,
                bold=True,
            ),
            svg_text(
                501.45,
                18.9,
                "INFERENCE-TIME FRAMEWORK",
                size=3.35,
                anchor="end",
                color=PERCEPTION,
                bold=True,
            ),
            svg_line(17.24, 27.25, 501.45, 27.25, color="#DCE2EA", width=0.55),
            '<rect x="17.05" y="39.65" width="1.65" height="9.0" rx="0.82" fill="'
            + PERCEPTION
            + '" />',
            '<rect x="17.05" y="163.75" width="1.65" height="9.0" rx="0.82" fill="'
            + ACTION
            + '" />',
            '<rect x="181.15" y="137.15" width="1.65" height="9.0" rx="0.82" fill="'
            + WORLD
            + '" />',
            '<rect x="0" y="296.0" width="518.688" height="39.52" fill="#ffffff" />',
            svg_line(17.24, 300.2, 501.45, 300.2, color="#E4E7EC", width=0.48),
            svg_text(
                17.24,
                313.15,
                "Fixed visual tokens  →  candidate-conditioned latent futures  →  conservative H4 selection",
                size=3.45,
                anchor="start",
                color=MUTED,
                bold=True,
            ),
            "</g>",
        ]
    )


def build_d_overlay() -> str:
    parts = [
        '<g id="panel_d_typography_overlay">',
        '<rect x="353.55" y="109.8" width="161.55" height="93.3" fill="#fcfafb" />',
        '<rect x="357.7" y="112.25" width="1.65" height="8.8" '
        'rx="0.82" fill="' + SCORER + '" />',
        svg_text(
            361.65,
            118.75,
            "D  TRAJECTORY-QUALITY SCORING",
            size=5.65,
            anchor="start",
            color=SCORER_DARK,
            bold=True,
        ),
        svg_line(361.65, 123.3, 510.0, 123.3, color="#eadce4", width=0.58),
    ]

    input_lines = (
        (139.03, 133.30, "Route · 192-D", PERCEPTION),
        (160.92, 155.15, "Future · 796-D", WORLD),
        (182.81, 177.05, "Action · 40-D", ACTION),
    )
    for line_y, label_y, label, color in input_lines:
        parts.append(
            svg_arrow(
                353.55,
                line_y,
                380.0,
                line_y,
                color=color,
                width=0.78,
            )
        )
        parts.append(
            svg_text(
                366.75,
                label_y,
                label,
                size=2.9,
                color=color,
                bold=True,
            )
        )

    # Compact Concat block.
    parts.append(
        svg_box(
            380.0,
            130.0,
            20.8,
            63.0,
            fill=WHITE,
            stroke=LIGHT,
            stroke_width=0.58,
            radius=2.35,
        )
    )
    parts.extend(
        [
            svg_text(390.4, 153.2, "Concat", size=3.55, bold=True),
            svg_text(390.4, 164.0, "1028-D", size=3.15, bold=True),
            svg_text(390.4, 176.5, "frozen", size=2.4, color=MUTED),
            svg_text(390.4, 179.4, "z-score", size=2.4, color=MUTED),
        ]
    )

    # Fan-out bus and three compact MLPs.
    parts.append(svg_line(400.8, 160.92, 405.1, 160.92, width=0.56))
    parts.append(svg_line(405.1, 137.75, 405.1, 184.15, width=0.56))

    mlps = (
        (130.0, 137.75, 1701),
        (153.17, 160.92, 1702),
        (176.40, 184.15, 1703),
    )
    for top, center, seed in mlps:
        parts.append(svg_arrow(405.1, center, 411.0, center, width=0.56))
        parts.append(
            svg_box(
                411.0,
                top,
                37.5,
                15.5,
                fill=SCORER_SOFT,
                stroke=SCORER_MID,
                stroke_width=0.5,
                radius=1.85,
            )
        )
        parts.append(
            svg_text(
                413.0,
                top + 6.2,
                f"MLP · {seed}",
                size=3.25,
                anchor="start",
                bold=True,
            )
        )
        parts.append(
            svg_text(
                413.0,
                top + 12.7,
                "1028 → 3×640 → 2",
                size=2.15,
                anchor="start",
                color=MUTED,
            )
        )
        parts.append(svg_line(448.5, center, 451.0, center, width=0.54))

    parts.append(svg_line(451.0, 137.75, 451.0, 184.15, width=0.56))

    # One semantic output wire. Its caption sits in a detached information
    # chip so the reduced full-figure view never reads the label as a wire.
    parts.append(
        svg_arrow(
            451.0,
            160.92,
            471.5,
            160.92,
            color=SCORER,
            width=0.72,
        )
    )
    parts.append(
        svg_box(
            452.2,
            144.35,
            18.2,
            7.55,
            fill="#FFF8FB",
            stroke=SCORER_MID,
            stroke_width=0.44,
            radius=1.85,
        )
    )
    parts.append(
        svg_text(
            461.3,
            149.35,
            "3× [ATE, RTP]",
            size=1.9,
            color=SCORER_DARK,
            bold=True,
        )
    )

    # Tight selector with the exact safe-promotion computation distilled into
    # two cards. e/p denote the ATE/RTP gain branches and s their scales.
    parts.append(
        svg_box(
            471.5,
            129.2,
            26.5,
            58.3,
            fill="#FFFDFE",
            stroke=SCORER,
            stroke_width=0.72,
            radius=2.65,
        )
    )
    parts.extend(
        [
            svg_text(
                484.75,
                136.55,
                "Joint selector",
                size=3.0,
                color=SCORER_DARK,
                bold=True,
            ),
            svg_text(
                484.75,
                143.15,
                "ℓᵐ = μ(Δᵐ) − βσ(Δᵐ)",
                size=1.72,
                color=MUTED,
            ),
            svg_box(
                473.65,
                147.0,
                22.2,
                11.0,
                fill=WHITE,
                stroke=SCORER_MID,
                stroke_width=0.44,
                radius=1.45,
            ),
            svg_text(
                484.75,
                151.05,
                "conservative utility",
                size=1.46,
                color=MUTED,
            ),
            svg_text(
                484.75,
                155.8,
                "uₖ = min(ℓᵉₖ/sₑ, ℓᵖₖ/sₚ)",
                size=1.62,
                color=SCORER_DARK,
                bold=True,
            ),
            svg_box(
                473.65,
                163.4,
                22.2,
                16.0,
                fill="#F4F8F6",
                stroke="#B8D2C7",
                stroke_width=0.44,
                radius=1.45,
            ),
            svg_text(
                484.75,
                167.45,
                "safe promotion",
                size=1.46,
                color=SUCCESS,
            ),
            svg_text(
                484.75,
                172.15,
                "k* = argmaxₖ∈𝒱 uₖ",
                size=1.72,
                color=INK,
                bold=True,
            ),
            svg_text(
                484.75,
                176.85,
                "gate fail → incumbent",
                size=1.42,
                color=MUTED,
            ),
        ]
    )

    parts.append(
        svg_arrow(
            498.0,
            160.92,
            510.2,
            160.92,
            color=SUCCESS,
            width=0.92,
        )
    )
    parts.append(
        svg_box(
            498.45,
            164.25,
            12.0,
            7.3,
            fill=SUCCESS_SOFT,
            stroke="#A9CABB",
            stroke_width=0.42,
            radius=1.75,
        )
    )
    parts.append(
        svg_text(
            504.45,
            169.15,
            "selected H4",
            size=1.9,
            color=SUCCESS,
            bold=True,
        )
    )
    parts.append("</g>")
    return "\n".join(parts)


def polish_svg() -> None:
    source = restyle_base_svg(BASE_SVG.read_text(encoding="utf-8"))
    overlays = "\n".join(
        (
            build_visual_context_overlay(),
            build_publication_overlay(),
            build_d_overlay(),
        )
    )
    if "</svg>" not in source:
        raise ValueError(f"Invalid SVG: {BASE_SVG}")
    polished = source.replace("</svg>", f"{overlays}\n</svg>", 1)
    OUT_SVG.write_text(polished, encoding="utf-8")

    svg_bytes = polished.encode("utf-8")
    cairosvg.svg2png(
        bytestring=svg_bytes,
        write_to=str(OUT_PNG),
        output_width=1873,
        output_height=1211,
    )
    cairosvg.svg2pdf(bytestring=svg_bytes, write_to=str(OUT_PDF))


def drawio_style(
    *,
    fill: str = "#F5F6F6",
    stroke: str = "#D7D9DC",
    size: int = 11,
) -> str:
    return (
        "rounded=1;arcSize=10;whiteSpace=wrap;html=1;"
        f"fillColor={fill};strokeColor={stroke};"
        f"fontColor=#202226;fontFamily=Arial;fontSize={size};"
    )


def drawio_image_style(
    path: Path,
    *,
    stroke: str,
    font_color: str,
) -> str:
    # Escape the data-URI separator so it cannot be parsed as a style delimiter.
    uri = image_data_uri(path).replace(";", "%3B", 1)
    return (
        "shape=image;html=1;imageAspect=0;aspect=fixed;"
        "verticalLabelPosition=bottom;verticalAlign=top;spacingTop=4;"
        f"image={uri};strokeColor={stroke};strokeWidth=1;"
        f"fontColor={font_color};fontFamily=Arial;fontSize=11;"
    )


def drawio_text_style(
    *,
    size: int,
    align: str = "center",
    color: str = "#202226",
    bold: bool = False,
) -> str:
    font_style = "fontStyle=1;" if bold else ""
    return (
        "text;html=1;strokeColor=none;fillColor=none;"
        f"align={align};verticalAlign=middle;fontFamily=Arial;"
        f"fontColor={color};fontSize={size};{font_style}"
    )


def update_drawio_style(cell: ET.Element, **values: str) -> None:
    tokens = [token for token in cell.get("style", "").split(";") if token]
    positions = {
        token.split("=", 1)[0]: index
        for index, token in enumerate(tokens)
        if "=" in token
    }
    for key, value in values.items():
        token = f"{key}={value}"
        if key in positions:
            tokens[positions[key]] = token
        else:
            positions[key] = len(tokens)
            tokens.append(token)
    cell.set("style", ";".join(tokens) + ";")
def set_geometry(
    cell: ET.Element,
    x: int,
    y: int,
    width: int,
    height: int,
) -> None:
    geometry = cell.find("mxGeometry")
    if geometry is None:
        geometry = ET.SubElement(cell, "mxGeometry", **{"as": "geometry"})
    geometry.set("x", str(x))
    geometry.set("y", str(y))
    geometry.set("width", str(width))
    geometry.set("height", str(height))


def upsert_vertex(
    root: ET.Element,
    cells: dict[str, ET.Element],
    cell_id: str,
    value: str,
    x: int,
    y: int,
    width: int,
    height: int,
    *,
    parent: str,
    style: str,
) -> ET.Element:
    cell = cells.get(cell_id)
    if cell is None:
        cell = ET.SubElement(
            root,
            "mxCell",
            id=cell_id,
            vertex="1",
            parent=parent,
        )
        cells[cell_id] = cell
    cell.set("value", value)
    cell.set("style", style)
    cell.set("vertex", "1")
    cell.set("parent", parent)
    set_geometry(cell, x, y, width, height)
    return cell


def apply_drawio_publication_style(
    root: ET.Element,
    cells: dict[str, ET.Element],
) -> None:
    title = cells["title"]
    title.set("value", "<b>DINOv3–LTWM trajectory planning</b>")
    title.set("style", drawio_text_style(size=22, align="left", bold=True))
    set_geometry(title, 40, 20, 760, 55)

    upsert_vertex(
        root,
        cells,
        "publication_note",
        "<b>INFERENCE-TIME FRAMEWORK</b>",
        1160,
        28,
        350,
        30,
        parent="1",
        style=drawio_text_style(
            size=11,
            align="right",
            color=PERCEPTION,
            bold=True,
        ),
    )
    upsert_vertex(
        root,
        cells,
        "publication_footer",
        "<b>Fixed visual tokens  →  candidate-conditioned latent futures  "
        "→  conservative H4 selection</b>",
        40,
        875,
        1470,
        35,
        parent="1",
        style=drawio_text_style(size=11, align="left", color=MUTED),
    )

    cells["a_label"].set(
        "style",
        drawio_text_style(size=14, align="left", color=PERCEPTION, bold=True),
    )
    cells["b_label"].set(
        "style",
        drawio_text_style(size=14, align="left", color=ACTION, bold=True),
    )
    for cell_id in ("history_label", "goal_label", "candidate_caption"):
        update_drawio_style(cells[cell_id], fontColor=MUTED)

    for cell_id in ("frame0", "frame1", "frame2", "frame3"):
        cells[cell_id].set(
            "style",
            drawio_image_style(
                VISUAL_ASSETS[cell_id],
                stroke=WORLD_DARK if cell_id == "frame3" else "#8DA4BA",
                font_color=MUTED,
            ),
        )
    cells["goal"].set("value", "")
    cells["goal"].set(
        "style",
        drawio_image_style(
            VISUAL_ASSETS["goal"],
            stroke=SUCCESS,
            font_color=SUCCESS,
        ),
    )

    update_drawio_style(
        cells["encoder"],
        fillColor=PERCEPTION_SOFT,
        strokeColor=WORLD_DARK,
        fontColor=WORLD_DARK,
        strokeWidth="2",
    )
    update_drawio_style(
        cells["dino"],
        fillColor="#E4EAF5",
        strokeColor="#A4B1CE",
    )
    update_drawio_style(
        cells["adapter"],
        fillColor="#F2EBF4",
        strokeColor="#C2AEC6",
    )

    update_drawio_style(
        cells["nomad"],
        fillColor=ACTION_SOFT,
        strokeColor=ACTION,
        strokeWidth="1.5",
    )
    update_drawio_style(
        cells["candidate_start"],
        fillColor=ACTION,
        strokeColor=ACTION,
    )
    update_drawio_style(
        cells["nomad_to_bundle"],
        strokeColor=ACTION,
        fontColor=ACTION,
    )
    for index in range(5):
        update_drawio_style(
            cells[f"candidate_curve{index}"],
            strokeColor=ACTION if index == 2 else "#BCA58F",
            strokeWidth="2" if index == 2 else "1.1",
        )

    update_drawio_style(
        cells["ltwm"],
        fillColor=WORLD_SOFT,
        strokeColor=WORLD,
        fontColor=WORLD_DARK,
        strokeWidth="2.5",
    )
    update_drawio_style(
        cells["rollout"],
        fillColor=WORLD_MID,
        strokeColor="#9AABCB",
    )
    for cell_id in ("h1", "h2", "h3", "h4"):
        update_drawio_style(
            cells[cell_id],
            fillColor=WHITE,
            strokeColor="#8295BC",
            fontColor=WORLD_DARK,
        )

    edge_colors = {
        "e1": PERCEPTION,
        "e2": "#7792AB",
        "e3": WORLD_DARK,
        "e4": ACTION,
        "e5": WORLD,
        "e6": PERCEPTION,
        "e7": ACTION,
    }
    for cell_id, color in edge_colors.items():
        update_drawio_style(cells[cell_id], strokeColor=color)

    update_drawio_style(
        cells["scorer"],
        fillColor="#FCFAFB",
        strokeColor="none",
        container="1",
        pointerEvents="0",
    )


def polish_drawio() -> None:
    tree = ET.parse(DRAWIO)
    root = tree.getroot().find("./diagram/mxGraphModel/root")
    if root is None:
        raise ValueError(f"Invalid draw.io XML: {DRAWIO}")
    cells = {cell.get("id", ""): cell for cell in root.findall("mxCell")}
    apply_drawio_publication_style(root, cells)

    for edge_id in ("e6", "e5", "e7", "quality_to_selector"):
        cells[edge_id].set("value", "")

    # Give the Future wire an orthogonal lead-in before the detached label lane.
    future_geometry = cells["e5"].find("mxGeometry")
    if future_geometry is None:
        raise ValueError("Edge e5 has no geometry")
    future_points = future_geometry.find("Array")
    if future_points is None:
        future_points = ET.SubElement(future_geometry, "Array", **{"as": "points"})
    for point in list(future_points):
        future_points.remove(point)
    for x, y in ((970, 504), (970, 540), (1095, 540)):
        ET.SubElement(future_points, "mxPoint", x=str(x), y=str(y))

    title = cells["d_title"]
    title.set(
        "style",
        drawio_text_style(
            size=14,
            align="left",
            color=SCORER_DARK,
            bold=True,
        ),
    )
    set_geometry(title, 0, 8, 410, 32)
    update_drawio_style(
        cells["d_rule"],
        fillColor=SCORER_MID,
        strokeColor=SCORER_MID,
    )

    for cell_id, label, y, color in (
        ("route_label", "Route · 192-D", 96, PERCEPTION),
        ("future_label", "Future · 796-D", 156, WORLD),
        ("action_label", "Action · 40-D", 216, ACTION),
    ):
        upsert_vertex(
            root,
            cells,
            cell_id,
            f"<b>{label}</b>",
            0,
            y,
            100,
            26,
            parent="scorer",
            style=drawio_text_style(
                size=11,
                color=color,
                bold=True,
            ),
        )

    concat = cells["concat_norm"]
    concat.set(
        "value",
        '<b>Concat</b><br><b>1028-D</b><br>'
        '<font color="#666A70">frozen z-score</font>',
    )
    concat.set(
        "style",
        drawio_style(fill=WHITE, stroke="#C7CCD4", size=11),
    )
    set_geometry(concat, 105, 90, 65, 200)

    fanout = cells["fanout_bus"]
    fanout.set(
        "style",
        "rounded=1;fillColor="
        + SCORER
        + ";strokeColor="
        + SCORER
        + ";",
    )
    set_geometry(fanout, 185, 108, 2, 164)

    for seed, y in ((1701, 81), (1702, 163), (1703, 245)):
        mlp = cells[f"mlp_{seed}"]
        mlp.set(
            "value",
            f"<b>MLP · {seed}</b><br>"
            '<font color="#626875">1028 → 3×640 → 2</font>',
        )
        mlp.set(
            "style",
            drawio_style(fill=SCORER_SOFT, stroke=SCORER_MID, size=11),
        )
        set_geometry(mlp, 205, y, 130, 54)

    output_bus = cells["output_bus"]
    output_bus.set(
        "style",
        "rounded=1;fillColor="
        + SCORER
        + ";strokeColor="
        + SCORER
        + ";",
    )
    set_geometry(output_bus, 370, 108, 2, 164)

    selector = cells["joint_selector"]
    selector.set("value", "<b>Joint selector</b>")
    selector.set(
        "style",
        drawio_style(fill="#FFFDFE", stroke=SCORER, size=11)
        + "verticalAlign=top;spacingTop=8;fontColor="
        + SCORER_DARK
        + ";",
    )
    set_geometry(selector, 425, 90, 86, 200)

    upsert_vertex(
        root,
        cells,
        "selector_lcb",
        "ℓᵐ = μ(Δᵐ) − βσ(Δᵐ)",
        6,
        31,
        74,
        24,
        parent="joint_selector",
        style=drawio_text_style(size=8, color=MUTED),
    )
    upsert_vertex(
        root,
        cells,
        "selector_utility",
        '<font color="#626875">conservative utility</font><br>'
        '<b>uₖ = min(ℓᵉₖ/sₑ, ℓᵖₖ/sₚ)</b>',
        7,
        59,
        72,
        39,
        parent="joint_selector",
        style=drawio_style(fill=WHITE, stroke=SCORER_MID, size=8),
    )
    upsert_vertex(
        root,
        cells,
        "selector_gate",
        '<font color="#2F7D62">safe promotion</font><br>'
        '<b>k* = argmax<sub>k∈𝒱</sub> uₖ</b><br>'
        '<font color="#626875">gate fail → incumbent</font>',
        7,
        117,
        72,
        56,
        parent="joint_selector",
        style=drawio_style(
            fill="#F4F8F6",
            stroke="#B8D2C7",
            size=8,
        ),
    )

    upsert_vertex(
        root,
        cells,
        "quality_label",
        '<b>3× [ATE, RTP]</b>',
        375,
        116,
        47,
        24,
        parent="scorer",
        style=drawio_style(
            fill="#FFF8FB",
            stroke=SCORER_MID,
            size=9,
        )
        + "fontColor="
        + SCORER_DARK
        + ";",
    )
    upsert_vertex(
        root,
        cells,
        "selected_h4",
        "<b>selected H4</b>",
        513,
        201,
        42,
        24,
        parent="scorer",
        style=drawio_style(
            fill=SUCCESS_SOFT,
            stroke="#A9CABB",
            size=9,
        )
        + "fontColor="
        + SUCCESS
        + ";",
    )
    upsert_vertex(
        root,
        cells,
        "selected_sink",
        "",
        552,
        189,
        2,
        2,
        parent="scorer",
        style=(
            "ellipse;html=1;fillColor=none;strokeColor=none;"
            "opacity=0;pointerEvents=0;"
        ),
    )
    cells["selector_to_selected"].set("target", "selected_sink")

    for edge_id in (
        "concat_to_fanout",
        "fanout_to_mlp_1701",
        "mlp_1701_to_output",
        "fanout_to_mlp_1702",
        "mlp_1702_to_output",
        "fanout_to_mlp_1703",
        "mlp_1703_to_output",
        "quality_to_selector",
    ):
        update_drawio_style(cells[edge_id], strokeColor=SCORER)
    update_drawio_style(
        cells["selector_to_selected"],
        strokeColor=SUCCESS,
        strokeWidth="1.6",
    )

    cem = cells["cem_note"]
    cem.set("value", "")
    cem.set("style", drawio_text_style(size=1, color="#FFFFFF"))
    set_geometry(cem, 423, 310, 1, 1)

    ET.indent(tree, space="  ")
    tree.write(DRAWIO, encoding="UTF-8", xml_declaration=True)


def main() -> None:
    polish_svg()
    polish_drawio()


if __name__ == "__main__":
    main()
