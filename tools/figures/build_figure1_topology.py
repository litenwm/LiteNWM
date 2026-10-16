#!/usr/bin/env python3
"""Build the topology-preserving, simplified Figure 1.

Unlike a generic left-to-right pipeline, this layout keeps the topology of the
original figure: visual context is upper-left, candidate actions are
lower-left, the frozen tokenizer is upper-middle, the candidate-conditioned
LTWM is lower-middle, and the scorer is a right-side fan-in hub.
"""

from __future__ import annotations

from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import cairosvg


THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
import build_figure1_topconf_simplified as base  # noqa: E402


ROOT = base.ROOT
OUT_DIR = ROOT / "figures" / "figure1_topology_simplified"
BASE = OUT_DIR / "figure1_topology_simplified"
OUT_SVG = BASE.with_suffix(".svg")
OUT_PNG = BASE.with_suffix(".png")
OUT_PDF = BASE.with_suffix(".pdf")
OUT_DRAWIO = BASE.with_suffix(".drawio")

W, H = 1320.0, 760.0

# Reuse the restrained palette and SVG primitives from the previous version.
INK = base.INK
MUTED = base.MUTED
HAIR = base.HAIR
PANEL = base.PANEL
WHITE = base.WHITE
BLUE = base.BLUE
BLUE_MID = base.BLUE_MID
BLUE_SOFT = base.BLUE_SOFT
ORANGE = base.ORANGE
ORANGE_MID = base.ORANGE_MID
ORANGE_SOFT = base.ORANGE_SOFT
PURPLE = base.PURPLE
PURPLE_MID = base.PURPLE_MID
PURPLE_SOFT = base.PURPLE_SOFT
GREEN = base.GREEN
GREEN_MID = base.GREEN_MID
GREEN_SOFT = base.GREEN_SOFT
FONT = base.FONT
ASSETS = base.ASSETS


def arrow_defs() -> list[str]:
    parts: list[str] = []
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
    return parts


def section_tag(x: float, y: float, number: str, title: str, color: str) -> str:
    return "\n".join(
        (
            f'<rect x="{x:g}" y="{y - 11:g}" width="4" height="22" rx="2" fill="{color}" />',
            base.text(x + 14, y + 2, f"{number}  {title}", size=14, color=color, weight=700),
        )
    )


def card(
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    fill: str,
    stroke: str,
    sw: float = 1.2,
    rx: float = 10.0,
) -> str:
    return base.rect(x, y, width, height, fill=fill, stroke=stroke, sw=sw, rx=rx)


def build_svg() -> str:
    parts: list[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'viewBox="0 0 {W:g} {H:g}" width="{W:g}pt" height="{H:g}pt">',
        "<defs>",
        *arrow_defs(),
        "</defs>",
        base.rect(0, 0, W, H, fill=WHITE, stroke="none", rx=0),
        base.text(
            34,
            34,
            "DINOv3–LTWM trajectory planning",
            size=23,
            color=INK,
            weight=700,
        ),
        base.text(
            34,
            59,
            "Visual context + candidate actions → candidate-conditioned latent futures → conservative H4 selection",
            size=11,
            color=MUTED,
            weight=600,
        ),
        base.text(
            1286,
            34,
            "INFERENCE-TIME FRAMEWORK",
            size=9,
            color=BLUE,
            weight=700,
            anchor="end",
            letter=0.8,
        ),
        base.line(34, 76, 1286, 76, color=HAIR, sw=1.0),
    ]

    # The large regions intentionally preserve the original spatial topology.
    parts.extend(
        (
            card(34, 106, 330, 246, fill=PANEL, stroke="#C9D4E1", sw=1.2),
            card(34, 394, 330, 252, fill=ORANGE_SOFT, stroke=ORANGE_MID, sw=1.2),
            card(410, 106, 350, 190, fill=BLUE_SOFT, stroke=BLUE_MID, sw=1.3),
            card(410, 344, 410, 302, fill="#F2F5FB", stroke=BLUE, sw=1.7),
            card(856, 164, 364, 430, fill=PURPLE_SOFT, stroke=PURPLE_MID, sw=1.3),
            section_tag(52, 131, "A", "VISUAL CONTEXT", BLUE),
            section_tag(52, 419, "B", "CANDIDATE ACTIONS", ORANGE),
            section_tag(428, 131, "C", "FROZEN VISUAL TOKENIZER", BLUE),
            section_tag(428, 369, "D", "CANDIDATE-CONDITIONED LTWM", BLUE),
            section_tag(874, 189, "E", "TRAJECTORY-QUALITY SCORING", PURPLE),
        )
    )

    # A — visual context.
    parts.extend(
        (
            base.text(58, 161, "RGB history", size=10, color=MUTED, weight=600),
            base.image_tile("t-3", 82, 151, 54, 31, stroke="#AEBBCB", sw=0.8),
            base.image_tile("t-2", 142, 151, 54, 31, stroke="#AEBBCB", sw=0.8),
            base.image_tile("t-1", 202, 151, 54, 31, stroke="#AEBBCB", sw=0.8),
            base.image_tile("t", 262, 151, 54, 31, stroke=BLUE, sw=1.4),
            base.text(109, 196, "t−3", size=8.5, color=MUTED, anchor="middle"),
            base.text(169, 196, "t−2", size=8.5, color=MUTED, anchor="middle"),
            base.text(229, 196, "t−1", size=8.5, color=MUTED, anchor="middle"),
            base.text(289, 196, "t", size=8.5, color=BLUE, weight=700, anchor="middle"),
            base.text(82, 240, "4-frame history", size=10, color=BLUE, weight=700),
            base.text(82, 269, "+ goal frame", size=10, color=GREEN, weight=700),
            base.image_tile("goal", 246, 247, 70, 42, stroke=GREEN, sw=1.4),
        )
    )
    # Merge history and goal into the tokenizer.
    parts.extend(
        (
            base.arrow_path("M 364 215 H 390 V 186 H 410", color=BLUE, sw=2.2),
            base.arrow_path("M 316 268 H 390 V 244 H 410", color=GREEN, sw=1.8),
            base.text(374, 177, "history", size=8.5, color=BLUE, weight=600),
            base.text(374, 255, "goal", size=8.5, color=GREEN, weight=600),
        )
    )

    # B — candidate action proposals.
    parts.extend(
        (
            card(58, 466, 100, 84, fill=WHITE, stroke=ORANGE, sw=1.1, rx=8),
            base.text(108, 496, "NoMaD", size=15, color=INK, weight=700, anchor="middle"),
            base.text(108, 520, "external proposal", size=8.5, color=MUTED, anchor="middle"),
            base.trajectory_fan(184, 456, 140, 112),
            base.text(190, 595, "K candidate trajectories", size=10.5, color=ORANGE, weight=700),
            base.text(190, 616, "H4 actions  [dx, dy, dψ]", size=9, color=MUTED),
            base.arrow_path("M 158 508 H 178", color=ORANGE, sw=2.0),
        )
    )

    # C — frozen tokenizer.
    parts.extend(
        (
            base.text(433, 161, "LOCKED", size=9, color=MUTED, weight=700, letter=0.5),
            card(438, 181, 142, 64, fill=WHITE, stroke=BLUE_MID, sw=1.0, rx=8),
            card(596, 181, 122, 64, fill="#F7EEF6", stroke=PURPLE_MID, sw=1.0, rx=8),
            base.text(509, 208, "DINOv3-L/16", size=13, color=INK, weight=700, anchor="middle"),
            base.text(509, 228, "14×14×1024", size=9, color=MUTED, anchor="middle"),
            base.text(657, 208, "Adapter", size=13, color=INK, weight=700, anchor="middle"),
            base.text(657, 228, "→ 16×16×384", size=9, color=MUTED, anchor="middle"),
            base.arrow_path("M 580 213 H 596", color=BLUE, sw=1.8),
            base.text(585, 273, "z history + z goal", size=10, color=BLUE, weight=700, anchor="middle"),
        )
    )

    # D — LTWM.
    parts.extend(
        (
            base.text(434, 405, "history tokens", size=10.5, color=BLUE, weight=700),
            base.text(680, 405, "candidate actions", size=10.5, color=ORANGE, weight=700, anchor="end"),
            card(442, 431, 112, 58, fill=WHITE, stroke=BLUE_MID, sw=1.0, rx=8),
            card(574, 431, 112, 58, fill=ORANGE_SOFT, stroke=ORANGE_MID, sw=1.0, rx=8),
            base.text(498, 459, "History encoder", size=10.5, color=INK, weight=700, anchor="middle"),
            base.text(630, 459, "Action prefix", size=10.5, color=INK, weight=700, anchor="middle"),
            base.arrow_path("M 554 460 H 574", color=MUTED, sw=1.5),
            card(514, 516, 174, 46, fill=BLUE, stroke=BLUE, sw=1.0, rx=8),
            base.text(601, 544, "factorized token rollout", size=11, color=WHITE, weight=700, anchor="middle"),
            base.arrow_path("M 498 489 V 538 H 514", color=BLUE, sw=1.6),
            base.arrow_path("M 630 489 V 516", color=ORANGE, sw=1.6),
            base.text(727, 457, "H1", size=9, color=MUTED, weight=700, anchor="middle"),
            base.text(727, 504, "H2", size=9, color=MUTED, weight=700, anchor="middle"),
            base.text(727, 551, "H3", size=9, color=MUTED, weight=700, anchor="middle"),
            base.text(727, 598, "H4", size=9, color=BLUE, weight=700, anchor="middle"),
            base.token_grid(700, 425, 52, 34, label="", emphasized=False),
            base.token_grid(700, 472, 52, 34, label="", emphasized=False),
            base.token_grid(700, 519, 52, 34, label="", emphasized=False),
            base.token_grid(700, 566, 52, 34, label="", emphasized=True),
            base.text(598, 603, "future latents  [K × 4 × 256 × 384]", size=10.5, color=BLUE, weight=700, anchor="middle"),
        )
    )
    # Encoder output down to LTWM; actions rightward to LTWM.
    parts.extend(
        (
            base.arrow_path("M 565 296 V 344 H 498 V 431", color=BLUE, sw=2.2),
            base.arrow_path("M 364 510 H 406 V 460 H 442", color=ORANGE, sw=2.2),
            base.text(379, 334, "history tokens", size=8.5, color=BLUE, weight=600),
            base.text(372, 500, "actions", size=8.5, color=ORANGE, weight=600),
        )
    )

    # E — scorer fan-in hub, matching the original right-side topology.
    parts.extend(
        (
            base.text(878, 225, "one feature vector per candidate", size=9.5, color=MUTED, weight=600),
            card(880, 250, 108, 190, fill=WHITE, stroke="#BFC8D3", sw=1.1, rx=8),
            base.text(934, 281, "Concat", size=13, color=INK, weight=700, anchor="middle"),
            base.text(934, 305, "1028-D", size=12, color=INK, weight=700, anchor="middle"),
            base.text(934, 350, "route 192-D", size=8.5, color=BLUE, weight=700, anchor="middle"),
            base.text(934, 377, "future 796-D", size=8.5, color=BLUE, weight=700, anchor="middle"),
            base.text(934, 404, "action 40-D", size=8.5, color=ORANGE, weight=700, anchor="middle"),
            base.text(934, 428, "frozen z-score", size=8, color=MUTED, anchor="middle"),
            card(1010, 250, 178, 190, fill=WHITE, stroke=PURPLE_MID, sw=1.1, rx=8),
            base.text(1099, 274, "shared MLP × 3 seeds", size=11.5, color=PURPLE, weight=700, anchor="middle"),
            card(1030, 293, 138, 28, fill=PURPLE_SOFT, stroke=PURPLE_MID, sw=0.8, rx=6),
            card(1030, 329, 138, 28, fill=PURPLE_SOFT, stroke=PURPLE_MID, sw=0.8, rx=6),
            card(1030, 365, 138, 28, fill=PURPLE_SOFT, stroke=PURPLE_MID, sw=0.8, rx=6),
            base.text(1099, 312, "MLP · 1701", size=9.5, color=INK, weight=700, anchor="middle"),
            base.text(1099, 348, "MLP · 1702", size=9.5, color=INK, weight=700, anchor="middle"),
            base.text(1099, 384, "MLP · 1703", size=9.5, color=INK, weight=700, anchor="middle"),
            base.text(1099, 416, "ATE ↓   RTP ↑   ± σ", size=10.5, color=PURPLE, weight=700, anchor="middle"),
            base.arrow_path("M 988 345 H 1010", color=PURPLE, sw=1.9),
            card(1010, 464, 178, 100, fill=WHITE, stroke=PURPLE, sw=1.3, rx=8),
            base.text(1099, 490, "Joint safe selector", size=11.5, color=PURPLE, weight=700, anchor="middle"),
            base.text(1099, 514, "uₖ = min(LCB ΔATE/sA,", size=9, color=INK, weight=600, anchor="middle"),
            base.text(1099, 533, "LCB ΔRTP/sR)", size=9, color=INK, weight=600, anchor="middle"),
            base.text(1099, 554, "both margins > 0; else incumbent", size=8.4, color=MUTED, anchor="middle"),
            base.arrow_path("M 1099 440 V 464", color=PURPLE, sw=2.0),
            base.arrow_path("M 1188 512 H 1246", color=GREEN, sw=2.4),
            card(1248, 478, 56, 68, fill=GREEN_SOFT, stroke=GREEN, sw=1.2, rx=9),
            base.text(1276, 508, "H4", size=13, color=GREEN, weight=700, anchor="middle"),
            base.text(1276, 529, "selected", size=7.5, color=GREEN, weight=700, anchor="middle"),
        )
    )

    # Three clean fan-in streams into the right-side concat hub.
    parts.extend(
        (
            base.arrow_path("M 760 218 H 812 V 336 H 880", color=BLUE, sw=2.2),
            base.text(812, 207, "current + goal latents", size=8.5, color=BLUE, weight=600),
            base.arrow_path("M 820 544 H 840 V 373 H 880", color=BLUE, sw=2.2),
            base.text(839, 532, "future latents", size=8.5, color=BLUE, weight=600, anchor="end"),
            base.arrow_path("M 364 625 H 800 V 400 H 880", color=ORANGE, sw=2.0),
            base.text(790, 640, "candidate actions", size=8.5, color=ORANGE, weight=600, anchor="end"),
        )
    )

    parts.extend(
        (
            base.line(34, 694, 1286, 694, color=HAIR, sw=1.0),
            base.text(
                34,
                722,
                "Fixed visual tokens → candidate-conditioned latent futures → conservative H4 selection",
                size=10.5,
                color=MUTED,
                weight=600,
            ),
            base.text(
                1286,
                722,
                "topology-preserving abstraction",
                size=9.5,
                color=BLUE,
                weight=700,
                anchor="end",
            ),
            "</svg>",
        )
    )
    return "\n".join(parts)


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
        id="figure1-topology-simple",
        name="Topology-Preserving Main Figure",
    )
    graph = ET.SubElement(
        diagram,
        "mxGraphModel",
        dx="1320",
        dy="760",
        grid="1",
        gridSize="10",
        guides="1",
        page="1",
        pageScale="1",
        pageWidth="1320",
        pageHeight="760",
        background="#FFFFFF",
    )
    root = ET.SubElement(graph, "root")
    ET.SubElement(root, "mxCell", id="0")
    ET.SubElement(root, "mxCell", id="1", parent="0")

    add = base.add_vertex
    style = base.cell_style
    tstyle = base.text_style

    add(root, "title", "<b>DINOv3–LTWM trajectory planning</b>", 34, 12, 700, 30, style_value=tstyle(size=20, bold=True))
    add(root, "subtitle", "Visual context + candidate actions → candidate-conditioned latent futures → conservative H4 selection", 34, 42, 850, 22, style_value=tstyle(size=10, color=MUTED))

    stages = (
        ("visual", "<b>A  VISUAL CONTEXT</b>", 34, 106, 330, 246, PANEL, "#C9D4E1", BLUE),
        ("actions", "<b>B  CANDIDATE ACTIONS</b>", 34, 394, 330, 252, ORANGE_SOFT, ORANGE_MID, ORANGE),
        ("tokenizer", "<b>C  FROZEN VISUAL TOKENIZER</b>", 410, 106, 350, 190, BLUE_SOFT, BLUE_MID, BLUE),
        ("ltwm_stage", "<b>D  CANDIDATE-CONDITIONED LTWM</b>", 410, 344, 410, 302, "#F2F5FB", BLUE, BLUE),
        ("scoring", "<b>E  TRAJECTORY-QUALITY SCORING</b>", 856, 164, 364, 430, PURPLE_SOFT, PURPLE_MID, PURPLE),
    )
    for stage_id, label, x, y, width, height, fill, stroke, color in stages:
        add(
            root,
            stage_id,
            label,
            x,
            y,
            width,
            height,
            style_value=style(
                fill=fill,
                stroke=stroke,
                color=color,
                size=13,
                bold=True,
                extra="container=1;pointerEvents=0;verticalAlign=top;align=left;spacingTop=10;spacingLeft=14;",
            ),
        )

    # Visual context child cards, relative to visual container.
    for tile_id, x in zip(("t-3", "t-2", "t-1", "t"), (48, 108, 168, 228)):
        add(root, f"vis_{tile_id}", "", x, 45, 54, 31, parent="visual", style_value=base.image_style(ASSETS[tile_id], stroke=BLUE if tile_id == "t" else "#AEBBCB"))
    add(root, "vis_goal", "", 212, 177, 70, 42, parent="visual", style_value=base.image_style(ASSETS["goal"], stroke=GREEN))
    add(root, "vis_context_note", "<b>4-frame history + goal</b>", 30, 130, 260, 22, parent="visual", style_value=tstyle(size=10, color=BLUE, bold=True))

    add(root, "nomad", "<b>NoMaD</b><br>K trajectories · H4 actions<br><font color=\"#A96A35\">external proposal policy</font>", 24, 72, 112, 78, parent="actions", style_value=style(fill=WHITE, stroke=ORANGE, size=10))
    add(root, "fan_note", "<b>K × 4 × 3</b><br>candidate action sequences", 156, 88, 142, 58, parent="actions", style_value=style(fill=WHITE, stroke=ORANGE_MID, size=10))

    add(root, "dino", "<b>DINOv3-L/16</b><br>14×14×1024", 28, 72, 142, 64, parent="tokenizer", style_value=style(fill=WHITE, stroke=BLUE_MID, size=10))
    add(root, "adapter", "<b>Adapter</b><br>16×16×384", 186, 72, 122, 64, parent="tokenizer", style_value=style(fill="#F7EEF6", stroke=PURPLE_MID, size=10))
    add(root, "token_note", "<b>z history + z goal</b> · 256 × 384 tokens", 54, 148, 242, 24, parent="tokenizer", style_value=tstyle(size=10, color=BLUE, bold=True, align="center"))

    add(root, "hist_enc", "<b>History encoder</b>", 32, 88, 112, 58, parent="ltwm_stage", style_value=style(fill=WHITE, stroke=BLUE_MID, size=10))
    add(root, "act_prefix", "<b>Action prefix</b>", 164, 88, 112, 58, parent="ltwm_stage", style_value=style(fill=ORANGE_SOFT, stroke=ORANGE_MID, size=10))
    add(root, "rollout_core", "<b>Factorized token rollout</b>", 104, 172, 174, 46, parent="ltwm_stage", style_value=style(fill=BLUE, stroke=BLUE, color=WHITE, size=10))
    for index, label in enumerate(("H1", "H2", "H3", "H4")):
        add(root, f"ltwm_{label.lower()}", f"<b>{label}</b>", 298, 80 + index * 48, 54, 34, parent="ltwm_stage", style_value=style(fill=WHITE, stroke=BLUE if label == "H4" else BLUE_MID, color=BLUE if label == "H4" else MUTED, size=9))
    add(root, "ltwm_note", "<b>future latents</b> [K × 4 × 256 × 384]", 56, 268, 260, 22, parent="ltwm_stage", style_value=tstyle(size=10, color=BLUE, bold=True, align="center"))

    add(root, "concat", "<b>Concat</b><br><b>1028-D</b><br><font color=\"#315E9A\">route 192-D</font><br><font color=\"#315E9A\">future 796-D</font><br><font color=\"#A96A35\">action 40-D</font>", 26, 82, 108, 190, parent="scoring", style_value=style(fill=WHITE, stroke="#BFC8D3", size=9))
    add(root, "mlp_stack", "<b>shared MLP ×3 seeds</b><br>1028 → 640 × 3 → 2<br><font color=\"#667180\">1701 · 1702 · 1703</font><br><br><b>ATE ↓  RTP ↑  ± σ</b>", 154, 82, 178, 190, parent="scoring", style_value=style(fill=WHITE, stroke=PURPLE_MID, size=10, color=PURPLE))
    add(root, "selector", "<b>Joint safe selector</b><br><br>uₖ = min(LCB ΔATE/sA,<br>LCB ΔRTP/sR)<br><br><font color=\"#667180\">both margins &gt; 0;<br>else → incumbent</font>", 154, 300, 178, 100, parent="scoring", style_value=style(fill=WHITE, stroke=PURPLE, size=9))
    add(root, "selected", "<b>SELECTED H4</b><br><font color=\"#667180\">execute first action</font>", 1248, 478, 56, 68, parent="1", style_value=style(fill=GREEN_SOFT, stroke=GREEN, size=9, color=GREEN))

    add = base.add_edge
    add(root, "visual_to_tok", "visual", "dino", color=BLUE, label="history + goal", exit_xy=(1, 0.45), entry_xy=(0, 0.5), points=((390, 215), (390, 214)))
    add(root, "goal_to_tok", "vis_goal", "adapter", color=GREEN, label="goal", exit_xy=(1, 0.5), entry_xy=(0.5, 1), points=((390, 304), (390, 245), (657, 245)))
    add(root, "dino_to_adapter", "dino", "adapter", color=BLUE, exit_xy=(1, 0.5), entry_xy=(0, 0.5))
    add(root, "tok_to_hist", "tokenizer", "hist_enc", color=BLUE, label="history tokens", exit_xy=(0.45, 1), entry_xy=(0.5, 0), points=((565, 330), (498, 330), (498, 432)))
    add(root, "actions_to_ltwm", "actions", "act_prefix", color=ORANGE, label="actions", exit_xy=(1, 0.5), entry_xy=(0.5, 1), points=((390, 510), (630, 510), (630, 500)))
    add(root, "hist_to_rollout", "hist_enc", "rollout_core", color=BLUE, exit_xy=(0.5, 1), entry_xy=(0.25, 0))
    add(root, "act_to_rollout", "act_prefix", "rollout_core", color=ORANGE, exit_xy=(0.5, 1), entry_xy=(0.75, 0))
    add(root, "rollout_to_h4", "rollout_core", "ltwm_h4", color=BLUE, exit_xy=(1, 0.5), entry_xy=(0, 0.5))
    add(root, "tok_to_concat", "tokenizer", "concat", color=BLUE, label="current + goal", exit_xy=(1, 0.5), entry_xy=(0, 0.2), points=((780, 214), (820, 214), (820, 264), (882, 264)))
    add(root, "future_to_concat", "ltwm_stage", "concat", color=BLUE, label="future", exit_xy=(1, 0.55), entry_xy=(0, 0.55), points=((830, 538), (840, 538), (840, 345), (882, 345)))
    add(root, "actions_to_concat", "actions", "concat", color=ORANGE, label="actions", exit_xy=(1, 0.9), entry_xy=(1, 0.8), points=((390, 670), (1000, 670), (1000, 416)))
    add(root, "concat_to_mlp", "concat", "mlp_stack", color=PURPLE, exit_xy=(1, 0.5), entry_xy=(0, 0.5))
    add(root, "mlp_to_selector", "mlp_stack", "selector", color=PURPLE, exit_xy=(0.5, 1), entry_xy=(0.5, 0))
    add(root, "selector_to_selected", "selector", "selected", color=GREEN, exit_xy=(1, 0.5), entry_xy=(0, 0.5))

    ET.indent(mxfile, space="  ")
    OUT_DRAWIO.write_bytes(ET.tostring(mxfile, encoding="UTF-8", xml_declaration=True))


def build_readme() -> None:
    (OUT_DIR / "README.md").write_text(
        """# Figure 1 — topology-preserving simplification

This version keeps the original spatial topology instead of forcing a
left-to-right pipeline:

- visual context: upper-left;
- candidate action proposals: lower-left;
- frozen DINOv3 tokenizer: upper-middle;
- candidate-conditioned LTWM: lower-middle;
- trajectory-quality scorer: right-side fan-in hub;
- selected H4: right-side output.

The abstraction preserves the repository contract: 4-frame history, goal
offset 4, frozen 1024→384 visual adapter, 256×384 stored tokens, K×4×3
candidate actions, H1–H4 dense future latents, 192-D route + 796-D rollout +
40-D action = 1028-D scorer input, three 640-wide MLP layers, three seeds,
ATE/RTP outputs, and incumbent fallback.
""",
        encoding="utf-8",
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    svg = build_svg()
    OUT_SVG.write_text(svg, encoding="utf-8")
    cairosvg.svg2png(
        bytestring=svg.encode("utf-8"),
        write_to=str(OUT_PNG),
        output_width=2640,
        output_height=1520,
    )
    cairosvg.svg2pdf(bytestring=svg.encode("utf-8"), write_to=str(OUT_PDF))
    build_drawio()
    build_readme()
    print(OUT_DRAWIO)
    print(OUT_SVG)
    print(OUT_PNG)
    print(OUT_PDF)


if __name__ == "__main__":
    main()
