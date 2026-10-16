#!/usr/bin/env python3
"""Apply a restrained, publication-style visual system to b2nnwm.drawio.

The source geometry and topology are intentionally preserved.  This script only
normalizes style tokens, typography, and a small number of labels so the figure
remains fully editable in draw.io while reading like a conference paper figure.
"""

from __future__ import annotations

import argparse
import re
import xml.etree.ElementTree as ET
from pathlib import Path


PREFIX = "6DSzBBY3_p7ScXEFss4Q-"

# Muted, print-safe palette: navy = learned visual features, violet = LTWM,
# ochre = action/CEM, teal = safety, coral = selected output.
PALETTE = {
    "#17212B": "#172B4D",
    "#17212b": "#172B4D",
    "#142B43": "#172B4D",
    "#142b43": "#172B4D",
    "#18283A": "#172B4D",
    "#18283a": "#172B4D",
    "#45669A": "#315E86",
    "#45669a": "#315E86",
    "#31527E": "#315E86",
    "#6557A7": "#6B5CA5",
    "#6557a7": "#6B5CA5",
    "#B27B48": "#A96D35",
    "#b27b48": "#A96D35",
    "#E77F2D": "#B97836",
    "#e77f2d": "#B97836",
    "#A96D35": "#A96D35",
    "#C55A4A": "#C65B4D",
    "#c55a4a": "#C65B4D",
    "#0C7976": "#347C78",
    "#0c7976": "#347C78",
    "#2C875E": "#347C78",
    "#2c875e": "#347C78",
    "#2D7A78": "#347C78",
    "#2d7a78": "#347C78",
    "#3E9562": "#347C78",
    "#3e9562": "#347C78",
    "#79A974": "#6D9A7A",
    "#79a974": "#6D9A7A",
    "#3A7F57": "#347C78",
    "#EEF3FA": "#F4F8FB",
    "#eef3fa": "#F4F8FB",
    "#EEF3F7": "#F4F8FB",
    "#eef3f7": "#F4F8FB",
    "#F3F0FA": "#F7F5FB",
    "#f3f0fa": "#F7F5FB",
    "#F3ECF3": "#F6F2F6",
    "#f3ecf3": "#F6F2F6",
    "#FFF0E2": "#FBF5EC",
    "#fff0e2": "#FBF5EC",
    "#F6F3ED": "#FBF8F2",
    "#f6f3ed": "#FBF8F2",
    "#FCEFEB": "#FCF1EE",
    "#fcefeb": "#FCF1EE",
    "#EFF7F5": "#F0F7F5",
    "#eff7f5": "#F0F7F5",
    "#F4F6F8": "#F4F7F9",
    "#f4f6f8": "#F4F7F9",
    "#F5F8FA": "#F5F8FA",
    "#DCE2E8": "#D6DEE7",
    "#dce2e8": "#D6DEE7",
    "#CDD5DE": "#D2DAE3",
    "#cdd5de": "#D2DAE3",
    "#627184": "#627184",
    "#647081": "#627184",
    "#647081": "#627184",
    "#8493A1": "#8FA0AE",
    "#8493a1": "#8FA0AE",
    "#B98F6C": "#B8906E",
    "#b98f6c": "#B8906E",
    "#B9C4CB": "#B7C4CD",
    "#b9c4cb": "#B7C4CD",
    "#B7C8D2": "#B7C8D2",
    "#b7c8d2": "#B7C8D2",
    "#687A87": "#6E8090",
    "#687a87": "#6E8090",
    "#C9D1D8": "#C9D2DA",
    "#c9d1d8": "#C9D2DA",
    "#E4E1D9": "#E2E5E7",
    "#e4e1d9": "#E2E5E7",
    "#83A4CC": "#7C9AB5",
    "#83a4cc": "#7C9AB5",
    "#6F8EAA": "#718EA7",
    "#6f8eaa": "#718EA7",
    "#8796A5": "#8B9AA8",
    "#8796a5": "#8B9AA8",
    "#9AA8B8": "#9AA9B7",
    "#9aa8b8": "#9AA9B7",
}


def parse_style(style: str) -> tuple[list[str], dict[str, str]]:
    """Parse an mxGraph style and collapse duplicate key/value tokens."""
    flags: list[str] = []
    values: dict[str, str] = {}
    for token in (style or "").split(";"):
        if not token:
            continue
        if "=" not in token:
            if token not in flags:
                flags.append(token)
            continue
        key, val = token.split("=", 1)
        if key not in values:
            values[key] = val
        else:
            # Last assignment is draw.io's effective value.  Preserve its key
            # position while replacing the value below.
            values[key] = val
    return flags, values


def style_string(flags: list[str], values: dict[str, str]) -> str:
    return ";".join(flags + [f"{k}={v}" for k, v in values.items()]) + ";"


def recolor(value: str) -> str:
    for old, new in PALETTE.items():
        value = value.replace(old, new)
    return value


def set_style(cell: ET.Element, **updates: object) -> None:
    flags, vals = parse_style(cell.get("style", ""))
    for key, value in updates.items():
        if value is None:
            vals.pop(key, None)
        else:
            vals[key] = str(value)
    cell.set("style", style_string(flags, vals))


def set_geometry(cell: ET.Element, *, x: float | None = None, y: float | None = None,
                 width: float | None = None, height: float | None = None) -> None:
    geo = cell.find("mxGeometry")
    if geo is None:
        geo = ET.SubElement(cell, "mxGeometry", {"as": "geometry"})
    for key, value in (("x", x), ("y", y), ("width", width), ("height", height)):
        if value is not None:
            geo.set(key, f"{value:g}")


def by_id(root: ET.Element, cid: str) -> ET.Element | None:
    return next((c for c in root.findall("mxCell") if c.get("id") == cid), None)


def by_value(root: ET.Element, text: str) -> list[ET.Element]:
    return [c for c in root.findall("mxCell") if text in (c.get("value") or "")]


def normalize_global(root: ET.Element) -> None:
    for cell in root.findall("mxCell"):
        style = recolor(cell.get("style", ""))
        flags, vals = parse_style(style)
        if cell.get("vertex") == "1" or cell.get("edge") == "1":
            vals.setdefault("fontFamily", "Arial")
        # Repeated legacy assignments make the XML hard to maintain and can
        # render differently between draw.io versions.  Keep one canonical key.
        if "fontFamily" in vals:
            vals["fontFamily"] = "Arial"
        if "shadow" in vals:
            vals["shadow"] = "0"
        if "strokeWidth" in vals:
            try:
                sw = float(vals["strokeWidth"])
                if sw >= 4:
                    vals["strokeWidth"] = "1.8"
                elif sw >= 3:
                    vals["strokeWidth"] = "1.6"
                elif sw > 2:
                    vals["strokeWidth"] = "1.4"
            except ValueError:
                pass
        if "arcSize" in vals:
            try:
                # Keep tiny pill ends gently rounded, but remove the UI-like
                # 50-pixel capsules from large architecture cards.
                arc = float(vals["arcSize"])
                if arc > 16:
                    vals["arcSize"] = "12"
            except ValueError:
                pass
        if cell.get("edge") == "1":
            vals.setdefault("fontFamily", "Arial")
            vals["fontSize"] = "8"
            vals["labelBackgroundColor"] = "#FFFFFF"
            if vals.get("endArrow") in {"block", "classic", "blockThin"}:
                vals["endArrow"] = "blockThin"
            vals.setdefault("endFill", "1")
            vals.setdefault("endSize", "6")
            if float(vals.get("strokeWidth", "1.2")) > 1.6:
                vals["strokeWidth"] = "1.3"
        cell.set("style", style_string(flags, vals))


def target_styles(root: ET.Element) -> None:
    # Page-level typography and legend.
    c = by_id(root, PREFIX + "1")
    if c is not None:
        set_style(c, fontFamily="Arial", fontSize=32, fontColor="#172B4D",
                  fontStyle=1, align="left", verticalAlign="middle")
    c = by_id(root, PREFIX + "2")
    if c is not None:
        set_style(c, fillColor="#F4F8FB", strokeColor="#7C9AB5", strokeWidth=1.0,
                  fontFamily="Arial", fontSize=10, fontColor="#315E86", fontStyle=1,
                  shadow=0, arcSize=10)
    for cid in (PREFIX + "3", PREFIX + "5"):
        c = by_id(root, cid)
        if c is not None:
            set_style(c, fillColor="#D6DEE7", strokeColor="#D6DEE7", strokeWidth=0)
    c = by_id(root, PREFIX + "4")
    if c is not None:
        set_style(c, fontFamily="Arial", fontSize=13, fontColor="#627184", fontStyle=0)
    for cid in (PREFIX + "6", PREFIX + "8", PREFIX + "10", PREFIX + "12", PREFIX + "14"):
        c = by_id(root, cid)
        if c is not None:
            set_style(c, strokeWidth=0, shadow=0)
    for cid in (PREFIX + "7", PREFIX + "9", PREFIX + "11", PREFIX + "13", PREFIX + "15", PREFIX + "16"):
        c = by_id(root, cid)
        if c is not None:
            set_style(c, fontFamily="Arial", fontSize=10, fontColor="#627184")

    # Section headers: modest hierarchy with slim color accents.
    for cid, color, size in (
        (PREFIX + "21", "#315E86", 23),
        (PREFIX + "18", "#6B5CA5", 23),
        (PREFIX + "109", "#A96D35", 23),
    ):
        c = by_id(root, cid)
        if c is not None:
            set_style(c, fontFamily="Arial", fontSize=size, fontColor=color, fontStyle=1)
    for cid, color in ((PREFIX + "20", "#315E86"), (PREFIX + "182", "#6B5CA5"), (PREFIX + "110", "#A96D35")):
        c = by_id(root, cid)
        if c is not None:
            set_style(c, fillColor=color, strokeColor=color, strokeWidth=0, shadow=0, arcSize=12)

    # Visual encoder / adapter / token projection.
    c = by_id(root, PREFIX + "134")
    if c is not None:
        set_style(c, fillColor="#F4F8FB", strokeColor="#315E86", strokeWidth=1.8,
                  fontFamily="Arial", shadow=0, arcSize=12)
    c = by_id(root, PREFIX + "135")
    if c is not None:
        set_style(c, fontFamily="Arial", fontSize=20, fontColor="#172B4D", fontStyle=1)
    c = by_id(root, PREFIX + "136")
    if c is not None:
        set_style(c, fillColor="#FFFFFF", strokeColor="#315E86", strokeWidth=1,
                  fontFamily="Arial", fontSize=10, fontColor="#315E86", shadow=0)
    c = by_id(root, PREFIX + "156")
    if c is not None:
        set_style(c, fillColor="#F6F2F6", strokeColor="#C4A8C0", strokeWidth=1.8,
                  fontFamily="Arial", fontSize=18, fontColor="#172B4D", fontStyle=1)
    c = by_id(root, PREFIX + "157")
    if c is not None:
        set_style(c, fillColor="#FFFFFF", strokeColor="#C4A8C0", strokeWidth=1,
                  fontFamily="Arial", fontSize=10, fontColor="#8A6384", shadow=0, arcSize=10)
    c = by_id(root, PREFIX + "158")
    if c is not None:
        set_style(c, fillColor="#FFFFFF", strokeColor="#7C9AB5", strokeWidth=1.4,
                  fontFamily="Arial", shadow=0, arcSize=10)
    c = by_id(root, PREFIX + "179")
    if c is not None:
        set_style(c, fontFamily="Arial", fontSize=10, fontColor="#315E86", fontStyle=1)

    # LTWM card and its compact stage labels.
    c = by_id(root, PREFIX + "183")
    if c is not None:
        set_style(c, fillColor="#F7F5FB", strokeColor="#6B5CA5", strokeWidth=1.8,
                  fontFamily="Arial", shadow=0, arcSize=12)
    c = by_id(root, PREFIX + "184")
    if c is not None:
        set_style(c, fillColor="#6B5CA5", strokeColor="#6B5CA5", strokeWidth=1,
                  fontFamily="Arial", fontSize=13, fontColor="#FFFFFF", fontStyle=1,
                  shadow=0, arcSize=10)
    for cid in (PREFIX + "185", PREFIX + "186", PREFIX + "187"):
        c = by_id(root, cid)
        if c is not None:
            set_style(c, fillColor="#6B5CA5", strokeColor="#6B5CA5", strokeWidth=0.8,
                      fontFamily="Arial", fontSize=10, fontColor="#FFFFFF", fontStyle=1,
                      shadow=0, arcSize=8)
    c = by_id(root, PREFIX + "198")
    if c is not None:
        set_style(c, fillColor="#FFFFFF", strokeColor="#A96D35", strokeWidth=1,
                  fontFamily="Arial", fontSize=10, fontColor="#A96D35", fontStyle=1,
                  shadow=0, arcSize=10)
    c = by_id(root, PREFIX + "199")
    if c is not None:
        set_style(c, fontFamily="Arial", fontSize=11, fontColor="#347C78", fontStyle=1)
    c = by_id(root, PREFIX + "240")
    if c is not None:
        set_style(c, fillColor="#FFFFFF", strokeColor="#6B5CA5", strokeWidth=1,
                  fontFamily="Arial", fontSize=10, fontColor="#6B5CA5", fontStyle=1,
                  shadow=0, arcSize=10)

    # Candidate-action card: one thin ochre boundary and a restrained input pill.
    c = by_id(root, PREFIX + "24")
    if c is not None:
        set_style(c, fillColor="#FBF8F2", strokeColor="#B97836", strokeWidth=1.5,
                  fontFamily="Arial", shadow=0, arcSize=12)
    c = by_id(root, PREFIX + "26")
    if c is not None:
        set_style(c, fillColor="#FBF5EC", strokeColor="#B97836", strokeWidth=1,
                  fontFamily="Arial", fontSize=10, fontColor="#B97836", fontStyle=1,
                  shadow=0, arcSize=10)
    c = by_id(root, PREFIX + "28")
    if c is not None:
        set_style(c, fillColor="#FBF8F2", strokeColor="#C9D2DA", strokeWidth=1.2,
                  fontFamily="Arial", shadow=0, arcSize=8)
    c = by_id(root, PREFIX + "102")
    if c is not None:
        set_style(c, fillColor="#B97836", strokeColor="#B97836", strokeWidth=0.8,
                  fontFamily="Arial", fontSize=10, fontColor="#172B4D", shadow=0,
                  arcSize=8)

    # The planner already contains the correct 3-head topology.  Give it the
    # same typography and remove the last remnants of UI-card styling.
    for cid, size, color in (
        (PREFIX + "581", 15, "#172B4D"),
        (PREFIX + "582", 8, "#627184"),
        (PREFIX + "600", 10, "#172B4D"),
        (PREFIX + "632", 10, "#172B4D"),
    ):
        c = by_id(root, cid)
        if c is not None:
            set_style(c, fontFamily="Arial", fontSize=size, fontColor=color)
    c = by_id(root, PREFIX + "583")
    if c is not None:
        set_style(c, strokeColor="#D2DAE3", strokeWidth=0.9, fillColor="#D2DAE3")
    c = by_id(root, PREFIX + "584")
    if c is not None:
        set_style(c, fillColor="#F0F7F5", strokeColor="#347C78", strokeWidth=1.4,
                  fontFamily="Arial", fontColor="#172B4D", shadow=0)
    c = by_id(root, PREFIX + "585")
    if c is not None:
        set_style(c, fillColor="#FBF8F2", strokeColor="#A96D35", strokeWidth=1.1,
                  fontFamily="Arial", shadow=0)
    for cid in (PREFIX + "587", PREFIX + "588", PREFIX + "589"):
        c = by_id(root, cid)
        if c is not None:
            set_style(c, fillColor="#DCE8F2", strokeColor="#718EA7", strokeWidth=0.9,
                      fontFamily="Arial", fontSize=9, fontColor="#172B4D")
    for cid in (PREFIX + "590", PREFIX + "591"):
        c = by_id(root, cid)
        if c is not None:
            set_style(c, strokeColor="#8B9AA8", strokeWidth=0.9, fontFamily="Arial",
                      fontSize=9, fontColor="#172B4D")
    for cid in (PREFIX + "604", PREFIX + "605", PREFIX + "606", PREFIX + "607", PREFIX + "611"):
        c = by_id(root, cid)
        if c is not None:
            set_style(c, fillColor="#F5F8FA", strokeColor="#315E86", strokeWidth=1,
                      fontFamily="Arial", fontColor="#172B4D", shadow=0)
    c = by_id(root, PREFIX + "611")
    if c is not None:
        set_style(c, fillColor="#F4F7F9", strokeColor="#9AA9B7", strokeWidth=1,
                  fontFamily="Arial", fontColor="#172B4D", shadow=0)
    c = by_id(root, PREFIX + "730")
    if c is not None:
        # The outer selected hypothesis group is kept as a group; its visible
        # frame is cell 586 below.
        pass
    c = by_id(root, PREFIX + "586")
    if c is not None:
        set_style(c, fillColor="#FFFFFF", strokeColor="#C65B4D", strokeWidth=1.4,
                  fontFamily="Arial", shadow=0)
    c = by_id(root, PREFIX + "599")
    if c is not None:
        set_style(c, fillColor="#FCF1EE", strokeColor="#C65B4D", strokeWidth=1,
                  fontFamily="Arial", fontSize=9, fontColor="#172B4D", shadow=0)
    c = by_id(root, PREFIX + "602")
    if c is not None:
        set_style(c, fontFamily="Arial", fontSize=10, fontColor="#172B4D", fontStyle=1)
    c = by_id(root, PREFIX + "603")
    if c is not None:
        set_style(c, fontFamily="Arial", fontSize=18, fontColor="#C65B4D", fontStyle=1)
    c = by_id(root, PREFIX + "601")
    if c is not None:
        set_style(c, fontFamily="Arial", fontSize=10, fontColor="#172B4D", fontStyle=1)
    for cid in (PREFIX + "594", PREFIX + "595", PREFIX + "596", PREFIX + "597", PREFIX + "598"):
        c = by_id(root, cid)
        if c is not None:
            set_style(c, fontFamily="Arial", fontSize=8, fontColor="#A96D35")


def transform(src: Path, dst: Path) -> None:
    tree = ET.parse(src)
    model = tree.getroot().find("./diagram/mxGraphModel")
    if model is None:
        raise ValueError("mxGraphModel not found")
    root = model.find("root")
    if root is None:
        raise ValueError("mxGraphModel/root not found")
    # A paper-ready figure should not open with an editor grid or guides over it.
    model.set("grid", "0")
    model.set("guides", "0")
    model.set("shadow", "0")
    normalize_global(root)
    target_styles(root)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tree.write(dst, encoding="utf-8", xml_declaration=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("src", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()
    transform(args.src, args.output)
    print(args.output)


if __name__ == "__main__":
    main()

