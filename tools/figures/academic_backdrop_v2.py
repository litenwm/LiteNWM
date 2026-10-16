#!/usr/bin/env python3
"""Make the academic restyle visually explicit on the existing b2nnwm figure.

This is a post-pass over the already-normalized XML.  It adds only background
panels and small section rules; all original nodes, labels, edges, and geometry
remain intact and editable.
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path


def add_cell(root: ET.Element, cid: str, *, x: float, y: float, w: float, h: float,
             style: str, value: str = "", parent: str = "1") -> None:
    cell = ET.Element("mxCell", {
        "id": cid,
        "parent": parent,
        "vertex": "1",
        "connectable": "0",
        "value": value,
        "style": style,
    })
    ET.SubElement(cell, "mxGeometry", {
        "x": f"{x:g}", "y": f"{y:g}", "width": f"{w:g}",
        "height": f"{h:g}", "as": "geometry",
    })
    # 0 and 1 are reserved root cells.  Inserting at index 2 places new
    # scenery behind every existing edge and vertex in the source file.
    root.insert(2, cell)


def panel(root: ET.Element, cid: str, x: float, y: float, w: float, h: float,
          fill: str, stroke: str) -> None:
    add_cell(
        root, cid, x=x, y=y, w=w, h=h,
        style=(
            "rounded=1;arcSize=8;whiteSpace=wrap;html=1;"
            f"fillColor={fill};strokeColor={stroke};strokeWidth=1;"
            "fontFamily=Arial;shadow=0;"
        ),
    )


def rule(root: ET.Element, cid: str, x: float, y: float, color: str) -> None:
    add_cell(
        root, cid, x=x, y=y, w=5, h=28,
        style=(
            "rounded=1;arcSize=6;whiteSpace=wrap;html=1;"
            f"fillColor={color};strokeColor={color};strokeWidth=0;"
            "fontFamily=Arial;shadow=0;"
        ),
    )


def kicker(root: ET.Element, cid: str, text: str, x: float, y: float, w: float,
           color: str) -> None:
    add_cell(
        root, cid, x=x, y=y, w=w, h=18, value=text,
        style=(
            "text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
            f"fontFamily=Arial;fontSize=9;fontColor={color};fontStyle=1;"
            "align=left;verticalAlign=middle;spacingLeft=2;"
        ),
    )


def transform(src: Path, dst: Path) -> None:
    tree = ET.parse(src)
    model = tree.getroot().find("./diagram/mxGraphModel")
    if model is None:
        raise ValueError("mxGraphModel not found")
    root = model.find("root")
    if root is None:
        raise ValueError("mxGraphModel/root not found")

    # Hide the editor grid for a paper-like canvas.
    model.set("grid", "0")
    model.set("guides", "0")
    model.set("shadow", "0")

    # Five semantic bands.  Their placement follows the source topology:
    # context/actions on the left, encoder bridge in the middle, LTWM above,
    # and the complete safe planner below.
    panel(root, "b2-v2-canvas", 58, 150, 1660, 795, "#FFFFFF", "#D8E0E8")
    panel(root, "b2-v2-context", 92, 172, 500, 278, "#F3F7FB", "#BFCFDF")
    panel(root, "b2-v2-actions", 92, 448, 500, 318, "#FCF7F0", "#D3B18E")
    panel(root, "b2-v2-encoder", 575, 172, 625, 275, "#F5F8FC", "#BFCDDC")
    panel(root, "b2-v2-ltwm", 1195, 172, 520, 275, "#F8F6FC", "#CFC5E3")
    panel(root, "b2-v2-planner", 638, 420, 1040, 490, "#FBFCFD", "#C5D0DB")

    # Section rules make the modules read as panels rather than floating UI
    # cards.  Kicker labels are intentionally short and non-redundant.
    rule(root, "b2-v2-rule-a", 105, 198, "#315E86")
    rule(root, "b2-v2-rule-c", 105, 474, "#A96D35")
    rule(root, "b2-v2-rule-enc", 588, 198, "#315E86")
    rule(root, "b2-v2-rule-b", 1208, 198, "#6B5CA5")
    rule(root, "b2-v2-rule-p", 650, 448, "#347C78")
    kicker(root, "b2-v2-kicker-a", "OBSERVATIONS", 116, 179, 130, "#627184")
    kicker(root, "b2-v2-kicker-c", "ACTION PROPOSALS", 116, 455, 150, "#8D694B")
    kicker(root, "b2-v2-kicker-enc", "FROZEN ENCODER", 600, 179, 130, "#627184")
    kicker(root, "b2-v2-kicker-b", "PREDICTED LATENT FUTURES", 1220, 179, 220, "#7669A3")
    kicker(root, "b2-v2-kicker-p", "CONSERVATIVE SELECTION", 662, 429, 210, "#347C78")

    # Make the title and subtitle visibly belong to the same publication plate.
    for cell in root.findall("mxCell"):
        if cell.get("id") == "6DSzBBY3_p7ScXEFss4Q-1":
            cell.set("style", "text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
                     "fontFamily=Arial;fontSize=31;fontColor=#172B4D;fontStyle=1;"
                     "align=left;verticalAlign=middle;")
        elif cell.get("id") == "6DSzBBY3_p7ScXEFss4Q-4":
            cell.set("style", "text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
                     "fontFamily=Arial;fontSize=13;fontColor=#627184;"
                     "align=left;verticalAlign=middle;")

    dst.parent.mkdir(parents=True, exist_ok=True)
    tree.write(dst, encoding="utf-8", xml_declaration=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("-o", "--output", type=Path, required=True)
    args = ap.parse_args()
    transform(args.src, args.output)
    print(args.output)


if __name__ == "__main__":
    main()

