#!/usr/bin/env python3
"""Add restrained visual hierarchy to the clean SAFE CANDIDATE PLANNER."""

from __future__ import annotations

import argparse
import base64
import xml.etree.ElementTree as ET
from pathlib import Path


P = "Lcp2UKG11IvjxIMB6Xdv-"
PLANNER = P + "2164"
PANEL = P + "2077"


def find(root: ET.Element, cid: str) -> ET.Element:
    for cell in root.findall("mxCell"):
        if cell.get("id") == cid:
            return cell
    raise KeyError(cid)


def geom(cell: ET.Element, x=None, y=None, w=None, h=None) -> ET.Element:
    old = cell.find("mxGeometry")
    if old is not None:
        cell.remove(old)
    attrs = {"as": "geometry"}
    for key, value in (("x", x), ("y", y), ("width", w), ("height", h)):
        if value is not None:
            attrs[key] = str(value)
    return ET.SubElement(cell, "mxGeometry", attrs)


def vertex(root, cid, value, style, x, y, w, h, parent=PLANNER):
    cell = ET.SubElement(
        root,
        "mxCell",
        {
            "id": cid,
            "parent": parent,
            "vertex": "1",
            "value": value,
            "style": style,
        },
    )
    geom(cell, x, y, w, h)
    return cell


def drop_prefix(root: ET.Element, prefix: str) -> None:
    doomed = {
        cell.get("id")
        for cell in root.findall("mxCell")
        if cell.get("id", "").startswith(prefix)
    }
    for cell in list(root.findall("mxCell")):
        if cell.get("id") in doomed:
            root.remove(cell)


def transform(src: Path, dst: Path, image_path: Path) -> None:
    tree = ET.parse(src)
    root = tree.getroot().find("./diagram/mxGraphModel/root")
    if root is None:
        raise ValueError("mxGraphModel/root not found")
    drop_prefix(root, "wm-balanced-")

    # A restrained journal palette: one cool base, three functional accents.
    ink = "#172B4D"
    navy = "#2F5D8A"
    risk_color = "#B2733D"
    green = "#2D7A78"
    tan = "#B2733D"
    coral = "#C65D4C"
    slate = "#65758A"

    # One strong title accent replaces extra stage captions and legends.
    title = find(root, P + "2080")
    geom(title, 34, 18, 520, 30)
    vertex(
        root,
        "wm-balanced-title-accent",
        "",
        "rounded=1;arcSize=20;html=1;fillColor=#C65D4C;strokeColor=none;",
        24,
        20,
        5,
        26,
        parent=PANEL,
    )

    # Card bodies receive very light semantic tints; the topology stays unchanged.
    cards = {
        P + "2078": ("#F8FAFC", navy),
        P + "2094": ("#FCFAF7", risk_color),
        "wm-simplified-safe-search": ("#FCFAF7", tan),
        "wm-simplified-safe-output": ("#F8FAFC", coral),
    }
    for cid, (fill, stroke) in cards.items():
        card = find(root, cid)
        card.set("value", "")
        card.set(
            "style",
            "rounded=1;arcSize=8;whiteSpace=wrap;html=1;"
            f"fillColor={fill};strokeColor={stroke};strokeWidth=1.5;"
            f"fontFamily=Arial;fontColor={ink};shadow=0;",
        )

    # Compact, color-coded stage headers make the sequence visible without a UI look.
    header_specs = (
        ("fusion", "01", "FEATURE FUSION", navy, 76, 345, 230, 34),
        ("risk", "02", "ENSEMBLE RISK", risk_color, 345, 370, 205, 34),
        ("cem", "04", "CEM REFINEMENT", tan, 380, 625, 340, 30),
        ("output", "05", "SELECTED HYPOTHESIS", coral, 800, 345, 210, 34),
    )
    for key, number, label, color, x, y, w, h in header_specs:
        vertex(
            root,
            f"wm-balanced-header-{key}",
            f"<font color=\"{color}\"><b>{number}</b></font>&nbsp;&nbsp;"
            f"<font color=\"{ink}\"><b>{label}</b></font>",
            "rounded=0;whiteSpace=wrap;html=1;"
            "fillColor=#EEF2F6;strokeColor=none;fontFamily=Arial;"
            "fontSize=10;align=left;verticalAlign=middle;spacingLeft=12;shadow=0;",
            x,
            y,
            w,
            h,
        )
        vertex(
            root,
            f"wm-balanced-rail-{key}",
            "",
            "rounded=1;arcSize=12;html=1;strokeColor=none;"
            f"fillColor={color};",
            x,
            y,
            5,
            h,
        )

    note_style = (
        "text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
        f"fontFamily=Arial;fontSize=9;fontColor={slate};"
        "align=center;verticalAlign=middle;"
    )
    vertex(root, "wm-balanced-fusion-note", "815-D concatenated descriptor", note_style, 94, 381, 194, 15)
    vertex(root, "wm-balanced-risk-note", "three independent scorers", note_style, 365, 406, 165, 15)
    vertex(root, "wm-balanced-cem-note", "3 rounds  ·  16 samples / round", note_style, 448, 656, 204, 15)
    vertex(
        root,
        "wm-balanced-output-h4",
        "<b>H4</b>",
        "text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
        f"fontFamily=Arial;fontSize=19;fontColor={ink};align=center;verticalAlign=middle;",
        860,
        380,
        90,
        24,
    )

    # The diamond remains the focal decision rather than becoming another rectangular card.
    safety = find(root, P + "2104")
    safety.set(
        "value",
        "<font style=\"font-size:17px;\"><b>SAFE?</b></font><br>"
        "<font style=\"font-size:9px;\">hard constraint gate</font>",
    )
    safety.set(
        "style",
        "rhombus;whiteSpace=wrap;html=1;fillColor=#EEF6F1;"
        f"strokeColor={green};strokeWidth=1.7;fontFamily=Arial;"
        f"fontColor={ink};align=center;verticalAlign=middle;shadow=0;",
    )
    vertex(
        root,
        "wm-balanced-safety-number",
        "<b>03</b>",
        "ellipse;whiteSpace=wrap;html=1;fillColor=#2D7A78;strokeColor=#FFFFFF;"
        "strokeWidth=1.5;fontFamily=Arial;fontSize=10;fontColor=#FFFFFF;"
        "align=center;verticalAlign=middle;shadow=0;",
        580,
        365,
        23,
        23,
    )

    # Replace the unreliable SVG data URI with a project-bound PNG asset.
    png_data = base64.b64encode(image_path.read_bytes()).decode("ascii")
    trajectory = find(root, "wm-clean-trajectory")
    trajectory.set("value", "")
    trajectory.set(
        "style",
        "shape=image;html=1;imageAspect=0;aspect=fixed;"
        f"image=data:image/png;base64,{png_data};strokeColor=none;fillColor=none;",
    )
    geom(trajectory, 814, 405, 182, 102)

    caption = find(root, "wm-clean-output-caption")
    caption.set("value", "candidate trajectories  ·  obstacle  ·  goal")
    geom(caption, 815, 508, 180, 14)
    action = find(root, "wm-clean-action")
    geom(action, 833, 531, 145, 42)

    # Subtle semantic emphasis inside the existing data glyphs.
    fusion_blocks = {
        "wm-clean-fusion-1": "#D9E7F2",
        "wm-clean-fusion-2": "#D3E1EE",
        "wm-clean-fusion-3": "#CBDCEB",
        "wm-clean-fusion-4": "#E5ECF2",
        "wm-clean-fusion-5": "#D9ECE8",
    }
    for cid, fill in fusion_blocks.items():
        cell = find(root, cid)
        style = cell.get("style", "")
        parts = [part for part in style.split(";") if part and not part.startswith("fillColor=")]
        cell.set("style", ";".join(parts) + f";fillColor={fill};")

    # Make the selected action read as the final commitment, not a generic data box.
    action.set(
        "style",
        "rounded=1;arcSize=6;whiteSpace=wrap;html=1;fillColor=#FCEFEA;"
        f"strokeColor={coral};strokeWidth=1.2;fontFamily=Arial;fontSize=9;"
        f"fontColor={ink};align=center;verticalAlign=middle;shadow=0;",
    )

    dst.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(dst, encoding="utf-8", xml_declaration=False)
    print(f"wrote {dst}; mxCells={len(root.findall('mxCell'))}; png_bytes={image_path.stat().st_size}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument(
        "--image",
        type=Path,
        default=Path("docs/diagrams/trajectory_planning_inset.png"),
    )
    args = parser.parse_args()
    transform(args.source, args.destination, args.image)


if __name__ == "__main__":
    main()
