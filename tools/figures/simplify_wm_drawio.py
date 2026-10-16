#!/usr/bin/env python3
"""Create a topology-preserving, paper-style simplification of wm.drawio.

The script intentionally leaves the visual encoder, world model, and candidate
proposal regions untouched.  It only replaces the dense SAFE CANDIDATE PLANNER
internals with four semantic blocks and a compact refinement loop.
"""

from __future__ import annotations

import argparse
import copy
import xml.etree.ElementTree as ET
from pathlib import Path


PREFIX = "Lcp2UKG11IvjxIMB6Xdv-"
ROOT_PARENT = PREFIX + "2164"


def cell_map(root: ET.Element) -> dict[str, ET.Element]:
    return {cell.get("id", ""): cell for cell in root.findall("mxCell")}


def geometry(
    cell: ET.Element,
    *,
    x: float | None = None,
    y: float | None = None,
    width: float | None = None,
    height: float | None = None,
    relative: bool = False,
) -> ET.Element:
    old = cell.find("mxGeometry")
    if old is not None:
        cell.remove(old)
    attrs: dict[str, str] = {"as": "geometry"}
    if relative:
        attrs["relative"] = "1"
    if x is not None:
        attrs["x"] = f"{x:g}"
    if y is not None:
        attrs["y"] = f"{y:g}"
    if width is not None:
        attrs["width"] = f"{width:g}"
    if height is not None:
        attrs["height"] = f"{height:g}"
    return ET.SubElement(cell, "mxGeometry", attrs)


def remove_ids(root: ET.Element, ids: set[str]) -> None:
    """Remove selected cells, their descendants, and dangling edges."""
    changed = True
    while changed:
        changed = False
        for cell in root.findall("mxCell"):
            if cell.get("parent") in ids and cell.get("id") not in ids:
                ids.add(cell.get("id", ""))
                changed = True

    for cell in list(root.findall("mxCell")):
        if (
            cell.get("id") in ids
            or cell.get("source") in ids
            or cell.get("target") in ids
        ):
            root.remove(cell)


def add_vertex(
    root: ET.Element,
    *,
    cell_id: str,
    value: str,
    style: str,
    x: float,
    y: float,
    width: float,
    height: float,
    parent: str = ROOT_PARENT,
) -> ET.Element:
    cell = ET.SubElement(
        root,
        "mxCell",
        {
            "id": cell_id,
            "parent": parent,
            "style": style,
            "value": value,
            "vertex": "1",
        },
    )
    geometry(cell, x=x, y=y, width=width, height=height)
    return cell


def add_edge(
    root: ET.Element,
    *,
    cell_id: str,
    source: str,
    target: str,
    style: str,
    value: str = "",
    points: tuple[tuple[float, float], ...] = (),
) -> ET.Element:
    cell = ET.SubElement(
        root,
        "mxCell",
        {
            "id": cell_id,
            "edge": "1",
            "parent": ROOT_PARENT,
            "source": source,
            "target": target,
            "style": style,
            "value": value,
        },
    )
    geo = geometry(cell, relative=True)
    if points:
        array = ET.SubElement(geo, "Array", {"as": "points"})
        for x, y in points:
            ET.SubElement(array, "mxPoint", {"x": f"{x:g}", "y": f"{y:g}"})
    return cell


def simplify(source: Path, destination: Path) -> tuple[int, int]:
    tree = ET.parse(source)
    graph_root = tree.getroot().find("./diagram/mxGraphModel/root")
    if graph_root is None:
        raise ValueError("Unsupported draw.io file: mxGraphModel/root not found")

    before = len(graph_root.findall("mxCell"))
    cells = cell_map(graph_root)

    required = {
        PREFIX + "2077",  # planner background/title group
        PREFIX + "2078",  # fusion input
        PREFIX + "2094",  # scorer
        PREFIX + "2104",  # risk/safety card
    }
    missing = sorted(required - cells.keys())
    if missing:
        raise ValueError(f"wm.drawio layout has changed; missing cells: {missing}")

    # Dense planner-only details.  Parent groups recursively remove their
    # map/grid children, per-seed cards, gate pills, CEM dots, and duplicate H4.
    remove_ids(
        graph_root,
        {
            PREFIX + "2040",
            PREFIX + "2082",
            PREFIX + "2083",
            PREFIX + "2084",
            PREFIX + "2085",
            PREFIX + "2086",
            PREFIX + "2087",
            PREFIX + "2088",
            PREFIX + "2089",
            PREFIX + "2090",
            PREFIX + "2091",
            PREFIX + "2092",
            PREFIX + "2093",
            PREFIX + "2095",
            PREFIX + "2096",
            PREFIX + "2097",
            PREFIX + "2098",
            PREFIX + "2099",
            PREFIX + "2100",
            PREFIX + "2101",
            PREFIX + "2102",
            PREFIX + "2103",
            PREFIX + "2105",
            PREFIX + "2106",
            PREFIX + "2107",
            PREFIX + "2108",
            PREFIX + "2109",
            PREFIX + "2110",
            PREFIX + "2111",
            PREFIX + "2112",
            PREFIX + "2113",
            PREFIX + "2114",
            PREFIX + "2115",
            PREFIX + "2116",
            PREFIX + "2117",
            PREFIX + "2119",
            PREFIX + "2154",
        },
    )

    cells = cell_map(graph_root)

    # Turn the existing planner group into a quiet visual container.  Keeping
    # its original location preserves the source figure's topology.
    planner = cells[PREFIX + "2077"]
    planner.set(
        "style",
        "rounded=1;arcSize=12;whiteSpace=wrap;html=1;"
        "fillColor=#FAFBFD;strokeColor=#B8C4D1;strokeWidth=2;"
        "fontFamily=Arial;shadow=0;",
    )

    tag = cells[PREFIX + "2079"]
    tag.set("value", "DECISION CORE")
    tag.set(
        "style",
        "rounded=1;arcSize=50;whiteSpace=wrap;html=1;"
        "fillColor=#2C875E;strokeColor=#2C875E;strokeWidth=1;"
        "fontFamily=Arial;fontSize=11;fontColor=#FFFFFF;fontStyle=1;",
    )
    geometry(tag, x=22, y=14, width=132, height=29)

    title = cells[PREFIX + "2080"]
    title.set("value", "SAFE CANDIDATE PLANNER")
    title.set(
        "style",
        "text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
        "fontFamily=Arial;fontSize=22;fontColor=#17212B;fontStyle=1;"
        "align=left;verticalAlign=middle;",
    )
    geometry(title, x=22, y=47, width=420, height=34)

    subtitle = cells[PREFIX + "2081"]
    subtitle.set("value", "score · filter · refine · fallback")
    subtitle.set(
        "style",
        "text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
        "fontFamily=Arial;fontSize=13;fontColor=#647081;"
        "align=left;verticalAlign=middle;",
    )
    geometry(subtitle, x=22, y=82, width=390, height=22)

    fusion = cells[PREFIX + "2078"]
    fusion.set(
        "value",
        "<b>Candidate fusion</b><br>"
        "<font style=\"font-size:11px;color:#647081;\">"
        "DINO · LTWM · action · guard</font>",
    )
    fusion.set(
        "style",
        "rounded=1;arcSize=16;whiteSpace=wrap;html=1;"
        "fillColor=#FFFFFF;strokeColor=#0C7976;strokeWidth=3;"
        "fontFamily=Arial;fontColor=#17212B;fontSize=16;fontStyle=0;"
        "align=center;verticalAlign=middle;spacing=8;shadow=0;",
    )
    geometry(fusion, x=92, y=400, width=132, height=112)

    scorer = cells[PREFIX + "2094"]
    scorer.set(
        "value",
        "<font color=\"#FFFFFF\"><b>Risk-aware scoring</b></font><br>"
        "<font style=\"font-size:11px;color:#D9EEEC;\">"
        "Arm-D ensemble ×3<br>quality + uncertainty (μ, σ)</font>",
    )
    scorer.set(
        "style",
        "rounded=1;arcSize=16;whiteSpace=wrap;html=1;"
        "fillColor=#173A4A;strokeColor=#0C7976;strokeWidth=3;"
        "fontFamily=Arial;fontColor=#FFFFFF;fontSize=16;"
        "align=center;verticalAlign=middle;spacing=9;shadow=0;",
    )
    geometry(scorer, x=270, y=362, width=210, height=126)

    safety = cells[PREFIX + "2104"]
    safety.set(
        "value",
        "<b>Safety filter</b><br>"
        "<font style=\"font-size:11px;color:#466A58;\">"
        "physical · support · harm<br>uncertainty-aware gate</font>",
    )
    safety.set(
        "style",
        "rounded=1;arcSize=16;whiteSpace=wrap;html=1;"
        "fillColor=#EDF7F1;strokeColor=#2C875E;strokeWidth=3;"
        "fontFamily=Arial;fontColor=#214C38;fontSize=16;"
        "align=center;verticalAlign=middle;spacing=9;shadow=0;",
    )
    geometry(safety, x=535, y=362, width=190, height=126)

    search_id = "wm-simplified-safe-search"
    output_id = "wm-simplified-safe-output"
    add_vertex(
        graph_root,
        cell_id=search_id,
        value=(
            "<b>Safe CEM refinement</b><br>"
            "<font style=\"font-size:11px;color:#795631;\">"
            "3 rounds · 16 proposals/round<br>cross-round safe archive</font>"
        ),
        style=(
            "rounded=1;arcSize=16;whiteSpace=wrap;html=1;"
            "fillColor=#FBF5EF;strokeColor=#B27B48;strokeWidth=3;"
            "fontFamily=Arial;fontColor=#6D4826;fontSize=16;"
            "align=center;verticalAlign=middle;spacing=9;shadow=0;"
        ),
        x=400,
        y=548,
        width=250,
        height=108,
    )
    add_vertex(
        graph_root,
        cell_id=output_id,
        value=(
            "<font color=\"#FFFFFF\"><b>SAFE H4</b></font><br>"
            "<font style=\"font-size:12px;color:#EAF6EF;\">"
            "best feasible plan<br>execute a₀ · replan</font><br>"
            "<font style=\"font-size:10px;color:#C9E6D5;\">"
            "no safe plan → incumbent</font>"
        ),
        style=(
            "rounded=1;arcSize=16;whiteSpace=wrap;html=1;"
            "fillColor=#2C875E;strokeColor=#246D4C;strokeWidth=3;"
            "fontFamily=Arial;fontColor=#FFFFFF;fontSize=18;"
            "align=center;verticalAlign=middle;spacing=9;shadow=1;"
        ),
        x=755,
        y=412,
        width=155,
        height=132,
    )

    edge_common = (
        "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;"
        "jettySize=auto;html=1;strokeWidth=3;endArrow=blockThin;"
        "endFill=1;endSize=9;fontFamily=Arial;fontSize=11;"
        "labelBackgroundColor=#FAFBFD;"
    )
    add_edge(
        graph_root,
        cell_id="wm-simplified-edge-fusion-score",
        source=PREFIX + "2078",
        target=PREFIX + "2094",
        style=edge_common
        + "strokeColor=#0C7976;exitX=1;exitY=0.5;entryX=0;entryY=0.65;",
    )
    add_edge(
        graph_root,
        cell_id="wm-simplified-edge-score-safety",
        source=PREFIX + "2094",
        target=PREFIX + "2104",
        style=edge_common
        + "strokeColor=#2C875E;exitX=1;exitY=0.5;entryX=0;entryY=0.5;",
    )
    add_edge(
        graph_root,
        cell_id="wm-simplified-edge-safety-search",
        source=PREFIX + "2104",
        target=search_id,
        style=edge_common
        + "strokeColor=#B27B48;exitX=0.5;exitY=1;entryX=0.75;entryY=0;",
        value="safe elites",
        points=((630, 520), (590, 520)),
    )
    add_edge(
        graph_root,
        cell_id="wm-simplified-edge-refine-loop",
        source=search_id,
        target=PREFIX + "2094",
        style=(
            edge_common
            + "strokeColor=#8A94A2;dashed=1;dashPattern=7 5;"
            "exitX=0;exitY=0.58;entryX=0.52;entryY=1;"
        ),
        value="refine ×3",
        points=((340, 610), (340, 520), (380, 520)),
    )
    add_edge(
        graph_root,
        cell_id="wm-simplified-edge-search-output",
        source=search_id,
        target=output_id,
        style=edge_common
        + "strokeColor=#2C875E;strokeWidth=4;"
        "exitX=1;exitY=0.5;entryX=0;entryY=0.72;",
        value="best safe",
        points=((700, 602), (700, 507)),
    )

    # Reduce version-specific noise without changing the surrounding layout.
    top_badge = cells.get(PREFIX + "2")
    if top_badge is not None:
        top_badge.set("value", "INFERENCE-TIME FRAMEWORK")
    title_note = cells.get(PREFIX + "4")
    if title_note is not None:
        title_note.set(
            "value",
            "visual context + candidate actions → latent futures → safe H4",
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(destination, encoding="utf-8", xml_declaration=False)
    after = len(graph_root.findall("mxCell"))
    return before, after


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()

    before, after = simplify(args.source, args.destination)
    print(f"wrote {args.destination}")
    print(f"mxCells: {before} -> {after} ({before - after} removed)")


if __name__ == "__main__":
    main()
