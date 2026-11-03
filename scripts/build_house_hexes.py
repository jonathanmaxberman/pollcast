"""Build the House hex tile layout: one equal-size hexagon per district.

Usage: uv run python scripts/build_house_hexes.py <districts.csv>

Hex cells are laid over the Albers USA state map (us-atlas, 975x610), then
each state's seats are assigned to cells by solving a minimum-cost
assignment: a cell's cost for a state is its distance to that state's real
shape, plus a small pull toward the state's center so blobs stay compact.
Cells may sit a little offshore so small dense states (NJ, MA, MD) have room.
Within a state, district numbers run north to south, west to east.

The output, src/pollcast/web/house_hexes.json, is committed; rerun this only
when apportionment changes.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[1]
TOPOLOGY = ROOT / "data" / "cache" / "us-atlas-3.0.1-states-albers-10m.json"
OUT = ROOT / "src" / "pollcast" / "web" / "house_hexes.json"

ABBR = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
    "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "FL": "Florida", "GA": "Georgia",
    "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
    "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts",
    "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi", "MO": "Missouri", "MT": "Montana",
    "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey",
    "NM": "New Mexico", "NY": "New York", "NC": "North Carolina", "ND": "North Dakota",
    "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island",
    "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah",
    "VT": "Vermont", "VA": "Virginia", "WA": "Washington", "WV": "West Virginia",
    "WI": "Wisconsin", "WY": "Wyoming",
}


def decode_topology(topo: dict) -> dict[str, MultiPolygon]:
    """Minimal TopoJSON decoder for the us-atlas states object."""
    scale, translate = topo["transform"]["scale"], topo["transform"]["translate"]
    arcs = []
    for arc in topo["arcs"]:
        x = y = 0
        pts = []
        for dx, dy in arc:
            x += dx
            y += dy
            pts.append((x * scale[0] + translate[0], y * scale[1] + translate[1]))
        arcs.append(pts)

    def ring(indexes):
        pts = []
        for i in indexes:
            a = arcs[i] if i >= 0 else arcs[~i][::-1]
            pts.extend(a if not pts else a[1:])
        return pts

    states = {}
    for g in topo["objects"]["states"]["geometries"]:
        polys = g["arcs"] if g["type"] == "MultiPolygon" else [g["arcs"]]
        shapes = [Polygon(ring(p[0]), [ring(h) for h in p[1:]]) for p in polys]
        states[g["properties"]["name"]] = MultiPolygon(shapes).buffer(0)
    return states


def hex_centers(bounds, size):
    """Pointy-top hex grid in offset coordinates (odd rows shifted right)."""
    w, h = math.sqrt(3) * size, 1.5 * size
    x0, y0, x1, y1 = bounds
    cells = []
    for row in range(int((y1 - y0) / h) + 2):
        for col in range(int((x1 - x0) / w) + 2):
            cx = x0 + col * w + (w / 2 if row % 2 else 0)
            cy = y0 + row * h
            cells.append((col, row, cx, cy))
    return cells


def neighbors(col, row):
    odd = row % 2
    return [(col - 1, row), (col + 1, row),
            (col - 1 + odd, row - 1), (col + odd, row - 1),
            (col - 1 + odd, row + 1), (col + odd, row + 1)]


def components(cells: set) -> int:
    seen, n = set(), 0
    for c in cells:
        if c in seen:
            continue
        n += 1
        stack = [c]
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            stack += [nb for nb in neighbors(*cur) if nb in cells and nb not in seen]
    return n


def main(districts_csv: str, size: float = 15.5, offshore: float = 2.2, pull: float = 0.6):
    rows = list(csv.reader(open(districts_csv)))
    districts = [r[0] for r in rows if len(r) > 2 and len(r[0]) == 5 and r[0][2] == "-"]
    seats = defaultdict(list)
    for d in districts:
        seats[ABBR[d[:2]]].append(d)

    shapes = decode_topology(json.loads(TOPOLOGY.read_text()))
    nation = unary_union([shapes[s] for s in seats])
    allowed = nation.buffer(size * offshore)
    cells = [c for c in hex_centers(nation.bounds, size) if allowed.contains(Point(c[2], c[3]))]
    print(f"{len(cells)} candidate cells for {len(districts)} districts")

    slots = [(state, d) for state in sorted(seats) for d in sorted(seats[state])]
    pts = [Point(c[2], c[3]) for c in cells]
    cost = np.zeros((len(slots), len(cells)))
    by_state = {}
    for state in seats:
        shape, center = shapes[state], shapes[state].representative_point()
        # Big states need a larger tolerance before the centroid pull matters.
        radius = math.sqrt(shape.area / math.pi)
        by_state[state] = np.array(
            [shape.distance(p) ** 2 + pull * (center.distance(p) / max(radius, size)) ** 2 * size**2
             for p in pts]
        )
    for i, (state, _) in enumerate(slots):
        cost[i] = by_state[state]
    r, c = linear_sum_assignment(cost)

    layout = {}
    owned = defaultdict(list)
    for i, j in zip(r, c):
        owned[slots[i][0]].append(cells[j][:2])
    for state, cs in owned.items():
        # District numbers: north to south, then west to east.
        for d, cell in zip(sorted(seats[state]), sorted(cs, key=lambda t: (t[1], t[0]))):
            layout[d] = list(cell)

    broken = {s: components(set(map(tuple, cs))) for s, cs in owned.items()}
    broken = {s: n for s, n in broken.items() if n > 1}
    print("states split into pieces:", broken or "none")
    cols = [v[0] for v in layout.values()]
    rows_ = [v[1] for v in layout.values()]
    OUT.write_text(json.dumps({
        "source": "Generated by scripts/build_house_hexes.py from us-atlas state shapes",
        "grid": "pointy-top hexes, odd rows offset right",
        "cols": max(cols) + 1, "rows": max(rows_) + 1,
        "districts": dict(sorted(layout.items())),
    }, indent=0))
    print(f"wrote {OUT} ({len(layout)} districts, {max(cols)+1}x{max(rows_)+1} grid)")


if __name__ == "__main__":
    main(*sys.argv[1:2])
