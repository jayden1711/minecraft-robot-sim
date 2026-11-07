"""Walkable-space analysis of an exported region MJCF (collider boxes + marker sites), for scripted paths and for the fit check.

Everything is in MuJoCo metres. One cell is one Minecraft block column, `Layout.cell` metres on a side (the world scale,
`meters_per_block`; read from the export's sibling .json header, 1.0 when there is none). A column's walking height is the top of
the lowest solid that leaves `clear` metres of headroom; columns without any solid (holes) are not walkable. Neighbouring columns
are connected when their heights differ by at most `max_step`. A cell is sampled at its centre, so at 0.5 m/block a 1-block
feature is 0.5 m wide: the planner does not check the body's width, the course design does (see results/m1/arena_course.md).
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Layout:
    sx: int  # cells along MuJoCo x
    sy: int  # cells along MuJoCo -y (rows)
    boxes: list = field(default_factory=list)  # (pos, half) np arrays
    markers: dict = field(default_factory=dict)  # kind -> list of (centre, half)
    cell: float = 1.0  # metres per cell (= meters_per_block)

    def column_intervals(self, cx: int, cy: int):
        """Solid z intervals over the centre of column (cx, cy); cy is the row index (MuJoCo y in (-(cy+1)*cell, -cy*cell))."""
        x, y = (cx + 0.5) * self.cell, -(cy + 0.5) * self.cell
        iv = []
        for p, h in self.boxes:
            if abs(p[0] - x) < h[0] and abs(p[1] - y) < h[1]:
                iv.append((p[2] - h[2], p[2] + h[2]))
        iv.sort()
        merged = []
        for lo, hi in iv:
            if merged and lo <= merged[-1][1] + 1e-6:
                merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
            else:
                merged.append((lo, hi))
        return merged

    def height(self, cx: int, cy: int, clear: float = 1.5):
        iv = self.column_intervals(cx, cy)
        for k, (lo, hi) in enumerate(iv):
            nxt = iv[k + 1][0] if k + 1 < len(iv) else np.inf
            if nxt - hi >= clear:
                return hi
        return None


def load_layout(xml_path: str | Path, cell: float | None = None) -> Layout:
    xml_path = Path(xml_path)
    if cell is None:
        js = xml_path.with_suffix(".json")
        cell = float(json.loads(js.read_text()).get("meters_per_block", 1.0)) if js.exists() else 1.0
    root = ET.parse(xml_path).getroot()
    boxes, markers = [], {}
    for g in root.iter("geom"):
        if g.get("name", "").startswith(("col_", "colp_")):
            boxes.append((np.array(g.get("pos").split(), float), np.array(g.get("size").split(), float)))
    for s in root.iter("site"):
        kind = s.get("name").rsplit("_", 1)[0]
        markers.setdefault(kind, []).append((np.array(s.get("pos").split(), float), np.array(s.get("size").split(), float)))
    ext = max([p[0] + h[0] for p, h in boxes] + [cell]), max([-(p[1] - h[1]) for p, h in boxes] + [cell])
    return Layout(int(np.ceil(ext[0] / cell - 1e-9)), int(np.ceil(ext[1] / cell - 1e-9)), boxes, markers, cell)


def cell_of(p, lay: Layout):
    return int(np.floor(p[0] / lay.cell)), int(np.floor(-p[1] / lay.cell))


def hazard_cells(lay: Layout, margin: float = 0.10):
    """Cells whose centre is inside a hazard box grown by `margin` metres (the task rule of tasks.py)."""
    out = set()
    for c, h in lay.markers.get("hazard", []):
        for cx in range(lay.sx):
            for cy in range(lay.sy):
                x, y = (cx + 0.5) * lay.cell, -(cy + 0.5) * lay.cell
                if abs(x - c[0]) <= h[0] + margin and abs(y - c[1]) <= h[1] + margin:
                    out.add((cx, cy))
    return out


def walkable_map(lay: Layout, max_step: float = 0.08, clear: float = 1.5, block_hazards: bool = True):
    """dict cell -> height for every cell reachable by neighbour steps of at most max_step from the spawn cell."""
    heights = {(cx, cy): lay.height(cx, cy, clear) for cx in range(lay.sx) for cy in range(lay.sy)}
    heights = {c: h for c, h in heights.items() if h is not None}
    bad = hazard_cells(lay) if block_hazards else set()
    sp = cell_of(lay.markers["spawn"][0][0], lay)
    seen = {sp: heights[sp]}
    q = deque([sp])
    while q:
        c = q.popleft()
        for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (c[0] + d[0], c[1] + d[1])
            if n in seen or n not in heights or n in bad:
                continue
            if abs(heights[n] - heights[c]) <= max_step:
                seen[n] = heights[n]
                q.append(n)
    return seen


def bfs_path(cells: dict, start, goal):
    prev = {start: None}
    q = deque([start])
    while q:
        c = q.popleft()
        if c == goal:
            break
        for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (c[0] + d[0], c[1] + d[1])
            if n in cells and n not in prev:
                prev[n] = c
                q.append(n)
    if goal not in prev:
        return None
    path, c = [], goal
    while c is not None:
        path.append(c)
        c = prev[c]
    return path[::-1]


def farthest_cell(cells: dict, start):
    dist = {start: 0}
    q = deque([start])
    while q:
        c = q.popleft()
        for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (c[0] + d[0], c[1] + d[1])
            if n in cells and n not in dist:
                dist[n] = dist[c] + 1
                q.append(n)
    return max(dist, key=dist.get)


def corners(path):
    """Cells where the path changes direction (plus the ends)."""
    out = [path[0]]
    for a, b, c in zip(path, path[1:], path[2:]):
        if (b[0] - a[0], b[1] - a[1]) != (c[0] - b[0], c[1] - b[1]):
            out.append(b)
    out.append(path[-1])
    return out


def cell_centre(c, z: float = 0.0, cell: float = 1.0):
    return np.array([(c[0] + 0.5) * cell, -(c[1] + 0.5) * cell, z])
