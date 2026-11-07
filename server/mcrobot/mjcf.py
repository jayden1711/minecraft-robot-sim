"""Region -> merged boxes -> MJCF. Deterministic: same region gives byte-identical XML."""

from __future__ import annotations

import colorsys
import hashlib
from dataclasses import dataclass, field

import numpy as np

from .boxes import merge_boxes
from .region import Kind, Region, block_box_to_mj

PLATE_HEIGHT = 1 / 16

_COLORS = {
    "gold_block": (1.0, 0.84, 0.1, 1.0),
    "red_wool": (0.75, 0.1, 0.1, 1.0),
    "lava": (1.0, 0.4, 0.0, 0.6),
    "command_block": (0.8, 0.55, 0.25, 1.0),
    "stone": (0.5, 0.5, 0.5, 1.0),
    "cobblestone": (0.45, 0.45, 0.45, 1.0),
    "stone_bricks": (0.62, 0.42, 0.36, 1.0),
    "dirt": (0.45, 0.32, 0.2, 1.0),
    "grass_block": (0.35, 0.6, 0.25, 1.0),
    "oak_planks": (0.65, 0.5, 0.3, 1.0),
    "sand": (0.85, 0.8, 0.55, 1.0),
    "white_wool": (0.92, 0.92, 0.92, 1.0),
}


def block_rgba(name: str) -> tuple[float, float, float, float]:
    bare = name.removeprefix("minecraft:")
    if bare in _COLORS:
        return _COLORS[bare]
    if bare.endswith("_pressure_plate"):
        return (0.7, 0.7, 0.2, 1.0)
    h = int(hashlib.md5(bare.encode()).hexdigest()[:6], 16)
    r, g, b = colorsys.hsv_to_rgb((h % 360) / 360, 0.35, 0.7)
    return (r, g, b, 1.0)


@dataclass
class Geom:
    center: np.ndarray
    half: np.ndarray
    rgba: tuple[float, float, float, float]
    name: str


@dataclass
class Marker:
    kind: Kind
    center: np.ndarray
    half: np.ndarray


@dataclass
class Scene:
    colliders: list[Geom] = field(default_factory=list)
    visuals: list[Geom] = field(default_factory=list)
    markers: list[Marker] = field(default_factory=list)
    n_solid_cells: int = 0
    extent: np.ndarray = field(default_factory=lambda: np.zeros(3))  # MuJoCo-frame box (x, y, z sizes), meters


def build_scene(region: Region, try_orders: bool = True) -> Scene:
    s = region.meters_per_block
    sx, sy, sz = region.size
    scene = Scene(extent=np.array([sx, sz, sy], float) * s)
    coll = region.collider_labels()
    scene.n_solid_cells = int(coll.sum())

    for i, b in enumerate(merge_boxes(coll, try_orders)):
        c, h = block_box_to_mj((b[0], b[1], b[2]), (0, 0, 0, b[3] - b[0], b[4] - b[1], b[5] - b[2]), s)
        scene.colliders.append(Geom(c, h, (0.5, 0.5, 0.5, 1.0), f"col_{i}"))

    # Partial blocks (slabs, stairs, fences...): the mod sends their exact collision boxes, emitted unmerged.
    nonfull = [i for i, p in enumerate(region.palette) if p.partial_boxes]
    j = 0
    for pi in nonfull:
        for x, y, z in np.argwhere(region.grid == pi):
            for lb in region.palette[pi].partial_boxes:
                c, h = block_box_to_mj((x, y, z), tuple(lb), s)
                scene.colliders.append(Geom(c, h, (0.5, 0.5, 0.5, 1.0), f"colp_{j}"))
                j += 1

    # Visuals: merge per palette entry (so colors survive); contype 0 so they never touch physics.
    vis = np.where(coll.astype(bool), region.grid.astype(np.int64) + 1, 0)
    for k, b in enumerate(merge_boxes(vis, try_orders)):
        c, h = block_box_to_mj((b[0], b[1], b[2]), (0, 0, 0, b[3] - b[0], b[4] - b[1], b[5] - b[2]), s)
        scene.visuals.append(Geom(c, h, block_rgba(region.palette[b[6] - 1].name), f"vis_{k}"))
    k = len(scene.visuals)
    for pi in nonfull:
        for x, y, z in np.argwhere(region.grid == pi):
            for lb in region.palette[pi].partial_boxes:
                c, h = block_box_to_mj((x, y, z), tuple(lb), s)
                scene.visuals.append(Geom(c, h, block_rgba(region.palette[pi].name), f"vis_{k}"))
                k += 1

    # Task markers: sensors only (sites), never colliders. Solid markers (gold, red wool, command block)
    # are also in `coll` above, so they remain physical; lava and plates have empty collision shapes.
    mk = region.marker_labels()
    for pi, p in enumerate(region.palette):
        kind = mk[region.grid == pi]
        if kind.size == 0 or kind[0] == 0:
            continue
        local = (0, 0, 0, 1, PLATE_HEIGHT, 1) if p.name.endswith("_pressure_plate") else (0, 0, 0, 1, 1, 1)
        for x, y, z in np.argwhere(region.grid == pi):
            c, h = block_box_to_mj((x, y, z), local, s)
            scene.markers.append(Marker(Kind(int(kind[0])), c, h))
            if not p.is_full:
                scene.visuals.append(Geom(c, h, block_rgba(p.name), f"vis_m{len(scene.visuals)}"))
    scene.markers.sort(key=lambda m: (int(m.kind), tuple(m.center)))
    return scene


def _f(a) -> str:
    return " ".join(f"{float(v):.6g}" for v in a)


def _overview_camera(extent: np.ndarray) -> str:
    # Looks at the region center from the south-west, above. xyaxes = camera x (right) and y (up) in world frame.
    center = np.array([extent[0] / 2, -extent[1] / 2, extent[2] / 3])
    d = max(float(extent[0]), float(extent[1]), float(extent[2]), 4.0)
    pos = center + np.array([-0.15, -0.5, 1.3]) * d
    fwd = center - pos
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return f'<camera name="overview" pos="{_f(pos)}" xyaxes="{_f(right)} {_f(up)}" fovy="45"/>'


_MARKER_RGBA = {
    Kind.GOAL: "1 0.84 0.1 0.25",
    Kind.HAZARD: "0.9 0.1 0.1 0.25",
    Kind.CHECKPOINT: "0.2 0.6 1 0.25",
    Kind.SPAWN: "0.2 1 0.3 0.25",
}


def to_mjcf(scene: Scene, model_name: str = "mc_region", visuals: bool = True, markers: bool = True) -> str:
    L = [f'<mujoco model="{model_name}">', '  <option timestep="0.005" gravity="0 0 -9.81"/>']
    L.append('  <visual><global offwidth="1280" offheight="720"/><headlight ambient="0.5 0.5 0.5"/></visual>')
    L.append("  <worldbody>")
    L.append('    <light pos="0 0 40" dir="0 0 -1" diffuse="0.6 0.6 0.6"/>')
    L.append("    " + _overview_camera(scene.extent))
    for g in scene.colliders:
        L.append(
            f'    <geom name="{g.name}" type="box" pos="{_f(g.center)}" size="{_f(g.half)}" '
            'group="3" friction="1 0.005 0.0001" rgba="0.5 0.5 0.5 1"/>'
        )
    if visuals:
        for g in scene.visuals:
            L.append(
                f'    <geom name="{g.name}" type="box" pos="{_f(g.center)}" size="{_f(g.half)}" '
                f'contype="0" conaffinity="0" group="2" rgba="{_f(g.rgba)}"/>'
            )
    if markers:
        counts: dict[Kind, int] = {}
        for m in scene.markers:
            i = counts.get(m.kind, 0)
            counts[m.kind] = i + 1
            L.append(
                f'    <site name="{m.kind.name.lower()}_{i}" type="box" pos="{_f(m.center)}" size="{_f(m.half)}" '
                f'group="4" rgba="{_MARKER_RGBA[m.kind]}"/>'
            )
    L.append("  </worldbody>")
    L.append("</mujoco>")
    return "\n".join(L) + "\n"


def mjcf_sha256(xml: str) -> str:
    return hashlib.sha256(xml.encode()).hexdigest()
