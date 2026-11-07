"""Block regions as received from the mod, task-block classification, and Minecraft <-> MuJoCo coordinates."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

import numpy as np

# Minecraft: x east, y up, z south. MuJoCo: x east, y north, z up. mj = (x, -z, y) * meters_per_block
# (a proper rotation, no mirroring). Positions are relative to the region origin (min corner).


class Kind(enum.IntEnum):
    GOAL = 1
    HAZARD = 2
    CHECKPOINT = 3
    SPAWN = 4


def classify_marker(name: str) -> Kind | None:
    """Task meaning of a block, by registry name. See PLAN.md 'Reward spec from blocks'."""
    bare = name.removeprefix("minecraft:")
    if bare == "gold_block":
        return Kind.GOAL
    if bare in ("lava", "red_wool"):
        return Kind.HAZARD
    if bare.endswith("_pressure_plate"):
        return Kind.CHECKPOINT
    if bare in ("command_block", "chain_command_block", "repeating_command_block"):
        return Kind.SPAWN
    return None


@dataclass
class PaletteEntry:
    name: str
    shape: str | list  # "empty" | "full" | [[x0,y0,z0,x1,y1,z1], ...] block-local collision boxes

    @property
    def is_full(self) -> bool:
        return self.shape == "full"

    @property
    def partial_boxes(self) -> list[list[float]]:
        return self.shape if isinstance(self.shape, list) else []


@dataclass
class Region:
    origin: tuple[int, int, int]
    size: tuple[int, int, int]
    palette: list[PaletteEntry]
    grid: np.ndarray  # uint16 (sx, sy, sz) palette indices
    dimension: str = "minecraft:overworld"
    meters_per_block: float = 1.0
    meta: dict = field(default_factory=dict)

    @classmethod
    def from_wire(cls, header: dict, grid: np.ndarray) -> "Region":
        pal = [PaletteEntry(p["name"], p["shape"]) for p in header["palette"]]
        return cls(
            origin=tuple(header["origin"]),
            size=tuple(header["size"]),
            palette=pal,
            grid=grid,
            dimension=header.get("dimension", "minecraft:overworld"),
            meters_per_block=float(header.get("meters_per_block", 1.0)),
            meta={k: header[k] for k in ("mc_version", "label") if k in header},
        )

    def collider_labels(self) -> np.ndarray:
        """1 where the block's collision shape is a full cube, else 0."""
        full = np.array([p.is_full for p in self.palette], dtype=np.uint8)
        return full[self.grid]

    def marker_labels(self) -> np.ndarray:
        """Kind value per cell (0 = no task marker)."""
        kinds = np.array([int(classify_marker(p.name) or 0) for p in self.palette], dtype=np.uint8)
        return kinds[self.grid]


def block_box_to_mj(
    cell: tuple[float, float, float], local: tuple[float, float, float, float, float, float], scale: float
) -> tuple[np.ndarray, np.ndarray]:
    """A box in Minecraft block units (cell + local 0..1 sub-box) -> MuJoCo (center, half-size) in meters."""
    cx, cy, cz = cell
    x0, y0, z0, x1, y1, z1 = local
    lo = np.array([cx + x0, -(cz + z1), cy + y0]) * scale
    hi = np.array([cx + x1, -(cz + z0), cy + y1]) * scale
    return (lo + hi) / 2, (hi - lo) / 2


def mc_to_mj(p: np.ndarray, origin: tuple[int, int, int], scale: float = 1.0) -> np.ndarray:
    """Minecraft world position (x,y,z) -> MuJoCo position relative to the region origin."""
    d = np.asarray(p, float) - np.asarray(origin, float)
    return np.array([d[0], -d[2], d[1]]) * scale


def mj_to_mc(p: np.ndarray, origin: tuple[int, int, int], scale: float = 1.0) -> np.ndarray:
    q = np.asarray(p, float) / scale
    return np.array([q[0], q[2], -q[1]]) + np.asarray(origin, float)
