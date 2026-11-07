"""Synthetic block regions (for tests and for the M0 export-matches-Minecraft comparison build)."""

from __future__ import annotations

import numpy as np

from .region import DEFAULT_METERS_PER_BLOCK, PaletteEntry, Region

FULL = "full"
EMPTY = "empty"
SLAB_BOTTOM = [[0, 0, 0, 1, 0.5, 1]]
PLATE = [[0.0625, 0, 0.0625, 0.9375, 0, 0.9375]]  # real plates have no collision; kept "empty" below

_SHAPES = {
    "air": EMPTY,
    "stone": FULL,
    "stone_bricks": FULL,
    "oak_planks": FULL,
    "gold_block": FULL,
    "red_wool": FULL,
    "command_block": FULL,
    "lava": EMPTY,
    "stone_pressure_plate": EMPTY,
    "stone_slab": SLAB_BOTTOM,
}


def from_layers(layers: list[list[str]], legend: dict[str, str], origin=(0, 64, 0), meters_per_block: float = DEFAULT_METERS_PER_BLOCK) -> Region:
    """layers[y][z] is a string over x. Legend maps a character to a block name ('.' defaults to air)."""
    legend = {".": "air", **legend}
    sy, sz, sx = len(layers), len(layers[0]), len(layers[0][0])
    names = sorted({legend[c] for L in layers for row in L for c in row})
    pal = [PaletteEntry(f"minecraft:{n}", _SHAPES[n]) for n in names]
    idx = {n: i for i, n in enumerate(names)}
    grid = np.zeros((sx, sy, sz), np.uint16)
    for y, L in enumerate(layers):
        for z, row in enumerate(L):
            assert len(row) == sx
            for x, c in enumerate(row):
                grid[x, y, z] = idx[legend[c]]
    return Region(tuple(origin), (sx, sy, sz), pal, grid, meters_per_block=meters_per_block)


def demo_maze(meters_per_block: float = DEFAULT_METERS_PER_BLOCK) -> Region:
    """9x4x9 build: stone floor, brick maze walls 2 high, spawn (command block), plate, slab, red wool, lava, gold goal."""
    floor = ["#########"] * 9
    floor[1] = "#C#######"
    floor[5] = "###L#####"
    floor[7] = "#######G#"
    walls = [
        "WWWWWWWWW",
        "W.......W",
        "W.WWWWW.W",
        "W.W...W.W",
        "W.W.W.W.W",
        "W...W.W.W",
        "WWWWW.W.W",
        "W.....W.W",
        "WWWWWWWWW",
    ]
    layer1 = [list(r) for r in walls]
    layer1[1][3] = "S"  # slab in a corridor
    layer1[5][1] = "P"  # pressure plate
    layer1[3][7] = "R"  # red wool
    layer1 = ["".join(r) for r in layer1]
    legend = {
        "#": "stone",
        "C": "command_block",
        "L": "lava",
        "G": "gold_block",
        "W": "stone_bricks",
        "S": "stone_slab",
        "P": "stone_pressure_plate",
        "R": "red_wool",
    }
    # y=0 floor (lava is a hole in the floor); y=1 wall layer with extras; y=2 wall layer; y=3 air
    air = ["........."] * 9
    return from_layers([floor, layer1, walls, air], legend, meters_per_block=meters_per_block)


def demo_maze_wide(meters_per_block: float = DEFAULT_METERS_PER_BLOCK) -> Region:
    """11x4x10 serpentine, 2-block-wide corridors (G1 body is 0.53 m), no step-ups, goal reachable.

    One red wool block and one lava hole each leave a free 1-block lane, so hazards must be avoided but never block the path.
    Path: spawn (2,1) east along z 1-2, down at x 8-9, west along z 4-5, down at x 1-2, east along z 7-8 to the gold block.
    """
    sx, sz = 11, 10
    walls = [["W"] * sx for _ in range(sz)]
    for z0 in (1, 4, 7):
        for z in (z0, z0 + 1):
            for x in range(1, sx - 1):
                walls[z][x] = "."
    for x in (8, 9):
        walls[3][x] = "."
    for x in (1, 2):
        walls[6][x] = "."
    floor = [["#"] * sx for _ in range(sz)]
    floor[1][2] = "C"  # spawn
    floor[8][9] = "G"  # goal
    floor[7][5] = "L"  # lava hole in the last corridor; lane z=8 stays open
    layer1 = [r[:] for r in walls]
    layer1[4][5] = "R"  # red wool in the middle corridor; lane z=5 stays open
    for x, z in ((8, 2), (2, 5), (7, 8)):  # checkpoints along the way
        layer1[z][x] = "P"
    strs = lambda g: ["".join(r) for r in g]  # noqa: E731
    legend = {"#": "stone", "C": "command_block", "L": "lava", "G": "gold_block", "W": "stone_bricks", "P": "stone_pressure_plate", "R": "red_wool"}
    air = ["." * sx] * sz
    return from_layers([strs(floor), strs(layer1), strs(walls), air], legend, meters_per_block=meters_per_block)
