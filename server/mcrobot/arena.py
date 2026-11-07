"""The sky arena: a floating demo stage for the G1, defined once here and used three ways.

  * `scripts/gen_arena_datapack.py` turns `build_arena()` into the data pack function that builds it in Minecraft;
  * `to_region()` turns it into the same Region the mod's exporter would send (checked against the real exporter by the smoke test),
    so the navigation training and the demo can use it without a client;
  * `plan.route` is the centre line the scripted G1 demo follows.

Coordinates in `Plan.blocks` are region-local blocks: x east (the course runs +x), y up, z south. Floor blocks are at y = FL.
World scale is `meters_per_block` (default 0.5 m): the G1 (1.32 m, 0.53 m wide with its arms down) is 2.6 blocks tall and about
1 block wide, a 2-block lane is 1 m. Every dimension below is chosen from the measured limits of the frozen walking policy
(results/m1/fit.md): step up <= 5 cm, gap <= 15 cm, corridor >= 0.6 m. See results/m1/arena_course.md for the section feasibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .region import DEFAULT_METERS_PER_BLOCK, PaletteEntry, Region

# World placement of the region's minimum corner (x, y, z). High above the terrain (about y 135) and away from the maze (x -18..-10).
ORIGIN = (96, 200, -160)
FL = 3  # local y of the floor layer; the floor surface is at FL + 1
SIZE = (92, 14, 34)

LANE_W = 6  # blocks (3 m at 0.5 m/block)
ZC = 15  # lane centre line at z = ZC (a block edge), lane occupies z 12..17
BEAM_W = 2  # blocks (1 m at the default scale). 1 block (0.5 m) is marginal in the sim: see results/m1/arena_course.md

FULL, EMPTY = "full", "empty"
_H = lambda h: [[0.0, 0.0, 0.0, 1.0, h, 1.0]]  # noqa: E731  bottom-attached box of height h (blocks)
SHAPES = {
    "air": EMPTY, "lava": EMPTY, "stone_pressure_plate": EMPTY,
    "white_concrete": FULL, "light_gray_concrete": FULL, "light_blue_concrete": FULL, "gray_concrete": FULL, "black_concrete": FULL,
    "quartz_block": FULL, "smooth_quartz": FULL, "quartz_pillar": FULL, "chiseled_quartz_block": FULL, "sea_lantern": FULL,
    "glass": FULL, "gold_block": FULL, "command_block": FULL, "dark_oak_planks": FULL, "blackstone": FULL, "polished_blackstone": FULL,
    "light_gray_carpet": _H(1 / 16), "repeater": _H(2 / 16), "spruce_trapdoor": _H(3 / 16), "quartz_slab": _H(0.5),
}
# micro-steps: each level is 1/16 block = 3.1 cm at 0.5 m/block (the policy's limit is 5 cm per step)
STEP_BLOCKS = {
    1: "minecraft:light_gray_carpet",
    2: "minecraft:repeater[delay=1,facing=east,locked=false,powered=false]",
    3: "minecraft:spruce_trapdoor[facing=north,half=bottom,open=false,powered=false,waterlogged=false]",
}


@dataclass
class Section:
    name: str
    x0: int  # first / last+1 local x of the section
    x1: int
    note: str = ""


@dataclass
class Plan:
    blocks: dict = field(default_factory=dict)  # (x, y, z) -> state string "minecraft:name[props]"
    sections: list = field(default_factory=list)
    route: list = field(default_factory=list)  # (x, z, label) region-local block coordinates, +x east, z south
    plates: list = field(default_factory=list)  # (x, z) checkpoint plates, one per section end
    spawn: tuple = (3, 15)
    size: tuple = SIZE
    walk: set = field(default_factory=set)  # (x, z) walkable cells

    def put(self, x, y, z, state):
        self.blocks[(x, y, z)] = state if ":" in state else "minecraft:" + state

    def fill(self, x0, y0, z0, x1, y1, z1, state):  # inclusive corners
        for x in range(x0, x1 + 1):
            for y in range(y0, y1 + 1):
                for z in range(z0, z1 + 1):
                    self.put(x, y, z, state)


def _lane(p: Plan, x0, x1, z0=ZC - LANE_W // 2, z1=ZC + LANE_W // 2 - 1, floor="white_concrete"):
    for x in range(x0, x1 + 1):
        for z in range(z0, z1 + 1):
            p.put(x, FL, z, floor)
            p.walk.add((x, z))


def build_arena(beam_w: int = BEAM_W) -> Plan:
    p = Plan()
    zl0, zl1 = ZC - LANE_W // 2, ZC + LANE_W // 2 - 1  # 12..17

    # --- start pad (x 0..7): quartz, the spawn command block, light strips
    _lane(p, 0, 7, floor="quartz_block")
    p.put(3, FL, ZC, "command_block")
    p.sections.append(Section("start pad", 0, 8, "spawn on the command block"))
    # --- S1 warm-up straight (x 8..15)
    _lane(p, 8, 15)
    p.sections.append(Section("straight", 8, 16))
    # --- S2 micro-stairs: 3 up, 1 top, 3 down; each riser 1/16 block
    _lane(p, 16, 25)
    for x, lv in zip(range(17, 24), (1, 2, 3, 3, 3, 2, 1)):
        for z in range(zl0, zl1 + 1):
            p.put(x, FL + 1, z, STEP_BLOCKS[lv])
    p.sections.append(Section("stairs", 16, 26, "micro-steps of 3.1 cm (carpet, repeater, trapdoor)"))
    # --- S3 slope: 2 blocks per level, up two levels and down
    _lane(p, 26, 39)
    for x, lv in zip(range(28, 38), (1, 1, 2, 2, 2, 2, 2, 2, 1, 1)):
        for z in range(zl0, zl1 + 1):
            p.put(x, FL + 1, z, STEP_BLOCKS[lv])
    p.sections.append(Section("slope", 26, 40, "gentle 2-level ramp, 2 blocks per level"))
    # --- S4 pit with a bridge: lava bowl either side, a 2-block bridge on the lane axis
    _lane(p, 40, 41)
    _lane(p, 48, 49)
    pit_x0, pit_x1 = 42, 47
    for x in range(pit_x0 - 1, pit_x1 + 2):
        for z in range(zl0 - 1, zl1 + 2):
            for y in range(0, FL):  # bowl: walls and bottom of blackstone under the lane and the pit
                p.put(x, y, z, "blackstone")
    for x in range(pit_x0, pit_x1 + 1):
        for z in range(zl0, zl1 + 1):
            if ZC - 1 <= z <= ZC:  # bridge cells (2 wide)
                p.put(x, FL, z, "dark_oak_planks")
                p.walk.add((x, z))
            else:
                p.put(x, 1, z, "blackstone")  # bowl bottom top at y=1: pit 2 deep
                p.put(x, 2, z, "lava")
    p.sections.append(Section("pit and bridge", 40, 50, "lava pit either side of a 2-block (1 m) bridge"))
    # --- S5 narrow beam over the void, `beam_w` blocks wide
    b0 = ZC - beam_w // 2
    for x in range(50, 58):
        for z in range(b0, b0 + beam_w):
            p.put(x, FL, z, "polished_blackstone")
            p.walk.add((x, z))
    p.sections.append(Section("beam", 50, 58, f"{beam_w}-block ({beam_w * 0.5:g} m) beam over the void"))
    # --- S6 turns: up the connector A, along the upper lane, down the connector B (4 turns of 90 degrees)
    _lane(p, 58, 63)  # lower lane to the connector
    _lane(p, 58, 63, z0=3, z1=zl1)  # connector A: x 58..63, z 3..17
    _lane(p, 64, 75, z0=3, z1=8)  # upper lane
    _lane(p, 70, 75, z0=3, z1=zl1)  # connector B: x 70..75
    _lane(p, 76, 83)
    p.sections.append(Section("turns", 58, 84, "four 90 degree turns, 3 m wide lane"))
    # --- finish (x 84..91): gold pad and a gate
    _lane(p, 84, 91, floor="quartz_block")
    for x in (86, 87, 88):
        for z in range(zl0, zl1 + 1):
            p.put(x, FL, z, "gold_block")
    for z in (zl0 - 1, zl1 + 1):
        p.fill(87, FL, z, 87, FL + 6, z, "quartz_pillar")
        p.put(87, FL + 7, z, "sea_lantern")
    p.fill(87, FL + 6, zl0, 87, FL + 6, zl1, "quartz_block")  # crossbar: 5 blocks = 2.5 m clear above the floor
    p.put(87, FL + 6, ZC - 1, "sea_lantern")
    p.put(87, FL + 6, ZC, "sea_lantern")
    p.sections.append(Section("finish gate", 84, 92, "gold goal pad, gate with 2.5 m of clearance"))

    # --- rails: glass, 2 high, on the cells beside every walkable lane except at the pit, the beam and the finish gate pillars
    rails_off = {(x, z) for x in range(42, 58) for z in range(0, SIZE[2])}
    for (x, z) in sorted(p.walk):
        for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (x + dx, z + dz)
            if n in p.walk or n in rails_off or (x + dx, FL, z + dz) in p.blocks:
                continue
            if 0 <= n[0] < SIZE[0] and 0 <= n[1] < SIZE[2]:
                p.put(n[0], FL, n[1], "white_concrete")
                p.fill(n[0], FL + 1, n[1], n[0], FL + 2, n[1], "glass")
    # sea lanterns in the floor along the lane edges, every 4th block
    for (x, z) in sorted(p.walk):
        if x % 4 == 2 and (x, z) not in rails_off and ((x, z - 1) not in p.walk or (x, z + 1) not in p.walk) and p.blocks.get((x, FL, z), "").endswith(("white_concrete", "quartz_block")):
            if (x, FL + 1, z) not in p.blocks:
                p.put(x, FL, z, "sea_lantern")
    # --- spectator stand south of the lane: floor, two tiers with slab seats, glass rail towards the course
    for x in range(26, 66):
        for z in range(21, 31):
            p.put(x, FL, z, "quartz_block")
    for x in range(27, 65):
        p.put(x, FL, 20, "quartz_block")
        p.fill(x, FL + 1, 20, x, FL + 2, 20, "glass")
        for z, h in ((24, 1), (26, 2), (28, 3)):
            p.fill(x, FL + 1, z, x, FL + h, z, "quartz_block")
            p.put(x, FL + h + 1, z, "quartz_slab[type=bottom,waterlogged=false]")
    for x in range(26, 66, 8):
        p.put(x, FL, 21, "sea_lantern")
    # --- spectator walkway: from the start pad to the stand (2 wide, along z=18..)
    for x in (2, 3):
        for z in range(zl1 + 2, 31):
            p.put(x, FL, z, "quartz_block")
    for x in range(2, 27):
        for z in (28, 29):
            p.put(x, FL, z, "quartz_block")

    # --- route (block coordinates, region-local; x east, z south): lane axis, bends at connector centres
    cA, cB = 61.0, 73.0
    p.route = [
        (3.5, ZC + 0.0, "start"), (15.5, ZC + 0.0, "straight"), (16.0, ZC + 0.0, "stairs"), (26.0, ZC + 0.0, "slope"), (40.0, ZC + 0.0, "pit"),
        (50.0, b0 + beam_w / 2, "beam"), (57.5, b0 + beam_w / 2, ""), (58.0, ZC + 0.0, "turns"), (cA, ZC + 0.0, ""), (cA, 6.0, ""), (cB, 6.0, ""), (cB, ZC + 0.0, ""),
        (84.0, ZC + 0.0, "finish"), (87.5, ZC + 0.0, "goal"),
    ]
    # checkpoints (pressure plates, one per section end) on the lane axis, just past each section
    for x in (15, 25, 39, 49, 57, 83):
        p.plates.append((x, ZC))
        p.put(x, FL + 1, ZC, "stone_pressure_plate")
    p.spawn = (3, ZC)
    return p


def to_region(plan: Plan | None = None, origin=ORIGIN, meters_per_block: float = DEFAULT_METERS_PER_BLOCK) -> Region:
    """The Region the mod's exporter would send for the arena's bounding box (origin, SIZE)."""
    plan = plan or build_arena()
    sx, sy, sz = plan.size
    states = ["minecraft:air"] + sorted(set(plan.blocks.values()))
    idx = {s: i for i, s in enumerate(states)}
    grid = np.zeros((sx, sy, sz), np.uint16)
    for (x, y, z), s in plan.blocks.items():
        grid[x, y, z] = idx[s]
    pal = []
    for s in states:
        base = s.split("[")[0].removeprefix("minecraft:")
        pal.append(PaletteEntry("minecraft:" + base, SHAPES[base]))
    return Region(tuple(origin), (sx, sy, sz), pal, grid, meters_per_block=meters_per_block)


def route_mj(plan: Plan, meters_per_block: float = DEFAULT_METERS_PER_BLOCK, floor_z: float | None = None):
    """Route waypoints in MuJoCo metres relative to the region origin: (x, y, label); y = -z * scale."""
    s = meters_per_block
    return [(x * s, -z * s, lab) for x, z, lab in plan.route]


def floor_top_m(meters_per_block: float = DEFAULT_METERS_PER_BLOCK) -> float:
    return (FL + 1) * meters_per_block


def cam_presets(origin=ORIGIN) -> dict[int, tuple[float, float, float, float, float]]:
    """World (x, y, z, yaw, pitch) of the four camera spots. Minecraft yaw: 0 = south (+z), -90 = east (+x), 180 = north, 90 = west."""
    ox, oy, oz = origin
    y = oy + FL + 1
    return {
        1: (ox - 5.0, y + 7.0, oz + ZC + 0.5, -90.0, 16.0),  # behind the start pad, looking along the course
        2: (ox + 30.5, y + 5.0, oz + 22.5, 180.0, 14.0),  # spectator stand, looking north at the stairs, slope and pit
        3: (ox + 46.5, y + 42.0, oz + 12.5, -90.0, 90.0),  # straight down over the middle of the course
        4: (ox + 98.0, y + 5.0, oz + ZC + 0.5, 90.0, 12.0),  # beyond the finish gate, looking back along the course
    }
