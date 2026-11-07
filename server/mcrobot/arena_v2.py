"""Sky arena v2: a TV-style obstacle course for the G1, defined once here and used three ways (like `arena`, which stays until the switch).

  * `scripts/gen_arena_v2_datapack.py` turns `build()` into the data pack functions that build it in Minecraft (blocks, labels, rams, cues);
  * `to_region()` gives the Region the mod's exporter would send for the arena's box, so the navigation training and the demo can use it
    without a client (the smoke test compares it with the real exporter);
  * `route_mj()` / `checkpoints_m()` / `pushes_m()` give the scripted demo's centre line, the checkpoint triggers and the push schedule.

Eight sections, each with its own colour band and a floating label: RAMP, BALANCE BEAM, SLALOM, DOORWAY, ICE PATCH, LAVA BRIDGE, PUSH ZONE,
STAIRS (a ramp until M2). Every dimension comes from the measured limits of the frozen walking policy (results/m1/fit.md and
results/m1/arena_course.md): step up <= 5 cm, corridor >= 0.6 m, body 1.32 m x 0.53 m; ice friction and push size from the probes in
arena_course.md. No redstone component appears anywhere (owner rule, tested): heights come from carpet, snow layers and trapdoors, the
push zone's pistons are animated by display entities from the mod's cue events.

Coordinates in `Plan.blocks` are region-local blocks: x east (the course runs +x), y up, z south. Floor blocks are at y = FL.
World scale is `meters_per_block` (default 0.5 m): the G1 is 2.6 blocks tall and about 1 block wide.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .region import DEFAULT_METERS_PER_BLOCK, PaletteEntry, Region

ORIGIN = (96, 200, -240)  # world position of the region's minimum corner; the v1 arena is at (96, 200, -160), 80 blocks north of this
FL = 3  # local y of the floor layer; the floor surface is at FL + 1
SIZE = (152, 20, 34)
LANE_W = 6
ZC = 15  # lane centre line at z = ZC (a block edge); the lane occupies z 12..17
BEAM_W = 2  # blocks (1 m)
DOOR_W = 2  # blocks (1 m): wider than the 0.53 m body plus margin
PUSH_LANE_W = 4  # blocks (2 m) between the walls of the push zone
CLEAR_ROWS = 5  # blocks of headroom under every gate (2.5 m)

# push zone: three pairs of 2 x 2 pistons, alternating walls. force and duration from the push sweep in results/m1/arena_course.md
PUSH_FORCE_N = 100.0
PUSH_DURATION_S = 0.2
CUE_CHECKPOINT = 0  # cue n = CUE_CHECKPOINT + k for checkpoint k = 1..8
CUE_PUSH = 10  # cue n = CUE_PUSH + i for push i = 1..3

COLORS = ("white", "light_gray", "gray", "black", "lime", "yellow", "orange", "magenta", "cyan", "light_blue", "red", "purple", "blue", "pink", "green", "brown")
_H = lambda h: [[0.0, 0.0, 0.0, 1.0, h, 1.0]]  # noqa: E731  bottom-attached box of height h (blocks)
SHAPES: dict = {
    "air": "empty", "lava": "empty", "beacon": "full", "emerald_block": "full", "packed_ice": "full", "piston": "full",
    "quartz_block": "full", "sea_lantern": "full", "glass": "full", "gold_block": "full", "dark_oak_planks": "full", "blackstone": "full",
    "quartz_slab": _H(0.5), "bamboo_trapdoor": _H(3 / 16), "warped_trapdoor": _H(3 / 16),
}
for _c in COLORS:
    SHAPES[f"{_c}_concrete"] = "full"
    SHAPES[f"{_c}_carpet"] = _H(1 / 16)
    SHAPES[f"{_c}_banner"] = "empty"

# microsteps of the ramps: level n is n/16 of a block above the floor surface = 3.1 cm per level at 0.5 m/block (limit 5 cm per step)
SNOW2, SNOW3 = "minecraft:snow[layers=2]", "minecraft:snow[layers=3]"  # collision is one layer lower than the visible snow: 2/16 and 4/16 of a block


def shape_of(state: str):
    base = state.split("[")[0].removeprefix("minecraft:")
    if base == "snow":
        layers = int(state.split("layers=")[1].split("]")[0])
        return "empty" if layers <= 1 else _H((layers - 1) * 2 / 16)
    return SHAPES[base]


def block_ids(plan: "PlanV2") -> set[str]:
    """Every block id the arena can place, in any state of the lit/unlit switches (for the redstone-free test)."""
    ids = {s.split("[")[0].removeprefix("minecraft:") for s in plan.blocks.values()}
    for s in plan.strips:
        ids |= {s["off"], s["on"]}
    for g in plan.gates:
        ids |= {g["lamp_off"], g["lamp_on"]}
    for r in plan.rams:
        ids.add("piston_head")
    return ids


@dataclass
class SectionV2:
    key: str
    label: str
    sub: str
    x0: int  # first / last+1 local x
    x1: int
    color: str  # floor band (concrete colour)
    text_color: str  # colour of the floating label (hex)
    note: str = ""


@dataclass
class PlanV2:
    blocks: dict = field(default_factory=dict)  # (x, y, z) -> "minecraft:name[props]"
    sections: list = field(default_factory=list)
    gates: list = field(default_factory=list)  # dict(x, key, lamp_off, lamp_on, lamps [(x, y, z)])
    labels: list = field(default_factory=list)  # dict(key, text, sub, x, y, z (local, floats), color)
    strips: list = field(default_factory=list)  # dict(cue, key, x, z0, z1, off, on): the floor strip that lights up at checkpoint `cue`
    rams: list = field(default_factory=list)  # dict(push, tag, facing, cell (x, y, z) of the piston-head cell, hidden (dx, dy, dz) offset)
    pushes: list = field(default_factory=list)  # dict(cue, key, x (block, centre of the piston pair), wall "north"|"south", force_n, duration_s)
    route: list = field(default_factory=list)  # (x, z, label), blocks
    spawn: tuple = (3, ZC)
    size: tuple = SIZE
    walk: set = field(default_factory=set)

    def put(self, x, y, z, state):
        self.blocks[(x, y, z)] = state if ":" in state else "minecraft:" + state

    def fill(self, x0, y0, z0, x1, y1, z1, state):  # inclusive corners
        for x in range(x0, x1 + 1):
            for y in range(y0, y1 + 1):
                for z in range(z0, z1 + 1):
                    self.put(x, y, z, state)


SECTIONS = [  # key, label, sub-label, x0, x1, floor colour, label colour, note
    SectionV2("start", "START", "", 0, 8, "white", "#55FF55", "spawn pad"),
    SectionV2("ramp", "RAMP", "", 8, 24, "lime", "#80FF20", "3 levels of 3.1 cm, 1 m per level, up and down"),
    SectionV2("beam", "BALANCE BEAM", "", 24, 36, "yellow", "#FFE600", "1 m wide beam, 4 m long, over the void"),
    SectionV2("slalom", "SLALOM", "", 36, 62, "orange", "#FF8A00", "three pillars to weave around"),
    SectionV2("door", "DOORWAY", "", 62, 70, "magenta", "#FF4FD8", "1 m wide gate"),
    SectionV2("ice", "ICE PATCH", "(low friction)", 70, 82, "cyan", "#40E0FF", "3 m of packed ice, friction 0.25 in MuJoCo"),
    SectionV2("lava", "LAVA BRIDGE", "", 82, 96, "red", "#FF3B30", "4 m long, 1 m wide bridge over lava"),
    SectionV2("push", "PUSH ZONE", "(pistons shove it)", 96, 118, "purple", "#B45CFF", "three 100 N pushes from the side walls"),
    SectionV2("stairs", "STAIRS", "(ramp until M2)", 118, 138, "blue", "#4D79FF", "4 levels of 3.1 cm; real 0.25 m steps are M2"),
    SectionV2("finish", "FINISH", "", 138, 148, "white", "#FFD700", "finish gate, gold pad"),
]
SEC = {s.key: s for s in SECTIONS}

GATE_COLOR = {"start": "lime", "finish": "yellow"}
RAMP_PROFILE =[0, 1, 1, 2, 2, 3, 3, 3, 3, 2, 2, 1, 1, 0, 0, 0]  # level per block of the ramp section (16 blocks)
STAIRS_PROFILE = [0, 1, 1, 2, 2, 3, 3, 4, 4, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0, 0]  # 20 blocks
RAMP_BLOCKS = {1: "minecraft:lime_carpet", 2: SNOW2, 3: "minecraft:bamboo_trapdoor"}
STAIRS_BLOCKS = {1: "minecraft:blue_carpet", 2: SNOW2, 3: "minecraft:warped_trapdoor", 4: SNOW3}
SLALOM_PILLARS = [(41, 13), (48, 15), (55, 13)]  # (x of the first of 2 columns, z of the first of 2 rows); each pillar is 2 x 2 blocks, 5 high
SLALOM_AMP = 1.2  # blocks (0.6 m) the route swings to each side
PUSH_STATIONS = [(101, "north"), (106, "south"), (111, "north")]  # (x of the first of 2 piston columns, wall)
PUSH_WALL_X = (98, 113)  # first / last x of the narrowed corridor
PIT_X = (85, 92)  # lava pit, 8 blocks (4 m)
BEAM_X = (26, 33)  # beam over the void, 8 blocks (4 m)
ICE_X = (73, 78)  # 6 blocks (3 m) of packed ice
DOOR_X = (65, 66)  # door frame depth 2 blocks (1 m)
FINISH_PAD_X = (141, 143)


def _lane(p: PlanV2, x0, x1, z0=ZC - LANE_W // 2, z1=ZC + LANE_W // 2 - 1, floor="white_concrete"):
    for x in range(x0, x1 + 1):
        for z in range(z0, z1 + 1):
            p.put(x, FL, z, floor)
            p.walk.add((x, z))


def slalom_route() -> list[tuple[float, float, str]]:
    """Sinusoid through the three pillars: z = ZC - A sin(2 pi (x - 45.5) / 14) for x in 38.5..59.5, sampled every block (heading changes < 0.25 rad)."""
    return [(x, ZC - SLALOM_AMP * math.sin(2 * math.pi * (x - 45.5) / 14.0), "") for x in np.arange(38.5, 59.5 + 1e-9, 1.0)]


def build() -> PlanV2:
    p = PlanV2()
    p.sections = list(SECTIONS)
    zl0, zl1 = ZC - LANE_W // 2, ZC + LANE_W // 2 - 1  # 12..17

    for s in SECTIONS:  # colour bands (the lane floor of every section)
        _lane(p, s.x0, s.x1 - 1, floor=("quartz_block" if s.key in ("start", "finish") else f"{s.color}_concrete"))
    p.put(3, FL, ZC, "beacon")  # spawn marker (clean, no redstone look)
    for z in range(zl0, zl1 + 1):  # start line: black and white checkers
        p.put(6, FL, z, "black_concrete" if z % 2 else "white_concrete")
        p.put(7, FL, z, "white_concrete" if z % 2 else "black_concrete")

    # --- RAMP and STAIRS: micro-step profile, 1 block = 0.5 m per block of the profile, full lane width
    for sec, prof, blocks in ((SEC["ramp"], RAMP_PROFILE, RAMP_BLOCKS), (SEC["stairs"], STAIRS_PROFILE, STAIRS_BLOCKS)):
        assert len(prof) == sec.x1 - sec.x0
        for i, lv in enumerate(prof):
            if lv:
                for z in range(zl0, zl1 + 1):
                    p.put(sec.x0 + i, FL + 1, z, blocks[lv])

    # --- BALANCE BEAM: void on both sides of a 2-wide beam, 8 long; pads of the lane before and after
    b0 = ZC - BEAM_W // 2
    beam = SEC["beam"]
    for x in range(beam.x0, beam.x1):
        if BEAM_X[0] <= x <= BEAM_X[1]:
            for z in range(zl0, zl1 + 1):
                p.blocks.pop((x, FL, z), None)
                p.walk.discard((x, z))
            for z in range(b0, b0 + BEAM_W):
                p.put(x, FL, z, "yellow_concrete")
                p.walk.add((x, z))

    # --- SLALOM: 2 x 2 pillars, 5 high, black with two yellow top blocks
    for (px, pz) in SLALOM_PILLARS:
        for x in (px, px + 1):
            for z in (pz, pz + 1):
                p.fill(x, FL + 1, z, x, FL + 3, z, "black_concrete")
                p.fill(x, FL + 4, z, x, FL + 5, z, "yellow_concrete")
                p.walk.discard((x, z))

    # --- DOORWAY: wall across the lane with a 2-block opening, lintel above 4 blocks of clearance
    for x in range(DOOR_X[0], DOOR_X[1] + 1):
        for z in range(zl0, zl1 + 1):
            opening = b0 <= z < b0 + DOOR_W
            if opening:
                p.fill(x, FL + 5, z, x, FL + 6, z, "pink_concrete")  # lintel; the opening below it stays free (4 blocks = 2 m)
            else:
                p.fill(x, FL + 1, z, x, FL + 6, z, "magenta_concrete")
                p.walk.discard((x, z))

    # --- ICE PATCH
    for x in range(ICE_X[0], ICE_X[1] + 1):
        for z in range(zl0, zl1 + 1):
            p.put(x, FL, z, "packed_ice")

    # --- LAVA BRIDGE: bowl under the pit, lava either side of a 2-block bridge (as in v1: solid under the bridge)
    for x in range(PIT_X[0] - 1, PIT_X[1] + 2):
        for z in range(zl0 - 1, zl1 + 2):
            for y in range(0, FL):
                p.put(x, y, z, "blackstone")
    for x in range(PIT_X[0], PIT_X[1] + 1):
        for z in range(zl0, zl1 + 1):
            if b0 <= z < b0 + BEAM_W:
                p.put(x, FL, z, "dark_oak_planks")
            else:
                p.blocks.pop((x, FL, z), None)
                p.walk.discard((x, z))
                p.put(x, 1, z, "blackstone")
                p.put(x, 2, z, "lava")

    # --- PUSH ZONE: narrowed corridor (4 blocks), two walls 4 high with 2 x 2 pistons; the pistons' heads are display entities
    wn, ws = ZC - PUSH_LANE_W // 2 - 1, ZC + PUSH_LANE_W // 2  # wall cells z = 12 and 17
    for x in range(PUSH_WALL_X[0], PUSH_WALL_X[1] + 1):
        for z in (wn, ws):
            p.fill(x, FL + 1, z, x, FL + 4, z, "purple_concrete")
            p.walk.discard((x, z))
    for i, (px, wall) in enumerate(PUSH_STATIONS, start=1):
        z = wn if wall == "north" else ws
        facing = "south" if wall == "north" else "north"
        for x in (px, px + 1):
            for y in (FL + 2, FL + 3):
                p.put(x, y, z, f"minecraft:piston[extended=false,facing={facing}]")
                hz = z + (1 if wall == "north" else -1)  # the cell in front of the piston face, inside the corridor
                hidden = (0.0, 0.0, -1.0 if wall == "north" else 1.0)  # retracted: one block back, inside the piston's own cell
                p.rams.append({"push": i, "tag": f"mcr_ram_{i}", "facing": facing, "cell": (x, y, hz), "hidden": hidden})
        p.pushes.append({"cue": CUE_PUSH + i, "key": f"push_{i}", "x": px + 1.0, "wall": wall, "force_n": PUSH_FORCE_N, "duration_s": PUSH_DURATION_S})

    # --- FINISH: gold pad, finish line
    for x in range(FINISH_PAD_X[0], FINISH_PAD_X[1] + 1):
        for z in range(zl0, zl1 + 1):
            p.put(x, FL, z, "gold_block")
    for z in range(zl0, zl1 + 1):
        p.put(140, FL, z, "black_concrete" if z % 2 else "white_concrete")
        p.put(139, FL, z, "white_concrete" if z % 2 else "black_concrete")

    # --- rails: glass 2 high beside every walkable lane cell, except at the beam and the pit
    rails_off = {(x, z) for x in list(range(BEAM_X[0], BEAM_X[1] + 1)) + list(range(PIT_X[0], PIT_X[1] + 1)) for z in range(0, SIZE[2])}
    for (x, z) in sorted(p.walk):
        for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (x + dx, z + dz)
            if n in p.walk or n in rails_off or (n[0], FL, n[1]) in p.blocks:
                continue
            if 0 <= n[0] < SIZE[0] and 0 <= n[1] < SIZE[2]:
                p.put(n[0], FL, n[1], "white_concrete")
                p.fill(n[0], FL + 1, n[1], n[0], FL + 2, n[1], "glass")

    # --- gates: two posts and a crossbar at the start of each section (and over the start pad), label above, lamps on the posts
    gate_x = {"start": 5, **{k: SEC[k].x0 for k in ("ramp", "beam", "slalom", "door", "ice", "lava", "push", "stairs", "finish")}}
    for key, gx in gate_x.items():
        sec = SEC[key]
        mat = f"{GATE_COLOR.get(key, sec.color)}_concrete"
        for z in (zl0 - 1, zl1 + 1):
            p.fill(gx, FL + 1, z, gx, FL + 1 + CLEAR_ROWS, z, mat)
        p.fill(gx, FL + 1 + CLEAR_ROWS, zl0, gx, FL + 1 + CLEAR_ROWS, zl1, mat)  # crossbar: CLEAR_ROWS blocks of free height below
        lamps = [(gx, FL + 2 + CLEAR_ROWS, zl0 - 1), (gx, FL + 2 + CLEAR_ROWS, zl1 + 1)]
        for (lx, ly, lz) in lamps:
            p.put(lx, ly, lz, "gray_concrete")
        p.gates.append({"x": gx, "key": key, "lamp_off": "gray_concrete", "lamp_on": "sea_lantern", "lamps": lamps})
        p.labels.append({"key": key, "text": sec.label, "sub": sec.sub, "x": gx + 0.5, "y": FL + 1 + CLEAR_ROWS + 4.0, "z": float(ZC), "color": sec.text_color})

    # --- checkpoints: one per section end, a floor strip (gray, turns emerald), a banner marker, the next gate's lamps
    order = ["ramp", "beam", "slalom", "door", "ice", "lava", "push", "stairs"]
    for k, key in enumerate(order, start=1):
        sec = SEC[key]
        x = sec.x1 - 1
        for z in range(zl0, zl1 + 1):
            p.put(x, FL, z, "gray_concrete")
        p.strips.append({"cue": CUE_CHECKPOINT + k, "key": key, "x": x, "z0": zl0, "z1": zl1, "off": "gray_concrete", "on": "emerald_block"})
        p.put(x, FL + 1, ZC - 1, "minecraft:lime_banner[rotation=0]")  # the checkpoint marker (no collision); the flag stands beside the axis
    p.spawn = (3, ZC)

    # --- spectator stand south of the lane (x 60..120) and a walkway from the start pad
    for x in range(60, 120):
        for z in range(21, 31):
            p.put(x, FL, z, "quartz_block")
    for x in range(61, 119):
        p.put(x, FL, 20, "quartz_block")
        p.fill(x, FL + 1, 20, x, FL + 2, 20, "glass")
        for z, h in ((24, 1), (26, 2), (28, 3)):
            p.fill(x, FL + 1, z, x, FL + h, z, "quartz_block")
            p.put(x, FL + h + 1, z, "quartz_slab[type=bottom,waterlogged=false]")
    for x in range(2, 4):
        for z in range(zl1 + 2, 31):
            p.put(x, FL, z, "quartz_block")
    for x in range(2, 61):
        for z in (28, 29):
            p.put(x, FL, z, "quartz_block")

    # --- route (blocks, region-local; x east, z south): the lane axis, the slalom swings
    r = [(3.5, float(ZC), "start")]
    for key in ("ramp", "beam"):
        r.append((float(SEC[key].x0), float(ZC), key))
    r.append((float(SEC["slalom"].x0), float(ZC), "slalom"))
    r += [(float(x), float(z), lab) for x, z, lab in slalom_route()]
    for key in ("door", "ice", "lava", "push", "stairs"):
        r.append((float(SEC[key].x0), float(ZC), key))
    r.append((float(SEC["finish"].x0), float(ZC), "finish"))
    r.append((float(sum(FINISH_PAD_X) / 2 + 0.5), float(ZC), "goal"))
    p.route = r
    return p


def to_region(plan: PlanV2 | None = None, origin=ORIGIN, meters_per_block: float = DEFAULT_METERS_PER_BLOCK) -> Region:
    """The Region the mod's exporter would send for the arena's bounding box (origin, SIZE)."""
    plan = plan or build()
    sx, sy, sz = plan.size
    states = ["minecraft:air"] + sorted(set(plan.blocks.values()))
    idx = {s: i for i, s in enumerate(states)}
    grid = np.zeros((sx, sy, sz), np.uint16)
    for (x, y, z), s in plan.blocks.items():
        grid[x, y, z] = idx[s]
    pal = [PaletteEntry("minecraft:" + s.split("[")[0].removeprefix("minecraft:"), shape_of(s)) for s in states]
    return Region(tuple(origin), (sx, sy, sz), pal, grid, meters_per_block=meters_per_block)


def route_mj(plan: PlanV2, meters_per_block: float = DEFAULT_METERS_PER_BLOCK):
    """Route waypoints in MuJoCo metres relative to the region origin: (x, y, label); y = -z * scale."""
    s = meters_per_block
    return [(x * s, -z * s, lab) for x, z, lab in plan.route]


def floor_top_m(meters_per_block: float = DEFAULT_METERS_PER_BLOCK) -> float:
    return (FL + 1) * meters_per_block


def checkpoints_m(plan: PlanV2, meters_per_block: float = DEFAULT_METERS_PER_BLOCK) -> list[dict]:
    """Checkpoint triggers: the cue fires when the pelvis passes the centre of the strip (MuJoCo x, metres)."""
    s = meters_per_block
    return [{"cue": st["cue"], "key": st["key"], "label": SEC[st["key"]].label, "x_m": (st["x"] + 0.5) * s, "y_m": -ZC * s} for st in plan.strips]


def pushes_m(plan: PlanV2, meters_per_block: float = DEFAULT_METERS_PER_BLOCK) -> list[dict]:
    """The push schedule in MuJoCo terms. A north-wall piston pushes towards +z (south) = -y in MuJoCo; a south-wall piston towards +y."""
    s = meters_per_block
    out = []
    for pu in plan.pushes:
        out.append({"cue": pu["cue"], "key": pu["key"], "x_m": pu["x"] * s, "dir": [0.0, -1.0 if pu["wall"] == "north" else 1.0],
                    "force_n": pu["force_n"], "duration_s": pu["duration_s"], "wall": pu["wall"]})
    return out


def cam_presets(origin=ORIGIN) -> dict[int, tuple[float, float, float, float, float]]:
    """World (x, y, z, yaw, pitch) of the camera spots 5..8 (the v1 spots are 1..4). Minecraft yaw: 0 = south (+z), -90 = east, 180 = north, 90 = west."""
    ox, oy, oz = origin
    y = oy + FL + 1
    return {
        5: (ox - 5.0, y + 7.0, oz + ZC + 0.5, -90.0, 16.0),  # behind the start pad, looking along the course
        6: (ox + 104.5, y + 9.0, oz + 23.5, 180.0, 30.0),  # stand above the push zone, looking north at the door, ice, lava and pistons
        7: (ox + 73.0, y + 70.0, oz + 14.5, -90.0, 90.0),  # straight down over the middle of the course
        8: (ox + 158.0, y + 5.0, oz + ZC + 0.5, 90.0, 12.0),  # beyond the finish gate, looking back along the course
    }
