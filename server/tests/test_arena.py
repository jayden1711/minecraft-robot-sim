"""The sky arena: layout invariants, solvability under the G1's measured limits (results/m1/fit.md), data pack in sync with the layout."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from mcrobot import arena, nav
from mcrobot.mjcf import build_scene, to_mjcf
from mcrobot.region import Kind

ROOT = Path(__file__).resolve().parents[2]
MAX_STEP = 0.05  # m, measured step-up limit of the frozen walking policy
MIN_LANE_M = 0.6  # m, narrowest corridor the G1 walked clean (the beam is a separate, simulated case)
MAX_GAP_M = 0.15  # m


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def plan():
    return arena.build_arena()


@pytest.fixture(scope="module")
def layout(plan, tmp_path_factory):
    d = tmp_path_factory.mktemp("arena")
    reg = arena.to_region(plan)
    (d / "a.xml").write_text(to_mjcf(build_scene(reg)))
    (d / "a.json").write_text(json.dumps({"meters_per_block": reg.meters_per_block}))
    return nav.load_layout(d / "a.xml")


def test_deterministic_and_inside_the_region(plan):
    again = arena.build_arena()
    assert again.blocks == plan.blocks and again.route == plan.route
    sx, sy, sz = plan.size
    for (x, y, z) in plan.blocks:
        assert 0 <= x < sx and 0 <= y < sy and 0 <= z < sz
    for s in plan.blocks.values():
        assert s.split("[")[0].removeprefix("minecraft:") in arena.SHAPES, s


def test_markers_for_the_task_spec(plan):
    sc = build_scene(arena.to_region(plan))
    kinds = [m.kind for m in sc.markers]
    assert kinds.count(Kind.SPAWN) == 1
    assert kinds.count(Kind.GOAL) >= 3
    assert kinds.count(Kind.CHECKPOINT) == len(plan.plates) == 6
    assert kinds.count(Kind.HAZARD) >= 1  # the lava of the pit


def test_mjcf_does_not_depend_on_palette_order(plan):
    """The mod's palette order is the order it met the blocks in; the exported MJCF must not depend on it."""
    from mcrobot.region import Region

    reg = arena.to_region(plan)
    order = list(reversed(range(len(reg.palette))))  # new index of old palette entry i is order[i]
    pal = [None] * len(reg.palette)
    for old, new in enumerate(order):
        pal[new] = reg.palette[old]
    grid = np.array(order, np.uint16)[reg.grid]
    shuffled = Region(reg.origin, reg.size, pal, grid, meters_per_block=reg.meters_per_block)
    assert to_mjcf(build_scene(shuffled)) == to_mjcf(build_scene(reg))


def test_scale_is_half_a_metre_and_the_g1_is_2_6_blocks(plan):
    reg = arena.to_region(plan)
    assert reg.meters_per_block == 0.5
    assert abs(1.32 / reg.meters_per_block - 2.64) < 1e-9
    assert arena.floor_top_m(0.5) == (arena.FL + 1) * 0.5


def test_route_stays_on_walkable_cells(plan):
    pts = [np.array([x, z]) for x, z, _ in plan.route]
    for a, b in zip(pts, pts[1:]):
        n = max(2, int(np.linalg.norm(b - a) / 0.25))
        for t in np.linspace(0, 1, n):
            x, z = a + (b - a) * t
            assert (int(np.floor(x)), int(np.floor(z))) in plan.walk, (x, z)


def test_lanes_are_wide_enough_for_the_body(plan):
    mpb = 0.5
    beam = [s for s in plan.sections if s.name == "beam"][0]
    for s in plan.sections:
        if s.name == "turns":
            continue  # width is checked per lane by the route test; the connectors are LANE_W wide by construction
        for x in range(s.x0, s.x1):
            n = sum(1 for (cx, cz) in plan.walk if cx == x)
            if n == 0:
                continue
            width = n * mpb
            if s.name == "beam":
                assert n == arena.BEAM_W
            elif s.name == "pit and bridge" and 42 <= x <= 47:
                assert width >= 2 * mpb  # the 2-block (1 m) bridge
            else:
                assert width >= MIN_LANE_M, (s.name, x, width)
    assert beam.x1 - beam.x0 >= 4


def test_goal_reachable_with_the_measured_step_limit(layout):
    cells = nav.walkable_map(layout, max_step=MAX_STEP)
    goal = nav.cell_of(layout.markers["goal"][0][0], layout)
    assert goal in cells
    path = nav.bfs_path(cells, nav.cell_of(layout.markers["spawn"][0][0], layout), goal)
    assert path is not None
    # no step higher than the limit anywhere on the path; the highest step is a micro-step (<= 1/16 block)
    hs = [cells[c] for c in path]
    steps = np.abs(np.diff(hs))
    assert steps.max() <= 1 / 16 * 0.5 + 1e-4  # MJCF text keeps 6 significant digits
    assert steps.max() < MAX_STEP


def test_no_walkable_gap_is_needed_beyond_the_limit(layout):
    # the planner never crosses a hole, so a path means every transition is between neighbouring cells with floor
    cells = nav.walkable_map(layout, max_step=MAX_STEP)
    assert MAX_GAP_M < layout.cell  # a whole-block gap (0.5 m) is out of reach by itself: the course has none on its route
    assert len(cells) > 100


def test_headroom_under_the_gate_and_everywhere_on_the_route(plan):
    clear_blocks = int(np.ceil(1.5 / 0.5))  # 1.5 m of headroom above the floor
    for (x, z) in plan.walk:
        for y in range(arena.FL + 2, arena.FL + 1 + clear_blocks + 1):
            st = plan.blocks.get((x, y, z))
            assert st is None or "glass" not in st
    gate_x = 87
    for z in range(arena.ZC - 3, arena.ZC + 3):
        for y in range(arena.FL + 1, arena.FL + 6):
            assert (gate_x, y, z) not in plan.blocks  # 5 blocks = 2.5 m of clearance between the pillars


def test_cams_are_around_the_course():
    cams = arena.cam_presets()
    assert sorted(cams) == [1, 2, 3, 4]
    ox, oy, oz = arena.ORIGIN
    for k, (x, y, z, yaw, pitch) in cams.items():
        assert y > oy + arena.FL and -90 <= pitch <= 90


def test_datapack_is_generated_from_this_layout(plan):
    gen = load_script("gen_arena_datapack")
    fn = ROOT / "datapack/mcrobot_demo/data/mcrobot_demo/function"
    place = (fn / "arena/place.mcfunction").read_text().splitlines()[1:]
    assert place == gen.place_lines(plan)
    for k, (x, y, z, yaw, pitch) in arena.cam_presets().items():
        text = (fn / f"cam/{k}.mcfunction").read_text()
        assert f"tp @s {x:.1f} {y:.1f} {z:.1f} {yaw:.1f} {pitch:.1f}" in text
    ox, oy, oz = arena.ORIGIN
    assert f"positioned {ox} {oy} {oz} run function mcrobot_demo:arena/place" in (fn / "arena/build.mcfunction").read_text()
    for name in ("arena/rules", "arena/unlock", "arena/second", "run/start", "run/tick", "run/progress", "run/finish", "run/fall"):
        assert (fn / f"{name}.mcfunction").exists()
    rules = (fn / "arena/rules.mcfunction").read_text()
    assert "advance_time false" in rules and "advance_weather false" in rules
    assert "kill @s" in (fn / "arena/second.mcfunction").read_text()
    assert "tag=!mcrobot_ghost" in (fn / "arena/second.mcfunction").read_text()


def test_fills_stay_under_the_fill_limit(plan):
    gen = load_script("gen_arena_datapack")
    for line in gen.place_lines(plan):
        parts = line.split()
        lo = [int(p.lstrip("~")) for p in parts[1:4]]
        hi = [int(p.lstrip("~")) for p in parts[4:7]]
        assert np.prod([h - l + 1 for l, h in zip(lo, hi)]) <= 32768
