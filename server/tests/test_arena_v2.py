"""Sky arena v2: no redstone anywhere (owner rule), sections and labels, feasibility under the G1's measured limits (results/m1/fit.md,
results/m1/arena_course.md), push schedule determinism, checkpoint cues, data pack in sync with the layout, and the scripted demo
walking every section at the 0.5 m scale in MuJoCo (needs the simulation python)."""

import importlib.util
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from mcrobot import arena_v2 as A, jobs, nav
from mcrobot.mjcf import build_scene, to_mjcf
from mcrobot.pushes import PushSchedule
from mcrobot.region import Kind

ROOT = Path(__file__).resolve().parents[2]
FN = ROOT / "datapack/mcrobot_demo/data/mcrobot_demo/function"
OVERLAY = ROOT / "datapack/mcrobot_demo/overlay_26_3/data/mcrobot_demo/function"
MAX_STEP = 0.05  # m, measured step-up limit of the frozen walking policy
MIN_LANE_M = 0.6  # m, narrowest corridor walked clean
BODY_W = 0.53  # m, G1 width with the arms down
BODY_H = 1.32
MEASURED_LATERAL_FAIL_N = 300.0  # N for 0.2 s: the policy falls (results/m1/arena_course.md push sweep, lateral)
BANNED = ("redstone", "repeater", "comparator", "observer")
MPB = 0.5


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def plan():
    return A.build()


@pytest.fixture(scope="module")
def layout(plan, tmp_path_factory):
    d = tmp_path_factory.mktemp("arena_v2")
    reg = A.to_region(plan)
    (d / "a.xml").write_text(to_mjcf(build_scene(reg)))
    (d / "a.json").write_text(json.dumps({"meters_per_block": reg.meters_per_block}))
    return nav.load_layout(d / "a.xml")


# ---- owner rule: no redstone ---------------------------------------------------------------------------------------------------------

def test_no_redstone_block_in_layout_or_switchable_states(plan):
    ids = A.block_ids(plan)
    assert ids, "no block ids found"
    for i in ids:
        assert not any(b in i for b in BANNED), i
    reg = A.to_region(plan)
    for p in reg.palette:
        assert not any(b in p.name for b in BANNED), p.name


def test_no_redstone_in_any_arena_v2_function():
    """Every command file of arena v2 (and the cue macro, the 26.3 overlay, cams 5-8) is scanned for the banned words, so a block id in a
    setblock, fill or summon cannot slip in. Pistons and display entities are allowed."""
    files = sorted((FN / "arena_v2").rglob("*.mcfunction")) + sorted(OVERLAY.rglob("*.mcfunction")) + [FN / "arena_v2.mcfunction", FN / "run/cue.mcfunction"]
    files += [FN / f"cam/{k}.mcfunction" for k in (5, 6, 7, 8)]
    assert len(files) > 25
    for f in files:
        text = f.read_text().lower()
        for b in BANNED:
            assert b not in text, f"{b} in {f.relative_to(ROOT)}"


def test_redstone_ban_would_catch_a_repeater():
    """The check itself works: a plan with the v1 micro-step block fails it."""
    p = A.build()
    p.put(20, A.FL + 1, 15, "minecraft:repeater[delay=1,facing=east,locked=false,powered=false]")
    with pytest.raises(AssertionError):
        for i in A.block_ids(p):
            assert not any(b in i for b in BANNED), i


# ---- layout ----------------------------------------------------------------------------------------------------------------------------

def test_deterministic_inside_the_region_and_known_shapes(plan):
    again = A.build()
    assert again.blocks == plan.blocks and again.route == plan.route and again.pushes == plan.pushes
    sx, sy, sz = plan.size
    for (x, y, z), s in plan.blocks.items():
        assert 0 <= x < sx and 0 <= y < sy and 0 <= z < sz
        A.shape_of(s)  # raises for an unknown block


def test_eight_sections_with_distinct_colour_bands_and_labels(plan):
    keys = ["ramp", "beam", "slalom", "door", "ice", "lava", "push", "stairs"]
    secs = {s.key: s for s in plan.sections}
    assert [s.label for s in (secs[k] for k in keys)] == ["RAMP", "BALANCE BEAM", "SLALOM", "DOORWAY", "ICE PATCH", "LAVA BRIDGE", "PUSH ZONE", "STAIRS"]
    assert len({secs[k].color for k in keys}) == 8
    # contiguous, in order
    xs = [(s.x0, s.x1) for s in plan.sections]
    assert all(a[1] == b[0] for a, b in zip(xs, xs[1:])) and xs[0][0] == 0 and xs[-1][1] <= plan.size[0]
    # the band: the lane floor of each section is its colour (ice and lava cells aside)
    for k in ("ramp", "slalom", "door", "stairs"):
        s = secs[k]
        floor = [plan.blocks.get((x, A.FL, A.ZC + 1)) for x in range(s.x0 + 1, s.x1 - 1)]
        assert f"minecraft:{s.color}_concrete" in floor, k
    labels = {lb["key"]: lb for lb in plan.labels}
    assert {"start", "finish", *keys} <= set(labels)
    assert labels["start"]["text"] == "START" and labels["finish"]["text"] == "FINISH"
    assert "ramp until M2" in labels["stairs"]["sub"] and labels["stairs"]["text"] == "STAIRS"


def test_start_and_finish_are_marked(plan):
    reg = A.to_region(plan)
    sc = build_scene(reg)
    kinds = [m.kind for m in sc.markers]
    assert kinds.count(Kind.SPAWN) == 1 and kinds.count(Kind.GOAL) >= 3
    assert plan.blocks[(plan.spawn[0], A.FL, plan.spawn[1])] == "minecraft:beacon"
    gates = {g["key"]: g for g in plan.gates}
    assert gates["start"]["x"] < plan.sections[1].x0 and gates["finish"]["x"] == A.SEC["finish"].x0
    assert plan.blocks[(A.FINISH_PAD_X[0], A.FL, A.ZC)] == "minecraft:gold_block"


def test_every_checkpoint_has_a_strip_a_flag_and_a_gate_to_light(plan):
    assert [s["cue"] for s in plan.strips] == list(range(1, 9))
    sc = build_scene(A.to_region(plan))
    assert [m.kind for m in sc.markers].count(Kind.CHECKPOINT) == 8
    for st in plan.strips:
        assert plan.blocks[(st["x"], A.FL, A.ZC)] == f"minecraft:{st['off']}"
        assert plan.blocks[(st["x"], A.FL + 1, A.ZC - 1)].startswith("minecraft:lime_banner")
    cps = A.checkpoints_m(plan)
    assert [c["x_m"] for c in cps] == sorted(c["x_m"] for c in cps) and len(cps) == 8


def test_snow_never_gets_enough_light_to_melt(plan):
    """Snow layers melt when the block light on them is above 11 (random tick). Light falls off by 1 per block; every light source in either
    state of the checkpoint switches must be at least 5 blocks (L1) from every snow block."""
    lights = {"beacon", "sea_lantern"}
    sources = [pos for pos, s in plan.blocks.items() if s.split("[")[0].removeprefix("minecraft:") in lights]
    sources += [pos for g in plan.gates for pos in g["lamps"]]  # lit state
    snow = [pos for pos, s in plan.blocks.items() if s.startswith("minecraft:snow")]
    assert snow and sources
    for sx_, sy_, sz_ in snow:
        for lx, ly, lz in sources:
            assert abs(sx_ - lx) + abs(sy_ - ly) + abs(sz_ - lz) >= 5, ((sx_, sy_, sz_), (lx, ly, lz))


# ---- feasibility under the measured limits (geometry; the simulation is below) ----------------------------------------------------------

def test_goal_reachable_and_steps_within_the_limit(layout):
    cells = nav.walkable_map(layout, max_step=MAX_STEP)
    goal = nav.cell_of(layout.markers["goal"][0][0], layout)
    assert goal in cells
    path = nav.bfs_path(cells, nav.cell_of(layout.markers["spawn"][0][0], layout), goal)
    assert path is not None
    steps = np.abs(np.diff([cells[c] for c in path]))
    assert steps.max() <= 1 / 16 * MPB + 1e-4 < MAX_STEP  # the highest riser is 1/16 block = 3.1 cm


def test_route_stays_on_walkable_cells_and_clear_of_obstacles(plan):
    pts = [np.array([x, z]) for x, z, _ in plan.route]
    solids = [(x, z) for (x, y, z), s in plan.blocks.items() if y in (A.FL + 2, A.FL + 3) and "glass" not in s and "banner" not in s and "piston_head" not in s]
    solids_arr = np.array(solids, float) if solids else np.zeros((0, 2))
    clear_blocks = (BODY_W / 2 + 0.10) / MPB  # half body width plus 10 cm
    for a, b in zip(pts, pts[1:]):
        n = max(2, int(np.linalg.norm(b - a) / 0.25))
        for t in np.linspace(0, 1, n):
            x, z = a + (b - a) * t
            assert (int(np.floor(x)), int(np.floor(z))) in plan.walk, (x, z)
            # distance from the route point to the nearest solid block column (cell rectangle) at body height, excluding the glass rails
            dx = np.maximum(np.abs(solids_arr[:, 0] + 0.5 - x) - 0.5, 0)
            dz = np.maximum(np.abs(solids_arr[:, 1] + 0.5 - z) - 0.5, 0)
            assert np.hypot(dx, dz).min() >= clear_blocks, (x, z, np.hypot(dx, dz).min())


def test_widths_and_headroom_of_each_obstacle(plan):
    # balance beam, bridge, door opening, push corridor: all clear of the 0.6 m the policy needs
    beam_w = {sum(1 for (cx, cz) in plan.walk if cx == x) for x in range(A.BEAM_X[0], A.BEAM_X[1] + 1)}
    assert beam_w == {A.BEAM_W} and A.BEAM_W * MPB >= 1.0
    bridge_w = {sum(1 for (cx, cz) in plan.walk if cx == x) for x in range(A.PIT_X[0], A.PIT_X[1] + 1)}
    assert bridge_w == {2}
    door_w = {sum(1 for (cx, cz) in plan.walk if cx == x) for x in range(A.DOOR_X[0], A.DOOR_X[1] + 1)}
    assert door_w == {A.DOOR_W} and A.DOOR_W * MPB >= BODY_W + 0.4  # 1 m opening: 0.47 m of margin beside the body
    push_w = {sum(1 for (cx, cz) in plan.walk if cx == x) for x in range(A.PUSH_WALL_X[0], A.PUSH_WALL_X[1] + 1)}
    assert push_w == {A.PUSH_LANE_W}
    # headroom: nothing in the door opening up to 1.5 m, and nothing under any gate crossbar
    for x in range(A.DOOR_X[0], A.DOOR_X[1] + 1):
        for z in (A.ZC - 1, A.ZC):
            for y in range(A.FL + 1, A.FL + 1 + int(np.ceil(1.5 / MPB)) + 1):
                assert (x, y, z) not in plan.blocks
    for g in plan.gates:
        for z in range(A.ZC - 3, A.ZC + 3):
            for y in range(A.FL + 1, A.FL + 1 + A.CLEAR_ROWS):
                assert (g["x"], y, z) not in plan.blocks, g["key"]


def test_slalom_pillars_force_a_detour_and_the_route_swings_by_the_amplitude(plan):
    pillars = {(x, z) for (x, z) in plan.walk}
    assert not pillars & {(x, z) for (px, pz) in A.SLALOM_PILLARS for x in (px, px + 1) for z in (pz, pz + 1)}
    zs = [z for x, z, _ in plan.route if 38.5 <= x <= 59.5]
    assert max(zs) - A.ZC == pytest.approx(A.SLALOM_AMP, abs=0.05) and A.ZC - min(zs) == pytest.approx(A.SLALOM_AMP, abs=0.05)
    # a straight line along the lane axis would hit every pillar
    for (px, pz) in A.SLALOM_PILLARS:
        assert pz <= A.ZC <= pz + 2 or pz <= A.ZC - 1 < pz + 2 or pz <= A.ZC + 1 < pz + 2
    # heading change between route samples stays below the controller's blend threshold (0.35 rad)
    p = [np.array([x, z]) for x, z, _ in plan.route if 38.5 <= x <= 59.5]
    for a, b, c in zip(p, p[1:], p[2:]):
        h1, h2 = np.arctan2(*(b - a)[::-1]), np.arctan2(*(c - b)[::-1])
        assert abs(h2 - h1) < 0.30


def test_ice_patch_is_packed_ice_all_across_the_lane(plan):
    reg = A.to_region(plan)
    sc = build_scene(reg)
    ice_geoms = [g for g in sc.colliders if g.friction < 1.0]
    assert ice_geoms
    cells = {(x, z) for (x, y, z), s in plan.blocks.items() if y == A.FL and s == "minecraft:packed_ice"}
    assert cells == {(x, z) for x in range(A.ICE_X[0], A.ICE_X[1] + 1) for z in range(A.ZC - 3, A.ZC + 3)}
    xml = to_mjcf(sc)
    assert 'friction="0.25 0.005 0.0001" priority="2"' in xml


# ---- push schedule -----------------------------------------------------------------------------------------------------------------------

def test_push_schedule_is_deterministic_and_within_what_the_policy_survives(plan):
    pushes = A.pushes_m(plan)
    assert pushes == A.pushes_m(A.build()) and len(pushes) == 3
    assert [p["cue"] for p in pushes] == [11, 12, 13]
    assert [p["wall"] for p in pushes] == ["north", "south", "north"]
    for p in pushes:
        assert p["force_n"] * p["duration_s"] <= 0.5 * MEASURED_LATERAL_FAIL_N * 0.2  # at most half the impulse that makes the policy fall
        assert abs(np.hypot(*p["dir"]) - 1) < 1e-9
        assert A.PUSH_WALL_X[0] * MPB <= p["x_m"] <= A.PUSH_WALL_X[1] * MPB  # inside the narrowed corridor
    assert pushes[0]["dir"] == [0.0, -1.0] and pushes[1]["dir"] == [0.0, 1.0]  # north wall pushes south (-y in MuJoCo)


def test_push_schedule_replays_identically_and_fires_once_per_lap(plan):
    pushes = A.pushes_m(plan)

    def run():
        s = PushSchedule(pushes)
        log = []
        for i in range(0, 60000):  # a walk along +x at 0.45 m/s, 20 ms per step
            t = i * 0.02
            f, fired = s.update(t, (3.0 + 0.45 * t, -7.5))
            log.append((tuple(f), tuple(fired)))
            if 3.0 + 0.45 * t > 62:
                break
        return log

    a, b = run(), run()
    assert a == b
    fired = [c for _, cs in a for c in cs]
    assert fired == [11, 12, 13]
    nz = [f for f, _ in a if any(f)]
    assert 30 <= len(nz) <= 33  # 0.2 s / 0.02 s = 10 steps each (11 when the float time rounds up)
    assert all(abs(f[0]) < 1e-12 and abs(f[1]) == 100.0 and f[2] == 0 for f in nz)
    s = PushSchedule(pushes)
    s.update(0.0, (60.0, 0))
    s.reset()
    assert s.update(0.0, (0.0, 0))[1] == []  # reset re-arms: nothing fires before the first trigger
    assert [c for c in s.update(1.0, (60.0, 0))[1]] == [11, 12, 13]  # all three triggers already passed: they fire (once)
    assert s.update(1.02, (60.0, 0))[1] == []


def test_pursuit_blend_keeps_walking_through_small_turns_and_is_off_by_default():
    from mcrobot.pursuit import Pursuit

    wp = [np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([2.0, 0.2])]  # a 0.2 rad bend, within the blend threshold
    for blend, expect_stop in ((False, True), (True, False)):
        c = Pursuit(wp, blend=blend)
        c.k = 1
        cmd, _ = c.command(np.array([0.9, 0.0]), 0.0)  # inside the reach of the first corner
        assert (np.linalg.norm(cmd) == 0) == expect_stop


# ---- data pack in sync -----------------------------------------------------------------------------------------------------------------

def test_datapack_is_generated_from_this_layout(plan):
    gen = load_script("gen_arena_v2_datapack")
    place = (FN / "arena_v2/place.mcfunction").read_text().splitlines()[1:]
    assert place == gen.place_lines(plan)
    for line in place:
        parts = line.split()
        lo = [int(p.lstrip("~")) for p in parts[1:4]]
        hi = [int(p.lstrip("~")) for p in parts[4:7]]
        assert np.prod([h - l + 1 for l, h in zip(lo, hi)]) <= 32768
    ox, oy, oz = A.ORIGIN
    assert f"positioned {ox} {oy} {oz} run function mcrobot_demo:arena_v2/place" in (FN / "arena_v2/build.mcfunction").read_text()
    for k, (x, y, z, yaw, pitch) in A.cam_presets().items():
        assert f"tp @s {x:.1f} {y:.1f} {z:.1f} {yaw:.1f} {pitch:.1f}" in (FN / f"cam/{k}.mcfunction").read_text()
    assert sorted(A.cam_presets()) == [5, 6, 7, 8]
    for cue in [*range(1, 9), 11, 12, 13]:
        assert (FN / f"arena_v2/cue/{cue}.mcfunction").exists(), cue
    for cue in (11, 12, 13):
        assert (FN / f"arena_v2/cue_back/{cue}.mcfunction").exists()
    assert (FN / "run/cue.mcfunction").read_text().splitlines()[-1] == "$function mcrobot_demo:arena_v2/cue/$(n)"
    ents = (FN / "arena_v2/entities.mcfunction").read_text()
    for lb in plan.labels:
        assert f'text:{{text:"{lb["text"]}"' in ents
    assert ents.count("summon minecraft:text_display") == len(plan.labels) + sum(1 for lb in plan.labels if lb["sub"])
    # piston heads: display entities in both syntaxes; the 26.3 overlay differs only in the block_state form
    base, over = (FN / "arena_v2/rams.mcfunction").read_text(), (OVERLAY / "arena_v2/rams.mcfunction").read_text()
    assert base.count("block_display") == over.count("block_display") == len(plan.rams) == 12
    assert "Properties:" in base and "Properties:" not in over and 'block_state:{id:"minecraft:piston_head",properties:{' in over
    meta = json.loads((ROOT / "datapack/mcrobot_demo/pack.mcmeta").read_text())
    assert {"min_format": 121, "max_format": 121, "directory": "overlay_26_3"} in meta["overlays"]["entries"]


def test_v1_functions_gained_only_the_v2_hooks():
    start = (FN / "run/start.mcfunction").read_text()
    assert "arena_v2/reset" in start and "#v2" in start
    assert json.loads((ROOT / "datapack/mcrobot_demo/data/minecraft/tags/function/tick.json").read_text())["values"] == ["mcrobot_demo:arena/tick", "mcrobot_demo:arena_v2/tick"]


def test_v1_export_is_unchanged_by_the_friction_work():
    """The navigation training keeps using the v1 export: regenerating it with the current exporter gives the same bytes."""
    from mcrobot import arena

    old = ROOT / "results/m1/arena/arena_region.xml"
    if not old.exists():
        pytest.skip("results/m1/arena export not present")
    assert to_mjcf(build_scene(arena.to_region())) == old.read_text()


# ---- the scripted demo, in simulation --------------------------------------------------------------------------------------------------

sim = pytest.mark.skipif(not jobs.have_sim_python(), reason="simulation python (MCROBOT_SIM_PY) not available")


@pytest.fixture(scope="module")
def v2_files(tmp_path_factory):
    d = tmp_path_factory.mktemp("arena_v2_export")
    mod = load_script("export_arena_v2")
    import sys

    argv, sys.argv = sys.argv, ["export_arena_v2.py", "--out", str(d)]
    try:
        mod.main()
    finally:
        sys.argv = argv
    return d


def test_route_file_carries_checkpoints_pushes_blend_and_cams(v2_files):
    doc = json.loads((v2_files / "arena_route.json").read_text())
    assert doc["arena"] == "v2" and doc["blend"] is True and doc["meters_per_block"] == 0.5
    assert len(doc["checkpoints"]) == 8 and len(doc["pushes"]) == 3 and sorted(doc["cams"]) == ["5", "6", "7", "8"]
    assert [s["label"] for s in doc["sections"]][1:9] == ["RAMP", "BALANCE BEAM", "SLALOM", "DOORWAY", "ICE PATCH", "LAVA BRIDGE", "PUSH ZONE", "STAIRS"]
    hdr = json.loads((v2_files / "arena_region.json").read_text())
    assert hdr["meters_per_block"] == 0.5 and hdr["origin"] == list(A.ORIGIN)


@sim
def test_scripted_demo_walks_every_section_in_simulation(v2_files, tmp_path):
    out = tmp_path / "verify.json"
    r = subprocess.run([jobs.SIM_PY, str(ROOT / "scripts/verify_arena_v2.py"), "--out", str(out)], capture_output=True, timeout=900, cwd=ROOT,
                       env={"PYTHONPATH": str(ROOT / "server"), "PATH": "/usr/bin:/bin", "HOME": str(Path.home())})
    assert r.returncode == 0, r.stdout.decode()[-1500:] + r.stderr.decode()[-1500:]
    res = json.loads(out.read_text())
    assert res["ok"] and res["why"] == "reached the finish"
    for s in res["sections"]:
        assert s["t_in"] is not None, s["label"]  # every section was entered, so every one was passed
        assert s["min_pelvis_above_floor_m"] > 0.55, s["label"]  # never crouched near the floor
        assert s["max_tilt_deg"] < 30, s["label"]
    assert [p["cue"] for p in res["pushes"]] == [11, 12, 13]  # the schedule fired in order during the lap
    secs = {s["key"]: s for s in res["sections"]}
    assert secs["door"]["max_dev_m"] < 0.2 and secs["beam"]["max_dev_m"] < 0.2 and secs["lava"]["max_dev_m"] < 0.2
    assert secs["slalom"]["max_dev_m"] < 0.3


@sim
def test_demo_streams_checkpoint_cues_on_the_v2_route(v2_files):
    from mcrobot import protocol as P
    from mcrobot.protocol import MsgType

    argv = [jobs.SIM_PY, str(ROOT / "scripts/g1_demo.py"), str(v2_files / "arena_region.xml"), "--stdout", "--no-realtime", "--seconds", "32", "--style", "blocks",
            "--arena", str(v2_files / "arena_route.json")]
    r = subprocess.run(argv, capture_output=True, timeout=600, cwd=ROOT, env={"PYTHONPATH": str(ROOT / "server"), "PATH": "/usr/bin:/bin", "HOME": str(Path.home())})
    assert r.returncode == 0, r.stderr.decode()[-1500:]
    data, pos, ev = r.stdout, 0, []
    while pos < len(data):
        t, n = P.decode_header(data[pos:pos + P.HEADER.size])
        pos += P.HEADER.size
        if t == MsgType.GHOST_PARTS:
            o = P.parse_json(data[pos:pos + n])
            if o.get("op") == "run":
                ev.append(o)
        pos += n
    assert ev[0] == {"op": "run", "event": "start"}
    cues = [e["n"] for e in ev if e.get("event") == "cue"]
    assert cues == [1]  # the RAMP is done after about 25 s; the beam's end is not reached in 32 s
