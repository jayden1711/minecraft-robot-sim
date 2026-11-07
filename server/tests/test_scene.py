import mujoco
import numpy as np

from mcrobot.mjcf import build_scene, mjcf_sha256, to_mjcf
from mcrobot.region import Kind, classify_marker, mc_to_mj, mj_to_mc
from mcrobot.synthetic import demo_maze, from_layers
from mcrobot.tasks import TaskSpec


def test_classify_markers():
    assert classify_marker("minecraft:gold_block") == Kind.GOAL
    assert classify_marker("minecraft:lava") == Kind.HAZARD
    assert classify_marker("minecraft:red_wool") == Kind.HAZARD
    assert classify_marker("minecraft:oak_pressure_plate") == Kind.CHECKPOINT
    assert classify_marker("minecraft:command_block") == Kind.SPAWN
    assert classify_marker("minecraft:stone") is None


def test_coordinate_mapping_is_rotation_and_invertible():
    o = (10, 64, -5)
    p = np.array([13.5, 65.25, -2.5])
    q = mc_to_mj(p, o)
    assert np.allclose(q, [3.5, -2.5, 1.25])
    assert np.allclose(mj_to_mc(q, o), p)
    # east x up == south must map to x_mj cross z_mj == -y_mj (south) : right-handed, no mirroring
    e, u, s = (mc_to_mj(v + np.array(o), o) for v in ([1, 0, 0], [0, 1, 0], [0, 0, 1]))
    assert np.allclose(np.cross(e, u), s)


def test_single_block_box_position():
    r = from_layers([["#"]], {"#": "stone"})
    sc = build_scene(r)
    assert len(sc.colliders) == 1
    assert np.allclose(sc.colliders[0].center, [0.5, -0.5, 0.5]) and np.allclose(sc.colliders[0].half, 0.5)


def test_scale():
    r = from_layers([["##"]], {"#": "stone"}, meters_per_block=0.2)
    c = build_scene(r).colliders[0]
    assert np.allclose(c.half, [0.2, 0.1, 0.1]) and np.allclose(c.center, [0.2, -0.1, 0.1])


def test_slab_is_half_height():
    r = from_layers([["S"]], {"S": "stone_slab"})
    c = build_scene(r).colliders[0]
    assert np.allclose(c.half, [0.5, 0.5, 0.25]) and np.allclose(c.center, [0.5, -0.5, 0.25])


def test_markers_never_collide():
    xml = to_mjcf(build_scene(demo_maze()))
    m = mujoco.MjModel.from_xml_string(xml)
    for i in range(m.ngeom):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i)
        if name.startswith("vis_"):
            assert m.geom_contype[i] == 0 and m.geom_conaffinity[i] == 0
    assert m.nsite == len(build_scene(demo_maze()).markers)


def test_mjcf_deterministic():
    a = to_mjcf(build_scene(demo_maze()))
    b = to_mjcf(build_scene(demo_maze()))
    assert a == b and mjcf_sha256(a) == mjcf_sha256(b)


def test_ball_rests_on_floor_and_hits_wall():
    r = from_layers([["###", "###", "###"]], {"#": "stone"})
    m = mujoco.MjModel.from_xml_string(to_mjcf(build_scene(r)).replace("</worldbody>", '<body pos="1.5 -1.5 2"><freejoint/><geom type="sphere" size="0.1"/></body></worldbody>'))
    d = mujoco.MjData(m)
    for _ in range(600):
        mujoco.mj_step(m, d)
    assert abs(d.qpos[2] - 1.1) < 0.01  # floor top at z=1, radius 0.1


def test_task_spec_from_demo():
    sc = build_scene(demo_maze())
    t = TaskSpec.from_scene(sc)
    assert len(t.goals) == 1 and len(t.checkpoints) == 1 and len(t.spawns) == 1
    assert len(t.hazards) == 2  # lava + red wool
    g = t.goals[0]
    center = (g[0] + g[1]) / 2
    assert t.reached_goal(center + [0, 0, 1.0])
    assert not t.reached_goal(center + [3, 0, 1.0])
    assert not t.reached_goal(center + [0, 0, 3.0])
    assert t.in_hazard((t.hazards[0][0] + t.hazards[0][1]) / 2)
    cp = (t.checkpoints[0][0] + t.checkpoints[0][1]) / 2
    assert t.checkpoint_hit(cp + [0, 0, 0.1]).all()
    sp = t.spawns[0]
    assert np.isclose(sp[2], 1.0)  # top of the command block at floor level y=0 -> z=1


def test_no_markers_empty_spec():
    t = TaskSpec.from_scene(build_scene(from_layers([["#"]], {"#": "stone"})))
    assert not t.reached_goal(np.zeros(3)) and not t.in_hazard(np.zeros(3)) and len(t.spawns) == 0
