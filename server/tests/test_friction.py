"""Ice blocks become low-friction collider geoms in the exported MJCF, and in MuJoCo the robot's foot contact really uses that value."""

import numpy as np
import pytest

from mcrobot.mjcf import build_scene, to_mjcf
from mcrobot.region import BLOCK_FRICTION, DEFAULT_FRICTION, PaletteEntry, Region, block_friction


def strip_region(names, mpb=0.5):
    """A 1-block-high floor, one block per name along +x, each a full cube."""
    pal = [PaletteEntry("minecraft:air", "empty")] + [PaletteEntry(n, "full") for n in dict.fromkeys(names)]
    grid = np.zeros((len(names), 1, 3), np.uint16)
    for i, n in enumerate(names):
        grid[i, 0, :] = [p.name for p in pal].index(n)
    return Region((0, 0, 0), (len(names), 1, 3), pal, grid, meters_per_block=mpb)


def test_ice_blocks_have_a_low_friction_and_everything_else_the_default():
    for n in ("minecraft:packed_ice", "minecraft:ice", "minecraft:blue_ice"):
        assert block_friction(n) == BLOCK_FRICTION[n.removeprefix("minecraft:")] < DEFAULT_FRICTION
    for n in ("minecraft:stone", "minecraft:white_concrete", "minecraft:sea_lantern", "minecraft:snow"):
        assert block_friction(n) == DEFAULT_FRICTION


def test_exporter_writes_friction_and_priority_on_ice_geoms_only():
    scene = build_scene(strip_region(["minecraft:stone", "minecraft:packed_ice", "minecraft:packed_ice", "minecraft:stone"]))
    assert sorted(round(g.friction, 3) for g in scene.colliders) == [BLOCK_FRICTION["packed_ice"], DEFAULT_FRICTION, DEFAULT_FRICTION]
    lines = [l for l in to_mjcf(scene).splitlines() if 'name="col_' in l]
    ice = [l for l in lines if f'friction="{BLOCK_FRICTION["packed_ice"]:g} ' in l]
    plain = [l for l in lines if 'friction="1 0.005 0.0001"' in l]
    assert len(ice) == 1 and len(plain) == 2 and len(lines) == 3
    assert 'priority="2"' in ice[0] and 'condim="6"' in ice[0]  # above the robot's geoms (priority 1), same contact model as a normal floor
    assert all("priority" not in l for l in plain)


def test_a_region_without_ice_exports_exactly_as_before():
    xml = to_mjcf(build_scene(strip_region(["minecraft:stone", "minecraft:white_concrete"])))
    assert "priority" not in xml and "condim" not in xml and xml.count('friction="1 0.005 0.0001"') == 1  # both blocks merge into one box


def test_friction_classes_do_not_merge_across_materials():
    reg = strip_region(["minecraft:stone", "minecraft:packed_ice", "minecraft:stone"])
    assert len(build_scene(reg).colliders) == 3


def test_robot_foot_contact_uses_the_ice_friction(tmp_path):
    """MuJoCo takes the friction of the higher-priority geom of a pair: the robot's collision geoms have priority 1, so without the
    exporter's priority the floor's friction would be ignored (found while building the ice patch)."""
    mujoco = pytest.importorskip("mujoco")
    from mcrobot import g1

    if not g1.XML.exists():
        pytest.skip("G1 model not available (MCROBOT_G1_ROOT)")
    names = ["minecraft:stone"] * 6 + ["minecraft:packed_ice"] * 6  # x 0..3 m stone, 3..6 m ice at 0.5 m/block
    scene = build_scene(strip_region(names))
    p = tmp_path / "strip.xml"
    p.write_text(to_mjcf(scene, visuals=False))
    m = g1.build_model(p)
    d = mujoco.MjData(m)
    free = m.joint("floating_base_joint").id

    def foot_friction(x: float) -> list[float]:
        mujoco.mj_resetData(m, d)
        d.qpos[m.jnt_qposadr[free]:m.jnt_qposadr[free] + 7] = [x, -0.75, 0.5 + 0.77, 1, 0, 0, 0]  # floor top at 0.5 m
        for j, name in enumerate(g1.JOINTS):
            d.qpos[m.jnt_qposadr[m.joint(name).id]] = g1.DEFAULT[j]
        for _ in range(6):
            mujoco.mj_step(m, d)
        out = []
        for c in d.contact[:d.ncon]:
            if m.geom(c.geom1).name.startswith("col_") != m.geom(c.geom2).name.startswith("col_"):  # floor box against a robot geom
                out.append(float(c.friction[0]))
        return out

    on_stone, on_ice = foot_friction(1.0), foot_friction(4.5)
    assert on_stone and on_ice
    assert all(abs(f - DEFAULT_FRICTION) < 1e-6 for f in on_stone)
    assert all(abs(f - BLOCK_FRICTION["packed_ice"]) < 1e-6 for f in on_ice)
