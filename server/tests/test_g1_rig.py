"""The ghost rig's Minecraft display transformations place each box exactly where MuJoCo has the link (no Minecraft needed)."""

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
from mcrobot import g1, protocol  # noqa: E402

pytestmark = pytest.mark.skipif(not g1.XML.exists(), reason=f"G1 model not found under {g1.G1_ROOT} (set MCROBOT_G1_ROOT)")


def quat_rotate_xyzw(q, v):
    x, y, z, w = q
    u = np.array([x, y, z])
    return v + 2 * np.cross(u, np.cross(u, v) + w * v)


def posed_model(seed=0):
    m = g1.build_model(None, extra_boxes=[([0, 0, -0.5], [5, 5, 0.5])])
    d = mujoco.MjData(m)
    rng = np.random.default_rng(seed)
    d.qpos[:] = m.qpos0
    d.qpos[0:3] = [3.0, -2.0, 1.0]
    q = rng.normal(size=4)
    d.qpos[3:7] = q / np.linalg.norm(q)
    d.qpos[7:] = rng.uniform(-0.6, 0.6, size=m.nq - 7)
    mujoco.mj_kinematics(m, d)
    return m, d


@pytest.mark.parametrize("mpb", [1.0, 0.5, 0.2])
def test_cube_corners_land_on_link_box_corners(mpb):
    m, d = posed_model()
    origin = np.array([-18, 134, 30])
    rig = g1.Rig(m, origin, [3.0, -2.0, 1.0], meters_per_block=mpb)
    pl = rig.payload(d)
    anchor = np.array(pl["anchor"])
    assert len(pl["parts"]) == len(g1.RIG)
    for k, part in enumerate(pl["parts"]):
        lo, hi = rig.local[k]
        R = d.xmat[rig.bid[k]].reshape(3, 3)
        for corner in np.ndindex(2, 2, 2):
            v = np.array(corner, float)  # unit-cube vertex in Minecraft-local axes
            mc = anchor + np.array(part["t"]) + quat_rotate_xyzw(part["r"], np.array(part["s"]) * v)
            mj = np.array([mc[0] - origin[0], -(mc[2] - origin[2]), mc[1] - origin[1]]) * mpb
            # the same vertex from MuJoCo: Minecraft-local (x, z, -y) -> link-frame MuJoCo (x, y, z)
            vl = np.array([lo[0] + v[0] * (hi[0] - lo[0]), hi[1] - v[2] * (hi[1] - lo[1]), lo[2] + v[1] * (hi[2] - lo[2])])
            expect = d.xpos[rig.bid[k]] + R @ vl
            assert np.allclose(mj, expect, atol=2e-3), (g1.RIG[k][0], corner, mj, expect)


def test_payload_is_a_valid_frame():
    m, d = posed_model()
    rig = g1.Rig(m, [0, 64, 0], [3.0, -2.0, 1.0])
    body = protocol.json_payload(rig.payload(d, spawn=True))
    assert len(protocol.encode_frame(protocol.MsgType.GHOST_PARTS, body)) < 3500
