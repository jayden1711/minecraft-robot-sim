"""MuJoCo Warp buffer overflow on arena v2 must never go unnoticed: constraints dropped by a full buffer mean wrong physics."""

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mujoco_warp")
from mcrobot import g1, navenv  # noqa: E402

XML = Path(__file__).resolve().parents[2] / "results" / "m1" / "arena_v2" / "arena_region.xml"
pytestmark = [
    pytest.mark.skipif(not g1.ONNX.exists(), reason="unitree_rl_mjlab policy not found (MCROBOT_G1_ROOT)"),
    pytest.mark.skipif(not XML.exists(), reason="arena v2 export not found (scripts/export_arena_v2.py)"),
]
SECTIONS = ("start", "ramp", "slalom", "ice", "push", "stairs")  # one env starts at the beginning of each


def place_at_section(env, i, x, y):
    q = env.q_default_full.clone()
    z = float(env.tt.ground_at(torch.tensor([[x, y]], dtype=torch.float32, device=env.dev))[0]) + navenv.STAND_Z
    q[env.qa:env.qa + 3] = torch.tensor([x, y, z], dtype=torch.float32)
    env.qpos[i] = q
    env.qvel[i] = 0
    env.push_done[i] = env.pelvis()[i, 0] >= env.push_x


def test_default_buffers_hold_a_short_walk_through_every_section():
    env = navenv.NavEnv(str(XML), len(SECTIONS), "cpu", seed=0, p_spawn=1.0)
    secs = {s["key"]: s for s in json.loads(XML.with_name("arena_route.json").read_text())["sections"]}
    for i, k in enumerate(SECTIONS):
        place_at_section(env, i, secs[k]["x0_m"] + 0.5, -7.5)
    for _ in range(40):  # 4 s of walking forward, including the push zone's pushes and the ice
        env.step(torch.tensor([[0.6, 0.0, 0.0]] * len(SECTIONS)))
    assert int((env.overflow_flags != 0).sum()) == 0
    assert 0 < int(env.peak_nefc) < env.njmax and 0 < float(env.peak_ncon) < env.nconmax


def test_overflow_raises_instead_of_warning():
    env = navenv.NavEnv(str(XML), 2, "cpu", seed=0, p_spawn=1.0, njmax=8, nconmax=2)
    with pytest.raises(navenv.PhysicsOverflow, match="constraints were dropped"):
        for _ in range(10):
            env.step(torch.zeros(2, 3))
