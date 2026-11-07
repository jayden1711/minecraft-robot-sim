"""NavEnv on MuJoCo Warp (CPU device here): parity with the CPU reference walker, reward rules, resets. Training venv only."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mujoco_warp")
from mcrobot import arena, g1, navenv  # noqa: E402
from mcrobot.mjcf import build_scene, to_mjcf  # noqa: E402

pytestmark = pytest.mark.skipif(not g1.ONNX.exists(), reason="unitree_rl_mjlab policy not found (MCROBOT_G1_ROOT)")


@pytest.fixture(scope="module")
def xml(tmp_path_factory):
    d = tmp_path_factory.mktemp("arena")
    p = d / "arena_region.xml"
    p.write_text(to_mjcf(build_scene(arena.to_region())))
    (d / "arena_region.json").write_text('{"meters_per_block": 0.5}')
    return str(p)


@pytest.fixture(scope="module")
def env(xml):
    return navenv.NavEnv(xml, 2, "cpu", seed=3, p_spawn=1.0)


def place(env, i, xy, z, yaw=0.0):
    q = env.q_default_full.clone()
    q[env.qa:env.qa + 3] = torch.tensor([xy[0], xy[1], z], dtype=torch.float32)
    q[env.qa + 3:env.qa + 7] = torch.tensor([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)], dtype=torch.float32)
    env.qpos[i] = q
    env.qvel[i] = 0
    env.phase[i] = 0
    env.last_a[i] = 0


def test_reset_stands_the_g1_on_the_command_block(env):
    p = env.pelvis()
    assert torch.allclose(p[:, 2], torch.full((2,), env.floor_z + navenv.STAND_Z), atol=0.02)
    assert float((p[:, :2] - torch.tensor(env.task.spawn_xy, dtype=torch.float32)).abs().max()) < 0.2


def test_low_level_matches_the_cpu_reference_walker(env, xml):
    place(env, 0, env.task.spawn_xy, env.floor_z + 0.8)
    place(env, 1, env.task.spawn_xy, env.floor_z + 0.8)
    w = g1.G1Walker(g1.build_model(xml))
    w.reset(env.task.spawn_xy, 0.0, z=env.floor_z + 0.8)
    cmd = torch.tensor([[0.4, 0.0, 0.0]] * 2)
    for _ in range(80):  # 1.6 s
        env._ll_step(cmd)
        w.step(np.array([0.4, 0, 0]))
    a, b = env.pelvis()[0].numpy(), w.pelvis_pos()
    assert np.abs(a - b).max() < 0.02, (a, b)
    assert float(env.upright()[0]) > 0.99


def test_step_progress_reward_and_command_smoothing(env):
    place(env, 0, env.task.spawn_xy, env.floor_z + 0.8)
    place(env, 1, env.task.spawn_xy, env.floor_z + 0.8)
    env.reset(torch.arange(2))
    obs = env.obs()
    assert obs.shape == (2, navenv.OBS_DIM) and torch.isfinite(obs).all()
    assert float(obs[0, 0]) > 0.8  # the compass points ahead at the start
    tot = torch.zeros(2)
    for _ in range(30):
        obs, r, done, info = env.step(torch.tensor([[0.5, 0.0, 0.0]] * 2))
        tot += r
    assert float(tot.min()) > 0.2 and not bool(done.any())  # moving along the course pays; the time penalty is small
    assert float(env.cmd[0, 0]) > 0.2 and float(env.cmd[0, 0]) <= navenv.CMD_HI[0]


def test_lava_and_pit_end_the_episode_with_the_fail_penalty(env):
    hz = env.tt.haz[0].cpu().numpy()
    place(env, 0, ((hz[0] + hz[1]) / 2, (hz[2] + hz[3]) / 2), env.floor_z - 0.2)
    place(env, 1, (50 * 0.5 + 0.25, -(arena.ZC + 4) * 0.5), env.floor_z - 0.6)  # beside the beam, over the void
    obs, r, done, info = env.step(torch.zeros(2, 3))
    assert bool(done.all()) and bool(info["fail"].all())
    assert bool(info["lava"][0]) and bool(info["fell"][1])
    assert float(r.max()) < navenv.R_FAIL + 1.0
    assert float(env.pelvis()[:, 2].min()) > env.floor_z + 0.7  # both were reset onto the course


def test_standing_on_the_gold_pad_succeeds_and_pays_the_goal(env):
    gc = env.tt.goal.mean(0).cpu().numpy()
    place(env, 0, gc[:2], env.task.goal[1, 2] + 0.80)
    place(env, 1, env.task.spawn_xy, env.floor_z + 0.8)
    obs, r, done, info = env.step(torch.zeros(2, 3))
    assert bool(info["success"][0]) and not bool(info["success"][1]) and bool(done[0])
    assert float(r[0]) > navenv.R_GOAL - 1.0


@pytest.fixture(scope="module")
def xml2(tmp_path_factory):
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    d = tmp_path_factory.mktemp("arena_v2")
    subprocess.run([sys.executable, str(root / "scripts/export_arena_v2.py"), "--out", str(d)], check=True, env={"PYTHONPATH": str(root / "server")}, capture_output=True)
    return str(d / "arena_region.xml")


@pytest.fixture(scope="module")
def env2(xml2):
    return navenv.NavEnv(xml2, 2, "cpu", seed=5, p_spawn=1.0)


def test_v2_loads_markers_pushes_and_keeps_ice_priority(env2, xml2):
    t = env2.task
    assert t.goal_reachable and t.n_plates == 8 and len(env2.pushes) == 3
    m = env2.model
    ice = [g for g in range(m.ngeom) if m.geom_priority[g] == 2]
    assert ice and all(abs(m.geom_friction[g, 0] - 0.25) < 1e-6 for g in ice)  # the loader keeps the ice geoms' priority and friction


def test_push_fires_once_at_its_trigger_with_the_scheduled_force(env2):
    p0 = env2.pushes[0]  # 100 N towards -y at x_m
    place(env2, 0, (p0["x_m"] - 0.3, -7.5), env2.floor_z + 0.8)
    place(env2, 1, (p0["x_m"] + 0.5, -7.5), env2.floor_z + 0.8)
    env2._apply_pushes()  # env 0 is still before the trigger: nothing
    assert float(env2.xfrc[0, env2.root, 1]) == 0.0
    place(env2, 0, (p0["x_m"] + 0.01, -7.5), env2.floor_z + 0.8)
    env2.push_done[:] = False
    env2.push_left[:] = 0
    env2.push_done[1, 0] = True  # env 1 started beyond the trigger
    env2._apply_pushes()
    assert float(env2.xfrc[0, env2.root, 1]) == pytest.approx(-p0["force_n"]) and float(env2.xfrc[1, env2.root, 1]) == 0.0
    env2._apply_pushes()  # second nav step of 0.2 s
    assert float(env2.xfrc[0, env2.root, 1]) == pytest.approx(-p0["force_n"])
    env2._apply_pushes()  # over
    assert float(env2.xfrc[0, env2.root, 1]) == 0.0
    place(env2, 0, (p0["x_m"] + 0.1, -7.5), env2.floor_z + 0.8)
    env2._apply_pushes()  # does not fire again in the same episode
    assert float(env2.xfrc[0, env2.root, 1]) == 0.0


def test_reset_beyond_a_trigger_marks_that_push_done(env2):
    env2.reset(torch.arange(2))
    assert not bool(env2.push_done.any())  # p_spawn = 1: the start pad is before every trigger
    assert float(env2.xfrc[:, env2.root, :3].abs().max()) == 0.0


def test_timeout_truncates_not_fails(env):
    env.reset(torch.arange(2))
    env.max_t[:] = 2
    for _ in range(2):
        obs, r, done, info = env.step(torch.zeros(2, 3))
    assert bool(info["timeout"].all()) and not bool(info["fail"].any()) and "final_obs" in info
