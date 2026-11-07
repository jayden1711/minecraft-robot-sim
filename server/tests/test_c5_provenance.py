"""C5 (CRITIQUES.md): every ghost pose sent is a logged env state that can be re-derived from (seed, policy). A scripted animation cannot do that."""

import json
import socket
import time

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mujoco_warp")
from mcrobot import arena, g1, navenv, navppo  # noqa: E402
from mcrobot.mjcf import build_scene, to_mjcf  # noqa: E402
from mcrobot.navghost import FrameOut, GhostStreamer  # noqa: E402

pytestmark = pytest.mark.skipif(not g1.ONNX.exists(), reason="unitree_rl_mjlab policy not found (MCROBOT_G1_ROOT)")
STEPS = 12


@pytest.fixture(scope="module")
def xml(tmp_path_factory):
    d = tmp_path_factory.mktemp("arena")
    (d / "arena_region.xml").write_text(to_mjcf(build_scene(arena.to_region())))
    (d / "arena_region.json").write_text(json.dumps({"origin": [96, 200, -160], "size": [92, 14, 34], "meters_per_block": 0.5}))
    return d / "arena_region.xml"


def run(xml, env_seed, stream_dir=None):
    torch.manual_seed(0)
    env = navenv.NavEnv(str(xml), 2, "cpu", seed=env_seed)
    ac = navppo.ActorCritic()
    st, client, fo = None, None, None
    if stream_dir is not None:
        fo = FrameOut(None, stream_dir / "c5.sock")
        client = socket.socket(socket.AF_UNIX)
        client.connect(str(stream_dir / "c5.sock"))
        time.sleep(0.05)
        st = GhostStreamer(env, xml, fo, 2, hz=1e9)
    obs, traj = env.obs(), []
    for _ in range(STEPS):
        obs, _, _, _ = env.step(ac.act(obs))
        traj.append(env.pelvis().clone().numpy())
        if st is not None:
            st(env)
            client.setblocking(False)
            try:  # the mod reads its socket; a client that never reads is dropped (by design)
                while client.recv(1 << 20):
                    pass
            except BlockingIOError:
                pass
    if fo is not None:
        client.close()
        fo.close()
    return traj, (np.array(st.log) if st is not None else None)


def test_C5_logged_ghost_poses_are_reproducible_training_states(xml, tmp_path):
    traj, log = run(xml, env_seed=5, stream_dir=tmp_path)
    assert log is not None and len(log) == 2 * STEPS
    again, _ = run(xml, env_seed=5)  # same seeds, nothing streamed
    for step, ghost, x, y, z in log:
        ref = again[int(step) - 1][int(ghost)]
        assert np.abs(np.array([x, y, z]) - ref).max() < 0.02  # xpos of the last substep vs the pelvis qpos: a few mm


def test_C5_control_a_different_seed_does_not_match(xml, tmp_path):
    traj, log = run(xml, env_seed=5, stream_dir=tmp_path)
    other, _ = run(xml, env_seed=6)
    worst = max(np.abs(np.array([x, y, z]) - other[int(s) - 1][int(g)]).max() for s, g, x, y, z in log)
    assert worst > 0.05  # non-vacuous: the check can fail
