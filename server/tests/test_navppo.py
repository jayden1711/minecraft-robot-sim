"""PPO trainer: yield request saves latest.pt atomically and a new Trainer resumes from it; best/final contain no optimizer state."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mujoco_warp")
from mcrobot import arena, g1, navenv, navppo  # noqa: E402
from mcrobot.mjcf import build_scene, to_mjcf  # noqa: E402

pytestmark = pytest.mark.skipif(not g1.ONNX.exists(), reason="unitree_rl_mjlab policy not found (MCROBOT_G1_ROOT)")


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    d = tmp_path_factory.mktemp("arena")
    (d / "arena_region.xml").write_text(to_mjcf(build_scene(arena.to_region())))
    (d / "arena_region.json").write_text('{"meters_per_block": 0.5}')
    return navenv.NavEnv(str(d / "arena_region.xml"), 4, "cpu", seed=0)


class FakeYield:
    def __init__(self):
        self.flag = False

    def requested(self):
        return self.flag


def test_yield_saves_and_resume_continues(env, tmp_path):
    tr = navppo.Trainer(env, tmp_path, horizon=4, minibatches=2, log=lambda *_: None)
    y = FakeYield()
    assert tr.train(1, 600, y) == "done" and tr.it == 1
    y.flag = True
    assert tr.train(5, 600, y) == "yield"
    assert (tmp_path / "latest.pt").exists() and not (tmp_path / "latest.pt.tmp").exists()
    tr2 = navppo.Trainer(env, tmp_path, horizon=4, minibatches=2, log=lambda *_: None)
    assert tr2.it == 1 and tr2.steps == tr.steps
    for a, b in zip(tr.ac.parameters(), tr2.ac.parameters()):
        assert torch.equal(a, b)


def test_policy_checkpoint_has_no_optimizer_state_and_loads(env, tmp_path):
    tr = navppo.Trainer(env, tmp_path, horizon=2, minibatches=1, log=lambda *_: None)
    tr.save_policy("final.pt")
    ck = torch.load(tmp_path / "final.pt", weights_only=False)
    assert "opt" not in ck and "model" in ck
    ac = navppo.load_policy(tmp_path / "final.pt")
    assert ac.act(torch.zeros(1, navenv.OBS_DIM), deterministic=True).shape == (1, 3)


def test_timeout_bootstraps_but_terminal_does_not(env, tmp_path):
    tr = navppo.Trainer(env, tmp_path, horizon=3, minibatches=1, log=lambda *_: None)
    env.reset(torch.arange(4))
    env.max_t[:] = 2  # truncated by the time limit during the rollout
    obs_b, act_b, logp_b, adv, ret, st = tr.rollout()
    assert st["timeout"] >= 4 and st["success"] == 0 and torch.isfinite(adv).all() and torch.isfinite(ret).all()
