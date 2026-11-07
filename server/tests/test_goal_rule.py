"""The goal rule must fire for a standing G1 (pelvis 0.78 m above the floor) on the gold pad; the old absolute 0.5 m margin never did."""

import numpy as np

from mcrobot import arena, tasks
from mcrobot.mjcf import build_scene
from mcrobot.tasks import G1_BASE_HEIGHT, TaskSpec


def spec():
    return TaskSpec.from_scene(build_scene(arena.to_region()))


def test_standing_g1_on_gold_pad_reaches_goal_only_with_base_height():
    t = spec()
    g = t.goals[0]
    top = g[1][2]
    p = np.array([(g[0][0] + g[1][0]) / 2, (g[0][1] + g[1][1]) / 2, top + G1_BASE_HEIGHT])
    assert not t.reached_goal(p)  # the old rule: pelvis 0.78 m up is "not on it"
    assert t.reached_goal(p, base_height=G1_BASE_HEIGHT)


def test_goal_rule_rejects_flying_and_far_points():
    t = spec()
    g = t.goals[0]
    top = g[1][2]
    c = np.array([(g[0][0] + g[1][0]) / 2, (g[0][1] + g[1][1]) / 2])
    assert not t.reached_goal(np.array([*c, top + G1_BASE_HEIGHT + 0.7]), base_height=G1_BASE_HEIGHT)  # well above a standing pelvis
    assert not t.reached_goal(np.array([c[0] - 6.0, c[1], top + G1_BASE_HEIGHT]), base_height=G1_BASE_HEIGHT)  # the spawn end of the course
    assert not t.reached_goal(np.array([*c, top - 1.5]), base_height=G1_BASE_HEIGHT)  # fallen far below the pad


def test_plate_rule_uses_base_height_too():
    t = spec()
    cp = t.checkpoints[0]
    c = (cp[0] + cp[1]) / 2
    p = np.array([c[0], c[1], cp[1][2] + G1_BASE_HEIGHT])
    assert not t.checkpoint_hit(p)[0]
    hit = t.checkpoint_hit(p, base_height=G1_BASE_HEIGHT)
    assert hit[0] and hit.sum() == 1  # only the plate it stands on


def test_g1_base_height_matches_the_reset_pose():
    from mcrobot import g1

    assert abs(G1_BASE_HEIGHT - 0.78) < 1e-9 and tasks.G1_BASE_HEIGHT < 0.80 and g1.CTRL_DT == 0.02
