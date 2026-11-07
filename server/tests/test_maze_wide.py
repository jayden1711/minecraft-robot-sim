"""The G1 fit analysis: the owner's maze cannot be solved by a walker that steps up at most 5 cm and avoids hazards; maze_wide can."""

import json

import pytest

from mcrobot import nav
from mcrobot.mjcf import build_scene, to_mjcf
from mcrobot.synthetic import demo_maze, demo_maze_wide

MAX_STEP = 0.05  # m, measured for the frozen G1 walking policy (results/m1/fit.md)


def layout(region, tmp_path):
    p = tmp_path / "r.xml"
    p.write_text(to_mjcf(build_scene(region)))
    p.with_suffix(".json").write_text(json.dumps({"meters_per_block": region.meters_per_block}))  # the server writes this header next to the MJCF
    return nav.load_layout(p)


def goal_reachable(lay):
    cells = nav.walkable_map(lay, max_step=MAX_STEP)
    goal = nav.cell_of(lay.markers["goal"][0][0], lay)
    return goal in cells, cells


def test_owner_maze_goal_unreachable(tmp_path):
    ok, _ = goal_reachable(layout(demo_maze(), tmp_path))
    assert not ok


def test_wide_maze_goal_reachable_and_corridors_wide(tmp_path):
    lay = layout(demo_maze_wide(), tmp_path)
    ok, cells = goal_reachable(lay)
    assert ok
    assert len(lay.markers["hazard"]) == 2
    sp = nav.cell_of(lay.markers["spawn"][0][0], lay)
    path = nav.bfs_path(cells, sp, nav.cell_of(lay.markers["goal"][0][0], lay))
    assert path is not None and len(path) < 45


@pytest.mark.parametrize("region", [demo_maze_wide])
def test_wide_maze_has_no_step_ups(region, tmp_path):
    lay = layout(region(), tmp_path)
    _, cells = goal_reachable(lay)
    heights = {round(h, 3) for h in cells.values()}
    assert len(heights) == 1
