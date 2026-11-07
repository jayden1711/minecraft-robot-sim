"""Navigation task geometry on the sky arena (grid, geodesic field, rays, reward rules). Needs torch (the training venv)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from mcrobot import arena, navtask  # noqa: E402
from mcrobot.mjcf import build_scene, to_mjcf  # noqa: E402
from mcrobot.tasks import G1_BASE_HEIGHT  # noqa: E402


@pytest.fixture(scope="module")
def task(tmp_path_factory):
    d = tmp_path_factory.mktemp("arena")
    p = d / "arena.xml"
    p.write_text(to_mjcf(build_scene(arena.to_region())))
    (d / "arena.json").write_text('{"meters_per_block": 0.5}')
    return navtask.NavTask.from_xml(p)


def test_goal_reachable_and_field_length_is_the_route(task):
    assert task.goal_reachable and task.n_plates == 6
    tt = task.to("cpu")
    d = float(tt.field_near(torch.tensor(np.array([task.spawn_xy]), dtype=torch.float32)))
    assert 40.0 < d < 55.0  # the designed route is 51 m; the shortest walkable path cuts the corners of the connectors


def test_grid_classes_floor_pit_and_walls(task):
    tt = task.to("cpu")
    s = torch.tensor([task.spawn_xy], dtype=torch.float32)
    r, c, _ = tt.cell_idx(s)
    assert int(tt.kind[r, c]) == navtask.FLOOR
    # the pit beside the bridge is void/lava, the bridge itself floor
    mx = 44.5 * 0.5
    bridge = torch.tensor([[mx, -arena.ZC * 0.5 - 0.1]], dtype=torch.float32)
    beside = torch.tensor([[mx, -(arena.ZC - 2.5) * 0.5]], dtype=torch.float32)
    assert int(tt.kind[tuple(tt.cell_idx(bridge)[:2])]) == navtask.FLOOR
    assert int(tt.kind[tuple(tt.cell_idx(beside)[:2])]) == navtask.VOID
    # the glass rail beside the straight is a wall
    rail = torch.tensor([[12.0 * 0.5, -(arena.ZC - arena.LANE_W // 2 - 0.5) * 0.5]], dtype=torch.float32)
    assert int(tt.kind[tuple(tt.cell_idx(rail)[:2])]) == navtask.WALL


def test_rays_see_rails_and_void(task):
    tt = task.to("cpu")
    # standing on the straight, looking east: rails to both sides
    xy = torch.tensor([[12.0 * 0.5, -arena.ZC * 0.5]], dtype=torch.float32)
    wall, void = tt.rays(xy, torch.zeros(1))
    assert wall.shape == (1, navtask.N_RAYS) and float(wall.min()) < 0.6 and float(void.min()) == 1.0
    # on the beam: the void is close on both sides, no walls
    beam = torch.tensor([[54.0 * 0.5, -arena.ZC * 0.5]], dtype=torch.float32)
    wall, void = tt.rays(beam, torch.zeros(1))
    assert float(void[0, navtask.N_RAYS // 2 + navtask.N_RAYS // 4]) < 0.3  # a ray to the side (ang = +90 deg)
    assert float(wall.min()) == 1.0 or float(wall.min()) > 0.5


def test_lookahead_points_down_the_course(task):
    tt = task.to("cpu")
    xy = torch.tensor([[12.0 * 0.5, -arena.ZC * 0.5], [66.0 * 0.5, -6.0 * 0.5]], dtype=torch.float32)
    d = tt.look_at(xy)
    assert float(d[0, 0]) > 0.9  # on the straight: east
    assert float(d[1, 0]) > 0.5  # on the upper lane: eastwards (towards connector B)
    n = torch.linalg.norm(d, dim=1)
    assert torch.allclose(n, torch.ones(2), atol=1e-4)


def test_reward_rules(task):
    tt = task.to("cpu")
    z = task.floor_z + G1_BASE_HEIGHT
    gc = tt.goal.mean(0)
    assert bool(tt.reached_goal(torch.tensor([[gc[0], gc[1], z]]))[0])  # standing on the gold pad: the old absolute margin never fired
    assert not bool(tt.reached_goal(torch.tensor([[task.spawn_xy[0], task.spawn_xy[1], z]]))[0])
    hz = tt.haz[0]
    cx, cy = float((hz[0] + hz[1]) / 2), float((hz[2] + hz[3]) / 2)
    assert not bool(tt.in_hazard(torch.tensor([[cx, cy, z]]))[0])  # standing on the rim height above the pit
    assert bool(tt.in_hazard(torch.tensor([[cx, cy, task.floor_z - 0.4]]))[0])  # in the lava
    pl = tt.plates[0]
    assert bool(tt.on_plates(torch.tensor([[pl[0], pl[1], z]]))[0, 0])
    assert not bool(tt.on_plates(torch.tensor([[task.spawn_xy[0], task.spawn_xy[1], z]]))[0, 0])


def test_start_cells_cover_the_course(task):
    tt = task.to("cpu")
    assert tt.start_cells.shape[0] > 500
    assert float(tt.start_d.max()) > 45.0 and float(tt.start_d.min()) < 0.5
