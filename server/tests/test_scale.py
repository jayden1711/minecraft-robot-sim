"""World scale: one setting (metres of physics per block, default 0.5), written into every export header and honoured downstream."""

import json

import numpy as np
import pytest

from mcrobot import nav, protocol as P
from mcrobot.mjcf import build_scene, to_mjcf
from mcrobot.region import DEFAULT_METERS_PER_BLOCK, Region, mc_to_mj, mj_to_mc
from mcrobot.server import Server
from mcrobot.synthetic import demo_maze_wide, from_layers


def test_default_is_half_a_metre_per_block():
    assert DEFAULT_METERS_PER_BLOCK == 0.5
    assert from_layers([["#"]], {"#": "stone"}).meters_per_block == 0.5
    assert demo_maze_wide().meters_per_block == 0.5


def test_legacy_header_without_the_key_is_one_metre():
    header = {"origin": [0, 0, 0], "size": [1, 1, 1], "palette": [{"name": "minecraft:stone", "shape": "full"}]}
    assert Region.from_wire(header, np.zeros((1, 1, 1), np.uint16)).meters_per_block == 1.0
    assert Region.from_wire({**header, "meters_per_block": 0.5}, np.zeros((1, 1, 1), np.uint16)).meters_per_block == 0.5


@pytest.mark.parametrize("mpb", [1.0, 0.5, 0.25])
def test_colliders_and_markers_scale(mpb):
    r = from_layers([["#C"], ["#P"]], {"#": "stone", "C": "command_block", "P": "stone_pressure_plate"}, meters_per_block=mpb)
    sc = build_scene(r)
    top = max(g.center[2] + g.half[2] for g in sc.colliders)
    assert np.isclose(top, 2 * mpb)  # two block rows at the chosen scale
    width = max(g.center[0] + g.half[0] for g in sc.colliders)
    assert np.isclose(width, 2 * mpb)
    plate = [m for m in sc.markers if m.kind.name == "CHECKPOINT"][0]
    assert np.isclose(plate.half[2], mpb / 32)  # plates are 1/16 block thick, so 1/32 half height


def test_mc_mj_roundtrip_with_scale():
    o = (10, 64, -5)
    p = np.array([13.5, 65.25, -2.5])
    q = mc_to_mj(p, o, 0.5)
    assert np.allclose(q, [1.75, -1.25, 0.625])
    assert np.allclose(mj_to_mc(q, o, 0.5), p)


def test_nav_cells_follow_the_header_scale(tmp_path):
    r = demo_maze_wide()
    xml = tmp_path / "r.xml"
    xml.write_text(to_mjcf(build_scene(r)))
    assert nav.load_layout(xml).cell == 1.0  # no header next to it: legacy
    xml.with_suffix(".json").write_text(json.dumps({"meters_per_block": 0.5}))
    lay = nav.load_layout(xml)
    assert lay.cell == 0.5 and (lay.sx, lay.sy) == (11, 10)
    c = nav.cell_of(lay.markers["spawn"][0][0], lay)
    assert c == (2, 1)
    assert np.allclose(nav.cell_centre(c, 0.0, lay.cell)[:2], [1.25, -0.75])
    # the same build at 0.5 m/block: spawn and goal are connected for a walker with a 5 cm step limit
    cells = nav.walkable_map(lay, max_step=0.05)
    assert nav.cell_of(lay.markers["goal"][0][0], lay) in cells


def test_server_keeps_the_scale_of_each_export(tmp_path):
    srv = Server(port=0, out_dir=tmp_path)
    try:
        region = from_layers([["##"]], {"#": "stone"}, meters_per_block=0.5)
        header = {"origin": [0, 64, 0], "size": list(region.size), "palette": [{"name": e.name, "shape": e.shape} for e in region.palette],
                  "meters_per_block": 0.5}
        srv.handle_region(P.encode_region(header, region.grid))
        saved = json.loads((tmp_path / "region_0001.json").read_text())
        assert saved["meters_per_block"] == 0.5
        assert 'size="0.5 0.25 0.25"' in (tmp_path / "region_0001.xml").read_text()
    finally:
        srv.stop()
