"""Per-Minecraft-version checks: the Python side must accept what the mod built for each supported version sends.

The supported versions are the directories under mod/versions/ (the Stonecutter versions). MCROBOT_MC_VERSION restricts the
run to one of them (scripts/test_matrix.sh sets it for each leg of the matrix).
"""
import os
from pathlib import Path

import numpy as np
import pytest

from mcrobot import protocol as P
from mcrobot.region import Region
from mcrobot.server import Server

VERSIONS_DIR = Path(__file__).resolve().parents[2] / "mod" / "versions"
ALL = sorted(p.name for p in VERSIONS_DIR.iterdir() if p.is_dir())
ONLY = os.environ.get("MCROBOT_MC_VERSION")
VERSIONS = [ONLY] if ONLY else ALL


def test_env_version_is_supported():
    assert ONLY is None or ONLY in ALL, f"MCROBOT_MC_VERSION={ONLY} is not in mod/versions: {ALL}"


@pytest.mark.parametrize("mc", VERSIONS)
def test_region_from_each_version(mc, tmp_path):
    header = {
        "origin": [0, -64, 0],
        "size": [2, 2, 2],
        "palette": [
            {"name": "minecraft:air", "shape": "empty"},
            {"name": "minecraft:stone", "shape": "full"},
            {"name": "minecraft:gold_block", "shape": "full"},
        ],
        "dimension": "minecraft:overworld",
        "mc_version": mc,
        "meters_per_block": 1.0,
    }
    grid = np.zeros((2, 2, 2), dtype=np.uint16)  # (sx, sy, sz)
    grid[:, 0, :] = 1
    grid[0, 1, 0] = 2
    h, g = P.decode_region(P.encode_region(header, grid))
    region = Region.from_wire(h, g)
    assert region.meta["mc_version"] == mc
    assert region.dimension == "minecraft:overworld"

    srv = Server(port=0, out_dir=tmp_path / "out", token="")
    ack = srv.handle_region(P.encode_region(header, grid))
    assert ack["solid_cells"] == 5 and ack["collider_boxes"] >= 1 and ack["markers"] == 1
