"""Tests that answer CRITIQUES.md. Each test is named after the critique it answers."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from mcrobot import protocol as P
from mcrobot.server import Server
from mcrobot.synthetic import demo_maze

ROOT = Path(__file__).resolve().parents[2]


def _export_through_server(tmp_path) -> Path:
    region = demo_maze()
    header = {"origin": list(region.origin), "size": list(region.size),
              "palette": [{"name": e.name, "shape": e.shape} for e in region.palette]}
    srv = Server(port=0, out_dir=tmp_path)
    try:
        ack = srv.handle_region(P.encode_region(header, region.grid))
    finally:
        srv.stop()
    return Path(ack["mjcf"])


def _rollout(xml: Path, seed: int) -> str:
    # -I: isolated mode, no PYTHONPATH, so mcrobot cannot be imported: this process is "Minecraft disconnected"
    r = subprocess.run([sys.executable, "-I", str(ROOT / "scripts" / "rollout_check.py"), str(xml), str(seed)],
                       capture_output=True, text=True, check=True, timeout=120)
    return r.stdout.strip()


def test_C2_minecraft_physics_is_fake__exported_mjcf_runs_identically_without_minecraft(tmp_path):
    xml = _export_through_server(tmp_path)
    a, b = _rollout(xml, seed=3), _rollout(xml, seed=3)
    assert a == b, "same MJCF + same seed must give a bit-identical rollout"
    assert _rollout(xml, seed=4) != a, "different seed must differ (the check is not vacuous)"
    # and the ball actually interacted with the exported blocks: it rests above the floor (z=1 top) or on a wall
    z = float(a.split()[3])
    assert z > 1.0 and np.isfinite(z)


def test_C2_in_process_and_isolated_rollouts_agree(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    from rollout_check import rollout

    xml = _export_through_server(tmp_path)
    h_in, _ = rollout(xml.read_text(), 3, 1500)
    assert h_in == _rollout(xml, seed=3).split()[0]


@pytest.mark.skip(reason="C5 needs training logs: lands with M1 (see CRITIQUES.md)")
def test_C5_ghosts_are_logged_training_states():
    ...
