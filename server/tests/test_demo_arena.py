"""The scripted G1 demo on the sky arena: frames, run events, cull box, both ghost styles. Needs the simulation python (mujoco + onnxruntime)."""

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

from mcrobot import jobs, protocol as P
from mcrobot.protocol import MsgType

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(not jobs.have_sim_python(), reason="simulation python (MCROBOT_SIM_PY) not available")


@pytest.fixture(scope="module")
def arena_files(tmp_path_factory):
    d = tmp_path_factory.mktemp("arena_export")
    spec = importlib.util.spec_from_file_location("export_arena", ROOT / "scripts" / "export_arena.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    import sys

    argv, sys.argv = sys.argv, ["export_arena.py", "--out", str(d)]
    try:
        mod.main()
    finally:
        sys.argv = argv
    return d


def frames_of(data: bytes):
    pos, out = 0, []
    while pos < len(data):
        t, n = P.decode_header(data[pos:pos + P.HEADER.size])
        pos += P.HEADER.size
        out.append((t, P.parse_json(data[pos:pos + n])))
        pos += n
    return out


def run_demo(arena_files, style, seconds=12):
    argv = [jobs.SIM_PY, str(ROOT / "scripts/g1_demo.py"), str(arena_files / "arena_region.xml"), "--stdout", "--no-realtime", "--seconds", str(seconds),
            "--style", style, "--arena", str(arena_files / "arena_route.json")]
    r = subprocess.run(argv, capture_output=True, timeout=300, cwd=ROOT, env={"PYTHONPATH": str(ROOT / "server"), "PATH": "/usr/bin:/bin", "HOME": str(Path.home())})
    assert r.returncode == 0, r.stderr.decode()[-2000:]
    return frames_of(r.stdout)


def test_blocks_style_streams_ghost_parts_and_run_events(arena_files):
    fr = run_demo(arena_files, "blocks")
    ops = [(t, o) for t, o in fr if t == MsgType.GHOST_PARTS]
    runs = [o for _, o in ops if o.get("op") == "run"]
    assert runs[0] == {"op": "run", "event": "start"}
    parts = [o for _, o in ops if o.get("op") == "parts"]
    assert len(parts) > 100 and parts[0]["style"] == "blocks" and parts[0]["spawn"] is True
    assert all("b" in p for p in parts[0]["parts"]) and len(parts[0]["parts"]) == 17
    assert parts[0]["cull"] >= 90  # the arena is 92 blocks long
    assert ops[-1][1]["op"] == "clear"
    assert any(o.get("event") == "progress" for o in runs)


def test_header_scale_sets_the_ghost_size(arena_files):
    """The rig divides physics metres by the header's metres per block: the G1 pelvis-to-ankle chain is about 1.3 m = 2.6 blocks tall."""
    fr = run_demo(arena_files, "blocks", seconds=1)
    parts = [o for t, o in fr if t == MsgType.GHOST_PARTS and o.get("op") == "parts"][0]["parts"]
    ys = [p["t"][1] for p in parts]
    assert 2.0 < max(ys) - min(ys) < 3.2  # blocks (head link top minus foot), at 0.5 m/block


@pytest.mark.skipif(not (ROOT / "results/m1/pack/rig_models.json").exists(), reason="resource pack not built (scripts/build_pack.py)")
def test_models_style_uses_item_models(arena_files):
    fr = run_demo(arena_files, "models", seconds=2)
    first = [o for t, o in fr if t == MsgType.GHOST_PARTS and o.get("op") == "parts"][0]
    assert first["style"] == "models" and all(p["m"].startswith("mcrobot:g1/") for p in first["parts"])
    assert len(first["parts"]) >= 25
