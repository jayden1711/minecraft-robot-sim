"""/robot train and /robot stop through the real Server code with a fake gpu-run and a fake trainer (no GPU, no torch)."""

import json
import os
import socket
import stat
import sys
import time

from mcrobot import jobs, protocol
from mcrobot.protocol import MsgType
from mcrobot.server import Server

FAKE_TRAINER = '''
import os, signal, socket, sys, time
sys.path.insert(0, {srv!r})
from mcrobot import protocol
from mcrobot.protocol import MsgType
out = sys.argv[sys.argv.index("--out") + 1]
os.makedirs(out, exist_ok=True)
open(os.path.join(out, "trainer.pid"), "w").write(str(os.getpid()))
path = os.path.join(out, "ghosts.sock")
if os.path.exists(path):
    os.unlink(path)
s = socket.socket(socket.AF_UNIX)
s.bind(path)
s.listen(1)
signal.signal(signal.SIGTERM, lambda *a: sys.exit(0))
c, _ = s.accept()
c.sendall(protocol.encode_frame(MsgType.GHOST_PARTS, protocol.json_payload({{"op": "parts", "ghost": 0, "parts": []}})))
time.sleep(60)
'''

FAKE_GPU_RUN = '''#!/bin/bash
while [ "$1" != "--" ]; do shift; done
shift
exec "$@"
'''


def read_until(sock, pred, timeout=8.0):
    sock.settimeout(timeout)
    seen = []

    def recv(n):
        buf = bytearray()
        while len(buf) < n:
            c = sock.recv(n - len(buf))
            if not c:
                raise EOFError
            buf += c
        return bytes(buf)

    end = time.time() + timeout
    while time.time() < end:
        t, p = protocol.read_frame(recv)
        seen.append((t, json.loads(p)))
        if pred(seen[-1]):
            return seen
    raise AssertionError(seen)


def test_train_starts_a_preemptible_job_attaches_and_stop_ends_it(tmp_path, monkeypatch):
    root = tmp_path / "root"
    (root / "scripts").mkdir(parents=True)
    (root / "results/m1/arena_v2").mkdir(parents=True)
    (root / "results/m1/arena_v2/arena_region.xml").write_text("<mujoco/>")
    (root / "scripts/train_nav.py").write_text(FAKE_TRAINER.format(srv=os.path.dirname(os.path.dirname(protocol.__file__))))
    gr = tmp_path / "gpu-run"
    gr.write_text(FAKE_GPU_RUN)
    gr.chmod(gr.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setattr(jobs, "ROOT", root)
    monkeypatch.setattr(jobs, "TRAIN_PY", sys.executable)
    monkeypatch.setattr(jobs, "GPU_RUN", gr)
    monkeypatch.setattr(jobs, "NAV_OUT", root / "results/m1/nav/v2")
    srv = Server(port=0, out_dir=tmp_path / "exports")
    a, b = socket.socketpair()
    try:
        srv._command(a, "train")
        seen = read_until(b, lambda m: m[0] == MsgType.GHOST_PARTS and m[1].get("op") == "parts")
        texts = [m[1].get("text", "") for m in seen if m[0] == MsgType.STATUS]
        assert any("queued on the GPU" in t for t in texts) and any("trainer connected" in t for t in texts)
        assert sum("trainer connected" in t for t in texts) == 1  # chat only on the state change
        assert srv._job is not None and srv._job.running
        srv._command(a, "stop")
        seen = read_until(b, lambda m: m[0] == MsgType.GHOST_PARTS and m[1].get("op") == "clear")
        assert any(m[1].get("text") == "stopped" for m in seen if m[0] == MsgType.STATUS)
        assert srv._job is None and srv._attach is None
    finally:
        srv._stop_job()
        a.close()
        b.close()
        srv.stop()


def test_play_without_a_policy_says_so(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "NAV_OUT", tmp_path / "none")
    monkeypatch.setattr(jobs, "ROOT", tmp_path)
    srv = Server(port=0, out_dir=tmp_path / "exports")
    a, b = socket.socketpair()
    try:
        srv._command(a, "play")
        (t, o), = read_until(b, lambda m: True)
        assert t == MsgType.STATUS and "no trained policy" in o["text"]
    finally:
        a.close()
        b.close()
        srv.stop()
