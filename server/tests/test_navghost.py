"""Ghost swarm plumbing: the trainer's frame socket, the server's reconnecting attach (chat only on state changes), and the streamer's rate limit."""

import json
import socket
import threading
import time

import pytest

from mcrobot import jobs, protocol
from mcrobot.navghost import FrameOut
from mcrobot.protocol import MsgType


def read_frames(sock, n, timeout=3.0):
    sock.settimeout(timeout)

    def recv(k):
        buf = bytearray()
        while len(buf) < k:
            c = sock.recv(k - len(buf))
            if not c:
                raise EOFError
            buf += c
        return bytes(buf)

    return [protocol.read_frame(recv) for _ in range(n)]


def test_frameout_socket_announces_a_new_client_once_and_never_blocks(tmp_path):
    fo = FrameOut(None, tmp_path / "g.sock")
    assert not fo.active
    fo.emit(MsgType.STATUS, {"text": "nobody listens"})  # no client: dropped silently
    c = socket.socket(socket.AF_UNIX)
    c.connect(str(tmp_path / "g.sock"))
    time.sleep(0.05)
    assert fo.active and fo.new_client and not fo.new_client
    fo.emit(MsgType.STATUS, {"text": "hi"})
    (t, p), = read_frames(c, 1)
    assert t == MsgType.STATUS and json.loads(p)["text"] == "hi"
    c.close()
    for _ in range(5):  # a gone client is dropped without raising
        fo.emit(MsgType.STATUS, {"text": "x" * 100})
    fo.close()
    assert not (tmp_path / "g.sock").exists()


def test_attach_reconnects_and_reports_only_state_changes(tmp_path):
    path = tmp_path / "ghosts.sock"
    got, states = [], []
    att = jobs.Attach(path, lambda t, p: got.append((t, p)), states.append, retry_s=0.05)
    try:
        time.sleep(0.2)
        assert states == [] and not att.connected  # trainer not there yet (queued on the GPU): no chat
        a = FrameOut(None, path)
        deadline = time.time() + 3
        while not att.connected and time.time() < deadline:
            a.active  # accept
            time.sleep(0.02)
        a.emit(MsgType.STATUS, {"text": "one"})
        deadline = time.time() + 3
        while not got and time.time() < deadline:
            time.sleep(0.02)
        assert states == ["connected"] and got and json.loads(got[0][1])["text"] == "one"
        a.close()  # the trainer yields the GPU
        deadline = time.time() + 3
        while len(states) < 2 and time.time() < deadline:
            time.sleep(0.02)
        assert states == ["connected", "disconnected"]
        time.sleep(0.3)
        assert states == ["connected", "disconnected"]  # still gone: no repeated message
        b = FrameOut(None, path)  # resumed
        deadline = time.time() + 3
        while len(states) < 3 and time.time() < deadline:
            b.active
            time.sleep(0.02)
        assert states == ["connected", "disconnected", "connected"]
        b.close()
    finally:
        att.stop()


def test_trainer_pid_only_counts_a_live_train_script(tmp_path):
    assert jobs.trainer_pid(tmp_path) is None
    (tmp_path / "trainer.pid").write_text("999999999")
    assert jobs.trainer_pid(tmp_path) is None
    (tmp_path / "trainer.pid").write_text(str(threading.get_native_id()))  # this process is not train_nav
    assert jobs.trainer_pid(tmp_path) is None


torch = pytest.importorskip("torch")
pytest.importorskip("mujoco_warp")


def test_streamer_rate_limit_ids_and_spawn_flag(tmp_path):
    from mcrobot import arena, g1, navenv
    from mcrobot.mjcf import build_scene, to_mjcf
    from mcrobot.navghost import GhostStreamer

    if not g1.ONNX.exists():
        pytest.skip("unitree_rl_mjlab policy not found")
    xml = tmp_path / "arena_region.xml"
    xml.write_text(to_mjcf(build_scene(arena.to_region())))
    (tmp_path / "arena_region.json").write_text(json.dumps({"origin": [96, 200, -160], "size": [92, 14, 34], "meters_per_block": 0.5}))
    env = navenv.NavEnv(str(xml), 3, "cpu", seed=0)
    fo = FrameOut(None, tmp_path / "s.sock")
    st = GhostStreamer(env, xml, fo, n_ghosts=3, hz=4.0)
    c = socket.socket(socket.AF_UNIX)
    c.connect(str(tmp_path / "s.sock"))
    time.sleep(0.05)
    st(env)
    st(env)  # inside the 250 ms window: dropped, never queued
    assert st.sent == 1
    frames = read_frames(c, 3)
    objs = [json.loads(p) for _, p in frames]
    assert [o["ghost"] for o in objs] == [0, 1, 2] and all(o["spawn"] for o in objs) and len(objs[0]["parts"]) == len(g1.RIG)
    time.sleep(0.3)
    st(env)
    objs = [json.loads(p) for _, p in read_frames(c, 3)]
    assert st.sent == 2 and not any(o["spawn"] for o in objs)
    c.close()
    fo.close()
