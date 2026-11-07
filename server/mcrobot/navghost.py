"""Streams real env states of a few parallel envs to the Minecraft mod as ghost frames (MR protocol), rate-limited, latest-wins on the
mod side. Frames go to a file descriptor (--stdout) and/or to the client of a unix socket (the Python server attaches to a running
trainer there). Every pose sent is also kept (ghost_log.npz) so a test can replay what the player saw."""

from __future__ import annotations

import json
import os
import socket
import time
from pathlib import Path

import numpy as np

from . import g1, nav, protocol
from .protocol import MsgType


class FrameOut:
    """Frame sink: an fd (the original stdout) and/or one client of a unix socket at sock_path (a newer client replaces the old one)."""

    def __init__(self, fd: int | None = None, sock_path: str | Path | None = None):
        self.f = os.fdopen(fd, "wb", buffering=0) if fd is not None else None
        self.srv = None
        self.client = None
        self.sock_path = str(sock_path) if sock_path else None
        if self.sock_path:
            Path(self.sock_path).unlink(missing_ok=True)
            self.srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.srv.bind(self.sock_path)
            self.srv.listen(1)
            self.srv.setblocking(False)

    @property
    def active(self) -> bool:
        self._accept()
        return self.f is not None or self.client is not None

    @property
    def new_client(self) -> bool:
        """True once after a client attached (the ghost spawn frame must be sent again)."""
        v, self._fresh = getattr(self, "_fresh", False), False
        return v

    def _accept(self):
        if self.srv is None:
            return
        try:
            c, _ = self.srv.accept()
        except (BlockingIOError, InterruptedError):
            return
        if self.client is not None:
            self.client.close()
        c.settimeout(0.5)
        self.client = c
        self._fresh = True

    def emit(self, mtype, obj):
        data = protocol.encode_frame(mtype, protocol.json_payload(obj))
        self._accept()
        if self.f is not None:
            try:
                self.f.write(data)
            except OSError:
                self.f = None  # the server went away; training goes on without the display
        if self.client is not None:
            try:
                self.client.sendall(data)
            except OSError:  # a slow or gone client never blocks training
                self.client.close()
                self.client = None

    def close(self):
        if self.client is not None:
            self.client.close()
        if self.srv is not None:
            self.srv.close()
            Path(self.sock_path).unlink(missing_ok=True)


class GhostStreamer:
    def __init__(self, env, xml_path, out: FrameOut, n_ghosts: int = 20, hz: float = 4.0, style: str = "blocks", log_path: Path | None = None):
        header = json.loads(Path(xml_path).with_suffix(".json").read_text())
        lay = nav.load_layout(xml_path)
        mpb = float(header.get("meters_per_block", 1.0))
        centre = np.array([lay.sx / 2, -lay.sy / 2, 0.0]) * lay.cell + [0, 0, env.floor_z]
        self.rig = g1.Rig(env.model, header["origin"], centre, mpb, style=style)
        self.cull = float(max(header["size"][0], header["size"][2])) + 8.0
        self.out, self.n, self.period = out, min(n_ghosts, env.n), 1.0 / hz
        self.ticks = max(2, int(round(self.period * 20)))
        self.last = 0.0
        self.sent = 0
        self.calls = 0
        self.spawn_next = True
        self.log_path = log_path
        self.log = []  # (env step index, ghost, pelvis x y z) for every pose sent

    def __call__(self, env, trainer=None):
        self.calls += 1  # one call per navigation step of the env
        if not self.out.active:
            return
        if self.out.new_client:
            self.spawn_next = True
        now = time.perf_counter()
        if now - self.last < self.period:
            return
        self.last = now
        states = env.body_state(list(range(self.n)))
        for i, d in enumerate(states):
            payload = self.rig.payload(d, ghost=i, ticks=self.ticks, spawn=self.spawn_next)
            payload["cull"] = self.cull
            self.out.emit(MsgType.GHOST_PARTS, payload)
            if len(self.log) < 400_000:
                self.log.append((self.calls, i, *np.round(d.xpos[1][:3], 3)))
        self.spawn_next = False
        self.sent += 1

    def clear(self, reason: str):
        self.out.emit(MsgType.GHOST_PARTS, {"op": "clear", "reason": reason})
        if self.log_path is not None and self.log:
            np.savez_compressed(self.log_path, rows=np.array(self.log, np.float32))
