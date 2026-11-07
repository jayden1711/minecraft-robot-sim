"""Child processes that stream protocol frames on stdout (the G1 demo, later training). The server forwards the frames to the mod."""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Callable

from . import protocol
from .protocol import MsgType

ROOT = Path(__file__).resolve().parents[2]
SIM_PY = os.environ.get("MCROBOT_SIM_PY", str(Path.home() / "projects/robot-arena-sim/.venv/bin/python"))
SIM_RUN = Path.home() / "projects/bin/sim-run"


class Job:
    """argv runs under sim-run (nice 10, one thread) when it exists. stdout carries MR frames; stderr goes to log_path."""

    def __init__(self, name: str, argv: list[str], log_path: Path, forward: Callable[[MsgType, bytes], None], on_exit=None, gpu: bool = False):
        """gpu=True: argv is a gpu-run command (not wrapped in sim-run), its stdout goes to the log and frames come through an Attach instead."""
        self.name = name
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = open(log_path, "ab")
        self._frames = not gpu
        env = {**os.environ, "PYTHONPATH": str(ROOT / "server"), "PYTHONUNBUFFERED": "1"}
        cmd = ([str(SIM_RUN)] if SIM_RUN.exists() and not gpu else []) + argv
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE if self._frames else self._log, stderr=self._log, env=env, cwd=str(ROOT))
        self._forward = forward
        self._on_exit = on_exit
        self._stopping = False
        self._thread = threading.Thread(target=self._pump, name=f"job-{name}", daemon=True)
        self._thread.start()

    def _pump(self) -> None:
        out = self.proc.stdout

        def recv(n: int) -> bytes:
            b = out.read(n)
            if len(b) < n:
                raise EOFError
            return b

        try:
            while self._frames:
                mtype, payload = protocol.read_frame(recv)
                self._forward(mtype, payload)
        except (EOFError, protocol.ProtocolError, OSError, ValueError):
            pass
        finally:
            rc = self.proc.wait()
            self._log.close()
            if self._on_exit and not self._stopping:
                self._on_exit(self, rc)

    @property
    def running(self) -> bool:
        return self.proc.poll() is None

    def stop(self) -> None:
        self._stopping = True
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self._thread.join(2)


TRAIN_PY = os.environ.get("MCROBOT_TRAIN_PY", str(ROOT / ".venv-train/bin/python"))
GPU_RUN = Path.home() / "projects/bin/gpu-run"
NAV_ARENA = "arena_v2"  # the arena the navigation policy trains and plays on (v1 results stay in results/m1/nav/run1)
NAV_OUT = ROOT / "results/m1/nav/v2"


class Attach:
    """Forwards the frames of a running trainer (unix socket <out>/ghosts.sock) to the mod. Reconnects while the trainer yields the GPU and
    comes back; `on_state("connected" | "disconnected")` fires only on changes, so the chat is not spammed."""

    def __init__(self, sock_path: Path, forward: Callable[[MsgType, bytes], None], on_state=None, retry_s: float = 3.0):
        self.path, self._forward, self._on_state, self.retry = str(sock_path), forward, on_state, retry_s
        self._stop = threading.Event()
        self._sock = None
        self.frames = 0
        self._thread = threading.Thread(target=self._run, name="attach", daemon=True)
        self._thread.start()

    @property
    def connected(self) -> bool:
        return self._sock is not None

    def _run(self) -> None:
        import socket

        while not self._stop.is_set():
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                s.connect(self.path)
            except OSError:
                s.close()
                self._stop.wait(self.retry)
                continue
            self._sock = s
            if self._on_state:
                self._on_state("connected")

            def recv(n: int) -> bytes:
                buf = bytearray()
                while len(buf) < n:
                    chunk = s.recv(n - len(buf))
                    if not chunk:
                        raise EOFError
                    buf += chunk
                return bytes(buf)

            try:
                while not self._stop.is_set():
                    mtype, payload = protocol.read_frame(recv)
                    self.frames += 1
                    self._forward(mtype, payload)
            except (EOFError, protocol.ProtocolError, OSError, ValueError):
                pass
            finally:
                self._sock = None
                s.close()
            if not self._stop.is_set() and self._on_state:
                self._on_state("disconnected")

    def stop(self) -> None:
        self._stop.set()
        if self._sock is not None:
            try:
                self._sock.shutdown(2)
            except OSError:
                pass
        self._thread.join(2)


def trainer_pid(out: Path = NAV_OUT) -> int | None:
    """Pid of a live train_nav.py for this run directory (written by the script), else None."""
    p = out / "trainer.pid"
    try:
        pid = int(p.read_text())
        return pid if "train_nav" in Path(f"/proc/{pid}/cmdline").read_text() else None
    except (OSError, ValueError):
        return None


def have_sim_python() -> bool:
    return shutil.which(SIM_PY) is not None or Path(SIM_PY).exists()
