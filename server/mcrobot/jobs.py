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

    def __init__(self, name: str, argv: list[str], log_path: Path, forward: Callable[[MsgType, bytes], None], on_exit=None):
        self.name = name
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = open(log_path, "ab")
        env = {**os.environ, "PYTHONPATH": str(ROOT / "server"), "PYTHONUNBUFFERED": "1"}
        cmd = ([str(SIM_RUN)] if SIM_RUN.exists() else []) + argv
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=self._log, env=env, cwd=str(ROOT))
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
            while True:
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


def have_sim_python() -> bool:
    return shutil.which(SIM_PY) is not None or Path(SIM_PY).exists()
