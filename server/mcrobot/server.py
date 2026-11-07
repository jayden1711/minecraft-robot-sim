"""Blocking TCP server on loopback. One mod connection at a time (the mod is the only client)."""

from __future__ import annotations

import argparse
import hmac
import itertools
import json
import os
import socket
import threading
import time
from pathlib import Path

from . import __version__, protocol
from .mjcf import build_scene, mjcf_sha256, to_mjcf
from .protocol import MsgType, ProtocolError
from .region import Region

DEFAULT_PORT = 25599


class Server:
    def __init__(self, host: str = "127.0.0.1", port: int = DEFAULT_PORT, out_dir: str | Path = "results/m0/exports", token: str = ""):
        self.out_dir = Path(out_dir)
        self.token = token
        self.sock = socket.create_server((host, port))
        self.port = self.sock.getsockname()[1]
        self._stop = threading.Event()
        self._ids = itertools.count(self._next_export_id())
        self._send_lock = threading.Lock()
        self._job = None
        self._job_arg = None
        self.style: str | None = None  # ghost look chosen with /robot style; None = default_style()

    def _next_export_id(self) -> int:
        """One past the highest existing region_NNNN, so a restart never overwrites an earlier export."""
        ids = [int(p.stem.split("_")[1]) for p in self.out_dir.glob("region_*.xml") if p.stem.split("_")[1].isdigit()]
        return max(ids, default=0) + 1

    def latest_region(self) -> Path | None:
        """Newest export that has at least one collider box (empty exports are skipped)."""
        for p in sorted(self.out_dir.glob("region_*.xml"), reverse=True):
            if 'name="col_' in p.read_text() and p.with_suffix(".json").exists():
                return p
        return None

    def serve_forever(self) -> None:
        self.sock.settimeout(0.2)
        while not self._stop.is_set():
            try:
                conn, addr = self.sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            conn.settimeout(30)
            try:
                self._handle(conn)
            except (EOFError, ConnectionError, socket.timeout):
                pass
            except ProtocolError as e:
                self._send_error(conn, "protocol", str(e))
            finally:
                conn.close()

    def stop(self) -> None:
        self._stop.set()
        self.sock.close()

    @staticmethod
    def _recv_exact(conn: socket.socket):
        def f(n: int) -> bytes:
            buf = bytearray()
            while len(buf) < n:
                chunk = conn.recv(n - len(buf))
                if not chunk:
                    raise EOFError
                buf += chunk
            return bytes(buf)

        return f

    def _send(self, conn, mtype, obj_or_bytes) -> None:
        payload = obj_or_bytes if isinstance(obj_or_bytes, bytes) else protocol.json_payload(obj_or_bytes)
        with self._send_lock:
            conn.sendall(protocol.encode_frame(mtype, payload))

    def _stop_job(self) -> bool:
        job, self._job = self._job, None
        if job is None or not job.running:
            return False
        job.stop()
        return True

    def default_style(self) -> str:
        """Ghost look: "models" (item_display parts of the resource pack) when the pack's rig file exists, else "blocks"."""
        env = os.environ.get("MCROBOT_GHOST_STYLE")
        if env in ("models", "blocks"):
            return env
        from . import jobs

        return "models" if (jobs.ROOT / "results/m1/pack/rig_models.json").exists() else "blocks"

    def _command(self, conn, cmd: str, arg: str | None = None) -> None:
        from . import jobs

        if cmd == "style":
            if arg not in ("models", "blocks"):
                self._send(conn, MsgType.STATUS, {"text": "style: use /robot style models (needs the server resource pack) or /robot style blocks"})
                return
            self.style = arg
            self._send(conn, MsgType.STATUS, {"text": f"ghost style set to {arg}" + (": restart the demo with /robot demo" if self._job and self._job.running else "")})
            if self._job is not None and self._job.running and self._job.name == "demo":
                self._command(conn, "demo", self._job_arg)
            return
        if cmd == "demo":
            self._stop_job()
            self._job_arg = arg
            if arg in ("arena", "arena_v2"):
                sub, script = ("arena", "export_arena.py") if arg == "arena" else ("arena_v2", "export_arena_v2.py")
                region = jobs.ROOT / f"results/m1/{sub}/arena_region.xml"
                if not region.exists():
                    self._send(conn, MsgType.STATUS, {"text": f"demo {arg}: results/m1/{sub}/arena_region.xml missing (run scripts/{script})"})
                    return
            else:
                region = self.latest_region()
            if region is None:
                self._send(conn, MsgType.STATUS, {"text": "demo: no region exported yet (build the maze, then /robot export)"})
                return
            if not jobs.have_sim_python():
                self._send(conn, MsgType.STATUS, {"text": f"demo: simulation python not found ({jobs.SIM_PY}); set MCROBOT_SIM_PY"})
                return

            def forward(mtype, payload):
                try:
                    self._send(conn, mtype, payload)
                except OSError:
                    pass

            def on_exit(job, rc):
                if rc not in (0, -15):
                    forward(MsgType.STATUS, protocol.json_payload({"text": f"demo ended with exit code {rc} (see results/m1/demo.log)"}))
                    forward(MsgType.GHOST_PARTS, protocol.json_payload({"op": "clear", "reason": f"error: demo exited with code {rc}"}))

            style = self.style or self.default_style()
            argv = [jobs.SIM_PY, str(jobs.ROOT / "scripts/g1_demo.py"), str(region.resolve()), "--stdout", "--style", style]
            if arg in ("arena", "arena_v2"):
                argv += ["--arena", str(region.with_name("arena_route.json"))]
            self._job = jobs.Job("demo", argv, jobs.ROOT / "results/m1/demo.log", forward, on_exit)
            self._send(conn, MsgType.STATUS, {"text": f"demo: starting the G1 on {region.name} ({style} ghost; simulation, scripted commands; it appears in a few seconds)"})
        elif cmd == "stop":
            was = self._stop_job()
            self._send(conn, MsgType.STATUS, {"text": "stopped" if was else "nothing running"})
            self._send(conn, MsgType.GHOST_PARTS, {"op": "clear", "reason": "stopped by command"})
        else:
            self._send(conn, MsgType.STATUS, {"text": f"command '{cmd}' not implemented until a later milestone"})

    def _send_error(self, conn, code: str, message: str) -> None:
        try:
            self._send(conn, MsgType.ERROR, {"code": code, "message": message})
        except OSError:
            pass

    def _handle(self, conn: socket.socket) -> None:
        recv = self._recv_exact(conn)
        mtype, payload = protocol.read_frame(recv)
        if mtype != MsgType.HELLO:
            raise ProtocolError("first message must be HELLO")
        hello = protocol.parse_json(payload)
        version = protocol.negotiate(hello)
        if self.token and not hmac.compare_digest(str(hello.get("token", "")), self.token):
            raise ProtocolError("bad token")
        self._send(conn, MsgType.HELLO_ACK, {"proto": version, "server": __version__, "time": time.time()})
        conn.settimeout(None)  # the mod keeps the connection open while a job streams ghosts
        try:
            self._serve(conn, recv)
        finally:
            self._stop_job()

    def _serve(self, conn, recv) -> None:
        while True:
            mtype, payload = protocol.read_frame(recv)
            if mtype == MsgType.PING:
                self._send(conn, MsgType.PONG, protocol.parse_json(payload))
            elif mtype == MsgType.REGION:
                self._send(conn, MsgType.REGION_ACK, self.handle_region(payload))
            elif mtype == MsgType.COMMAND:
                msg = protocol.parse_json(payload)
                self._command(conn, msg.get("cmd", ""), msg.get("arg"))
            elif mtype == MsgType.BYE:
                return
            else:
                raise ProtocolError(f"unexpected message {mtype.name}")

    def handle_region(self, payload: bytes) -> dict:
        header, grid = protocol.decode_region(payload)
        region = Region.from_wire(header, grid)
        scene = build_scene(region)
        xml = to_mjcf(scene)
        rid = next(self._ids)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / f"region_{rid:04d}.xml"
        path.write_text(xml)
        (self.out_dir / f"region_{rid:04d}.json").write_text(json.dumps({**header, "palette": header["palette"]}, indent=1))
        return {
            "id": rid,
            "mjcf": str(path),
            "sha256": mjcf_sha256(xml),
            "solid_cells": scene.n_solid_cells,
            "collider_boxes": len(scene.colliders),
            "markers": len(scene.markers),
        }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--out", default="results/m0/exports")
    args = ap.parse_args()
    srv = Server(args.host, args.port, args.out, os.environ.get("MCROBOT_TOKEN", ""))
    print(f"mcrobot server {__version__} listening on {args.host}:{srv.port}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        srv.stop()


if __name__ == "__main__":
    main()
