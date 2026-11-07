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
        self._ids = itertools.count(1)

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

    @staticmethod
    def _send(conn, mtype, obj_or_bytes) -> None:
        payload = obj_or_bytes if isinstance(obj_or_bytes, bytes) else protocol.json_payload(obj_or_bytes)
        conn.sendall(protocol.encode_frame(mtype, payload))

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
        while True:
            mtype, payload = protocol.read_frame(recv)
            if mtype == MsgType.PING:
                self._send(conn, MsgType.PONG, protocol.parse_json(payload))
            elif mtype == MsgType.REGION:
                self._send(conn, MsgType.REGION_ACK, self.handle_region(payload))
            elif mtype == MsgType.COMMAND:
                cmd = protocol.parse_json(payload).get("cmd", "")
                self._send(conn, MsgType.STATUS, {"text": f"command '{cmd}' not implemented until a later milestone"})
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
