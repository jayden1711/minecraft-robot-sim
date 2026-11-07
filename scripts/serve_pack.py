"""Serve the G1 resource pack over HTTP for server.properties `resource-pack=`. Read-only, one file, exact name only.

usage: python scripts/serve_pack.py [--port 25580] [--dir results/m1/pack] [--host 0.0.0.0]
URL: http://<NUC LAN or Tailscale address>:<port>/mcrobot-g1-pack.zip
"""

from __future__ import annotations

import argparse
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NAME = "mcrobot-g1-pack.zip"


def make_server(directory: Path, host: str = "0.0.0.0", port: int = 25580) -> ThreadingHTTPServer:
    zpath = directory / NAME

    class Handler(BaseHTTPRequestHandler):
        server_version = "mcrobot-pack"

        def _send(self, head_only: bool) -> None:
            if self.path.split("?", 1)[0] != "/" + NAME or not zpath.is_file():
                self.send_error(404)
                return
            data = zpath.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "application/zip")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            if not head_only:
                self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802
            self._send(False)

        def do_HEAD(self) -> None:  # noqa: N802
            self._send(True)

        def log_message(self, fmt, *args) -> None:
            sys.stderr.write("pack-http %s %s\n" % (self.address_string(), fmt % args))

    return ThreadingHTTPServer((host, port), Handler)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=25580)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--dir", default=str(ROOT / "results/m1/pack"))
    a = ap.parse_args()
    srv = make_server(Path(a.dir), a.host, a.port)
    print(f"serving {Path(a.dir) / NAME} on {a.host}:{srv.server_address[1]}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        srv.server_close()


if __name__ == "__main__":
    main()
