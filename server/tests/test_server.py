import socket
import threading

import numpy as np
import pytest

from mcrobot import protocol as P
from mcrobot.protocol import MsgType
from mcrobot.server import Server
from mcrobot.synthetic import demo_maze


class Client:
    def __init__(self, port):
        self.s = socket.create_connection(("127.0.0.1", port), timeout=5)

    def send(self, t, obj=None, raw=None):
        self.s.sendall(P.encode_frame(t, raw if raw is not None else P.json_payload(obj or {})))

    def recv(self):
        def f(n):
            buf = b""
            while len(buf) < n:
                c = self.s.recv(n - len(buf))
                if not c:
                    raise EOFError
                buf += c
            return buf

        return P.read_frame(f)


@pytest.fixture
def server(tmp_path):
    srv = Server(port=0, out_dir=tmp_path / "out", token="secret")
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield srv
    srv.stop()
    th.join(2)


def hello(c, token="secret", proto=(1, 1)):
    c.send(MsgType.HELLO, {"proto": list(proto), "token": token, "mod": "test"})
    return c.recv()


def test_handshake_ping_region(server, tmp_path):
    c = Client(server.port)
    t, p = hello(c)
    assert t == MsgType.HELLO_ACK and P.parse_json(p)["proto"] == 1
    c.send(MsgType.PING, {"n": 42})
    t, p = c.recv()
    assert t == MsgType.PONG and P.parse_json(p) == {"n": 42}

    region = demo_maze()
    header = {"origin": list(region.origin), "size": list(region.size),
              "palette": [{"name": e.name, "shape": e.shape} for e in region.palette]}
    c.send(MsgType.REGION, raw=P.encode_region(header, region.grid))
    t, p = c.recv()
    ack = P.parse_json(p)
    assert t == MsgType.REGION_ACK and ack["collider_boxes"] > 0 and ack["markers"] == 5
    assert (tmp_path / "out" / "region_0001.xml").exists()
    c.send(MsgType.COMMAND, {"cmd": "train"})
    assert c.recv()[0] == MsgType.STATUS
    c.send(MsgType.BYE)


def test_bad_token_rejected(server):
    c = Client(server.port)
    t, p = hello(c, token="wrong")
    assert t == MsgType.ERROR


def test_no_common_version(server):
    c = Client(server.port)
    t, p = hello(c, proto=(2, 3))
    assert t == MsgType.ERROR


def test_must_hello_first(server):
    c = Client(server.port)
    c.send(MsgType.PING, {})
    assert c.recv()[0] == MsgType.ERROR


def test_garbage_does_not_kill_server(server):
    s = socket.create_connection(("127.0.0.1", server.port), timeout=5)
    s.sendall(b"GET / HTTP/1.1\r\n\r\n")
    s.close()
    c = Client(server.port)
    assert hello(c)[0] == MsgType.HELLO_ACK
