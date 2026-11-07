"""Server-list ping with a chosen client protocol number; prints the status JSON's version block.

Usage: ping_status.py HOST PORT PROTOCOL
ViaVersion answers with the *client's* protocol number when it can translate that client (and with the server's own number
when it cannot); a vanilla server always answers with its own. scripts/smoke_via.sh uses this as a protocol-level join check.
It does not log in: real client logins need a Mojang account (online-mode) and are not done by any script here.
"""
import json
import socket
import struct
import sys


def varint(n: int) -> bytes:
    out = b""
    while True:
        b = n & 0x7F
        n >>= 7
        out += bytes([b | (0x80 if n else 0)])
        if not n:
            return out


def read_varint(s: socket.socket) -> int:
    n = shift = 0
    while True:
        b = s.recv(1)
        if not b:
            raise EOFError
        n |= (b[0] & 0x7F) << shift
        if not b[0] & 0x80:
            return n
        shift += 7


def packet(pid: int, body: bytes = b"") -> bytes:
    data = varint(pid) + body
    return varint(len(data)) + data


def ping(host: str, port: int, protocol: int) -> dict:
    with socket.create_connection((host, port), timeout=10) as s:
        h = host.encode()
        s.sendall(packet(0, varint(protocol) + varint(len(h)) + h + struct.pack(">H", port) + varint(1)))
        s.sendall(packet(0))
        read_varint(s)  # packet length
        read_varint(s)  # packet id (0 = status response)
        n = read_varint(s)
        buf = b""
        while len(buf) < n:
            c = s.recv(n - len(buf))
            if not c:
                raise EOFError
            buf += c
    return json.loads(buf)


if __name__ == "__main__":
    host, port, proto = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    v = ping(host, port, proto)["version"]
    print(json.dumps({"asked": proto, "answered": v["protocol"], "name": v["name"]}))
