"""Wire protocol between the Fabric mod and the Python server (see PLAN.md, "Socket protocol").

Frame: magic "MR" (2 bytes) | framing version u8 | type u8 | payload length u32 big-endian | payload.
All integers/floats in binary payloads are big-endian so the Java side can use the default ByteBuffer order.
"""

from __future__ import annotations

import enum
import json
import struct
import zlib
from typing import Callable

import numpy as np

MAGIC = b"MR"
FRAMING_VERSION = 1
APP_PROTO_MIN = 1
APP_PROTO_MAX = 2
MAX_FRAME = 64 * 1024 * 1024
HEADER = struct.Struct(">2sBBI")


class ProtocolError(Exception):
    pass


class MsgType(enum.IntEnum):
    HELLO = 1  # mod -> server, JSON
    HELLO_ACK = 2  # server -> mod, JSON
    PING = 3  # either, JSON
    PONG = 4  # either, JSON (echoes PING)
    REGION = 5  # mod -> server, region payload
    REGION_ACK = 6  # server -> mod, JSON
    COMMAND = 7  # mod -> server, JSON {"cmd": "train", ...}
    STATUS = 8  # server -> mod, JSON {"text": ...} shown in chat
    GHOSTS = 9  # server -> mod, ghost payload
    ERROR = 10  # either, JSON {"code": ..., "message": ...}
    BYE = 11  # either, JSON
    GHOST_PARTS = 12  # server -> mod (app proto >= 2), JSON: display transformations of ghost parts


def encode_frame(msg_type: int, payload: bytes = b"") -> bytes:
    if len(payload) > MAX_FRAME:
        raise ProtocolError(f"payload {len(payload)} exceeds MAX_FRAME")
    return HEADER.pack(MAGIC, FRAMING_VERSION, int(msg_type), len(payload)) + payload


def decode_header(data: bytes) -> tuple[MsgType, int]:
    magic, version, mtype, length = HEADER.unpack(data)
    if magic != MAGIC:
        raise ProtocolError(f"bad magic {magic!r}")
    if version != FRAMING_VERSION:
        raise ProtocolError(f"unsupported framing version {version}")
    if length > MAX_FRAME:
        raise ProtocolError(f"frame length {length} exceeds MAX_FRAME")
    try:
        return MsgType(mtype), length
    except ValueError:
        raise ProtocolError(f"unknown message type {mtype}") from None


def read_frame(recv_exact: Callable[[int], bytes]) -> tuple[MsgType, bytes]:
    """recv_exact(n) must return exactly n bytes or raise EOFError."""
    mtype, length = decode_header(recv_exact(HEADER.size))
    return mtype, recv_exact(length) if length else b""


def json_payload(obj: dict) -> bytes:
    return json.dumps(obj, separators=(",", ":"), sort_keys=True).encode("utf-8")


def parse_json(payload: bytes) -> dict:
    try:
        obj = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ProtocolError(f"bad JSON payload: {e}") from None
    if not isinstance(obj, dict):
        raise ProtocolError("JSON payload must be an object")
    return obj


def negotiate(hello: dict) -> int:
    """Pick the highest app protocol version both sides support."""
    try:
        lo, hi = hello["proto"]
    except (KeyError, TypeError, ValueError):
        raise ProtocolError("HELLO needs 'proto': [min, max]") from None
    best = min(hi, APP_PROTO_MAX)
    if best < max(lo, APP_PROTO_MIN):
        raise ProtocolError(f"no common protocol: peer {lo}..{hi}, server {APP_PROTO_MIN}..{APP_PROTO_MAX}")
    return best


# ---- REGION payload ------------------------------------------------------------------------------------
# u32 header_len | JSON header | zlib(grid). grid = uint16 big-endian palette indices, index (y*sz + z)*sx + x
# (x fastest). Header: origin [x,y,z], size [sx,sy,sz], palette [{"name", "shape"}], dimension, mc_version.
# palette[i].shape: "empty" | "full" | [[x0,y0,z0,x1,y1,z1], ...] collision boxes in block-local 0..1 coordinates.


def encode_region(header: dict, grid: np.ndarray) -> bytes:
    sx, sy, sz = header["size"]
    if grid.shape != (sx, sy, sz):
        raise ProtocolError(f"grid shape {grid.shape} != size {(sx, sy, sz)}")
    raw = np.ascontiguousarray(grid.transpose(1, 2, 0)).astype(">u2").tobytes()
    h = json_payload(header)
    return struct.pack(">I", len(h)) + h + zlib.compress(raw, 6)


def decode_region(payload: bytes) -> tuple[dict, np.ndarray]:
    if len(payload) < 4:
        raise ProtocolError("region payload too short")
    (hlen,) = struct.unpack_from(">I", payload)
    if 4 + hlen > len(payload):
        raise ProtocolError("region header overruns payload")
    header = parse_json(payload[4 : 4 + hlen])
    try:
        sx, sy, sz = (int(v) for v in header["size"])
        n_pal = len(header["palette"])
    except (KeyError, TypeError, ValueError):
        raise ProtocolError("region header needs 'size' and 'palette'") from None
    if min(sx, sy, sz) < 1 or sx * sy * sz > MAX_FRAME // 2:
        raise ProtocolError(f"bad region size {(sx, sy, sz)}")
    try:
        raw = zlib.decompressobj().decompress(payload[4 + hlen :], sx * sy * sz * 2 + 1)
    except zlib.error as e:
        raise ProtocolError(f"bad region compression: {e}") from None
    if len(raw) != sx * sy * sz * 2:
        raise ProtocolError("region grid has wrong byte length")
    grid = np.frombuffer(raw, dtype=">u2").reshape(sy, sz, sx).transpose(2, 0, 1).astype(np.uint16)
    if grid.size and int(grid.max()) >= n_pal:
        raise ProtocolError("region grid references palette index out of range")
    return header, grid


# ---- GHOSTS payload ------------------------------------------------------------------------------------
# u32 train_step | u16 n | n * (u16 env_id | 7 * f32 [x,y,z (Minecraft world coords), qw,qx,qy,qz]).
# Poses are already converted from MuJoCo to Minecraft coordinates by the server; the mod stays dumb.

_GHOST_HEAD = struct.Struct(">IH")
_GHOST_ROW = struct.Struct(">H7f")


def encode_ghosts(step: int, env_ids: np.ndarray, poses: np.ndarray) -> bytes:
    if poses.shape != (len(env_ids), 7):
        raise ProtocolError("poses must be (n, 7)")
    out = [_GHOST_HEAD.pack(step, len(env_ids))]
    for e, p in zip(env_ids, poses):
        out.append(_GHOST_ROW.pack(int(e), *map(float, p)))
    return b"".join(out)


def decode_ghosts(payload: bytes) -> tuple[int, np.ndarray, np.ndarray]:
    step, n = _GHOST_HEAD.unpack_from(payload)
    if len(payload) != _GHOST_HEAD.size + n * _GHOST_ROW.size:
        raise ProtocolError("ghost payload has wrong length")
    ids = np.empty(n, np.uint16)
    poses = np.empty((n, 7), np.float32)
    for i in range(n):
        row = _GHOST_ROW.unpack_from(payload, _GHOST_HEAD.size + i * _GHOST_ROW.size)
        ids[i] = row[0]
        poses[i] = row[1:]
    return step, ids, poses
