import numpy as np
import pytest

from mcrobot import protocol as P


def reader(data: bytes):
    pos = [0]

    def f(n):
        if pos[0] + n > len(data):
            raise EOFError
        out = data[pos[0] : pos[0] + n]
        pos[0] += n
        return out

    return f


def test_frame_roundtrip():
    f = P.encode_frame(P.MsgType.PING, b'{"a":1}')
    t, p = P.read_frame(reader(f))
    assert t == P.MsgType.PING and p == b'{"a":1}'


def test_empty_payload():
    t, p = P.read_frame(reader(P.encode_frame(P.MsgType.BYE)))
    assert t == P.MsgType.BYE and p == b""


def test_golden_header_bytes():
    # Java side must produce exactly this
    assert P.encode_frame(P.MsgType.HELLO, b"xy") == b"MR\x01\x01\x00\x00\x00\x02xy"


def test_bad_magic_version_type_length():
    with pytest.raises(P.ProtocolError):
        P.read_frame(reader(b"XX\x01\x03\x00\x00\x00\x00"))
    with pytest.raises(P.ProtocolError):
        P.read_frame(reader(b"MR\x09\x03\x00\x00\x00\x00"))
    with pytest.raises(P.ProtocolError):
        P.read_frame(reader(b"MR\x01\x63\x00\x00\x00\x00"))
    with pytest.raises(P.ProtocolError):
        P.read_frame(reader(b"MR\x01\x03\xff\xff\xff\xff"))


def test_truncated_is_eof():
    with pytest.raises(EOFError):
        P.read_frame(reader(P.encode_frame(P.MsgType.PING, b"abcdef")[:-2]))


def test_negotiate():
    assert P.negotiate({"proto": [1, 5]}) == 1
    with pytest.raises(P.ProtocolError):
        P.negotiate({"proto": [2, 3]})
    with pytest.raises(P.ProtocolError):
        P.negotiate({})


def make_region(seed=0):
    rng = np.random.default_rng(seed)
    grid = rng.integers(0, 3, (5, 4, 6)).astype(np.uint16)
    header = {
        "origin": [10, 64, -3],
        "size": [5, 4, 6],
        "palette": [{"name": "minecraft:air", "shape": "empty"}, {"name": "minecraft:stone", "shape": "full"},
                    {"name": "minecraft:stone_slab", "shape": [[0, 0, 0, 1, 0.5, 1]]}],
    }
    return header, grid


def test_region_roundtrip():
    h, g = make_region()
    h2, g2 = P.decode_region(P.encode_region(h, g))
    assert h2 == h and (g2 == g).all()


def test_region_index_order_is_x_fastest():
    h, g = make_region()
    payload = P.encode_region(h, g)
    import struct, zlib

    (hl,) = struct.unpack_from(">I", payload)
    raw = np.frombuffer(zlib.decompress(payload[4 + hl :]), ">u2")
    sx, sy, sz = g.shape
    assert raw[(2 * sz + 3) * sx + 1] == g[1, 2, 3]


def test_region_rejects_bad_palette_index_and_size():
    h, g = make_region()
    g[0, 0, 0] = 9
    with pytest.raises(P.ProtocolError):
        P.decode_region(P.encode_region(h, g))
    h, g = make_region()
    h["size"] = [0, 4, 6]
    with pytest.raises(P.ProtocolError):
        P.decode_region(P.encode_region({**h, "size": [5, 4, 6]}, g)[:10])


def test_region_zip_bomb_bounded():
    import struct, zlib

    h = P.json_payload({"size": [4, 4, 4], "palette": [{"name": "a", "shape": "full"}]})
    bomb = struct.pack(">I", len(h)) + h + zlib.compress(b"\0" * 10_000_000)
    with pytest.raises(P.ProtocolError):
        P.decode_region(bomb)


def test_ghosts_roundtrip():
    ids = np.array([3, 7, 49], np.uint16)
    poses = np.random.default_rng(1).normal(size=(3, 7)).astype(np.float32)
    step, i2, p2 = P.decode_ghosts(P.encode_ghosts(1234, ids, poses))
    assert step == 1234 and (i2 == ids).all() and np.allclose(p2, poses)
