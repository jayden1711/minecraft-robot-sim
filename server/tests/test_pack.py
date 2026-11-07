"""The G1 resource pack, its rig style (item_display payloads) and the pack HTTP server. No Minecraft needed."""

import hashlib
import importlib.util
import json
import sys
import threading
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
from mcrobot import g1  # noqa: E402

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


bp = load_script("build_pack")
pytestmark = pytest.mark.skipif(not (bp.MENAGERIE / "g1.xml").exists(), reason="Menagerie G1 checkout not found")


@pytest.fixture(scope="module")
def pack(tmp_path_factory):
    out = tmp_path_factory.mktemp("pack")
    report = bp.build(out)
    return out, report


def test_pack_is_reproducible(pack, tmp_path):
    out, report = pack
    r2 = bp.build(tmp_path)
    assert r2["sha1"] == report["sha1"]
    assert (out / "mcrobot-g1-pack.sha1").read_text().strip() == hashlib.sha1((out / "mcrobot-g1-pack.zip").read_bytes()).hexdigest()


def test_pack_contents(pack):
    out, report = pack
    z = zipfile.ZipFile(out / "mcrobot-g1-pack.zip")
    names = z.namelist()
    assert names == sorted(names)
    assert not any("logo" in n.lower() for n in names)
    assert "LICENSE-menagerie-g1.txt" in names and "NOTICE.txt" in names
    assert b"BSD" in z.read("LICENSE-menagerie-g1.txt") or b"Redistribution" in z.read("LICENSE-menagerie-g1.txt")
    mc = json.loads(z.read("pack.mcmeta"))["pack"]
    assert mc["min_format"] <= 75 and mc["max_format"] >= 97  # 1.21.11 is 75.0, 26.3 is 97.1
    textures = [n for n in names if n.endswith(".png")]
    assert textures == ["assets/mcrobot/textures/item/g1_palette.png"]
    assert len(z.read(textures[0])) < 2000
    assert sum(i.file_size for i in z.infolist()) < 200_000
    items = [n for n in names if n.startswith("assets/mcrobot/items/g1/")]
    assert len(items) >= 28
    for n in items:
        ref = json.loads(z.read(n))["model"]["model"]
        ns, path = ref.split(":")
        assert ns == "mcrobot" and f"assets/mcrobot/models/{path}.json" in names
    for n in names:
        if n.startswith("assets/mcrobot/models/item/g1/"):
            m = json.loads(z.read(n))
            assert m["textures"]["0"] == "mcrobot:item/g1_palette"
            assert 0 < len(m["elements"]) <= bp.BOX_BUDGET
            for e in m["elements"]:
                assert min(e["from"]) >= -16 and max(e["to"]) <= 32
                assert all(a < b for a, b in zip(e["from"], e["to"]))
                for f in e["faces"].values():
                    assert all(0 <= v <= 16 for v in f["uv"])


def test_rig_models_file(pack):
    out, report = pack
    spec = json.loads((out / "rig_models.json").read_text())
    assert spec["pack_sha1"] == report["sha1"]
    assert spec["model_scale_m"] == report["model_scale_m"]
    assert not any("logo" in p["body"] for p in spec["parts"])
    assert 28 <= len(spec["parts"]) <= 31


def posed(model):
    d = mujoco.MjData(model)
    free = model.joint("floating_base_joint").id
    a = model.jnt_qposadr[free]
    d.qpos[a:a + 7] = [1.0, -2.0, 0.9, 0.9, 0.1, -0.2, 0.3]
    d.qpos[a + 3:a + 7] /= np.linalg.norm(d.qpos[a + 3:a + 7])
    rng = np.random.default_rng(1)
    for j in g1.JOINTS:
        d.qpos[model.jnt_qposadr[model.joint(j).id]] = rng.uniform(-0.4, 0.4)
    mujoco.mj_kinematics(model, d)
    return d


@pytest.mark.skipif(not g1.XML.exists(), reason="G1 model not found")
@pytest.mark.parametrize("mpb", [0.5, 1.0])
def test_models_style_places_boxes_on_the_meshes(pack, monkeypatch, mpb):
    out, report = pack
    monkeypatch.setenv("MCROBOT_RIG_MODELS", str(out / "rig_models.json"))
    prev = load_script("preview_pack")
    m = g1.build_model(None, extra_boxes=[([0, 0, -0.5], [5, 5, 0.5])])
    d = posed(m)
    origin = [-18, 134, 30]
    rig = g1.Rig(m, origin, [1.0, -2.0, 0.9], mpb, style="models")
    pl = rig.payload(d)
    assert pl["style"] == "models"
    assert len(json.dumps(pl)) < 12000
    assert all(set(p) == {"m", "t", "r", "s"} for p in pl["parts"])
    assert {p["s"][0] for p in pl["parts"]} == {round(report["model_scale_m"] / mpb, 5)}
    # model centre lands on the link point centre_link
    anchor = np.array(pl["anchor"])
    for k, p in enumerate(pl["parts"]):
        mc = anchor + np.array(p["t"])
        mj = np.array([mc[0] - origin[0], -(mc[2] - origin[2]), mc[1] - origin[1]]) * mpb
        expect = d.xpos[rig.bid[k]] + d.xmat[rig.bid[k]].reshape(3, 3) @ rig.mcentre[k]
        assert np.allclose(mj, expect, atol=2e-3)
    # the element boxes (from the payload alone) stay inside the original mesh bounding box of their link, with one voxel of slack
    meshes = bp.body_meshes(mujoco.MjModel.from_xml_path(str(bp.MENAGERIE / "g1.xml")))
    boxes = prev.world_boxes(rig, d, out / "mcrobot-g1-pack.zip", origin)
    per_body = {}
    k = 0
    with zipfile.ZipFile(out / "mcrobot-g1-pack.zip") as z:
        for i, p in enumerate(pl["parts"]):
            n = len(json.loads(z.read(f"assets/mcrobot/models/item/{p['m'].split(':')[1]}.json"))["elements"])
            per_body[rig.m.body(rig.bid[i]).name] = boxes[k:k + n]
            k += n
    slack = report["model_scale_m"] / 16 * 1.01
    for body, bxs in per_body.items():
        allv = np.vstack([v for _, v, _ in meshes[body]])
        b = rig.m.body(body).id
        R = d.xmat[b].reshape(3, 3)
        lo, hi = allv.min(0), allv.max(0)
        for c, h, q, _ in bxs:
            ctr_link = R.T @ (c - d.xpos[b])
            half_link = h  # axes already link-aligned
            assert (ctr_link - half_link >= lo - slack).all() and (ctr_link + half_link <= hi + slack).all(), body
        # and together they span most of it
        ext = np.array([max((R.T @ (c - d.xpos[b]) + h)[i] for c, h, _, _ in bxs) - min((R.T @ (c - d.xpos[b]) - h)[i] for c, h, _, _ in bxs)
                        for i in range(3)])
        assert (ext >= (hi - lo) - 2 * slack).all(), (body, ext, hi - lo)


@pytest.mark.skipif(not g1.XML.exists(), reason="G1 model not found")
def test_blocks_style_unchanged_and_marked():
    m = g1.build_model(None, extra_boxes=[([0, 0, -0.5], [5, 5, 0.5])])
    rig = g1.Rig(m, [0, 64, 0], [0.0, 0.0, 0.0])
    pl = rig.payload(mujoco.MjData(m))
    assert pl["style"] == "blocks" and all("b" in p for p in pl["parts"])


def test_missing_rig_models_is_a_clear_error(monkeypatch, tmp_path):
    monkeypatch.setenv("MCROBOT_RIG_MODELS", str(tmp_path / "nope.json"))
    with pytest.raises(FileNotFoundError, match="build_pack"):
        g1.load_rig_models()


def test_pack_http_server(pack):
    out, report = pack
    sp = load_script("serve_pack")
    srv = sp.make_server(out, "127.0.0.1", 0)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        port = srv.server_address[1]
        r = urllib.request.urlopen(f"http://127.0.0.1:{port}/mcrobot-g1-pack.zip")
        data = r.read()
        assert hashlib.sha1(data).hexdigest() == report["sha1"]
        assert int(r.headers["Content-Length"]) == len(data)
        for bad in ("/", "/rig_models.json", "/../pack.mcmeta"):
            with pytest.raises(urllib.error.HTTPError):
                urllib.request.urlopen(f"http://127.0.0.1:{port}{bad}")
    finally:
        srv.shutdown()
        srv.server_close()
