"""Render the G1 pack models at the policy's default standing pose in MuJoCo, from the Rig(style="models") payload and the
element boxes inside the built pack zip (so it checks the whole chain: pack JSON -> payload -> world), to
results/m1/pack/preview_front.png and preview_side.png. CPU rendering (Mesa llvmpipe), never the GPU.

usage: PYTHONPATH=server python scripts/preview_pack.py [--pack results/m1/pack] [--mpb 0.5]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", "/usr/share/glvnd/egl_vendor.d/50_mesa.json")

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "server"))
from mcrobot import g1  # noqa: E402

PALETTE_RGB = {  # mirrors scripts/build_pack.py PALETTE order (cell index = label - 1)
    0: (92, 97, 104), 1: (192, 197, 203), 2: (112, 172, 222), 3: (28, 28, 32), 4: (240, 240, 240), 5: (24, 34, 56)}


def qrot_xyzw(q, v):
    x, y, z, w = q
    u = np.array([x, y, z])
    return v + 2 * np.cross(u, np.cross(u, v) + w * v)


def world_boxes(rig: "g1.Rig", d: mujoco.MjData, pack_zip: Path, origin) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, tuple]]:
    """Every model element as (MuJoCo-frame centre, half size in link axes, link wxyz quat, rgb), computed from the payload only."""
    pl = rig.payload(d)
    anchor = np.array(pl["anchor"])
    origin = np.asarray(origin, float)
    out = []
    with zipfile.ZipFile(pack_zip) as z:
        for k, part in enumerate(pl["parts"]):
            model = json.loads(z.read(f"assets/mcrobot/models/item/{part['m'].split(':')[1]}.json"))
            for e in model["elements"]:
                lo, hi = np.array(e["from"], float), np.array(e["to"], float)
                ctr_blocks = (lo + hi) / 32.0 - 0.5  # element centre relative to the model centre, in model blocks (16 units)
                size = np.array(part["s"]) * (hi - lo) / 16.0  # element size in Minecraft blocks, Minecraft-local axes
                mc = anchor + np.array(part["t"]) + qrot_xyzw(part["r"], np.array(part["s"]) * ctr_blocks)
                mj = np.array([mc[0] - origin[0], -(mc[2] - origin[2]), mc[1] - origin[1]]) * rig.mpb
                half = np.array([size[0], size[2], size[1]]) * rig.mpb / 2  # Minecraft-local (x, y, z) -> link (x, z, y)
                uv = next(iter(e["faces"].values()))["uv"]
                label = int(uv[1] // 4) * 4 + int(uv[0] // 4)
                out.append((mj, half, d.xquat[rig.bid[k]].copy(), PALETTE_RGB[label]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pack", default=str(ROOT / "results/m1/pack"))
    ap.add_argument("--mpb", type=float, default=0.5)
    a = ap.parse_args()
    pack = Path(a.pack)
    os.environ["MCROBOT_RIG_MODELS"] = str(pack / "rig_models.json")
    m = g1.build_model(None, extra_boxes=[([0, 0, -0.5], [5, 5, 0.5])])
    d = mujoco.MjData(m)
    w = g1.G1Walker.__new__(g1.G1Walker)
    w.m, w.d = m, d
    jid = [m.joint(n).id for n in g1.JOINTS]
    d.qpos[m.jnt_qposadr[m.joint("floating_base_joint").id] + 2] = 0.78
    d.qpos[m.jnt_qposadr[m.joint("floating_base_joint").id] + 3] = 1.0
    for i, j in enumerate(jid):
        d.qpos[m.jnt_qposadr[j]] = g1.DEFAULT[i]
    mujoco.mj_kinematics(m, d)
    rig = g1.Rig(m, [0, 64, 0], [0.0, 0.0, 0.0], a.mpb, style="models")
    boxes = world_boxes(rig, d, pack / "mcrobot-g1-pack.zip", [0, 64, 0])

    L = ['<mujoco><visual><global offwidth="900" offheight="900"/><headlight ambient="0.55 0.55 0.55"/></visual><worldbody>',
         '<geom type="plane" size="2 2 0.1" rgba="0.8 0.85 0.8 1"/><light pos="1 -2 3" dir="-0.3 0.6 -1"/>']
    for i, (c, h, q, rgb) in enumerate(boxes):
        L.append(f'<geom type="box" pos="{" ".join(f"{v:.5f}" for v in c)}" size="{" ".join(f"{max(v, 1e-4):.5f}" for v in h)}" '
                 f'quat="{" ".join(f"{v:.6f}" for v in q)}" rgba="{rgb[0] / 255:.3f} {rgb[1] / 255:.3f} {rgb[2] / 255:.3f} 1"/>')
    L.append("</worldbody></mujoco>")
    pm = mujoco.MjModel.from_xml_string("".join(L))
    pd = mujoco.MjData(pm)
    mujoco.mj_forward(pm, pd)
    r = mujoco.Renderer(pm, 900, 900)
    from PIL import Image

    for name, az in (("front", 180), ("side", 90)):
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [0.0, 0.0, 0.65]
        cam.distance = 2.6
        cam.azimuth = az
        cam.elevation = -8
        r.update_scene(pd, camera=cam)
        out = pack / f"preview_{name}.png"
        Image.fromarray(r.render()).save(out)
        print("wrote", out, f"({len(boxes)} boxes)")


if __name__ == "__main__":
    main()
