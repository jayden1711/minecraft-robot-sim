"""Build the server resource pack that shows the Unitree G1 in Minecraft (one item_display per link, no client mod).

Source: the visual meshes of the MuJoCo Menagerie G1 (BSD-3-Clause, pinned rev below). Vanilla item models are cuboid-only, so every
link is voxelised (surface rasterisation + exterior flood fill, per mesh) and greedy-merged into a few axis-aligned cuboid elements.
Colours are plain (a generated 4x4-cell palette PNG); the Unitree logo mesh and every original texture are left out.

Outputs (git-ignored, results/m1/pack/): mcrobot-g1-pack.zip (byte-reproducible), mcrobot-g1-pack.sha1, rig_models.json (read by
mcrobot.g1.Rig(style="models")), build_report.json. This script is the only producer of those files.

usage: PYTHONPATH=server python scripts/build_pack.py [--out results/m1/pack] [--xml path/to/g1.xml]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import zipfile
import zlib
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "server"))
from mcrobot.boxes import merge_boxes  # noqa: E402

MENAGERIE = Path.home() / "projects/robot-arena-sim/third_party/mujoco_menagerie/unitree_g1"
MENAGERIE_REV = "98a70c569c4c1b6da467a20012f766ee1f8dc521"
PACK_MIN_FORMAT = 75  # 1.21.11 resource pack format 75.0
PACK_MAX_FORMAT = 99  # 26.3 is 97.1; a wide upper bound is harmless
PACK_ID = "mcrobot-g1-pack.zip"
SCALES_M = (0.256, 0.32, 0.384, 0.48)  # metres per 16 model units; first one that fits every link
BOX_BUDGET = 150
EXCLUDE_MESHES = {"logo_link"}

# palette: name -> RGB (cell index = list position, 4x4 px cells in a 16x16 texture)
PALETTE = [
    ("grey", (92, 97, 104)),
    ("light_grey", (192, 197, 203)),
    ("blue", (112, 172, 222)),
    ("black", (28, 28, 32)),
    ("white", (240, 240, 240)),
    ("visor", (24, 34, 56)),
]
COL = {n: i + 1 for i, (n, _) in enumerate(PALETTE)}  # label values (0 = empty)


def mesh_colour(mesh: str) -> str:
    if mesh == "head_link":
        return "white"
    if any(k in mesh for k in ("hip_yaw", "shoulder")):
        return "blue"
    if any(k in mesh for k in ("knee", "elbow", "torso")):
        return "light_grey"
    if "ankle_roll" in mesh or "rubber_hand" in mesh:
        return "black"
    return "grey"


def body_meshes(model: mujoco.MjModel) -> dict[str, list[tuple[str, np.ndarray, np.ndarray]]]:
    """body -> [(mesh name, vertices in the link frame (n,3), faces (m,3))] for the visual (contype 0) mesh geoms, logo excluded."""
    out: dict[str, list] = {}
    for g in range(model.ngeom):
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or model.geom_contype[g] or model.geom_conaffinity[g]:
            continue
        mid = model.geom_dataid[g]
        name = model.mesh(mid).name
        if name in EXCLUDE_MESHES or "logo" in name.lower():
            continue
        v = model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid] + model.mesh_vertnum[mid]].astype(float)
        f = model.mesh_face[model.mesh_faceadr[mid]:model.mesh_faceadr[mid] + model.mesh_facenum[mid]]
        Rg = np.zeros(9)
        mujoco.mju_quat2Mat(Rg, model.geom_quat[g])
        v = v @ Rg.reshape(3, 3).T + model.geom_pos[g]
        out.setdefault(model.body(model.geom_bodyid[g]).name, []).append((name, v, f.astype(np.int64)))
    return out


def to_model_units(p: np.ndarray, centre: np.ndarray, u: float) -> np.ndarray:
    """Link-frame metres -> model units (x = link x, y = link z, z = -link y), bounding-box centre at (8, 8, 8)."""
    d = (p - centre) / u
    return np.stack([8 + d[..., 0], 8 + d[..., 2], 8 - d[..., 1]], axis=-1)


def sample_surface(tri: np.ndarray, step: float = 0.5) -> np.ndarray:
    """Points on triangles (t,3,3) so that no point of the surface is further than ~step from a sample."""
    e = np.maximum.reduce([np.linalg.norm(tri[:, 1] - tri[:, 0], axis=1), np.linalg.norm(tri[:, 2] - tri[:, 1], axis=1),
                           np.linalg.norm(tri[:, 0] - tri[:, 2], axis=1)])
    n = np.clip(np.ceil(e / step).astype(int), 1, 64)
    pts = []
    for k in np.unique(n):
        t = tri[n == k]
        for a in range(k + 1):
            for b in range(k + 1 - a):
                w = np.array([a, b, k - a - b], float) / k
                pts.append(np.tensordot(t, w, axes=([1], [0])))
    return np.vstack(pts)


def fill_interior(surface: np.ndarray) -> np.ndarray:
    """Voxels not reachable from outside through empty voxels (6-connected); the grid must have an empty border."""
    ext = np.zeros(surface.shape, bool)
    ext[0, :, :] = ext[-1, :, :] = ext[:, 0, :] = ext[:, -1, :] = ext[:, :, 0] = ext[:, :, -1] = True
    ext &= ~surface
    while True:
        n = ext.copy()
        n[1:] |= ext[:-1]
        n[:-1] |= ext[1:]
        n[:, 1:] |= ext[:, :-1]
        n[:, :-1] |= ext[:, 1:]
        n[:, :, 1:] |= ext[:, :, :-1]
        n[:, :, :-1] |= ext[:, :, 1:]
        n &= ~surface
        if (n == ext).all():
            break
        ext = n
    return ~ext


def mesh_volume(v: np.ndarray, f: np.ndarray) -> float:
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    return abs(float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum()) / 6)


def voxelise_body(meshes, centre: np.ndarray, u: float):
    """Returns (labels grid with 1 voxel padding, origin index i0 in model units, per-mesh stats)."""
    pts_by_mesh = []
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for name, v, f in meshes:
        m = to_model_units(v, centre, u)
        pts_by_mesh.append((name, m, f))
        lo = np.minimum(lo, m.min(0))
        hi = np.maximum(hi, m.max(0))
    i0 = np.floor(lo).astype(int) - 1
    shape = tuple((np.ceil(hi).astype(int) + 2 - i0).tolist())
    labels = np.zeros(shape, np.uint8)
    stats = []
    for name, m, f in pts_by_mesh:
        p = sample_surface(m[f])
        idx = np.floor(p).astype(int) - i0
        surf = np.zeros(shape, bool)
        surf[idx[:, 0], idx[:, 1], idx[:, 2]] = True
        solid = fill_interior(surf)
        col = COL[mesh_colour(name)]
        labels[solid] = col
        if name == "head_link":  # visor: dark band on the front (+x) face of the head, upper-middle height
            xs, ys, zs = np.nonzero(solid)
            xmax = xs.max()
            ymin, ymax = ys.min(), ys.max()
            band = (xs >= xmax - 1) & (ys >= ymin + 0.35 * (ymax - ymin)) & (ys <= ymin + 0.7 * (ymax - ymin))
            labels[xs[band], ys[band], zs[band]] = COL["visor"]
        vol_units = float(solid.sum())
        stats.append({"mesh": name, "voxel_volume": vol_units, "mesh_volume": mesh_volume(m, f), "voxels": int(solid.sum())})
    return labels, i0, stats


def cell_uv(label: int) -> list[float]:
    i = label - 1
    cx, cy = i % 4, i // 4
    return [cx * 4 + 1, cy * 4 + 1, cx * 4 + 3, cy * 4 + 3]


FACES = {  # name: (axis, side)
    "east": (0, 1), "west": (0, 0), "up": (1, 1), "down": (1, 0), "south": (2, 1), "north": (2, 0),
}


def model_json(labels: np.ndarray, i0: np.ndarray) -> tuple[dict, int]:
    boxes = merge_boxes(labels, try_orders=True)
    elements = []
    for x0, y0, z0, x1, y1, z1, lab in boxes:
        lo, hi = (x0, y0, z0), (x1, y1, z1)
        faces = {}
        for fname, (ax, side) in FACES.items():
            sl = [slice(lo[k], hi[k]) for k in range(3)]
            sl[ax] = slice(hi[ax], hi[ax] + 1) if side else slice(lo[ax] - 1, lo[ax])
            if (labels[tuple(sl)] == 0).any():
                faces[fname] = {"uv": cell_uv(lab), "texture": "#0"}
        if not faces:  # fully enclosed by other voxels: never visible
            continue
        elements.append({"from": [int(lo[k] + i0[k]) for k in range(3)], "to": [int(hi[k] + i0[k]) for k in range(3)], "faces": faces})
    return {
        "textures": {"0": "mcrobot:item/g1_palette", "particle": "mcrobot:item/g1_palette"},
        "gui_light": "side",
        "elements": elements,
    }, len(elements)


def palette_png() -> bytes:
    rows = []
    for py in range(16):
        row = bytearray([0])
        for px in range(16):
            i = (py // 4) * 4 + px // 4
            r, g, b = PALETTE[i][1] if i < len(PALETTE) else (255, 0, 255)
            row += bytes([r, g, b, 255])
        rows.append(bytes(row))
    raw = b"".join(rows)

    def chunk(t: bytes, d: bytes) -> bytes:
        c = struct.pack(">I", len(d)) + t + d
        return c + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 16, 16, 8, 6, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def dumps(o) -> bytes:
    return (json.dumps(o, sort_keys=True, separators=(",", ":")) + "\n").encode()


NOTICE = """mcrobot G1 resource pack (generated by scripts/build_pack.py of minecraft-robot-sim, Apache-2.0).

The item models are simplified voxel (cuboid) approximations of the Unitree G1 visual meshes from MuJoCo Menagerie
(unitree_g1, revision {rev}), which are BSD-3-Clause licensed; the licence text is in LICENSE-menagerie-g1.txt and must stay
with every copy of this pack. No Unitree logo, branding or original texture is included; the only texture is a generated
palette of plain colours. This pack is a simulation display aid, not an official model.
"""


def build(out: Path, xml: Path = MENAGERIE / "g1.xml") -> dict:
    out.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(xml))
    bodies = body_meshes(model)
    chosen = None
    for scale in SCALES_M:
        u = scale / 16
        ok = True
        for meshes in bodies.values():
            allv = np.vstack([v for _, v, _ in meshes])
            ext = (allv.max(0) - allv.min(0)) / u
            if ext.max() > 46:
                ok = False
        if ok:
            chosen = scale
            break
    if chosen is None:
        raise RuntimeError("a link does not fit the model range at any scale")
    u = chosen / 16

    files: dict[str, bytes] = {}
    parts, report = [], {"model_scale_m": chosen, "menagerie_rev": MENAGERIE_REV, "bodies": {}}
    for body in sorted(bodies):
        meshes = sorted(bodies[body], key=lambda t: t[0] == "head_link")  # head last so its visor wins
        allv = np.vstack([v for _, v, _ in meshes])
        centre = (allv.min(0) + allv.max(0)) / 2
        labels, i0, stats = voxelise_body(meshes, centre, u)
        mj, n = model_json(labels, i0)
        if n > BOX_BUDGET:
            raise RuntimeError(f"{body}: {n} elements exceeds the budget {BOX_BUDGET}")
        for e in mj["elements"]:
            if min(e["from"]) < -16 or max(e["to"]) > 32:
                raise RuntimeError(f"{body}: element outside -16..32")
        files[f"assets/mcrobot/models/item/g1/{body}.json"] = dumps(mj)
        files[f"assets/mcrobot/items/g1/{body}.json"] = dumps({"model": {"type": "minecraft:model", "model": f"mcrobot:item/g1/{body}"}})
        parts.append({"body": body, "model": f"mcrobot:g1/{body}", "centre_link": [round(float(c), 6) for c in centre]})
        report["bodies"][body] = {"elements": n, "meshes": stats}
    files["assets/mcrobot/textures/item/g1_palette.png"] = palette_png()
    files["pack.mcmeta"] = dumps({"pack": {"description": "minecraft-robot-sim: simulated Unitree G1 (voxel models, BSD-3 meshes)",
                                           "min_format": PACK_MIN_FORMAT, "max_format": PACK_MAX_FORMAT}})
    files["LICENSE-menagerie-g1.txt"] = (MENAGERIE / "LICENSE").read_bytes()
    files["NOTICE.txt"] = NOTICE.format(rev=MENAGERIE_REV).encode()

    zpath = out / PACK_ID
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name in sorted(files):
            zi = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = 0o644 << 16
            zi.create_system = 3
            z.writestr(zi, files[name], compresslevel=9)
    sha1 = hashlib.sha1(zpath.read_bytes()).hexdigest()
    (out / "mcrobot-g1-pack.sha1").write_text(sha1 + "\n")
    (out / "rig_models.json").write_text(json.dumps({"model_scale_m": chosen, "pack_sha1": sha1, "parts": parts}, indent=1) + "\n")
    report.update({"sha1": sha1, "zip_bytes": zpath.stat().st_size, "parts": len(parts),
                   "elements_total": sum(b["elements"] for b in report["bodies"].values())})
    (out / "build_report.json").write_text(json.dumps(report, indent=1) + "\n")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "results/m1/pack"))
    ap.add_argument("--xml", default=str(MENAGERIE / "g1.xml"))
    a = ap.parse_args()
    r = build(Path(a.out), Path(a.xml))
    print(f"pack {r['zip_bytes']} bytes, {r['parts']} parts, {r['elements_total']} elements, scale {r['model_scale_m']} m/16 units, sha1 {r['sha1']}")


if __name__ == "__main__":
    main()
