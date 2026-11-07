"""M0 check: does an exported region (from the real game) match the data-pack maze (mcrobot_demo:maze)?

Reads the exported MJCF (and its JSON header), voxelizes its collider and visual boxes back into cells, and compares them cell by cell
with the expected layout (`mcrobot.synthetic.demo_maze`, the source the data pack is generated from). Also compares markers and the box merge.

usage: compare_m0.py region.xml [--shot screenshot.png --out-dir DIR]
Writes (in DIR): checks.json, mujoco_view.png, and side_by_side.png when --shot exists. Prints a text summary.
Run with PYTHONPATH=server.
"""

import argparse
import hashlib
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", "/usr/share/glvnd/egl_vendor.d/50_mesa.json")

from mcrobot.mjcf import block_rgba, build_scene, to_mjcf
from mcrobot.region import Region
from mcrobot.synthetic import demo_maze


def floats(s):
    return np.array([float(v) for v in s.split()])


def mj_box_to_cells(pos, half):
    """MuJoCo box (center, half) -> list of region cells (x, y, z) fully covered, in Minecraft cell indices (mj = (x, -z, y))."""
    lo, hi = pos - half, pos + half
    cells = []
    for x in range(int(round(lo[0])), int(round(hi[0]))):
        for z in range(int(round(-hi[1])), int(round(-lo[1]))):
            for y in range(int(round(lo[2])), int(round(hi[2]))):
                cells.append((x, y, z))
    return cells


def parse(xml_path):
    root = ET.parse(xml_path).getroot()
    geoms = {"col": [], "colp": [], "vis": [], "site": []}
    for g in root.iter("geom"):
        n = g.get("name")
        kind = "colp" if n.startswith("colp_") else "col" if n.startswith("col_") else "vis"
        geoms[kind].append((n, floats(g.get("pos")), floats(g.get("size")), floats(g.get("rgba"))))
    for s in root.iter("site"):
        geoms["site"].append((s.get("name"), floats(s.get("pos")), floats(s.get("size")), floats(s.get("rgba"))))
    return geoms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xml")
    ap.add_argument("--shot", default=None)
    ap.add_argument("--out-dir", default="results/m0/compare")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    xml_text = Path(a.xml).read_text()
    header = json.loads(Path(a.xml).with_suffix(".json").read_text())
    res = {"xml": a.xml, "sha256": hashlib.sha256(xml_text.encode()).hexdigest(), "origin": header["origin"], "size": header["size"]}

    exp_region = demo_maze(float(header.get("meters_per_block", 1.0)))
    exp_scene = build_scene(exp_region)
    exp_full = exp_region.collider_labels().astype(bool)
    sx, sy, sz = exp_region.size

    g = parse(a.xml)

    # ---- collider cells ----
    got_cells = {}
    for n, p, h, _ in g["col"]:
        for c in mj_box_to_cells(p, h):
            got_cells[c] = got_cells.get(c, 0) + 1
    overlap = sum(1 for v in got_cells.values() if v > 1)
    exp_cells = {tuple(int(v) for v in c) for c in np.argwhere(exp_full)}
    got_set = set(got_cells)
    res["solid_cells_expected"] = len(exp_cells)
    res["solid_cells_exported"] = len(got_set)
    res["collider_box_volume_cells"] = float(sum(np.prod(2 * h) for _, _, h, _ in g["col"]))
    res["missing_cells"] = sorted(exp_cells - got_set)
    res["extra_cells"] = sorted(got_set - exp_cells)
    res["overlapping_cells"] = overlap
    res["cells_outside_expected_footprint"] = sorted(c for c in got_set if not (0 <= c[0] < sx and 0 <= c[1] < sy and 0 <= c[2] < sz))

    # ---- partial blocks ----
    exp_partial = []
    for pi, p in enumerate(exp_region.palette):
        for c in np.argwhere(exp_region.grid == pi):
            for lb in p.partial_boxes:
                exp_partial.append((tuple(int(v) for v in c), tuple(lb)))
    got_partial = []
    for n, p, h, _ in g["colp"]:
        lo, hi = p - h, p + h
        cell = (int(np.floor(lo[0] + 1e-9)), int(np.floor(lo[2] + 1e-9)), int(np.floor(-hi[1] + 1e-9)))
        local = (lo[0] - cell[0], lo[2] - cell[1], -hi[1] - cell[2], hi[0] - cell[0], hi[2] - cell[1], -lo[1] - cell[2])
        got_partial.append((cell, tuple(round(float(v), 6) for v in local)))
    res["partial_expected"] = [[list(c), list(b)] for c, b in exp_partial]
    res["partial_exported"] = [[list(c), list(b)] for c, b in got_partial]
    res["partial_match"] = sorted(exp_partial) == sorted(got_partial)

    # ---- merge ----
    res["collider_boxes_full_exported"] = len(g["col"])
    res["collider_boxes_partial_exported"] = len(g["colp"])
    res["collider_boxes_total_exported"] = len(g["col"]) + len(g["colp"])
    res["collider_boxes_total_expected"] = len(exp_scene.colliders)
    res["unmerged_would_be"] = len(exp_cells) + len(exp_partial)

    # ---- block type per cell, from visual geom colors ----
    name_by_rgba = {tuple(round(float(v), 4) for v in block_rgba(p.name)): p.name for p in exp_region.palette}
    got_name = {}
    for n, p, h, rgba in g["vis"]:
        if n.startswith("vis_m"):
            continue
        nm = name_by_rgba.get(tuple(round(float(v), 4) for v in rgba), f"unknown rgba {rgba.tolist()}")
        for c in mj_box_to_cells(p, h):
            got_name[c] = nm
    wrong = []
    for c in exp_cells:
        e = exp_region.palette[int(exp_region.grid[c])].name
        if got_name.get(c) != e:
            wrong.append((c, e, got_name.get(c)))
    res["block_type_mismatches"] = [[list(c), e, o] for c, e, o in sorted(wrong)]

    # ---- markers ----
    def mk(scene_markers):
        return sorted((m.kind.name.lower(), tuple(round(float(v), 4) for v in m.center), tuple(round(float(v), 4) for v in m.half)) for m in scene_markers)

    got_markers = sorted(
        (re.sub(r"_\d+$", "", n), tuple(round(float(v), 4) for v in p), tuple(round(float(v), 4) for v in h)) for n, p, h, _ in g["site"]
    )
    exp_markers = mk(exp_scene.markers)
    res["markers_expected"] = [[k, list(c)] for k, c, _ in exp_markers]
    res["markers_exported"] = [[k, list(c)] for k, c, _ in got_markers]
    res["markers_match"] = got_markers == exp_markers
    res["marker_cells_exported"] = [[k, [int(np.floor(c[0])), int(np.floor(c[2])), int(np.floor(-c[1]))]] for k, c, _ in got_markers]

    # ---- byte-level: same geoms as the expected scene? ----
    exp_xml = to_mjcf(exp_scene)
    strip = lambda t: [l for l in t.splitlines() if "<camera" not in l]
    res["xml_identical_except_camera"] = strip(exp_xml) == strip(xml_text)
    res["xml_identical_fully"] = exp_xml == xml_text

    res["ok"] = (
        not res["missing_cells"]
        and not res["extra_cells"]
        and overlap == 0
        and res["partial_match"]
        and not res["block_type_mismatches"]
        and res["markers_match"]
        and res["collider_boxes_total_exported"] == res["collider_boxes_total_expected"]
    )
    (out / "checks.json").write_text(json.dumps(res, indent=1, default=str))

    # ---- render ----
    import mujoco
    from PIL import Image

    m = mujoco.MjModel.from_xml_path(a.xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, 720, 1280)
    opt = mujoco.MjvOption()
    opt.sitegroup[4] = 1
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [4.5, -4.5, 0.8]
    cam.distance = 17.0
    cam.azimuth = 45.0  # looking toward the north-east from the south-west
    cam.elevation = -48.0
    r.update_scene(d, camera=cam, scene_option=opt)
    img = Image.fromarray(r.render())
    img.save(out / "mujoco_view.png")
    cam.azimuth, cam.elevation, cam.distance = 90.0, -88.0, 16.0  # top-down, markers visible (sites are drawn)
    r.update_scene(d, camera=cam, scene_option=opt)
    Image.fromarray(r.render()).save(out / "mujoco_top.png")
    if a.shot and Path(a.shot).exists():
        s = Image.open(a.shot).convert("RGB")
        h = 720
        s = s.resize((int(s.width * h / s.height), h))
        canvas = Image.new("RGB", (s.width + img.width, h), (255, 255, 255))
        canvas.paste(s, (0, 0))
        canvas.paste(img, (s.width, 0))
        canvas.save(out / "side_by_side.png")
    print(json.dumps({k: v for k, v in res.items() if not isinstance(v, list) or len(v) < 8}, indent=1, default=str))
    sys.exit(0 if res["ok"] else 1)


if __name__ == "__main__":
    main()
