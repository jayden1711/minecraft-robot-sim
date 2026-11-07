"""Fit check of the Unitree G1 (frozen unitree_rl_mjlab walking policy) against a maze: body dimensions from the MJCF at the policy's
default pose, then simulated trials in MuJoCo: straight walks through corridors of varying width, over steps of varying height,
and across floor gaps. Writes JSON to --out. Simulation only.

usage: PYTHONPATH=server python scripts/fit_g1.py --out results/m1/fit.json
"""

from __future__ import annotations

import argparse
import json
import math

import mujoco
import numpy as np

from mcrobot import g1


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def body_dims() -> dict:
    m = g1.build_model(None, extra_boxes=[([0, 0, -0.5], [5, 5, 0.5])])
    w = g1.G1Walker(m)
    w.reset([0, 0], 0.0, z=0.80)
    for _ in range(100):  # let it settle standing still
        w.step([0, 0, 0])
    d = w.d
    pts = []
    for gi in range(m.ngeom):
        if m.geom_type[gi] != mujoco.mjtGeom.mjGEOM_MESH or m.geom_contype[gi]:
            continue
        mid = m.geom_dataid[gi]
        v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        pts.append((m.body(m.geom_bodyid[gi]).name, v @ d.geom_xmat[gi].reshape(3, 3).T + d.geom_xpos[gi]))
    allp = np.vstack([p for _, p in pts])
    yaw = w.yaw()
    c, s = math.cos(-yaw), math.sin(-yaw)
    R = np.array([[c, -s], [s, c]])
    loc = (allp[:, :2] - w.pelvis_pos()[:2]) @ R.T  # body frame: x forward, y left

    def extent(sel):
        q = np.vstack([(p[:, :2] - w.pelvis_pos()[:2]) @ R.T for n, p in pts if sel(n)])
        return [round(float(q[:, 0].max() - q[:, 0].min()), 3), round(float(q[:, 1].max() - q[:, 1].min()), 3)]

    return {
        "whole_body_xy_m (forward, lateral), arms at sides": [round(float(np.ptp(loc[:, 0])), 3), round(float(np.ptp(loc[:, 1])), 3)],
        "torso_and_head_xy_m": extent(lambda n: n in ("torso_link", "pelvis")),
        "arms_xy_m (shoulder to wrist, both sides)": extent(lambda n: "shoulder" in n or "elbow" in n or "wrist" in n),
        "left_foot_xy_m": extent(lambda n: n == "left_ankle_roll_link"),
        "height_m": round(float(allp[:, 2].max() - allp[:, 2].min()), 3),
        "pelvis_height_m": round(float(w.pelvis_pos()[2]), 3),
    }


class Course:
    """Flat floor along +x. Optional walls at y = +-width/2 (height 2), steps (box across the path) and gaps (floor removed)."""

    def __init__(self, width=None, step=None, gap=None, length=8.0):
        x0, x1 = -1.0, length + 1.0  # floor spans x0..x1, top at z = 0
        if gap:
            g0, g1_ = 2.0, 2.0 + gap
            self.boxes = [([(x0 + g0) / 2, 0, -0.25], [(g0 - x0) / 2, 3, 0.25]), ([(g1_ + x1) / 2, 0, -0.25], [(x1 - g1_) / 2, 3, 0.25])]
        else:
            self.boxes = [([(x0 + x1) / 2, 0, -0.25], [(x1 - x0) / 2, 3, 0.25])]
        self.walls = []
        if width:
            for sgn in (-1, 1):
                self.walls.append(([(x0 + x1) / 2, sgn * (width / 2 + 0.1), 1.0], [(x1 - x0) / 2, 0.1, 1.0]))
        if step:  # a platform of this height from x = 3 to the end
            self.boxes.append(([(3.0 + x1) / 2, 0, step / 2], [(x1 - 3.0) / 2, 3, step / 2]))
        self.length = length


def run(course: Course, seconds=14.0, speed=0.45) -> dict:
    extra = [(p, h) for p, h in course.boxes] + [(p, h) for p, h in course.walls]
    m = g1.build_model(None, extra_boxes=extra)
    wall_geoms = {m.geom(f"extra_{len(course.boxes) + i}").id for i in range(len(course.walls))}
    w = g1.G1Walker(m)
    w.reset([0.0, 0.0], 0.0, z=0.80)
    d = w.d
    touches = 0
    steps = int(seconds / g1.CTRL_DT)
    ok_end = False
    for i in range(steps):
        p = w.pelvis_pos()
        cmd = np.array([speed, float(np.clip(-1.5 * p[1], -0.3, 0.3)), float(np.clip(2.0 * wrap(0.0 - w.yaw()), -1, 1))])
        w.step(cmd)
        if wall_geoms:
            for c in d.contact[: d.ncon]:
                if (c.geom1 in wall_geoms) != (c.geom2 in wall_geoms):
                    touches += 1
                    break
        if w.fallen():
            return {"ok": False, "why": "fell", "t": round(i * g1.CTRL_DT, 1), "wall_contact_steps": touches, "x": round(float(p[0]), 2)}
        if w.pelvis_pos()[0] > course.length - 1.5:
            ok_end = True
            break
    return {"ok": ok_end, "why": "reached" if ok_end else "stuck", "x": round(float(w.pelvis_pos()[0]), 2), "wall_contact_steps": touches}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/m1/fit.json")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    out = {"body": body_dims(), "corridor": {}, "step": {}, "gap": {}}
    print(json.dumps(out["body"], indent=1), flush=True)
    widths = [0.5, 0.6, 0.7, 0.8, 1.0] if not a.quick else [0.7, 1.0]
    for wd in widths:
        out["corridor"][str(wd)] = run(Course(width=wd))
        print("corridor", wd, out["corridor"][str(wd)], flush=True)
    steps = [0.03, 0.05, 0.08, 0.10, 0.15, 0.25, 0.5] if not a.quick else [0.05, 0.5]
    for h in steps:
        out["step"][str(h)] = run(Course(step=h))
        print("step", h, out["step"][str(h)], flush=True)
    gaps = [0.1, 0.2, 0.3, 0.5, 1.0] if not a.quick else [0.3, 1.0]
    for gp in gaps:
        out["gap"][str(gp)] = run(Course(gap=gp))
        print("gap", gp, out["gap"][str(gp)], flush=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
