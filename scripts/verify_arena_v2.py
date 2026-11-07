"""Simulate the frozen G1 walking policy along sky arena v2 in MuJoCo, with the push schedule and the ice friction, and report per section
whether it gets through. Everything is simulated (MuJoCo physics, scripted velocity commands, no learning). Writes JSON to --out.

usage: PYTHONPATH=server python scripts/verify_arena_v2.py [--speed 0.45] [--trials N] [--out results/m1/arena_v2/verify.json] [--no-push] [--ice-mu 0.3]
--trials N: perturbed starts (lateral offset +-0.2 m, heading +-0.3 rad, speeds 0.35/0.45/0.55); the push schedule stays on.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from mcrobot import arena_v2 as A, g1
from mcrobot.mjcf import build_scene, to_mjcf
from mcrobot.pursuit import Pursuit, drop_collinear
from mcrobot.pushes import PushSchedule


def dist_to_polyline(p, pts) -> float:
    best = 1e9
    for a, b in zip(pts, pts[1:]):
        ab = b - a
        t = float(np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0, 1))
        best = min(best, float(np.linalg.norm(p - (a + t * ab))))
    return best


def make_model(plan, mpb: float, scratch: str = "results/_scratch/arena_v2_check.xml"):
    region = A.to_region(plan, meters_per_block=mpb)
    Path(scratch).parent.mkdir(parents=True, exist_ok=True)
    Path(scratch).write_text(to_mjcf(build_scene(region), visuals=False))
    return g1.build_model(scratch)


def run_route(plan, model, mpb: float, speed: float, dy: float = 0.0, dyaw: float = 0.0, push: bool = True, max_seconds: float = 400.0):
    w = g1.G1Walker(model)
    pts = A.route_mj(plan, mpb)
    zf = A.floor_top_m(mpb)
    ctl_pts = drop_collinear([np.array([x, y]) for x, y, _ in pts])
    d0 = ctl_pts[1] - ctl_pts[0]
    w.reset(ctl_pts[0] + np.array([0.0, dy]), math.atan2(d0[1], d0[0]) + dyaw, z=zf + 0.80)
    ctl = Pursuit(ctl_pts, v=speed, blend=True)
    sched = PushSchedule(A.pushes_m(plan, mpb) if push else [])
    secs = [{"key": s.key, "label": s.label, "x0": s.x0 * mpb, "x1": s.x1 * mpb, "t_in": None, "t_out": None, "max_dev_m": 0.0, "min_pelvis_above_floor_m": 9.0,
             "max_tilt_deg": 0.0} for s in plan.sections]
    pushes_log = []
    t = 0.0
    for i in range(int(max_seconds / g1.CTRL_DT)):
        t = i * g1.CTRL_DT
        cmd, _ = ctl.command(w.pelvis_pos(), w.yaw())
        if ctl.done:
            break
        force, fired = sched.update(t, w.pelvis_pos()[:2])
        for cue in fired:
            pushes_log.append({"cue": cue, "t": round(t, 2), "x_m": round(float(w.pelvis_pos()[0]), 2), "y_m": round(float(w.pelvis_pos()[1]), 2)})
        w.set_push(force)
        w.step(cmd)
        p = w.pelvis_pos()
        dev = dist_to_polyline(p[:2], ctl_pts)
        for s in secs:
            if s["x0"] <= p[0] < s["x1"]:
                if s["t_in"] is None:
                    s["t_in"] = round(t, 1)
                s["t_out"] = round(t, 1)
                s["max_dev_m"] = max(s["max_dev_m"], round(dev, 3))
                s["min_pelvis_above_floor_m"] = min(s["min_pelvis_above_floor_m"], round(float(p[2] - zf), 3))
                s["max_tilt_deg"] = max(s["max_tilt_deg"], round(math.degrees(math.acos(max(-1.0, min(1.0, w.upright())))), 1))
        if w.fallen():
            where = next((s["label"] for s in secs if s["x0"] <= p[0] < s["x1"]), "?")
            return {"ok": False, "why": f"fell in section '{where}'", "t": round(t, 1), "x_m": round(float(p[0]), 2), "y_m": round(float(p[1]), 2), "sections": secs, "pushes": pushes_log}
    final = w.pelvis_pos()
    end = pts[-1]
    ok = ctl.done and math.hypot(final[0] - end[0], final[1] - end[1]) < 0.6
    return {"ok": bool(ok), "why": "reached the finish" if ok else "did not finish", "t": round(t, 1), "sections": secs, "pushes": pushes_log,
            "route_len_m": round(sum(float(np.linalg.norm(b - a)) for a, b in zip(ctl_pts, ctl_pts[1:])), 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--speed", type=float, default=0.45)
    ap.add_argument("--mpb", type=float, default=0.5)
    ap.add_argument("--trials", type=int, default=0)
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--out", default="results/m1/arena_v2/verify.json")
    a = ap.parse_args()
    plan = A.build()
    model = make_model(plan, a.mpb)
    res = run_route(plan, model, a.mpb, a.speed, push=not a.no_push)
    res.update({"speed_m_s": a.speed, "meters_per_block": a.mpb, "push": not a.no_push})
    print(json.dumps(res, indent=1), flush=True)
    if a.trials:
        rng = np.random.default_rng(0)
        trials = []
        for k in range(a.trials):
            sp = float(rng.choice([0.35, 0.45, 0.55]))
            dy, dyaw = float(rng.uniform(-0.2, 0.2)), float(rng.uniform(-0.3, 0.3))
            r = run_route(plan, model, a.mpb, sp, dy, dyaw, push=not a.no_push)
            trials.append({"speed": sp, "dy_m": round(dy, 3), "dyaw_rad": round(dyaw, 3), "ok": r["ok"], "why": r["why"], "t": r["t"]})
            print(trials[-1], flush=True)
        res["trials"] = trials
        res["trials_ok"] = sum(t["ok"] for t in trials)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
