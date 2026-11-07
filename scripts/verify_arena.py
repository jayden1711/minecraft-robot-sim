"""Simulate the frozen G1 walking policy along the sky arena route in MuJoCo and report, per section, whether it gets through.

Everything is simulated (MuJoCo physics, scripted velocity commands, no learning). Writes JSON to --out (default
results/m1/arena_verify.json). With --trials N it also runs perturbed starts (lateral offset and heading error) at several speeds and
reports how many complete the course: the margin behind the single nominal run.

usage: PYTHONPATH=server python scripts/verify_arena.py [--beam 1] [--speed 0.45] [--mpb 0.5] [--trials 0]
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np

from mcrobot import arena, g1
from mcrobot.mjcf import build_scene, to_mjcf
from mcrobot.pursuit import Pursuit, drop_collinear


def dist_to_polyline(p, pts) -> float:
    best = 1e9
    for a, b in zip(pts, pts[1:]):
        ab = b - a
        t = float(np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0, 1))
        best = min(best, float(np.linalg.norm(p - (a + t * ab))))
    return best


def make_model(plan, mpb: float):
    region = arena.to_region(plan, meters_per_block=mpb)
    path = "results/_scratch/arena_check.xml"
    with open(path, "w") as f:
        f.write(to_mjcf(build_scene(region), visuals=False))
    return g1.build_model(path)


def run_route(plan, model, mpb: float, speed: float, dy: float = 0.0, dyaw: float = 0.0, max_seconds: float = 400.0):
    w = g1.G1Walker(model)
    pts = arena.route_mj(plan, mpb)
    zf = arena.floor_top_m(mpb)
    wp = [np.array([x, y]) for x, y, _ in pts]
    ctl_pts = drop_collinear(wp)
    d0 = ctl_pts[1] - ctl_pts[0]
    start = ctl_pts[0] + np.array([0.0, dy])
    w.reset(start, math.atan2(d0[1], d0[0]) + dyaw, z=zf + 0.80)
    ctl = Pursuit(ctl_pts, v=speed)
    secs = [{"name": s.name, "x0": s.x0 * mpb, "x1": s.x1 * mpb, "t_in": None, "t_out": None, "max_dev_m": 0.0} for s in plan.sections]
    min_z = 9.0
    for i in range(int(max_seconds / g1.CTRL_DT)):
        cmd, _ = ctl.command(w.pelvis_pos(), w.yaw())
        t = i * g1.CTRL_DT
        if ctl.done:
            break
        w.step(cmd)
        p = w.pelvis_pos()
        min_z = min(min_z, p[2] - zf)
        dev = dist_to_polyline(p[:2], ctl_pts)
        for s in secs:
            if s["x0"] <= p[0] < s["x1"]:
                if s["t_in"] is None:
                    s["t_in"] = round(t, 1)
                s["t_out"] = round(t, 1)
                s["max_dev_m"] = max(s["max_dev_m"], round(dev, 3))
        if w.fallen():
            where = next((s["name"] for s in secs if s["x0"] <= p[0] < s["x1"]), "?")
            return {"ok": False, "why": f"fell in section '{where}'", "t": round(t, 1), "x_m": round(float(p[0]), 2), "y_m": round(float(p[1]), 2), "sections": secs}
    final = w.pelvis_pos()
    ok = ctl.done and math.hypot(final[0] - pts[-1][0], final[1] - pts[-1][1]) < 0.6
    return {"ok": bool(ok), "why": "reached the finish" if ok else "did not finish", "t": round(t, 1), "min_pelvis_above_floor_m": round(min_z, 3), "sections": secs,
            "route_len_m": round(sum(float(np.linalg.norm(b - a)) for a, b in zip(ctl_pts, ctl_pts[1:])), 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--beam", type=int, default=arena.BEAM_W)
    ap.add_argument("--speed", type=float, default=0.45)
    ap.add_argument("--mpb", type=float, default=0.5)
    ap.add_argument("--trials", type=int, default=0, help="perturbed runs: lateral start offset +-0.2 m, heading +-0.3 rad, speeds 0.35/0.45/0.55")
    ap.add_argument("--out", default="results/m1/arena_verify.json")
    a = ap.parse_args()
    plan = arena.build_arena(beam_w=a.beam)
    model = make_model(plan, a.mpb)
    res = run_route(plan, model, a.mpb, a.speed)
    res.update({"beam_w_blocks": a.beam, "speed_m_s": a.speed, "meters_per_block": a.mpb})
    print(json.dumps(res, indent=1), flush=True)
    if a.trials:
        rng = np.random.default_rng(0)
        trials = []
        for k in range(a.trials):
            sp = float(rng.choice([0.35, 0.45, 0.55]))
            dy, dyaw = float(rng.uniform(-0.2, 0.2)), float(rng.uniform(-0.3, 0.3))
            r = run_route(plan, model, a.mpb, sp, dy, dyaw)
            trials.append({"speed": sp, "dy_m": round(dy, 3), "dyaw_rad": round(dyaw, 3), "ok": r["ok"], "why": r["why"], "t": r["t"]})
            print(trials[-1], flush=True)
        res["trials"] = trials
        res["trials_ok"] = sum(t["ok"] for t in trials)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
