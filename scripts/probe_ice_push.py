"""Limits probes for the frozen G1 walking policy (simulation only): ice friction sweep (3 m patch) and pelvis push sweep (lateral, back, forward;
force for 0.2 s while walking at 0.45 m/s). The numbers behind arena v2 (results/m1/arena_course.md): BLOCK_FRICTION and PUSH_FORCE_N.
usage: PYTHONPATH=server python scripts/probe_ice_push.py ice|push
"""
import json
import math
import sys

import mujoco
import numpy as np

from mcrobot import g1
from mcrobot.pursuit import Pursuit

mode = sys.argv[1]


def model_with(ice_mu=None, ice=(3.0, 6.0)):
    if ice_mu is None:
        boxes = [([10, 0, -0.5], [20, 5, 0.5])]
    else:
        a, b = ice
        boxes = [([(a - 5) / 2, 0, -0.5], [(a + 5) / 2, 5, 0.5]), ([(a + b) / 2, 0, -0.5], [(b - a) / 2, 5, 0.5]), ([(b + 25) / 2, 0, -0.5], [(25 - b) / 2, 5, 0.5])]
    m = g1.build_model(None, extra_boxes=boxes)
    if ice_mu is not None:
        gid = m.geom("extra_1").id
        m.geom_friction[gid, 0] = ice_mu
        m.geom_priority[gid] = 2
    return m


def walk(m, t_end, push=None, v=0.45, yaw0=0.0):
    w = g1.G1Walker(m)
    w.reset(np.array([0.0, 0.0]), yaw0, z=0.80)
    ctl = Pursuit([np.array([0.0, 0.0]), np.array([30.0, 0.0])], v=v)
    root = w.root
    tr = []
    for i in range(int(t_end / g1.CTRL_DT)):
        t = i * g1.CTRL_DT
        cmd, _ = ctl.command(w.pelvis_pos(), w.yaw())
        w.d.xfrc_applied[root, :3] = 0
        if push and push["t"] <= t < push["t"] + push["T"]:
            w.d.xfrc_applied[root, :3] = push["F"]
        w.step(cmd)
        p = w.pelvis_pos()
        tr.append((t, p[0], p[1], w.yaw()))
        if w.fallen():
            return False, t, np.array(tr)
    return True, t_end, np.array(tr)


if mode == "ice":
    for mu in (1.0, 0.6, 0.45, 0.3, 0.2, 0.15, 0.1, 0.07, 0.05):
        m = model_with(mu)
        ok, t, tr = walk(m, 18.0)
        on = tr[(tr[:, 1] > 3.0) & (tr[:, 1] < 6.0)]
        v_on = (on[-1, 1] - on[0, 1]) / max(on[-1, 0] - on[0, 0], 1e-6) if len(on) > 2 else float("nan")
        print(f"mu={mu:5.2f} ok={ok} t_end={t:5.1f} max|y|={np.abs(tr[:,2]).max():.3f} max|yaw|={np.degrees(np.abs(tr[:,3]).max()):5.1f}deg speed_on_ice={v_on:.2f}", flush=True)
else:
    m = model_with(None)
    mass = float(m.body_subtreemass[m.body("pelvis").id])
    print("mass", round(mass, 1), flush=True)
    for dirname, d in (("left(+y)", (0, 1, 0)), ("right(-y)", (0, -1, 0)), ("back(-x)", (-1, 0, 0)), ("fwd(+x)", (1, 0, 0))):
        for F in (40, 80, 120, 160, 200, 250, 300, 400):
            for T in (0.2,):
                ok, t, tr = walk(m, 12.0, push={"t": 4.0, "T": T, "F": np.array(d, float) * F})
                after = tr[tr[:, 0] > 4.0]
                dy = np.abs(after[:, 2]).max() if len(after) else 0
                print(f"{dirname:10s} F={F:4d}N T={T:.1f}s impulse={F*T:5.1f}Ns dv={F*T/mass:.2f}m/s ok={ok} t_end={t:5.1f} max|y|after={dy:.2f}", flush=True)
                if not ok:
                    break
            if not ok:
                break
