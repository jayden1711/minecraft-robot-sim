"""DEMO (not learned navigation): one G1 on the frozen unitree_rl_mjlab walking policy, driven by SCRIPTED velocity commands along a
path through an exported region, stepped by MuJoCo in real time and streamed as ghost frames.

Region mode: the path is planned from the exported MJCF (BFS over cells the G1 can walk: step height <= --max-step, hazards avoided). If
the gold block is reachable the walk goes spawn -> goal and back; otherwise spawn -> the farthest reachable cell and back.
Arena mode (--arena route.json): the G1 follows the sky arena's designed route, start to finish, one run per lap, and sends the run
events (start, progress, finish with the simulated time, fall) that drive the in-game boss bar timer and scoreboard.
The commands (vx, vy, yaw rate) come from a hand-written pursuit controller; no learning is involved. Physics is MuJoCo's; this is a simulation.

usage: g1_demo.py region.xml [--stdout] [--seconds N] [--no-realtime] [--style models|blocks] [--arena route.json]
With --stdout the MR-framed GHOST_PARTS/STATUS frames go to stdout (the Python server forwards them to the Minecraft mod);
without it a one-line summary per second goes to stderr. Needs mujoco + onnxruntime: run with MCROBOT_SIM_PY (see server.py).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from mcrobot import g1, nav, protocol
from mcrobot.protocol import MsgType
from mcrobot.pursuit import Pursuit, drop_collinear, route_progress
from mcrobot.pushes import PushSchedule

FINISH_HOLD_S = 4.0  # stand on the finish pad this long before the next lap
FALL_HOLD_S = 3.0


def plan(xml: str, max_step: float):
    lay = nav.load_layout(xml)
    cells = nav.walkable_map(lay, max_step=max_step)
    sp = nav.cell_of(lay.markers["spawn"][0][0], lay)
    goal = nav.cell_of(lay.markers["goal"][0][0], lay) if "goal" in lay.markers else None
    path = nav.bfs_path(cells, sp, goal) if goal in cells else None
    reached_goal = path is not None
    if path is None:
        path = nav.bfs_path(cells, sp, nav.farthest_cell(cells, sp))
    floor_z = cells[sp]
    return lay, path, floor_z, reached_goal


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("region_xml")
    ap.add_argument("--stdout", action="store_true")
    ap.add_argument("--seconds", type=float, default=0.0, help="stop after N seconds of sim time (0 = loop until killed)")
    ap.add_argument("--no-realtime", action="store_true")
    ap.add_argument("--max-step", type=float, default=0.04, help="region mode: tallest step the planner allows (m); the G1's measured limit is 0.05")
    ap.add_argument("--send-every", type=int, default=2, help="send a frame every N policy steps (50 Hz / N)")
    ap.add_argument("--speed", type=float, default=0.45)
    ap.add_argument("--style", choices=("models", "blocks"), default="blocks", help="ghost look: item models of the resource pack, or block displays")
    ap.add_argument("--arena", default="", help="arena route.json (written by scripts/export_arena.py): follow the designed route and send run events")
    a = ap.parse_args()

    header = json.loads(Path(a.region_xml).with_suffix(".json").read_text())
    mpb = float(header.get("meters_per_block", 1.0))
    arena_mode = bool(a.arena)
    if arena_mode:
        info = json.loads(Path(a.arena).read_text())
        lay = nav.load_layout(a.region_xml)
        floor_z = float(info["floor_top_m"])
        route = [np.array([x, y]) for x, y, _ in info["route_m"]]
        way = drop_collinear(route)
        tour = [np.array([p[0], p[1], floor_z]) for p in way]
        print(f"arena demo: route {len(way)} corners, {sum(float(np.linalg.norm(b - c)) for b, c in zip(way, way[1:])):.1f} m, scale {mpb} m/block", file=sys.stderr)
    else:
        lay, path, floor_z, reached_goal = plan(a.region_xml, a.max_step)
        way = nav.corners(path)
        pts = [nav.cell_centre(c, floor_z, lay.cell) for c in way]
        tour = pts + pts[-2::-1]  # there and back
        print(f"demo path: {len(path)} cells, {len(way)} corners, goal {'reached' if reached_goal else 'NOT reachable (walks to the farthest reachable cell)'}", file=sys.stderr)

    model = g1.build_model(a.region_xml)
    w = g1.G1Walker(model)
    centre = np.array([lay.sx / 2, -lay.sy / 2, 0.0]) * lay.cell + [0, 0, floor_z]
    rig = g1.Rig(model, header["origin"], centre, mpb, style=a.style)
    cull = float(max(header["size"][0], header["size"][2])) + 8.0  # blocks: the parts move up to half the region away from the anchor

    out = sys.stdout.buffer if a.stdout else None

    def emit(mtype, obj):
        if out is not None:
            out.write(protocol.encode_frame(mtype, protocol.json_payload(obj)))
            out.flush()

    def run_event(event, **kw):
        if arena_mode:
            emit(MsgType.GHOST_PARTS, {"op": "run", "event": event, **kw})

    # arena v2 route files also carry checkpoint triggers (cue events for the in-game glow) and the push schedule (forces on the pelvis)
    blend = arena_mode and bool(info.get("blend"))
    checkpoints = sorted(info.get("checkpoints", []), key=lambda c: c["x_m"]) if arena_mode else []
    sched = PushSchedule(info.get("pushes", []) if arena_mode else [])
    cp_next = 0

    def start():
        nonlocal cp_next
        d0 = tour[1][:2] - tour[0][:2]
        w.reset(tour[0][:2], math.atan2(d0[1], d0[0]), z=floor_z + 0.80)
        sched.reset()
        cp_next = 0
        return Pursuit(tour, v=a.speed, blend=blend)

    ctl = start()
    emit(MsgType.STATUS, {"text": f"G1 demo ({a.style} ghost): scripted velocity commands on the frozen walking policy (simulation in MuJoCo, not learned navigation)"})
    t_wall0 = time.perf_counter()
    step = 0
    last_spawn = -1e9
    resets = 0
    lap_t0 = 0.0
    laps = 0
    hold_until = -1.0  # arena: standing after a finish or a fall until this sim time
    last_pct = -1
    run_event("start")
    while True:
        t_sim = step * g1.CTRL_DT
        if a.seconds and t_sim >= a.seconds:
            break
        if t_sim < hold_until:
            cmd = np.zeros(3)
        else:
            if hold_until > 0:  # hold over: next lap
                hold_until = -1.0
                ctl = start()
                lap_t0 = t_sim
                last_pct = -1
                run_event("start")
            cmd, _ = ctl.command(w.pelvis_pos(), w.yaw())
            if ctl.done:
                if arena_mode:
                    laps += 1
                    run_event("progress", pct=100)
                    run_event("finish", cs=int(round((t_sim - lap_t0) * 100)))
                    emit(MsgType.STATUS, {"text": f"G1 demo: finished the course in {t_sim - lap_t0:.1f} s (simulated time), lap {laps}"})
                    hold_until = t_sim + FINISH_HOLD_S
                    cmd = np.zeros(3)
                else:
                    ctl = start()
                    emit(MsgType.STATUS, {"text": "G1 demo: loop restarted"})
                    continue
        if t_sim >= hold_until:
            force, fired = sched.update(t_sim, w.pelvis_pos()[:2])
            for n in fired:
                run_event("cue", n=n)  # the in-game piston extends as the push starts
            while cp_next < len(checkpoints) and w.pelvis_pos()[0] >= checkpoints[cp_next]["x_m"]:
                run_event("cue", n=checkpoints[cp_next]["cue"])
                cp_next += 1
        else:
            force = np.zeros(3)
        w.set_push(force)
        w.step(cmd)
        step += 1
        if w.fallen() and t_sim >= hold_until:
            resets += 1
            if arena_mode:
                run_event("fall")
                emit(MsgType.STATUS, {"text": "G1 demo: the G1 fell, resetting it at the start"})
                hold_until = t_sim + FALL_HOLD_S
                w.reset(tour[0][:2], math.atan2(*(tour[1][:2] - tour[0][:2])[::-1]), z=floor_z + 0.80)
            else:
                emit(MsgType.STATUS, {"text": "G1 demo: the G1 fell, resetting it at the spawn"})
                ctl = start()
        if arena_mode and hold_until < 0 and step % 25 == 0:
            pct = int(100 * route_progress(ctl.wp, ctl.k, w.pelvis_pos()))
            if pct != last_pct:
                last_pct = pct
                run_event("progress", pct=pct)
        if step % a.send_every == 0:
            spawn = t_sim - last_spawn > 5.0
            if spawn:
                last_spawn = t_sim
            payload = rig.payload(w.d, ghost=0, ticks=2, spawn=spawn)
            payload["cull"] = cull
            emit(MsgType.GHOST_PARTS, payload)
        if not a.no_realtime:
            lag = t_wall0 + step * g1.CTRL_DT - time.perf_counter()
            if lag > 0:
                time.sleep(lag)
        if step % 50 == 0:
            p = w.pelvis_pos()
            print(f"t={t_sim:6.1f}s pos=({p[0]:.2f},{p[1]:.2f},{p[2]:.2f}) yaw={math.degrees(w.yaw()):6.1f} cmd={np.round(cmd, 2)} leg={ctl.k}/{len(tour) - 1} resets={resets}", file=sys.stderr)
    emit(MsgType.GHOST_PARTS, {"op": "clear", "reason": "demo ended"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
