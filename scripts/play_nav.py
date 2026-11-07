"""Play the trained navigation policy on the arena: one G1 from the command block, simulated in real time on the CPU (MuJoCo Warp's CPU device,
same code as training), streamed as a ghost with the run events of the arena timer. Simulation only.

usage: play_nav.py region.xml --policy best.pt [--stdout] [--style blocks|models] [--seconds N] [--no-realtime] [--seed N]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server"))

FINISH_HOLD_S, FALL_HOLD_S = 4.0, 3.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("region_xml")
    ap.add_argument("--policy", required=True)
    ap.add_argument("--stdout", action="store_true")
    ap.add_argument("--style", choices=("models", "blocks"), default="blocks")
    ap.add_argument("--seconds", type=float, default=0.0)
    ap.add_argument("--no-realtime", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    frame_fd = None
    if a.stdout:
        frame_fd = os.dup(1)
        os.dup2(2, 1)
    import torch
    import warp as wp

    wp.config.quiet = True
    from mcrobot import navenv, navppo, protocol
    from mcrobot.navghost import FrameOut, GhostStreamer
    from mcrobot.protocol import MsgType

    fo = FrameOut(frame_fd)
    env = navenv.NavEnv(a.region_xml, 1, "cpu", seed=a.seed, p_spawn=1.0, n_spawn_forced=1)
    ac = navppo.load_policy(a.policy, "cpu")
    streamer = GhostStreamer(env, a.region_xml, fo, 1, hz=1e9, style=a.style)
    streamer.ticks = 2
    fo.emit(MsgType.STATUS, {"text": f"play: {Path(a.policy).name} on the arena (simulation: the frozen G1 walking policy under a learned navigation policy)"})

    def run_event(event, **kw):
        fo.emit(MsgType.GHOST_PARTS, {"op": "run", "event": event, **kw})

    d0 = float(env.tt.field_near(torch.as_tensor(env.task.spawn_xy, dtype=torch.float32)[None])[0])
    zero_act = (-env.cmd_lo / (env.cmd_hi - env.cmd_lo) * 2 - 1)[None]  # the action that asks for a zero velocity command
    hold_until, step, last_pct, t0 = -1.0, 0, -1, time.perf_counter()
    obs = env.obs()
    run_event("start")
    while True:
        t_sim = step * 0.1
        if a.seconds and t_sim >= a.seconds:
            break
        act = zero_act if t_sim < hold_until else ac.act(obs, deterministic=True)
        obs, r, done, info = env.step(act)  # on done, env resets itself onto the start (spawn is forced)
        step += 1
        streamer(env)
        if done[0]:
            if info["success"][0]:
                run_event("progress", pct=100)
                run_event("finish", cs=int(round(int(info["ep_len"][0]) * 10)))
                fo.emit(MsgType.STATUS, {"text": f"play: reached the finish in {int(info['ep_len'][0]) * 0.1:.1f} s of simulated time"})
                hold_until = t_sim + FINISH_HOLD_S
            else:
                why = "fell into the lava" if info["lava"][0] else ("fell" if info["fell"][0] else "ran out of time")
                run_event("fall")
                fo.emit(MsgType.STATUS, {"text": f"play: the G1 {why}; resetting at the start"})
                hold_until = t_sim + FALL_HOLD_S
            last_pct = -1
        elif t_sim >= hold_until >= 0:
            hold_until = -1.0
            run_event("start")
        pct = int(100 * max(0.0, 1.0 - float(env.prev_field[0]) / max(d0, 1e-6)))
        if hold_until < 0 and pct != last_pct and step % 5 == 0:
            last_pct = pct
            run_event("progress", pct=min(pct, 99))
        if not a.no_realtime:
            lag = t0 + step * 0.1 - time.perf_counter()
            if lag > 0:
                time.sleep(lag)
        if step % 50 == 0:
            p = env.pelvis()[0]
            print(f"t={t_sim:6.1f}s pelvis=({float(p[0]):.2f},{float(p[1]):.2f},{float(p[2]):.2f}) field={float(env.prev_field[0]):.1f}", file=sys.stderr, flush=True)
    streamer.clear("play ended")
    return 0


if __name__ == "__main__":
    sys.exit(main())
