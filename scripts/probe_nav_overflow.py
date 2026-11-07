"""Measure the peak constraint rows (nefc, per world) and contacts per world of NavEnv on an arena, to size NJMAX / NCONMAX.

Two phases, simulation only: a scripted lap (envs start on the command block and walk the shortest path with the pelvis pushes,
ice and slalom included) and a stress phase (random starts, random commands, so robots fall and pile into walls and edges).
Run it with large njmax / nconmax so nothing overflows while measuring:
  ~/projects/bin/gpu-run --light --need-mib 2500 -- .venv-train/bin/python scripts/probe_nav_overflow.py results/m1/arena_v2/arena_region.xml
"""

from __future__ import annotations

import argparse
import json

import torch

from mcrobot import navenv


def drive(env, steps, scripted, vx=0.45):
    lo, hi = env.cmd_lo, env.cmd_hi
    a_vx = float((vx - lo[0]) / (hi[0] - lo[0]) * 2 - 1)
    obs, done_total, success = env.obs(), 0, 0
    for _ in range(steps):
        if scripted:
            err = torch.atan2(obs[:, 1], obs[:, 0])
            a = torch.zeros(env.n, 3, device=env.dev)
            a[:, 0] = torch.where(err.abs() < 0.6, torch.full_like(err, a_vx), torch.full_like(err, -0.3))
            a[:, 2] = (1.8 * err).clamp(-1, 1)
        else:
            a = torch.rand(env.n, 3, device=env.dev, generator=env.gen) * 2 - 1
        obs, _, done, info = env.step(a)
        done_total += int(done.sum())
        success += int(info["success"].sum())
    return done_total, success


def main():
    p = argparse.ArgumentParser()
    p.add_argument("region_xml")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--envs", type=int, default=256)
    p.add_argument("--lap-steps", type=int, default=1500)
    p.add_argument("--stress-steps", type=int, default=600)
    p.add_argument("--njmax", type=int, default=2000)
    p.add_argument("--nconmax", type=int, default=400)
    a = p.parse_args()
    out = {}
    for name, scripted, steps, p_spawn in (("lap", True, a.lap_steps, 1.0), ("stress", False, a.stress_steps, 0.0)):
        env = navenv.NavEnv(a.region_xml, a.envs, a.device, seed=1, p_spawn=p_spawn, njmax=a.njmax, nconmax=a.nconmax)
        done, success = drive(env, steps, scripted)
        out[name] = {"envs": a.envs, "steps": steps, "episodes_ended": done, "successes": success,
                     "peak_nefc": int(env.peak_nefc), "peak_contacts_per_world": round(float(env.peak_ncon), 1),
                     "overflow_worlds": int((env.overflow_flags != 0).sum())}
        print(name, json.dumps(out[name]), flush=True)
    print("recommended NJMAX >= %d, NCONMAX >= %d (current %d, %d)" % (
        1.5 * max(v["peak_nefc"] for v in out.values()), 1.5 * max(v["peak_contacts_per_world"] for v in out.values()),
        navenv.NJMAX, navenv.NCONMAX))


if __name__ == "__main__":
    main()
