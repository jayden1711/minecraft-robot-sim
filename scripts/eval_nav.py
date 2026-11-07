"""Evaluate a navigation policy from the command block, deterministic actions, several seeds. Simulation only.

Run through gpu-run (normal job):  ~/projects/bin/gpu-run --need-mib 3500 --label minecraft-robot-sim -- \
    .venv-train/bin/python scripts/eval_nav.py results/m1/arena_v2/arena_region.xml --policy results/m1/nav/v2/best.pt --out results/m1/nav/eval_v2_best.json
Each env runs exactly one episode from the spawn (position +-0.1 m, heading +-0.15 rad, the training spawn noise); the episode ends on the
finish, the lava or a fall, or after its time limit (15 s + 0.3 m/s over the path length). --scripted runs the compass-following baseline
(no learning) the same way. Reports success rate, mean time to finish, falls, lava, plates per episode, per seed and pooled.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server"))


def by_section(env, res, end_x) -> dict:
    """Failures (falls, lava, timeouts) per course section: where (pelvis x) the episode ended, binned by arena_route.json's sections."""
    route = Path(env.region_xml).with_name("arena_route.json")
    if not route.exists():
        return {}
    out = {}
    for sec in json.loads(route.read_text())["sections"]:
        inside = (end_x >= sec["x0_m"]) & (end_x < sec["x1_m"])
        out[sec["key"]] = {k: int((inside & res[k]).sum()) for k in ("fell", "lava", "timeout")}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("region_xml")
    ap.add_argument("--policy", default="")
    ap.add_argument("--scripted", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--envs", type=int, default=200, help="episodes per seed")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    import torch
    import warp as wp

    wp.config.quiet = True
    from mcrobot import navenv, navppo

    ac = None if a.scripted else navppo.load_policy(a.policy, a.device)
    runs = []
    for seed in a.seeds:
        env = navenv.NavEnv(a.region_xml, a.envs, a.device, seed=seed, p_spawn=1.0, n_spawn_forced=a.envs)
        n = a.envs
        fin = torch.zeros(n, dtype=torch.bool, device=env.dev)
        res = {k: torch.zeros(n, dtype=torch.bool, device=env.dev) for k in ("success", "lava", "fell", "timeout")}
        t_end = torch.zeros(n, device=env.dev)
        plates = torch.zeros(n, device=env.dev)
        end_x = torch.zeros(n, device=env.dev)
        obs = env.obs()
        lo, hi = env.cmd_lo, env.cmd_hi
        a_vx = float((0.45 - lo[0]) / (hi[0] - lo[0]) * 2 - 1)
        t0 = time.time()
        step = 0
        while not bool(fin.all()):
            if a.scripted:
                err = torch.atan2(obs[:, 1], obs[:, 0])
                act = torch.zeros(n, 3, device=env.dev)
                act[:, 0] = torch.where(err.abs() < 0.6, torch.full_like(err, a_vx), torch.full_like(err, -0.3))
                act[:, 2] = (1.8 * err).clamp(-1, 1)
            else:
                act = ac.act(obs, deterministic=True)
            obs, r, done, info = env.step(act)
            step += 1
            new = done & ~fin
            for k in res:
                res[k] |= new & info[k]
            t_end = torch.where(new, info["ep_len"].float() * 0.1, t_end)
            plates = torch.where(new, info["plates"].float(), plates)
            end_x = torch.where(new, info["x"], end_x)
            fin |= new
            if step % 200 == 0:
                print(f"seed {seed} step {step}: {int(fin.sum())}/{n} finished, {time.time() - t0:.0f} s", file=sys.stderr, flush=True)
            if step > 4000:
                break
        s = res["success"]
        runs.append({"seed": seed, "episodes": n, "success": float(s.float().mean()), "lava": float(res["lava"].float().mean()), "fell": float(res["fell"].float().mean()),
                     "timeout": float(res["timeout"].float().mean()), "unfinished": int((~fin).sum()),
                     "time_to_finish_s_mean": float(t_end[s].mean()) if bool(s.any()) else None, "time_to_finish_s_min": float(t_end[s].min()) if bool(s.any()) else None,
                     "time_to_finish_s_max": float(t_end[s].max()) if bool(s.any()) else None, "plates_mean": float(plates.mean()),
                     "by_section": by_section(env, res, end_x)})
        print(json.dumps(runs[-1]), file=sys.stderr, flush=True)
        del env
    succ = [r["success"] for r in runs]
    sections: dict = {}
    for r in runs:
        for key, d in r["by_section"].items():
            acc = sections.setdefault(key, {"fell": 0, "lava": 0, "timeout": 0})
            for k in acc:
                acc[k] += d[k]
    out = {"policy": "scripted compass baseline" if a.scripted else a.policy, "region": a.region_xml, "runs": runs, "failures_by_section_pooled": sections,
           "pooled": {"episodes": sum(r["episodes"] for r in runs), "success_mean": sum(succ) / len(succ), "success_min": min(succ), "success_max": max(succ),
                      "falls_mean": sum(r["fell"] for r in runs) / len(runs), "lava_mean": sum(r["lava"] for r in runs) / len(runs)},
           "note": "simulation: MuJoCo Warp, frozen unitree_rl_mjlab walking policy, no real robot"}
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out["pooled"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
