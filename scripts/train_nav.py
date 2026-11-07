"""Train the navigation policy (velocity commands on the frozen G1 walking policy) on an exported region, PPO on MuJoCo Warp.

Run it only through gpu-run:  ~/projects/bin/gpu-run --preemptible --need-mib 4000 --label minecraft-robot-sim -- \
    .venv-train/bin/python scripts/train_nav.py results/m1/arena_v2/arena_region.xml --out results/m1/nav/v2 [--stdout]
It resumes from <out>/latest.pt, saves it on a yield request (SIGUSR1 / yield file) or SIGTERM, and exits 0. MR frames (ghosts, status
lines) go to the unix socket <out>/ghosts.sock (the Python server attaches to it) and, with --stdout, to the original stdout;
anything else printed goes to stderr. Simulation only: no real robot, no Minecraft physics.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("region_xml")
    ap.add_argument("--out", default=str(ROOT / "results/m1/nav/v2"))
    ap.add_argument("--envs", type=int, default=2048)
    ap.add_argument("--horizon", type=int, default=32)
    ap.add_argument("--iters", type=int, default=100000)
    ap.add_argument("--minutes", type=float, default=1e9, help="stop (final.pt) after this long in this process; resume-able runs count only their own time")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--ghosts", type=int, default=20)
    ap.add_argument("--ghost-hz", type=float, default=4.0)
    ap.add_argument("--style", choices=("models", "blocks"), default="blocks")
    ap.add_argument("--stdout", action="store_true", help="send MR frames (ghosts, status) on stdout")
    ap.add_argument("--finish", action="store_true", help="when the iteration budget is reached: write final.pt and drop latest.pt")
    a = ap.parse_args()

    frame_fd = None
    if a.stdout:  # warp and torch print banners on stdout: keep the original stdout for frames only
        frame_fd = os.dup(1)
        os.dup2(2, 1)
    import warp as wp

    wp.config.quiet = True
    from mcrobot import navenv, navppo
    from mcrobot.navghost import FrameOut, GhostStreamer
    from mcrobot.protocol import MsgType

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fo = FrameOut(frame_fd, out / "ghosts.sock")  # the Python server attaches here (/robot train); nothing is sent while nobody listens
    (out / "trainer.pid").write_text(str(os.getpid()))

    def say(text: str):
        print(text, file=sys.stderr, flush=True)

    def status(text: str):
        say(text)
        fo.emit(MsgType.STATUS, {"text": text})

    y = navppo.Yield()
    env = navenv.NavEnv(a.region_xml, a.envs, a.device, seed=a.seed, n_spawn_forced=a.ghosts)
    streamer = GhostStreamer(env, a.region_xml, fo, a.ghosts, a.ghost_hz, a.style, out / "ghost_log.npz") if a.ghosts else None
    tr = navppo.Trainer(env, out, horizon=a.horizon, hook=streamer, log=say)
    (out / "config.json").write_text(json.dumps({**vars(a), "obs_dim": navenv.OBS_DIM, "nav_repeat": navenv.NAV_REPEAT}, indent=1))
    status(f"train: PPO on {a.region_xml.split('/')[-1]}, {a.envs} parallel G1s on MuJoCo Warp ({a.device}), {a.ghosts} shown as ghosts (simulation; iteration {tr.it})")
    t_status = time.time()
    why = "done"
    t_begin = time.time()
    while True:
        why = tr.train(min(tr.it + 5, a.iters), max(a.minutes * 60 - (time.time() - t_begin), 0.0), y)
        if why == "yield":
            break
        if time.time() - t_begin >= a.minutes * 60:
            break
        if time.time() - t_status > 30:
            t_status = time.time()
            rec = json.loads((out / "log.jsonl").read_text().splitlines()[-1])
            status(f"train: iteration {rec['it']}, {rec['spawn_success'] * 100:.0f}% of runs from the start reach the finish, {rec['success'] * 100:.0f}% of all episodes (simulation)")
        if tr.it >= a.iters:
            break
    if streamer is not None:
        streamer.clear("trainer yielded the GPU" if why == "yield" else "training ended")
    if why == "done" and a.finish:
        tr.save_policy("final.pt")
        trash = ROOT / "results/_trash"
        trash.mkdir(parents=True, exist_ok=True)
        if (out / "latest.pt").exists():
            shutil.move(str(out / "latest.pt"), str(trash / f"latest_{out.name}_{int(time.time())}.pt"))
    status(f"train: stopped ({why}) at iteration {tr.it}")
    fo.close()
    (out / "trainer.pid").unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
