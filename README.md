# minecraft-robot-sim

Train robots in Minecraft. Build a world, mark the task with blocks, type `/robot train`, watch robots learn live in-game, export a policy
that runs in plain MuJoCo (or later on a real robot).

**This is a simulation.** Minecraft is the world editor and the display. All physics is MuJoCo; Minecraft's physics is never used.
No result is claimed for real robots until milestone M4 measures it.

Status: milestone M0 (mod + server handshake, block-region to MuJoCo export). See [PLAN.md](PLAN.md) for the design,
[CRITIQUES.md](CRITIQUES.md) for the objections and the tests that answer them, [HANDOFF.md](HANDOFF.md) for the current state and next steps.

## How it fits together

```
vanilla Minecraft client --> headless Fabric server (mcrobot mod) --TCP loopback--> Python server (mcrobot) --> MuJoCo / mjlab
```

- `mod/`: server-side Fabric mod (Java 21). Reads blocks, sends them over a framed socket, will show ghost robots as display entities. No client mod is needed.
- `server/`: Python package `mcrobot`: protocol, block-region to merged boxes to MJCF, task markers, socket server. Tests in `server/tests/`.
- `scripts/`: synthetic export, MuJoCo render (CPU), rollout check, server launcher, demo data pack generator.
- `datapack/mcrobot_demo/`: data pack that builds the M0 demo maze in game (generated).
- `server-mc/`, `results/`, `.venv/`: git-ignored. Minecraft's own files never enter git.

Task blocks: gold block = goal, lava or red wool = hazard, pressure plate = checkpoint, command block = spawn.
Commands (op only): `/robot ping`, `/robot export <from> <to>`, `/robot spawn <g1|go2|rover>`, `/robot train`, `/robot play`, `/robot stop`
(the last four are forwarded to the trainer and arrive with M1).

## Run the tests (no Minecraft needed)

```
python3 -m venv .venv && .venv/bin/pip install numpy mujoco pytest pillow
PYTHONPATH=server ~/projects/bin/sim-run .venv/bin/python -m pytest server/tests
JAVA_HOME=~/opt/jdk-21.0.13+9 MCROBOT_PY=$PWD/.venv/bin/python mod/gradlew -p mod build
```

Java toolchain: Temurin JDK 21 unpacked in `~/opt` (no sudo), Gradle via the wrapper. Versions are pinned in PLAN.md.

## Running with Minecraft

Requires you to accept Mojang's EULA yourself; see HANDOFF.md for the exact steps. Minecraft is proprietary and is not part of this repository.

## License

Apache-2.0 (LICENSE). Third-party components and their licenses: [THIRD_PARTY.md](THIRD_PARTY.md).
Not affiliated with Mojang or Microsoft.
