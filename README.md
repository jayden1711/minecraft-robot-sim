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

- `mod/`: server-side Fabric mod, one codebase for several Minecraft versions via Stonecutter (Java 17/21/25 by version). Reads blocks, sends them over a framed socket, will show ghost robots as display entities. No client mod is needed.
- `server/`: Python package `mcrobot`: protocol, block-region to merged boxes to MJCF, task markers, socket server. Tests in `server/tests/`.
- `scripts/`: test matrix and smoke tests, server setup/launcher (pinned downloads in `pins.tsv`), synthetic export, MuJoCo render (CPU), rollout check, demo data pack generator.
- `datapack/mcrobot_demo/`: data pack that builds the M0 demo maze in game (generated).
- `server-mc/`, `server-mc-test/`, `results/`, `.venv/`: git-ignored. Minecraft's own files never enter git.

Task blocks: gold block = goal, lava or red wool = hazard, pressure plate = checkpoint, command block = spawn.
Commands (op only): `/robot ping`, `/robot export <from> <to>`, `/robot spawn <g1|go2|rover>`, `/robot train`, `/robot play`, `/robot stop`
(the last four are forwarded to the trainer and arrive with M1).

## Supported Minecraft versions

One codebase ([Stonecutter](https://stonecutter.kikugie.dev) 0.9.8), built and tested for every row. Status as of 2026-11-06, from the last local run of `make test-matrix-smoke`.

| Minecraft | Fabric Loader | Fabric API | Mappings | Java | Mod build | Python + Java + mod-to-server tests | Real server boot with the mod | Client versions that can join the NUC server |
|-----------|---------------|------------|----------|------|-----------|-------------------------------------|-------------------------------|---------------------------------------------|
| **26.3** (latest, primary) | 0.20.1 | 0.165.2+26.3 | none: unobfuscated (Mojang names) | 25 | pass | pass | pass | NUC server runs this version with ViaFabric + ViaVersion + ViaBackwards 5.12.2: native 26.3; 1.9 to 26.2 through Via (see HANDOFF.md for which were checked) |
| 1.21.11 | 0.20.1 | 0.141.6+1.21.11 | Mojang official | 21 | pass | pass | pass | not served on the NUC (vanilla 1.21.11 clients join via Via on the 26.3 server) |
| 1.20.1 | 0.20.1 | 0.92.12+1.20.1 | Mojang official | 17 | pass | pass | pass | not served on the NUC (vanilla 1.20.1 clients join via Via on the 26.3 server) |

Details, exact pins and the Via setup: [PLAN.md](PLAN.md) ("Multi-version support"); licenses: [THIRD_PARTY.md](THIRD_PARTY.md). Via runs only on the NUC server (GPL-3.0; not shipped here).

## Run the tests (no Minecraft needed)

```
python3 -m venv .venv && .venv/bin/pip install numpy mujoco pytest pillow
make test-matrix           # every supported version: Python tests, mod build, Java tests, mod <-> Python server integration test
make test-matrix-smoke     # plus a real Fabric server boot per version (needs MCROBOT_EULA_FILE=<your accepted eula.txt>)
```

The matrix script takes versions as arguments (`scripts/test_matrix.sh 26.3`), waits while NUC load is 20 or more, and runs at nice 10 with 2 Gradle workers.
Java toolchain: Temurin JDKs 17, 21 and 25 unpacked in `~/opt` (no sudo; Gradle itself runs on 25), Gradle via the wrapper. `make build-all` writes `mod/dist/mcrobot-<mod>+<mc>.jar`.
`.github/workflows/ci.yml` runs the same matrix (not pushed; there is no remote).

## Running with Minecraft

Requires you to accept Mojang's EULA yourself; see HANDOFF.md for the exact steps. Minecraft is proprietary and is not part of this repository.

## License

Apache-2.0 (LICENSE). Third-party components and their licenses: [THIRD_PARTY.md](THIRD_PARTY.md).
Not affiliated with Mojang or Microsoft.
