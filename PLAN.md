# PLAN

Vision: build a world in Minecraft, mark the task with blocks, type `/robot train`, watch robots learn live in-game, export a policy
that runs in plain MuJoCo (or later on a real robot). **This is a simulation project.** Minecraft is the editor and the display;
the physics is MuJoCo, never Minecraft's. Nothing here is a claim about real robots until M4 says so.

```
 owner's Minecraft client (vanilla)                         NUC
        |  (Minecraft protocol, over the network)   +-------------------------------------+
        +------------------------------------------>| headless Fabric server (JDK 25)     |
                                                    |   mcrobot mod (server-side only)    |
                                                    +------------------+------------------+
                                                                       | TCP, loopback, framed (below)
                                                    +------------------v------------------+
                                                    | Python server (mcrobot)             |
                                                    |   region -> boxes -> MJCF -> mjlab   |
                                                    |   MuJoCo Warp training (gpu-run)    |
                                                    |   ghosts <- sampled env states      |
                                                    +-------------------------------------+
```

## Milestones

| M | Goal | Done when |
|---|------|-----------|
| M0 | Mod + server handshake; export a small build to MuJoCo and show it matches | handshake on the real Fabric server; MuJoCo render and in-game screenshot of the same build in `results/m0/`; Python + Java tests green |
| M1 | Rover (diff-drive) learns "reach the gold block" in a player-built maze; ghosts visible in game; `/robot play` works | success-rate curve with seeds; ghosts logged and replayed (critique C5) |
| M2 | Go2, then G1: locomotion to a goal over player-built terrain, starting from an existing open locomotion policy | license of the base policy checked and recorded in THIRD_PARTY.md first |
| M2 (next, after M1 trains) | **Real stairs**: fine-tune the frozen G1 walking policy to climb real 0.25 m slab steps at the 0.5 m/block scale (today's limit is a 5 cm step), then replace the STAIRS ramp of arena v2 by real slab steps and relabel it | the policy climbs the slab staircase in MuJoCo with seeds; `arena_v2.STAIRS_*` and the label "(ramp until M2)" updated; the G1 policy licence recorded in THIRD_PARTY.md first (that row still says unknown) |
| M3 | Export: the trained policy runs in plain MuJoCo without Minecraft; success rates with seeds | critique C2 and C5 tests pass on the exported artifact |
| M4 | (later, with pocket-vla) rebuild the owner's room to scale, train the rover, run on the real rover | measured, with real-vs-sim gap stated |

## Multi-version support (owner decision 2026-11-06)

Primary target: the **latest Minecraft Java release, 26.3** (Mojang's `latest.release` in `version_manifest_v2.json` on 2026-11-06; released 2026-09-15).
Also built and tested for **1.21.11** and **1.20.1**. One codebase with [Stonecutter](https://stonecutter.kikugie.dev) 0.9.8: `mod/src/` is shared,
version differences are `//? if >=1.21.11 { ... //?} else { /* ... */ //?}` comment blocks (today only two: the command permission check and the
dimension id accessor). `mod/stonecutter.gradle` marks 26.3 as the active version (the form of the code checked in); every `:<version>:build` generates its own
processed sources, so building any version needs no switching.

- **Mappings: Mojang's official names on every version.** 26.x ships unobfuscated with Mojang names and has no Yarn at all
  (`meta.fabricmc.net/v2/versions/yarn/26.3` is empty), so using Mojang names on 1.20.1 and 1.21.11 too keeps the source identical. This replaces the Yarn pin of M0.
  Mojang's mapping terms (header of the downloaded file): usable for development, not redistributable complete and unmodified. We never ship them; Loom fetches them at build time.
- Loom 1.19.4 has two plugins from one jar: `net.fabricmc.fabric-loom` (no remap; 26.x) and `net.fabricmc.fabric-loom-remap` (1.20.1, 1.21.11).
  `mod/build.gradle` picks by `stonecutter.eval(version, '<26')`.
- **Gradle must run on JDK 25** (Loom 1.19.4's plugin variant requires Java 25 and Gradle API 9.8). Each version is compiled with `--release` of its own Java (17 / 21 / 25) and its tests run on that JDK
  (`MCROBOT_JDK_<n>`, defaults to `~/opt/jdk-*`).
- Test matrix: `scripts/test_matrix.sh` (`make test-matrix`): per version, the Python tests, the mod build, the Java tests, and the mod-to-Python-server integration test (`InteropTest`, which the script requires to have run, not skipped).
  `--smoke` (`make test-matrix-smoke`) also boots a real Fabric server per version on a throwaway flat world with that version's mod jar, runs `robot ping` and `robot export` from the console against real blocks, and checks the ack; the 26.3 leg also runs with Via and checks protocol translation.
  `.github/workflows/ci.yml` runs the non-smoke matrix (never pushed: there is no remote). Its `${{ env.JAVA_HOME_n_X64 }}` hand-off from setup-java is [verify].

### Pinned versions (checked against Mojang, meta.fabricmc.net, maven.fabricmc.net, maven.kikugie.dev and the Modrinth API on 2026-11-06)

| Component | 26.3 (primary) | 1.21.11 | 1.20.1 |
|-----------|----------------|---------|--------|
| Minecraft protocol number | 777 | 774 | 763 |
| Fabric Loader | 0.20.1 | 0.20.1 | 0.20.1 |
| Fabric API | 0.165.2+26.3 | 0.141.6+1.21.11 | 0.92.12+1.20.1 |
| Mappings | none (unobfuscated, Mojang names) | Mojang official | Mojang official |
| Java (build `--release` / server JDK) | 25 | 21 | 17 |
| Loom plugin | `net.fabricmc.fabric-loom` | `net.fabricmc.fabric-loom-remap` | `net.fabricmc.fabric-loom-remap` |

| Shared pin | Version |
|------------|---------|
| Fabric Loom | 1.19.4 |
| Stonecutter | 0.9.8 |
| Gradle | 9.8.2 (wrapper, in git; latest is 9.9.0, not needed) |
| JDKs | Temurin 25.0.5+8, 21.0.13+9, 17.0.21+9 in `~/opt` (user space) |
| Fabric installer (server launcher) | 1.1.2 |
| Python | 3.12, `.venv/` |
| MuJoCo (pip) | 4.0.0; check exact API drift before pinning the trainer |
| mjlab, mujoco_warp | pinned at the start of M1 [verify versions and licences] |
| Mod version | 0.0.2 (jars are named `mcrobot-<mod>+<mc>.jar`) |

Server-side Via (NUC server only, 26.3): **ViaFabric 0.4.22+188-26.x** (the Fabric platform for ViaVersion; works on a dedicated server, `environment: *`) with **ViaVersion 5.12.2** (release, overrides the 5.12.1-SNAPSHOT that ViaFabric embeds)
and **ViaBackwards 5.12.2** (release). ViaFabricPlus is client-side only and is not used. ViaRewind (1.8 and older clients) is not installed. ViaFabric is on Modrinth's alpha channel (the only channel for 26.x).
URLs and SHA-256 are in `scripts/pins.tsv`; jars live only in git-ignored server dirs (`server-mc/mods/`, `server-mc-test/`).

Client versions: ViaBackwards 5.12.2 contains the protocol chain from 1.9 up to 26.3 (`Protocol1_9_1To1_9` ... `Protocol26_3To26_2`), so older clients 1.9 through 26.2 are claimed by Via, plus native 26.3.
Newer clients than 26.3 are not covered until Via adds them. `scripts/ping_status.py` and `scripts/smoke_server.sh --via` check, per protocol number, that the server answers a status request with the client's number (Via translating) and that a plain server does not (control).
That is a protocol-level check, not a login: a real client has never joined from this repo.

## Server layout: headless Fabric server on the NUC

`server-mc/` (git-ignored) holds the Fabric server launcher, Mojang's server jar (downloaded by the launcher on first run, never committed),
`mods/` (Fabric API + the built `mcrobot` jar + the Via jars) and the world. `scripts/setup_server.sh 26.3 server-mc --via --eula-from <file>` creates it from the pins;
`scripts/run_server.sh` starts it at nice 10 with 2 CPU threads and a 2 GB heap on the JDK the version needs.
Minecraft listens on 25566 (the default 25565 left free), `online-mode=true`, whitelist on. The owner's vanilla client connects to `<nuc>:25566`.
How the NUC is reachable (Tailscale, LAN, SSH tunnel) is the owner's call; nothing is exposed by this repo. The Python bridge port
(25599) binds to loopback only and requires a shared token when `MCROBOT_TOKEN` is set.
The server only runs when the owner starts it, and it needs the owner to accept Mojang's EULA (HANDOFF.md).

## Does the player need a client mod? **No.**

Everything the player sees is vanilla-renderable:
- Commands (`/robot ...`) are server commands, and chat/action-bar messages come from the server.
- **Ghost robots are `block_display` / `item_display` entities** spawned and moved by the server (Minecraft 1.19.4+ display entities
  take a transformation, translation/rotation/scale, with built-in interpolation). Vanilla clients render them with no mod and no resource pack.
- Optional later (not planned): a client mod for a HUD/reward overlay or a custom robot model. A resource pack would give nicer robot models but
  forces a download; the default is plain stacked block displays.

## Socket protocol (v1)

Loopback TCP, mod = client, Python = server, one connection. Implemented in `server/mcrobot/protocol.py` and `mod/.../Frames.java`;
a golden-bytes test on both sides pins the framing.

Frame: `"MR"` | framing version u8 (=1) | type u8 | payload length u32 | payload. Big-endian everywhere (Java default). Max payload 64 MiB.
Anything else (bad magic, version, type, length) closes the connection with an `ERROR` where possible.

| type | name | dir | payload |
|------|------|-----|---------|
| 1 | HELLO | mod->py | JSON `{proto:[min,max], mod, token}` |
| 2 | HELLO_ACK | py->mod | JSON `{proto: chosen, server, time}` (highest common app version) |
| 3/4 | PING/PONG | either | JSON, echoed |
| 5 | REGION | mod->py | `u32 hlen` + JSON header + zlib(grid) (below) |
| 6 | REGION_ACK | py->mod | JSON `{id, mjcf, sha256, solid_cells, collider_boxes, markers}` |
| 7 | COMMAND | mod->py | JSON `{cmd: spawn/train/play/export/stop, arg}` |
| 8 | STATUS | py->mod | JSON `{text}` shown in chat/action bar |
| 9 | GHOSTS | py->mod | `u32 train_step, u16 n, n x (u16 env_id, f32 x y z qw qx qy qz)`; poses already in Minecraft world coordinates |
| 10 | ERROR | either | JSON `{code, message}` |
| 11 | BYE | either | JSON |

Versioning: the framing byte only changes for incompatible framing. Application semantics are negotiated in HELLO (`proto` range); a peer with no common version gets `ERROR` and a closed socket. New optional JSON fields do not bump the version.

REGION grid: `uint16` palette indices, big-endian, `index = (y*sz + z)*sx + x`, zlib level 6. Header JSON: `origin [x,y,z]`, `size [sx,sy,sz]`, `dimension`, `mc_version`, `meters_per_block`, and `palette: [{name, shape}]` where `shape` is `"empty"`, `"full"`, or a list of block-local collision boxes `[x0,y0,z0,x1,y1,z1]`
(read from the block's real collision shape, so slabs, stairs and fences are exact). Threading: the mod reads blocks only on the server thread, then hands bytes to its single I/O thread; the server never blocks the Minecraft tick on the network.
Limits: 4M cells per export (mod side); the Python side accepts up to the 64 MiB frame limit and its decompressor is bounded (zip-bomb test).

## Block -> collider merge

1. Colliders: every cell whose palette shape is `full` gets label 1. Partial blocks keep their exact boxes (not merged; few in practice).
2. Greedy merge (`server/mcrobot/boxes.py`): scan cells in (y, z, x) order; at each unclaimed cell, extend along x while cells are unclaimed and equal,
   then extend that run along z while the whole row matches, then extend the slab along y. Claim the box. This is O(cells) and gives a disjoint exact cover.
3. The scan is run for all 6 axis priorities and the result with the fewest boxes is kept; output is sorted so the MJCF is byte-identical for identical input.
   Greedy is not minimal (minimum box cover is NP-hard in general); it is good on floors and walls, which is what players build.
4. Visual geoms are merged separately per palette entry (colors survive) with `contype=conaffinity=0`, in geom group 2; colliders are in group 3 (hidden).
   Training MJCF can drop visuals (`to_mjcf(visuals=False)`).
5. Test: boxes are disjoint and cover exactly the solid cells on random grids; a flat floor is 1 box; the maze needs far fewer boxes than cells.

6. Materials (arena v2): colliders are labelled by friction class (`Region.friction_classes`): packed ice, ice, blue ice and frosted ice are separate boxes with a low sliding friction
   (`region.BLOCK_FRICTION`), written on the geom together with `priority="2" condim="6"`. The priority matters: MuJoCo takes a contact's friction from the geom with the higher priority, and the G1's
   own collision geoms have priority 1, so a floor geom at the default priority would be ignored. A region without slippery blocks exports exactly as before. Tests: `test_friction.py`.

Known limit: boxes are axis-aligned and exact; there is no mesh for diagonal terrain because Minecraft has none. Contact quality at box seams is a
real MuJoCo concern (a foot can catch an internal edge between adjacent boxes); merging reduces seams and the merged-vs-unmerged comparison is a test in M1.

## Coordinates and units

Minecraft: x east, y up, z south (right-handed with east x up = south). MuJoCo: x east, y north, z up. Mapping, relative to the region's minimum corner:
`mj = (x, -z, y) * meters_per_block`. It is a proper rotation (no mirror), tested. Region origin is stored so poses map back exactly (`mj_to_mc`).

**World scale is one setting, `meters_per_block`, default 0.5** (owner decision 2026-11-06; it was 1.0 in M0). Set in the mod's `config/mcrobot.properties` (`meters_per_block`) or the environment variable
`MCROBOT_METERS_PER_BLOCK`; validated to [0.01, 100]; written into every export header (`Region.meters_per_block`); headers without the key mean 1.0. Argument for 0.5: the Unitree G1 (1.32 m) is then 2.6 blocks tall
and its 0.53 m body about one block wide, which reads right next to the world; the M0 rover is ~0.3 m wide, so its corridors can stay 1 block (0.5 m). The cost: a block is half as big against the robot's
limits (a G1 corridor needs 2 blocks, ceilings 3, a carpet is a climbable 3 cm step, a full block never is). The setting is global to the server: exports made at another scale stay valid because each carries its own.
Everything reads the header value: the collider merge and the marker boxes (`mjcf.py`), the navigation grid (`nav.Layout.cell`), the ghost transforms (`g1.Rig`: block ghosts scale the box by `1/meters_per_block`, item-model ghosts the
model by `model_scale_m / meters_per_block`), the demo. Tests: `test_scale.py`, the Rig test at 1.0, 0.5 and 0.2. M2 locomotion worlds may use ~0.2 m/block ([verify] against what the base policy can climb).
Rewards, thresholds, and policies are always in MuJoCo meters.

## Reward spec from blocks

Marker blocks are classified by registry name (`region.classify_marker`). Markers are sensors in MJCF (sites, group 4), never extra colliders; solid markers
(gold block, red wool, command block) are also ordinary solid blocks, so they stay physical.

| Block | Meaning | Rule (MuJoCo meters; `server/mcrobot/tasks.py`) |
|-------|---------|---|
| `gold_block` | goal | reached when base position is within 0.35 m of the block's box horizontally/below and up to 0.5 m above; success, `+10`, episode terminates |
| `lava`, `red_wool` | hazard | inside the box plus 0.10 m horizontal margin, up to 0.5 m above: `-10`, terminates |
| any `*_pressure_plate` | checkpoint | within 0.5 m horizontally of the plate: `+1` once per plate per episode (unordered in M1) |
| `command_block` (and chain/repeating) | spawn | robot reset on top of one, chosen at random; yaw random; xy jitter 0.2 m |

Per-step shaping: `+k * (d_prev - d)` where `d` is the **geodesic** distance to the nearest goal, computed by flood fill over free space in the exported block grid (Euclidean distance misleads in mazes);
small time penalty and action-rate penalty. Also terminates on: falling below the region floor, rover flipped, timeout. Errors the player sees in chat: no goal block, no spawn block.
Future: per-task parameters (time limit, reward scale) from sign text. Reward weights are config, not block-encoded, in M1.

## Ghosts: sampling and rate limiting

Goal: the player watches the policy improve. Ghosts are not an animation: every pose sent is a real env state from the training run (critique C5).
- ~50 env ids are chosen once per run (evenly spaced over `num_envs`, recorded in the run log); they never change during a run.
- A sampler thread reads the poses of those envs from the already-computed sim state every `ghost_every` training iterations (default: time-based, 5 Hz wall-clock);
  one batched device-to-host copy of (50, 7) per sample. Nothing is re-simulated for display.
- Latest-wins: the sender keeps only the newest sample; if the socket or game is slow, older samples are dropped, never queued, so training is never blocked by display.
- The mod updates each display entity with a 4-tick interpolation duration so 5 Hz updates look smooth at 20 TPS.
- Budget: 50 ghosts x 5 Hz x ~60 B of entity data is ~15 KB/s per connected client, plus the bridge frame of ~0.4 KB per sample.
- Every sample sent is appended to `results/<run>/ghost_log.npz` (step, env_id, pose), the artifact the C5 test replays.
- Each ghost is 1 to 3 block displays (rover: a flat box and two wheels; legged robots: a body box plus coarse leg boxes from the same state's link poses in M2), tinted by episode status (running/success/failed) from the same log.
- Visual style (M1, owner decision 2026-11-06): the demo G1 can also be drawn as one `item_display` per link with a model from a server resource pack (generated from the Menagerie G1 meshes as cuboid
  elements, plain colours, no logos; `scripts/build_pack.py`). The payload part carries `"m": "mcrobot:g1/<link>"` instead of `"b": <block>`; the mod summons an `item_display` whose stack is a stick with the
  `minecraft:item_model` component (>= 1.21.11; 1.20.1 has no such component and draws a stick). The pack is offered through `server.properties` (`resource-pack` URL on the NUC, SHA-1, fixed id,
  `require-resource-pack=false`); an item display is centred on its entity, a block display is not, which `g1.Rig` accounts for. `/robot style blocks|models` switches; block ghosts remain the fallback and the swarm style.
  Display culling: a display entity is culled by its own position, so the payload carries `cull` (blocks) and the mod sets `width`/`height`/`view_range` on the entities (a robot walking 40 blocks from its anchor stays visible).
- Pack formats (read from `version.json` in the server jars): resource format 75.0 on 1.21.11, 97.1 on 26.3; the pack declares `min_format` 75, `max_format` 99, which one pack serves on both (item model
  definitions under `assets/<ns>/items/` exist on both). ViaBackwards passes the resource-pack push through (the client downloads from the URL) and should map the `item_model` component 1:1: [verify] with a real 1.21.11 client.
- Arena run events: the demo sends `{"op":"run","event":"start|progress|finish|fall",...}` in the ghost stream; the mod maps them to fixed data pack functions (`mcrobot_demo:run/*`, integers only) for the boss bar and scoreboard.

## Python server and training (M1, as built: the G1 on the sky arena)

M1 changed from the rover in a maze to the **Unitree G1 humanoid on the sky arena** (owner decisions 2026-11-06). Two levels:

- **Low level, frozen:** the `unitree_rl_mjlab` velocity policy v0 (Apache-2.0; flat-ground trained, 98 inputs, three ELU layers), used unchanged: (vx, vy, yaw rate) in, 29 joint targets out at 50 Hz, the same PD gains and 4 x 5 ms physics substeps as the policy's own deployment config. `navenv.LowLevel` ports it to a torch MLP so thousands of copies run on the GPU; `g1.G1Walker` (CPU MuJoCo + onnxruntime) is the reference the batched version is tested against.
- **High level, learned:** a PPO policy (`navppo.py`) that outputs a velocity command every 0.1 s (smoothed, inside the walking policy's command range). Observation (45 numbers): a compass (direction to the next point on the shortest walkable path to the gold block, in the body frame, and the path distance), 16 wall rays and 16 void/lava rays (the only view of the world, sampled from the exported boxes), base velocity, gravity, the previous command, and the fraction of plates collected. Rewards: progress along the shortest path (dense), +1 per pressure plate (once), +10 and end at the gold pad, -10 and end for lava, leaving the course, or falling, a small time and command-change cost. The goal rule is relative to the robot's base height (`tasks.G1_BASE_HEIGHT`): the old absolute margin never fired for a standing G1.
- **Physics:** MuJoCo Warp (`mujoco_warp`, the backend of mjlab) directly, one world per env, the arena's collider boxes plus the G1; mjlab itself is only the version pin (THIRD_PARTY.md). Episodes start on the command block (a share of them, always for the 20 ghost envs) or on random free cells along the course, so every section is practised; evaluation always starts on the command block.
- **Under `gpu-run --preemptible`:** `scripts/train_nav.py` checkpoints atomically, saves on SIGUSR1/yield file/SIGTERM and exits, and resumes from `latest.pt`; only latest, best and final are kept (NUC rule).
- **Ghost swarm:** the first 20 envs are streamed to the mod as display-entity ghosts at 4 Hz (latest-wins; every pose sent is logged). The trainer serves the frames on `<out>/ghosts.sock`; the Python server attaches there (`jobs.Attach`, reconnects when the trainer yields and resumes, chat only on state changes). `/robot train` starts the trainer if none runs (or attaches to a running one), `/robot play` runs the best policy for one G1 on the CPU with the arena timer, `/robot stop` ends it (for a trainer started outside the game, only the display).
- `scripts/eval_nav.py` evaluates from the command block over seeds; `--scripted` is a compass-following baseline without learning.

## M2: real stairs (design; full text in `docs/m2-plan.md`)

Goal: the frozen G1 walking policy (`unitree_rl_mjlab` policy v0, Apache-2.0, recorded in THIRD_PARTY.md) is **fine-tuned** to climb and descend real 0.25 m slab steps (half a block at 0.5 m/block); then arena v2's STAIRS section becomes real stairs and the navigation policy uses them. All simulation (MuJoCo Warp for training); no real-robot claim.

- **Starting point:** only the exported ONNX exists (no checkpoint, no critic). The actor and its observation normaliser are imported into an rsl_rl checkpoint for the mjlab 1.6.0 runner; the critic is fresh and is trained alone first (actor frozen) so that a random critic cannot damage the actor. An identity test (imported actor equals the ONNX within 1e-5) runs before any training.
- **Which task trains:** v0 was trained in the checkout's own velocity task (gait-phase observation, `feet_gait`), which pins `mjlab==1.2.0`; importing it against the installed 1.6.0 fails at once. Plan: port that task onto 1.6.0 (same physics stack as the nav env) behind a fidelity gate against the CPU reference walker; fallbacks are a second venv with 1.2.0 or our own PPO.
- **Curriculum:** step height from the measured limit (`results/m1/fit.md`) to 0.25 m in 9 linear levels, then harder variants (tread 0.25 m, faster commands, jitter); treads {0.25, 0.5, 0.75, 1.0} m plus jitter; flights of 2 to 10 steps, up, down and up-down; promote on "far platform reached", demote on a fall; 20 % of envs always sample lower levels. Terrain is built from boxes in the exporter's way (0.5 m cells, so seams are in the training data).
- **Rewards:** every term of the shipped velocity task stays (tracking is the objective; climbing is what tracking requires). Changes: foot clearance measured above the terrain under the foot with a step-aware target, new foot-stumble (riser collision) and undesired-contact (no crawling) penalties, a looser posture prior on stairs tiles only.
- **Observations:** none changed in plan A (blind, 98 inputs, drops into `LowLevel`, `G1Walker`, `deploy.yaml` unchanged). Plan B adds a height scan; its costs (interface, real-robot claim, nav env gather, training time, full regression again) are listed in the doc and it is gated, not assumed. Fallback F: base policy on flat, fine-tuned on stairs, selected by a mode bit from the navigator.
- **Keeping flat skills:** 25 % original flat tiles and 10 % micro-steps, an anchor loss to the frozen base actor on those tiles, learning rate 1e-4, a frozen normaliser, and a **regression gate** (original task, arena v2 sections, limit probes, behavioural distance) with paired seeds and margins fixed before training.
- **Compute (RTX 5080, `gpu-run --preemptible`):** main line 0.5 G env-steps at the largest env count that fits (1024 to 4096), hard cap 12 GPU-hours (controls 8, evaluation 2); throughput is unmeasured and the first GPU job measures it. Checkpoints: latest, best (by the gate score, not return), final; terrain levels and curriculum state travel in the checkpoint so preemption does not restart the curriculum.
- **Evaluation:** tiers 0.25 / 0.1875 / 0.125 m (reported at the highest passed); headline cell n = 256 per direction, pass at 95 % observed and 90 % lower Wilson bound; seeds: train {0,1,2}, validation 2000 to 2007, test 1000 to 1031 used once; paired bootstrap over seeds for differences.
- **Arena:** STAIRS = 6 steps up, an 8-block platform (1.5 m), 6 steps down in the same 20 blocks; label "(0.25 m steps)". `arena_v2` gets a `stairs` mode (`ramp` stays the default until the gates pass, so M1 results stay comparable). The navigator's map must stop calling a step above 0.25 m a wall (`navtask.WALL_ABOVE` is relative to the global floor today): walkability from the step between neighbouring cells. Then scripted lap, nav retraining (warm start vs scratch control) and per-section evaluation.
- **Critiques:** C10 "it was just retrained", C11 "stairs are a trivial terrain curriculum", C12 "it forgot flat walking", C13 "the navigator never saw the stairs as walkable", each with a planned test in CRITIQUES.md.

## Export (M3)

A run produces `policy.onnx` (or TorchScript) + `scene.xml` + `task.json` (rewards, spawn points, observation spec) + a `rollout.py` that needs only `mujoco` and `numpy` (+ `onnxruntime`). Success rate is evaluated by that script over N held-out seeds with Minecraft not running.

## Safety and hygiene

Commands are op-only (level 2). The bridge accepts loopback only. No Minecraft jar or asset is committed or redistributed; the launcher fetches them on the NUC.
Results, logs, run directories, clips and screenshots stay in `results/` (ignored). CPU sims run through `sim-run`, GPU only through `gpu-run --preemptible`, total NUC load kept under 20.
