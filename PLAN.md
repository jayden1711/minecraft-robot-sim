# PLAN

Vision: build a world in Minecraft, mark the task with blocks, type `/robot train`, watch robots learn live in-game, export a policy
that runs in plain MuJoCo (or later on a real robot). **This is a simulation project.** Minecraft is the editor and the display;
the physics is MuJoCo, never Minecraft's. Nothing here is a claim about real robots until M4 says so.

```
 owner's Minecraft client (vanilla)                         NUC
        |  (Minecraft protocol, over the network)   +-------------------------------------+
        +------------------------------------------>| headless Fabric server (JDK 21)     |
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
| M3 | Export: the trained policy runs in plain MuJoCo without Minecraft; success rates with seeds | critique C2 and C5 tests pass on the exported artifact |
| M4 | (later, with pocket-vla) rebuild the owner's room to scale, train the rover, run on the real rover | measured, with real-vs-sim gap stated |

## Pinned versions (checked against meta.fabricmc.net / Maven on 2026-11-06)

| Component | Pin | Why |
|-----------|-----|-----|
| Minecraft (Java Edition) | **1.21.11** | newest release before the 26.x versioning; Fabric API, Loom and Yarn all exist for it. 26.x exists but I have no offline knowledge of its API, so I did not pin it. Revisit after M1 |
| Fabric Loader | 0.20.1 | latest stable for 1.21.11 |
| Fabric API | 0.141.6+1.21.11 | latest for 1.21.11 |
| Fabric Loom | 1.14.10 | needs Gradle >= 9.2 |
| Mappings | Yarn 1.21.11+build.6 (CC0) | avoids Mojang-mappings licence questions [verify: Mojang mappings terms] |
| Gradle | 9.8.2 (wrapper, in git) | |
| JDK | Temurin 21.0.13+9 in `~/opt/jdk-21.0.13+9` | user space, no sudo |
| Python | 3.12, `.venv/` | |
| MuJoCo (pip) | 4.0.0 | installed; check exact API drift before pinning the trainer |
| mjlab, mujoco_warp | pinned at the start of M1 | [verify] versions and licences |

## Server layout: headless Fabric server on the NUC

`server-mc/` (git-ignored) holds the Fabric server launcher, Mojang's server jar (downloaded by the launcher on first run, never committed),
`mods/` (Fabric API + the built `mcrobot` jar) and the world. `scripts/run_server.sh` starts it at nice 10 with 2 CPU threads and a 2 GB heap.
Minecraft listens on 25566 (the default 25565 left free), `online-mode=true`. The owner's vanilla client connects to `<nuc>:25566`.
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

Known limit: boxes are axis-aligned and exact; there is no mesh for diagonal terrain because Minecraft has none. Contact quality at box seams is a
real MuJoCo concern (a foot can catch an internal edge between adjacent boxes); merging reduces seams and the merged-vs-unmerged comparison is a test in M1.

## Coordinates and units

Minecraft: x east, y up, z south (right-handed with east x up = south). MuJoCo: x east, y north, z up. Mapping, relative to the region's minimum corner:
`mj = (x, -z, y) * meters_per_block`. It is a proper rotation (no mirror), tested. Region origin is stored so poses map back exactly (`mj_to_mc`).

**1 block = 1 m by default.** Argument: the rover is ~0.3 m wide so 1 m corridors are comfortable; "rebuild the owner's room to scale" (M4) is
simplest at 1 block = 1 m with sub-block detail from slabs (0.5 m) and furniture as exact collision boxes; and the player's intuition (a block is a metre) is a feature.
Where it breaks: a Go2 (~0.3 m tall at the hip) cannot climb a 1 m step and a G1 does not fit a 1 m cube tunnel in a believable way. So each region carries
`meters_per_block` (already in the protocol and exporter, tested). M2 locomotion worlds use ~0.2 m/block ([verify] against what the base policy can climb), where a one-block step is a 0.2 m step. Ghosts are then drawn
scaled by `1/meters_per_block` so the robot looks right against the blocks. Rewards, thresholds, and policies are always in MuJoCo meters.

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

## Python server and training (M1 outline)

`/robot train` re-exports the saved region (edits are picked up), builds the MJCF, makes the env (rover first: diff-drive, ray/lidar-free observation of
heading, relative goal vector, and a coarse local occupancy patch), trains with PPO under `gpu-run --preemptible` (checkpoint on SIGUSR1/yield-file, resume on start),
streams ghosts and STATUS lines. `/robot play` runs the latest policy in one env, shown as one ghost; `/robot stop` checkpoints and halts.
Commands beyond the owner's list that M1 adds: `/robot region <from> <to>` (saves the arena). The M0 mod implements `ping`, `export`, and forwards `spawn/train/play/stop`.

## Export (M3)

A run produces `policy.onnx` (or TorchScript) + `scene.xml` + `task.json` (rewards, spawn points, observation spec) + a `rollout.py` that needs only `mujoco` and `numpy` (+ `onnxruntime`). Success rate is evaluated by that script over N held-out seeds with Minecraft not running.

## Safety and hygiene

Commands are op-only (level 2). The bridge accepts loopback only. No Minecraft jar or asset is committed or redistributed; the launcher fetches them on the NUC.
Results, logs, run directories, clips and screenshots stay in `results/` (ignored). CPU sims run through `sim-run`, GPU only through `gpu-run --preemptible`, total NUC load kept under 20.
