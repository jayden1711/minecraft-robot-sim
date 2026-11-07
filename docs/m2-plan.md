# M2: real stairs (design, 2026-11-07)

Status: **design only.** Nothing in this document has been trained or run, except the offline checks listed in section 1 (file reads and one CPU import test).
Every number below is a design choice, a threshold fixed in advance, or an estimate marked as such; measured results live in `results/` and are cited by path, never copied here.
Everything is **simulation**: physics is MuJoCo (MuJoCo Warp for training), never Minecraft's. Nothing here is a claim about a real robot.

## 0. Goal and done-when

Fine-tune the frozen open G1 walking policy so that it climbs and descends **real slab steps of 0.25 m** (half a block at the 0.5 m/block scale), then replace the STAIRS ramp of arena v2 by real steps and let the navigation policy use them.
Today's limit is the step the frozen policy can climb, measured in `results/m1/fit.md` (called **h0** below); it is far below 0.25 m, and the ramp (3.1 cm risers) exists only because of that.

Done when (all of them, pre-registered in section 8):
1. The fine-tuned policy passes the **stairs gate** (section 8.2) on seeds never used in training or selection.
2. It passes the **regression gate** (section 8.3): the flat-ground skills and the arena v2 sections are not worse than the base policy beyond the stated margins.
3. Arena v2 builds with real slab steps in the STAIRS section, exports, and a scripted lap through the whole arena passes (section 10).
4. The nav policy is retrained on it and evaluated per section (section 10.4).
5. `THIRD_PARTY.md` carries the policy licence (done in this branch, with `[verify]` items), and CRITIQUES C10 to C12 have run their tests (section 11).

If the 0.25 m target is not reached, the milestone is reported at the highest **tier** reached (section 8.1) and the arena is built with that riser; nothing is rounded up.

## 1. What was checked offline while planning

| Fact | Source |
|------|--------|
| Base policy: `unitree_rl_mjlab` policy v0, `deploy/robots/g1/config/policy/velocity/v0/exported/policy.onnx`, checkout rev `1425b15f73bd4095f0df53709d7c389c3eb9e790` (read from `MCROBOT_G1_ROOT`, default `~/projects/robot-arena-sim/third_party/unitree_rl_mjlab`) | `git -C <checkout> log -1` |
| Licence: Apache-2.0, standard text (201 lines, the file is spelled `LICENCE`); the checkout has no `NOTICE` file and the README states no separate terms for the weights | `LICENCE`, `README.md` (grep for "licen" finds nothing) |
| The checkout ships **only the exported ONNX** of the policy: no `.pt` checkpoint, no critic, no optimizer state, no action std. Its README trains with `scripts/train.py Unitree-G1-Flat --env.scene.num-envs=4096` (rsl_rl through mjlab) | file listing, `README.md` |
| Actor: 98 inputs (`base_ang_vel` 3, `projected_gravity` 3, command 3, gait phase 2, `joint_pos_rel` 29, `joint_vel_rel` 29, `last_action` 29), obs normaliser (mean/std), MLP 512-256-128 ELU, 29 outputs; 50 Hz policy, 4 x 5 ms physics, per-joint PD gains and action scales in `deploy.yaml` | `deploy.yaml`, `navenv.LowLevel` |
| v0 is the **flat** task: the flat env config deletes the `height_scan` observation and the terrain curriculum. The rough config (not what v0 was trained on) has a 17 x 11 height scan at the pelvis | `src/tasks/velocity/config/g1/env_cfgs.py` |
| The checkout pins `mjlab==1.2.0`; the training venv has **mjlab 1.6.0**. Importing the checkout's task code against 1.6.0 **fails at the first import** (`cannot import name 'update_assets' from 'mjlab.utils.os'`, raised from the Go2 asset module, so the G1 module is not even reached). Script: `results/_scratch/m2_import_check.py`, CPU only, no GPU | run 2026-11-07 |
| mjlab 1.6.0's own G1 velocity task has **no gait-phase observation and no `feet_gait` reward**, so it is not the task v0 was trained in | `mjlab/tasks/velocity/velocity_env_cfg.py` (grep) |
| mjlab 1.6.0 provides box terrain generators (`BoxPyramidStairsTerrainCfg`, inverted, random grid), the terrain-level curriculum, `TerrainHeightSensorCfg` (foot height above terrain). Its stairs presets stop at 0.10 m risers | `mjlab/terrains/`, `config.py` |
| mjlab 1.6.0's runner `load()` accepts `actor_state_dict` / `critic_state_dict`; actor keys are `mlp.*`, `obs_normalizer.{_mean,_var,_std,count}`, `distribution.std_param`; it also restores `common_step_counter` | `mjlab/rl/runner.py`, `rsl_rl/models/mlp_model.py` (rsl_rl 5.4.2) |
| GPU lane today: one non-preemptible job holds the GPU (`gpu-run --status`), 85 GB free on `/`, load 13 to 14 | 2026-11-07 |

Not checked (no web access): whether Unitree publishes any further terms for weights or models beyond the repo licence, upstream issues about stairs, and whether a pip cache holds `mjlab==1.2.0`. All `[verify]`.

## 2. Base policy, licence, fine-tuning

- **Base:** policy v0 above. It is the policy M1 already uses unchanged (`navenv.LowLevel`, `g1.G1Walker`).
- **Is fine-tuning allowed?** Apache-2.0 section 2 grants the right to make and distribute derivative works. Conditions (section 4) apply only if we *distribute*: give recipients the licence text, keep notices, state that the files were changed (there is no `NOTICE` to carry). Section 6 (no trademark licence): we do not call the result "Unitree" anything.
  The fine-tuned weights are our own result, **git-ignored under `results/m2/`** like every weight (NUC rule 7); nothing is distributed. If the owner ever releases them: ship the LICENCE text, a "modified from unitree_rl_mjlab v0" statement, and this training recipe. This is a reading of the licence for planning, not legal advice; the owner decides.
- **Record first:** the `THIRD_PARTY.md` row that said "unknown yet" now records the policy, revision and licence (this branch). `[verify]`: any model-specific terms upstream; the G1 mesh/xml licence inside the checkout (Menagerie's G1 is BSD-3-Clause, separately recorded); a Go2 base policy (the checkout has a Go2 task config; whether it ships a Go2 policy was not checked, and this plan covers the G1 only).

## 3. How the checkpoint is loaded, and which stack trains

### 3.1 Importing the weights (no `.pt` exists)

`scripts/m2_import_actor.py` (new) reads the ONNX initializers exactly as `navenv.LowLevel` does and writes an rsl_rl-5.x checkpoint for the **mjlab 1.6.0 runner**:

| Checkpoint key | From |
|---|---|
| `actor_state_dict["mlp.{0,2,4,6}.{weight,bias}"]` | the four ONNX linear layers |
| `actor_state_dict["obs_normalizer._mean"]`, `_var`, `_std`, `count` | the ONNX normaliser mean and std (`_var = std^2`; `count` set large so the statistics do not drift); the exact epsilon handling is read from `rsl_rl/modules/normalization.py` |
| `actor_state_dict["distribution.std_param"]` | not in the ONNX: initialised to a small action std (section 4.3) |
| `critic_state_dict` | **fresh**: there is no critic in the file |
| `optimizer_state_dict`, `iter`, `infos` | empty / zero |

**Identity test (before any training):** the imported actor's deterministic output equals the ONNX output within 1e-5 on 10 000 recorded observations (`test_m2_import.py`), and the exported ONNX of a freshly imported actor loads through `navenv.LowLevel` unchanged (guards the initializer-naming risk: `LowLevel` finds `onnx::Div_24`, a name that depends on the exporter version).

**Normaliser: frozen** for the whole fine-tune. The base function was trained with these statistics; letting them drift changes what every weight means. The new tiles shift the observation distribution only mildly (same 98 proprioceptive inputs).

### 3.2 Which environment trains (the real obstacle)

v0 was trained in the checkout's task (gait-phase observation, `feet_gait`, its action scales). Fine-tuning in any other task would silently change the problem, so the task must be that one, on a stack we can run.

| Option | What | Cost / risk |
|---|---|---|
| **T2 (recommended)** | Port the checkout's `src/tasks/velocity` + G1 asset config onto the installed mjlab 1.6.0 and mujoco_warp 3.11.0 (copy the needed files into `server/mcrobot/m2/` with the Apache notice kept; fix the API drift). Physics is then the same stack as the nav env and the evaluation | the drift is an unknown size; budget 1 day, abandon to T1 if exceeded |
| T1 | A second venv with `mjlab==1.2.0` and its pinned mujoco/warp, task code as shipped | no porting, but a second physics version (so training physics differs from the nav env's), needs network or a wheel cache `[verify]` |
| T3 | Our own PPO on MuJoCo Warp as `navppo.py` does | most code, full control; only if T2 and T1 both fail |

**Fidelity gate for T2 (G1 below), a hard stop if it fails:** run the imported actor, unchanged, in the ported flat task and compare with the CPU reference `g1.G1Walker` (ONNX + CPU MuJoCo) on the same scripted commands and initial states: tracking error and survival of the ported env must match the reference within the CIs of section 8. If the port reproduces the base policy's behaviour badly, fine-tuning there would be fine-tuning against the wrong physics or observations.
Terrain tile size: the generator's default is 8 x 8 m; flights of up to 8 m plus platforms need a longer tile (planned 12 x 4 m) `[verify]` that the generator accepts non-square tiles.

## 4. Training design

### 4.1 Terrain (what the robot practises on)

Built from boxes the way the Minecraft exporter builds them (lane-wide steps split into 0.5 m cells, so the seams of the real arena are in the training geometry too: critique C3 test 2). `server/mcrobot/m2terrain.py` (new) is the generator; a test checks that its geometry for a given step list equals the exporter's output for the same blocks.

| Tile family | Share (start; tuned) | Notes |
|---|---|---|
| Flat, original v0 distribution (plane, original commands, original randomisation) | 25 % | the "original tasks"; this is what must not be forgotten |
| Micro-steps and ramps (risers 0.03 to 0.0625 m, 1 to 4 cells per level) | 10 % | keeps the arena's RAMP skill |
| Stairs up | 25 % | |
| Stairs down | 20 % | |
| Up, platform, down | 15 % | the arena's shape |
| Small random boxes (up to 0.03 m) | 5 % | seam and clutter robustness |

Stairs parameters (per tile, sampled): riser `h` from the level ladder below; tread depth from {0.25, 0.5, 0.75, 1.0} m plus a continuous jitter (Minecraft builds multiples of 0.25 m; a policy that learns one tread and one gait period could count steps instead of adapting; held-out treads are an evaluation, section 8.2); per-step riser jitter up to 10 % (the arena is exact, the jitter is for robustness); 2 to 10 steps per flight; lane 3 m wide; platforms 1 to 2 m.
Buildable riser heights (collision heights): carpet 0.031 m, snow layers 0.0625 m per layer above the first (2 layers 0.0625 m, 3 layers 0.125 m, 4 layers 0.1875 m, 5 layers or a bottom slab 0.25 m) `[verify]` the 4- and 5-layer collision boxes with the real exporter, the lower ones are already in arena v2.

### 4.2 Curriculum: step height from h0 to 0.25 m

Levels L = 0 to 8: `h_L = h0 + (0.25 - h0) * L / 8` (linear; h0 is the measured limit of fit.md, read from the file by the script so the doc carries no result number). Levels 9 to 11 stay at 0.25 m and add difficulty: tread 0.25 m only (stair-block geometry, foot length 0.21 m), speed commands up to 0.7 m/s, riser jitter 0.02 m, a stress riser of 0.275 m (reported as the frontier, not required).

Promotion rule (replaces mjlab's `terrain_levels_vel`, which only checks distance walked): per env, **promote** when the episode ended with the far platform reached and no fall; **demote** on a fall or on less than half the flight traversed. 20 % of envs sample their level uniformly from 0 to the current maximum, so low levels are never dropped from the data (a guard against forgetting the gentler terrain). The mean level, the per-level success and the per-family success go in the run log every 10 iterations. Levels and `common_step_counter` are saved in the checkpoint `infos` (a resume after a preemption must not restart the curriculum).

Commands on stairs tiles: forward-biased (vx 0.2 to 0.7 m/s, |vy| up to 0.2, |yaw rate| up to 0.5, 5 % standing); on flat tiles the original ranges of `deploy.yaml`.

### 4.3 Algorithm settings (starting points, to be tuned by the first probe)

- Warm start: actor from the ONNX, **critic fresh**. A random critic gives meaningless advantages and wrecks a good actor, so phase P1 trains the critic only (actor learning rate 0) for about 200 iterations on the full tile mix at levels 0 to 2.
- Then actor learning rate **1e-4** (the from-scratch config uses 1e-3), adaptive schedule with `desired_kl` 0.005, entropy coefficient 0 to 0.005, initial action std 0.2 to 0.3 (the from-scratch config starts at 1.0; the base's final std is not in the ONNX). Too large knocks the walker off its gait, too small cannot find the swing height for a riser: a first-probe sweep over {0.15, 0.3}.
- **Anchor loss** on flat and micro tiles only: `beta * mean ||pi_theta(o) - pi_0(o)||^2` against the frozen base actor (kept as a second copy on the GPU), `beta` 1.0 decaying to 0.1 over P2. This is the main guard against forgetting (section 6).
- Domain randomisation unchanged from the flat task (foot friction 0.3 to 1.6, push events, COM offset, encoder bias, observation noise).
- PPO otherwise as the shipped config (24 steps per env, 5 epochs, 4 minibatches, gamma 0.99, lambda 0.95, clip 0.2).

### 4.4 Rewards

Keep every term of the shipped velocity task at its weight: `track_linear_velocity`, `track_angular_velocity`, `body_orientation_l2`, `pose` (variable posture), `body_ang_vel`, `angular_momentum`, `is_terminated` (-200), `joint_acc_l2`, `joint_pos_limits`, `action_rate_l2`, `foot_gait` (period 0.6), `foot_clearance`, `foot_slip`, `soft_landing`, `stand_still`, `self_collisions`. The original velocity-tracking reward stays the main objective: climbing is what tracking a forward command requires on a staircase, so no separate "height gain" reward is added.

Changes, each with its reason:

| Term | Change | Why |
|---|---|---|
| `foot_clearance` | clearance measured above the **terrain under the foot** (mjlab's foot height sensor), target swing height `max(0.10 m, h_ahead + 0.05 m)` where `h_ahead` is the tallest terrain in a small window in front of the foot (privileged, used only in the reward) | the shipped term measures height above the plane; on a staircase it would punish climbing |
| new `feet_stumble` | penalty when the horizontal contact force on a foot or ankle link exceeds a multiple of its vertical force (toe against a riser) | the collision term for the riser |
| new `undesired_contacts` | penalty for knee, shin, hip and hand links touching terrain | blocks crawling and knee-climbing as a way to "finish" the flight; the evaluation also counts it as a failure |
| `pose` | on stairs tiles only, the walking std of hip pitch, knee and ankle pitch is doubled | a 0.25 m step needs leg excursions the flat posture prior penalises; applying it to flat tiles would loosen the flat gait |
| terminations | add out-of-terrain bounds; keep the 70 degree fall limit and the 20 s timeout | |

### 4.5 Observations: none changed (plan A), and what a height scan would cost (plan B)

**Plan A (first): the 98 inputs stay exactly as they are.** The policy is blind; it learns the riser from foot contact, IMU and joint feedback. Consequences: the new weights drop into `LowLevel`, `G1Walker`, `deploy.yaml` and (M3) `rollout.py` with no code change; the critic still gets the privileged height scan and foot data (asymmetric actor-critic, as the shipped task does), which helps value estimation on stairs.
Honest prior (a design expectation, not a result): blind climbing of a 0.25 m riser is feasible in the literature on small humanoids only with high swing heights; blind **descent** is harder (the first drop arrives unannounced). Plan A is run first because a negative result is informative and the interface stays unchanged.

**Plan B (conditional, gate G3): add a height scan to the actor.**

| Cost | Detail |
|---|---|
| Network | input 98 -> 98 + K (the shipped rough design K = 187: a 17 x 11 grid, 0.1 m spacing, yaw-aligned at the pelvis, scaled by 1/5, noise 0.1). New first-layer columns initialised to zero, so the function starts identical to the base; the normaliser gets mean 0 / std 1 for them |
| Interface | `LowLevel`, `G1Walker`, `deploy.yaml`, `rollout.py` and any future real robot need the scan. On a real robot that means a terrain map from depth or lidar plus state estimation: a different, bigger claim than "joints and IMU". The README and PLAN would say so |
| Nav env | each low-level step gathers K heights per env from a precomputed ground-height map of the arena (`navtask.ground`), yaw-aligned: one gather of K x N values, cheap; but pillars, lintels and gates must be treated consistently (the scan sees a pillar as a tall column the robot never trained on: add random boxes to the training tiles) |
| Training | ray or height-field queries per env per step: planned +20 to 40 % step time `[verify]` by measurement |
| Evaluation | the full regression gate again, plus a scan-noise and scan-dropout test |

A middle option, observation history (stacked previous observations), is not planned: it changes the interface like B without adding perception.
Fallback F, if neither A nor B passes both gates: two frozen low-level policies (base for flat, fine-tuned for stairs) selected by a mode bit that the nav policy outputs; costs one more action dimension for the navigator and makes the regression gate trivial for flat sections, but the stairs policy still has to pass its own gate.

## 5. Compute budget (RTX 5080 16 GB, shared; nothing here is measured)

Rules: GPU only through `~/projects/bin/gpu-run --preemptible --need-mib <N>`; check `cut -d' ' -f1 /proc/loadavg` before every job and wait above 20 (owner decision 2026-11-06); one preemptible job at a time; the free-space floor is 50 GB; CPU sims through `sim-run`, at most 4.

| Item | Plan |
|---|---|
| Envs | the largest of {1024, 2048, 4096} whose VRAM fits the free memory minus 1 GB, found by the G1 probe; 24 steps per env per iteration. The VRAM per env is not known (the nav env used about 1 GB at its earlier buffers); `--need-mib` is set from the probe |
| Env-steps | main line **0.5 G** (about 5000 iterations at 4096 envs): P1 critic warm-up 0.02 G, P2 curriculum ramp 0.3 G, P3 consolidation at full difficulty with the anchor decayed 0.18 G. Controls (section 11): 0.25 G (from-scratch run for C10, forgetting control for C12 shortened). Plan B if gated in: 0.3 G |
| Hours | `hours = env_steps / throughput / 3600`; throughput of this task on this GPU is **unknown `[verify]`**: at 20k / 40k / 80k env-steps per second the main line is about 7 / 3.5 / 1.7 h of GPU time. The lane is shared, so wall time is planned at twice that. First action of G1 is to measure throughput |
| Hard caps | main line 12 GPU-hours, controls and variants 8, evaluation 2; at each cap the run stops and the report says where it stood |
| Chunking | one `gpu-run --preemptible` command; the trainer checks a stop flag every iteration (an iteration is a few seconds), saves atomically on SIGUSR1 / `$GPU_RUN_YIELD_FILE` / SIGTERM within the 60 s yield window, and resumes from `latest.pt` including terrain levels, curriculum stage, RNG and optimizer state |
| Checkpoint rule | **latest + best + final** only (NUC rule). `best` is chosen by the **gate score** (stairs success, subject to the regression gate on the validation seeds), never by training return. Optimizer state is dropped from `final`. Each kept checkpoint has its ONNX export beside it; checkpoints cited by a report are kept |
| Telemetry kept per eval | weight distance from the base per layer (for C10), anchor loss, per-family success, contact statistics, joint-limit saturation fraction, physics-overflow counter (as `NavEnv` has, critique C9) |

Planned command (not run): `~/projects/bin/gpu-run --preemptible --need-mib <probe> --label minecraft-robot-sim -- .venv-train/bin/python scripts/train_m2.py --out results/m2/ft --init results/m2/base_actor.pt`.

## 6. Keeping the flat-ground skills

Four mechanisms, all measured (section 8.3):
1. **Data:** 25 % of tiles are the original flat task, 10 % micro-steps, plus 20 % of all envs at uniformly sampled lower levels.
2. **Anchor loss** to the frozen base actor on flat and micro tiles.
3. **Small learning rate and a frozen normaliser**; the checkpoint with the best stairs score *among those that pass the regression gate* is `best`.
4. **A regression eval that is known to have power:** a deliberately forgetting control (stairs only, no anchor, no flat tiles) must fail it (critique C12).

## 7. Failure modes

| # | Failure | How we would see it | Mitigation / next step |
|---|---------|---------------------|------------------------|
| 1 | Forgets flat walking | R1 and R2 fail; behavioural distance to the base on flat observations grows | anchor, mixture, lower LR; if it persists, fallback F |
| 2 | Random critic ruins the actor | return collapses in the first iterations after the unfreeze | critic warm-up P1 is mandatory; unfreeze LR 1e-4 |
| 3 | Imported actor is not the base | identity test or fidelity gate fails | stop before training; fix the import or the port |
| 4 | Environment port is not v0's task | fidelity gate: ported env vs `G1Walker` mismatch | fix; else T1 |
| 5 | Crawling, knee or hand contacts, sliding up on the shins | `undesired_contacts` statistics; evaluation counts it as a failure | penalty weight; contact termination if needed |
| 6 | Jumping up steps with both feet | both feet off the ground at once, landing force spikes (`soft_landing`) | `foot_gait` stays on; flagged in the evaluation, not counted as success if flight time exceeds a threshold |
| 7 | Overfits one tread / gait period | success drops on held-out treads (0.35, 0.65, 0.8 m) | tread randomisation; wider jitter |
| 8 | Descent fails while climbing works | up and down are reported separately | plan B (scan) |
| 9 | Kinematic or torque limit at 0.25 m | joint-limit saturation fraction near 1 at levels near the top; the curriculum stalls | tiers (section 8.1): accept 0.1875 m or 0.125 m, built with snow layers; plan B |
| 10 | Seam snagging between boxes | success differs merged vs per-block geometry (C3 test 2) | train on per-cell geometry (already planned); fillet edges |
| 11 | Curriculum stalls from lateral drift | per-level success flat, mean level stuck | promotion criterion uses the platform reached, lane is 3 m; loosen lateral tolerance |
| 12 | Physics buffer overflow on dense step contacts | counter in the log; the run raises (C9 control, now also in the training env) | size the buffers from a measured peak |
| 13 | New ONNX not loadable by `LowLevel` | the load test | map by structure, not by the exporter's generated names |
| 14 | GPU physics is not bit-reproducible `[verify]` | repeat of a fixed eval differs | seeds fix draws, not trajectories; report with CIs, say so |
| 15 | Nav stack cannot use the new steps | the compass route ends at the stairs (section 10.3) | the nav perception fix is part of M2, with a test |

## 8. Pre-registered pass criteria and evaluation protocol

Defined before training. Changing a threshold after seeing results needs a dated note in the run's report.

### 8.1 Tiers (what is reported if 0.25 m is not reached)

| Tier | Riser (collision height) | Built with |
|---|---|---|
| 3 (target) | 0.25 m | bottom slab / half block |
| 2 | 0.1875 m | 4 snow layers `[verify]` |
| 1 | 0.125 m | 3 snow layers |

The arena's STAIRS section uses the highest tier that passes the stairs gate and its label states the riser.

### 8.2 Stairs gate

A **stairs trial** is one flight, one robot, start at the foot of the flight (up) or on the top platform (down), command along the flight; **success** = reaches the far platform (xy inside it, pelvis at least 0.70 m above the platform surface) within 3x the nominal time, no fall (torso more than 70 degrees from upright, or pelvis lower than 0.45 m above the local ground), no knee / shin / hand contact with terrain for more than 0.2 s, lateral deviation under 1.0 m.

- **Headline cell:** riser at the tier under test, tread 0.5 m (slab), 6-step flights, up and down reported separately, speed 0.45 m/s, start perturbations on (lateral offset up to 0.2 m, heading up to 0.3 rad), **n = 256 trials per direction** (32 seeds x 8 perturbations). **Pass:** observed success at least 95 % **and** the lower end of the 95 % Wilson interval at least 90 %, for up **and** for down.
- **Grid (reported, not gated):** risers {0.0625, 0.125, 0.1875, 0.25, 0.275, 0.30} x treads {0.25, 0.5, 1.0} x flights {2, 6} x speeds {0.3, 0.45, 0.6} x up/down, 64 trials per cell; the failure frontier is a result.
- **Held-out geometry (reported, gated at half the headline margin):** treads 0.35, 0.65, 0.8 m, riser jitter 0.02 m, a non-uniform flight (risers 0.2 / 0.25 / 0.3 mixed).
- **Seam test:** the headline cell on the arena's merged boxes versus one box per block; the difference must have a 95 % CI that excludes a loss of more than 5 points.

### 8.3 Regression gate (the flat skills and the arena as it is)

Paired design: base policy v0 and the candidate run on **identical seeds** (same terrain, commands, randomisation draws, pushes), so differences are not seed noise.

| Set | What | Pass |
|---|---|---|
| R1 original task | the ported flat velocity task at the deployed command ranges (`deploy.yaml`: vx -0.5 to 1.0, vy +-0.5, yaw rate +-1.0), 32 seeds x 256 envs, 20 s episodes, original randomisation and pushes. Metrics: survival, RMS linear and yaw-rate tracking error, stand-still joint motion, foot slip, action rate | survival: lower 95 % bound of (candidate - base) at least -2 points; tracking error ratio candidate/base: upper 95 % bound at most 1.10; the base itself must survive at least 90 %, otherwise the comparison is flagged as meaningless |
| R2 arena v2 sections | every non-stairs section (RAMP, BALANCE BEAM, SLALOM, DOORWAY, ICE PATCH, LAVA BRIDGE, PUSH ZONE) entered at the section start with the scripted pursuit controller, the `verify_arena_v2.py` perturbations (offset +-0.2 m, heading +-0.3 rad, speeds 0.35 / 0.45 / 0.55 m/s, pushes on), 96 trials per section; plus the whole lap on arena v2 with the ramp | per section: paired success difference lower 95 % bound at least -3 points; maximum lateral deviation (median) at most 1.25x the base; whole-lap success not lower |
| R3 envelope | the limit probes of fit.md and `probe_ice_push.py` on the same grids: ice friction at which it falls, push force at which it falls (left, right), 1-block beam, micro-step ramps | each threshold not worse than the base by more than one grid step |
| R4 behavioural distance | mean and 95th percentile of the action difference candidate vs base on 100 000 flat-tile observations from base rollouts | reported (feeds C10), no threshold |

### 8.4 Seeds and intervals

- **Seed sets:** training seeds {0, 1, 2}; validation (checkpoint selection) 2000 to 2007; **test 1000 to 1031, used once for the final report**. No test seed is used in training, tuning or selection.
- **Training repeats:** the final recipe is trained with seed 0 for the whole pipeline; seeds 1 and 2 are added **only if budget allows**, shortened if needed. If only one training seed exists the report says "one training seed" and the CIs cover evaluation noise, not training-run variance.
- **Intervals:** binary outcomes: Wilson 95 % for a single proportion; paired differences: bootstrap over seeds (10 000 resamples, percentile). Continuous metrics: bootstrap over seeds. `n` is printed next to every number. The stats code is tested on synthetic data (`test_m2_eval_stats.py`: known proportion, coverage).
- **Determinism:** CPU MuJoCo evaluations are bit-reproducible (two runs, hashes recorded); GPU runs are re-run twice and the spread reported `[verify]`.

## 9. What changes in the code (list; none written yet)

| Area | Change |
|---|---|
| New | `scripts/m2_import_actor.py`, `scripts/train_m2.py`, `scripts/eval_m2.py`, `server/mcrobot/m2/` (ported task, terrain, rewards, anchor loss, eval stats) |
| `server/mcrobot/arena_v2.py` | `STAIRS_PROFILE` / `STAIRS_BLOCKS` replaced by a step list; `STAIRS_RISE_M`, a `stairs` mode `"ramp"` (default until the gates pass) or `"real"`; label sub-line from the riser (`(0.25 m steps)`); `scripts/export_arena_v2.py --stairs real` writes `results/m2/arena_v2_stairs/` so the ramp results of M1 stay comparable |
| `scripts/gen_arena_v2_datapack.py` | slab placement (`type=bottom`), regenerated function; `test_arena_v2.py` keeps the no-redstone rule |
| `server/mcrobot/navtask.py` | walkability from **local step height** between neighbouring cells instead of height above the global floor (section 10.3) |
| `server/mcrobot/g1.py`, `navenv.py` | policy path from `MCROBOT_G1_POLICY` (default: the base) so the base stays the default until M2 passes |
| `scripts/verify_arena_v2.py` | pelvis height checked against the local ground, `--stairs real` |
| Tests | `test_m2_import.py`, `test_m2_terrain.py`, `test_m2_arena_stairs.py`, `test_m2_eval_stats.py`, additions to `test_navtask.py` and `test_navenv.py` |

## 10. The arena change and the nav retraining

### 10.1 The STAIRS section

Today (arena v2): section 8, x 118 to 137, 20 blocks, 6 blocks (3 m) wide, 4 levels of 3.1 cm (`STAIRS_PROFILE`), label sub-line "(ramp until M2)".
Planned (0.5 m per block): **6 steps up, an 8-block platform, 6 steps down** = 6 + 8 + 6 = 20 blocks. Step k has its top at k half-blocks (0.25 m x k): a bottom slab, a full block, a full block and a slab, and so on; tread 1 block (0.5 m), riser 0.25 m, platform height 1.5 m. Same lane width, same blue floor band, same gate; label sub-line becomes "(0.25 m steps)" (taken from `STAIRS_RISE_M`). The exporter reads the real collision shapes from the game, so slabs are exact partial boxes (`PLAN.md` "Block -> collider merge"); the byte-identical check against the synthetic export (smoke test) covers them.
Visual: the slab and the full blocks should read as one staircase in blue tones; the material choice (full blue concrete under a slab of a similar colour) is `[verify]` by eye in a client.
The 1.5 m platform stays under the gate and label heights; nothing else of the arena moves, so cams, push schedule and route x positions are unchanged.
If the stairs gate passes only at a lower tier (section 8.1) the step count and platform length are recomputed for that riser (a 0.1875 m riser needs 8 steps for 1.5 m) and the section still fits in 20 blocks with a shorter platform.

### 10.2 Rollout of the change

`stairs` mode stays `"ramp"` and the v2 results remain valid until the M2 gates pass; then the default flips to `"real"`, the ramp stays selectable for regression runs. The live Minecraft server is **not** touched by this work (deploy is the owner's or the organizer's step: regenerate the data pack, rebuild the arena).

### 10.3 The navigator cannot see steps as walkable today

`navtask.WALL_ABOVE = 0.25`: a cell whose top is more than 0.25 m above the global floor is a **wall** (for the free-space and the geodesic route and for the 16 wall rays). The second step of a 0.25 m staircase is already a wall by that rule, and the platform at 1.5 m is a wall. Without a change the compass route does not pass the stairs, whatever the low-level policy can do.
Fix (part of M2, tested): classify walkability from the **step height between neighbouring cells** (walkable when every move between adjacent walkable cells rises or drops by at most `STEP_MAX` = riser + 0.02 m); a cell is a wall when it is too tall relative to every neighbour (pillars, gate posts, lintel walls). Stair cells then count as FLOOR, so the stored `ground` height (it is `floor_z` for every non-FLOOR cell today) returns their tops for the above-ground lookup (`navtask.py` line 232) and for the pelvis-above-ground test of `verify_arena_v2.py`. The goal and hazard rules read block tops, not `floor_z`, and need no change. The geodesic field then runs through the stairs. Tests: the route of arena v2 with real stairs is connected from START to FINISH; pillars of the SLALOM and the DOORWAY walls stay walls; the ramp mode produces the same map as today (byte-identical route).

### 10.4 Export and nav retraining

1. The fine-tuned policy is exported to ONNX (`results/m2/ft/final.onnx`); `MCROBOT_G1_POLICY` selects it. `LowLevel` and `G1Walker` need no code change under plan A.
2. **Feasibility before learning:** the scripted lap (`verify_arena_v2.py --stairs real`, perturbed, 48 laps) with the new policy: pass at least 46 of 48 (95 %), Wilson interval reported. It uses the compass-following controller with no learning, so it isolates the low-level policy.
3. **Nav retraining:** the high level is a 45-input policy; its input layout is unchanged, so the v2 ramp checkpoint (if one exists by then) can warm-start. Control: a from-scratch nav run with the same budget; if the warm start is not better, use from scratch. Trained on the real-stairs arena in the same preemptible way as v2 (`train_nav.py`); the nav env needs the per-section speed cap only if the evaluation shows falls on the stairs at the speeds the navigator commands.
4. Evaluate with `scripts/eval_nav.py` from the command block on 20 held-out seeds, per-section failures, plus the scripted baseline: success with 95 % Wilson intervals, time to finish, falls per section; compared with the v2 ramp results of M1 (labelled as ramp results).
5. Ghosts and `/robot play` use the CPU reference (`G1Walker`) with the new ONNX; they show the G1 climbing the 0.25 m steps in the game. Nothing else in the mod changes.

## 11. Critiques and their tests

C10, C11 and C12 in `CRITIQUES.md` (each with a planned test and its status "designed"); C13 records the nav-perception finding of section 10.3.

## 12. Schedule and gates

| Gate | Work | Pass | If not |
|------|------|------|--------|
| G0 | import script, identity test, terrain generator and its geometry test, arena step list and tests, `navtask` fix and tests (CPU only; can run now, no GPU) | tests green; identity within 1e-5 | fix before any GPU time |
| G1 | environment port (T2), fidelity gate against `G1Walker`; throughput and VRAM probe (short GPU job) | fidelity within CIs; throughput measured, `--need-mib` set | T1, then T3; if throughput makes the budget impossible, shrink the plan and say so |
| G2 | P1 critic warm-up, P2 curriculum ramp | levels advance; R1 holds along the way (checked every 100 iterations on validation seeds) | tune std / LR / anchor; stop at the cap |
| G3 | P3 consolidation, full stairs and regression gates on validation seeds | both gates pass | plan B (section 4.5), then fallback F; or report the tier reached |
| G4 | test seeds, once; controls for C10 / C12 | report written to `results/m2/` | |
| G5 | arena `real` mode, scripted lap, nav retraining and evaluation | section 10.4 items 2 and 4 | report per-section failures |

Ordering note: G0 does not need the GPU and can run while the GPU is busy; the first GPU job is the G1 probe, after the T5 arms and nav v2 training the owner has already queued.
