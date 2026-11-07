"""Batched G1 navigation environment on MuJoCo Warp (the physics backend of mjlab), simulation only.

Low level: the frozen unitree_rl_mjlab velocity policy (g1.ONNX), ported to a torch MLP and run for every env every 20 ms with the
same PD gains, default pose, action scale and 4 x 5 ms physics substeps as g1.G1Walker (the CPU reference the tests compare with).
High level: the learned navigation policy outputs a velocity command (vx, vy, yaw rate) every NAV_REPEAT low-level steps (0.1 s).
Task: reach the gold pad (+10, ends), do not fall into lava / off the course / over (-10, ends), pressure plates pay +1 once each,
dense shaping by progress along the shortest walkable path (navtask). Episodes start on the command block or on random free cells.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import torch

from . import g1, navtask

NAV_REPEAT = 5  # low-level steps per navigation step: 0.1 s
CMD_LO = (-0.3, -0.3, -1.0)  # vx, vy, yaw rate the navigation policy may ask for (inside g1.CMD_RANGE)
CMD_HI = (0.8, 0.3, 1.0)
CMD_SMOOTH = 0.5  # command = SMOOTH * previous + (1 - SMOOTH) * requested
OBS_DIM = 45
STAND_Z = 0.80  # reset pelvis height above the surface
FALL_UPRIGHT, FALL_BELOW_SURFACE = 0.4, 0.40  # tipped over / pelvis this close to the surface it should stand on
VOID_DROP = 0.30  # pelvis below the floor level by this much: it has left the course
R_PROGRESS, R_PLATE, R_GOAL, R_FAIL, R_TIME, R_SMOOTH = 1.0, 1.0, 10.0, -10.0, -0.005, -0.02
# Per-world MuJoCo Warp buffers (constraint rows, contacts). Sized from the measured peaks on arena v2 (scripts/probe_nav_overflow.py)
# with headroom; any overflow raises PhysicsOverflow because dropped constraints mean wrong physics.
NJMAX, NCONMAX = 900, 160


class PhysicsOverflow(RuntimeError):
    pass


class LowLevel(torch.nn.Module):
    """The frozen ONNX walking policy: (obs - mean) / std, three ELU layers, linear head."""

    def __init__(self, onnx_path=g1.ONNX, device="cpu"):
        super().__init__()
        import onnx
        from onnx import numpy_helper

        init = {t.name: numpy_helper.to_array(t) for t in onnx.load(str(onnx_path)).graph.initializer}
        self.register_buffer("mean", torch.tensor(init["obs_normalizer._mean"][0]))
        self.register_buffer("std", torch.tensor(init["onnx::Div_24"][0]))
        self.layers = torch.nn.ModuleList()
        for i in (0, 2, 4, 6):
            lin = torch.nn.Linear(init[f"mlp.{i}.weight"].shape[1], init[f"mlp.{i}.weight"].shape[0])
            lin.weight.data = torch.tensor(init[f"mlp.{i}.weight"])
            lin.bias.data = torch.tensor(init[f"mlp.{i}.bias"])
            self.layers.append(lin)
        self.requires_grad_(False)
        self.to(device)

    @torch.no_grad()
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = (obs - self.mean) / self.std
        for lin in self.layers[:-1]:
            x = torch.nn.functional.elu(lin(x))
        return self.layers[-1](x)


def load_pushes(xml_path) -> list:
    """The arena's scheduled pushes: "pushes" of arena_route.json next to the region MJCF (none for arenas without one)."""
    route = Path(xml_path).with_name("arena_route.json")
    return list(json.loads(route.read_text()).get("pushes", [])) if route.exists() else []


def quat_yaw(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def wrap(a: torch.Tensor) -> torch.Tensor:
    return (a + math.pi) % (2 * math.pi) - math.pi


class NavEnv:
    def __init__(self, xml_path, n_envs: int, device: str = "cuda:0", seed: int = 0, nconmax: int = NCONMAX, njmax: int = NJMAX,
                 p_spawn: float = 0.2, task: navtask.NavTask | None = None, n_spawn_forced: int = 0, pushes=None):
        import warp as wp
        import mujoco_warp as mjw

        self.wp, self.mjw = wp, mjw
        self.region_xml = str(xml_path)
        self.n, self.dev = n_envs, torch.device(device)
        wp.set_device(device)
        self.cuda = self.dev.type == "cuda"
        if self.cuda:  # torch ops share warp's stream so the aliased arrays are never read mid-write
            torch.cuda.set_stream(torch.cuda.ExternalStream(wp.get_stream(device).cuda_stream))
        self.gen = torch.Generator(device=self.dev)
        self.gen.manual_seed(seed)
        self.p_spawn = p_spawn
        self.n_spawn_forced = n_spawn_forced  # the first envs always start on the command block (the ghosts)

        self.task = task or navtask.NavTask.from_xml(xml_path)
        self.tt = self.task.to(self.dev)
        self.floor_z = self.task.floor_z
        self.model = g1.build_model(xml_path)
        m = self.model
        self.ll = LowLevel(device=self.dev)
        f = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=self.dev)  # noqa: E731
        self.default, self.kp, self.kd, self.scale, self.effort = f(g1.DEFAULT), f(g1.KP), f(g1.KD), f(g1.ACTION_SCALE), f(g1.EFFORT)
        self.q_idx = torch.as_tensor([m.jnt_qposadr[m.joint(n).id] for n in g1.JOINTS], device=self.dev)
        self.v_idx = torch.as_tensor([m.jnt_dofadr[m.joint(n).id] for n in g1.JOINTS], device=self.dev)
        self.act_idx = torch.as_tensor([m.actuator(n).id for n in g1.JOINTS], device=self.dev)
        free = m.joint("floating_base_joint").id
        self.qa, self.va = int(m.jnt_qposadr[free]), int(m.jnt_dofadr[free])
        self.cmd_lo, self.cmd_hi = f(CMD_LO), f(CMD_HI)
        self.cmd_clip_lo = f([r[0] for r in g1.CMD_RANGE])
        self.cmd_clip_hi = f([r[1] for r in g1.CMD_RANGE])
        self.pushes = load_pushes(xml_path) if pushes is None else list(pushes)
        npush = len(self.pushes)
        self.push_x = f([p["x_m"] for p in self.pushes])
        self.push_f = f([[p["dir"][0] * p["force_n"], p["dir"][1] * p["force_n"], 0.0] for p in self.pushes]).reshape(npush, 3)
        self.push_steps = torch.as_tensor([max(1, round(p["duration_s"] / (g1.CTRL_DT * NAV_REPEAT))) for p in self.pushes], dtype=torch.long, device=self.dev)
        self.root = int(m.body("pelvis").id)

        d0 = mujoco.MjData(m)
        mujoco.mj_resetData(m, d0)
        d0.qpos[self.qa + 3] = 1.0
        d0.qpos[self.q_idx.cpu().numpy()] = g1.DEFAULT
        d0.qpos[self.qa:self.qa + 3] = [self.task.spawn_xy[0], self.task.spawn_xy[1], self.floor_z + STAND_Z]
        mujoco.mj_forward(m, d0)
        self.wm = mjw.put_model(m)
        self.wd = mjw.put_data(m, d0, nworld=n_envs, nconmax=nconmax, njmax=njmax)
        self.qpos, self.qvel, self.ctrl = wp.to_torch(self.wd.qpos), wp.to_torch(self.wd.qvel), wp.to_torch(self.wd.ctrl)
        self.warm = wp.to_torch(self.wd.qacc_warmstart)
        self.overflow_flags, self.nefc_t, self.nacon_t = wp.to_torch(self.wd.overflow), wp.to_torch(self.wd.nefc), wp.to_torch(self.wd.nacon)
        self.njmax, self.nconmax = njmax, nconmax
        self.peak_nefc = torch.zeros((), dtype=torch.int32, device=self.dev)
        self.peak_ncon = torch.zeros((), dtype=torch.float32, device=self.dev)  # contacts per world, averaged over the worlds (the buffer is pooled)
        self.xfrc = wp.to_torch(self.wd.xfrc_applied)
        self.q_default_full = torch.as_tensor(d0.qpos, dtype=torch.float32, device=self.dev).clone()
        mjw.step(self.wm, self.wd)  # compile
        self.graph = None
        if self.cuda and wp.is_mempool_enabled(device):
            try:
                with wp.ScopedCapture() as cap:
                    mjw.step(self.wm, self.wd)
                self.graph = cap.graph
            except Exception as e:  # noqa: BLE001 - fall back to launching the kernels one by one
                import sys

                print(f"navenv: CUDA graph capture failed ({e}); stepping without a graph", file=sys.stderr)

        n = n_envs
        z = lambda *s, dt=torch.float32: torch.zeros(*s, dtype=dt, device=self.dev)  # noqa: E731
        self.phase, self.last_a, self.cmd = z(n), z(n, 29), z(n, 3)
        self.t, self.max_t = z(n, dt=torch.long), z(n, dt=torch.long)
        self.got = z(n, navtask.MAX_PLATES, dt=torch.bool)
        self.prev_field = z(n)
        self.ep_return, self.ep_len = z(n), z(n, dt=torch.long)
        self.start_d, self.start_x = z(n), z(n)
        self.from_spawn = z(n, dt=torch.bool)
        self.push_done, self.push_left = z(n, npush, dt=torch.bool), z(n, npush, dt=torch.long)
        self.reset(torch.arange(n, device=self.dev))

    # ---- physics --------------------------------------------------------------------------------------------------------------
    def _phys_step(self):
        if self.graph is not None:
            self.wp.capture_launch(self.graph)
        else:
            self.mjw.step(self.wm, self.wd)
        self.peak_nefc = torch.maximum(self.peak_nefc, self.nefc_t.max().to(torch.int32))
        self.peak_ncon = torch.maximum(self.peak_ncon, self.nacon_t[0].float() / self.n)

    def check_overflow(self):
        """Raise if any world overflowed a constraint or contact buffer in any substep so far (the flag is sticky)."""
        if self.overflow_flags.any().item():
            bad = int((self.overflow_flags != 0).sum())
            raise PhysicsOverflow(
                f"MuJoCo Warp buffer overflow in {bad} of {self.n} worlds (flags {sorted(set(self.overflow_flags.tolist()))}; peak nefc "
                f"{int(self.peak_nefc)} vs njmax {self.njmax}, peak contacts/world {float(self.peak_ncon):.0f} vs nconmax {self.nconmax}): "
                "constraints were dropped, the physics is wrong; raise NJMAX/NCONMAX")

    def _grav_ang(self):
        q = self.qpos[:, self.qa + 3:self.qa + 7]
        w, x, y, z = q.unbind(-1)
        grav = torch.stack([-2 * (x * z - w * y), -2 * (y * z + w * x), -(1 - 2 * (x * x + y * y))], -1)
        return grav, self.qvel[:, self.va + 3:self.va + 6]

    def _ll_obs(self, cmd: torch.Tensor) -> torch.Tensor:
        grav, angv = self._grav_ang()
        self.phase = (self.phase + g1.CTRL_DT / g1.GAIT_PERIOD) % 1.0
        ang = 2 * math.pi * self.phase
        still = torch.linalg.norm(cmd, dim=1) < 0.1
        ph = torch.stack([torch.sin(ang), torch.cos(ang)], -1) * (~still)[:, None]
        return torch.cat([angv, grav, cmd, ph, self.qpos[:, self.q_idx] - self.default, self.qvel[:, self.v_idx], self.last_a], -1)

    def _apply_pushes(self):
        """PushSchedule semantics, batched: a push starts when the pelvis first reaches its trigger x and lasts its duration (in nav steps)."""
        if not self.pushes:
            return
        trig = (self.pelvis()[:, 0:1] >= self.push_x[None]) & ~self.push_done
        self.push_done |= trig
        self.push_left = torch.where(trig, self.push_steps[None].expand_as(self.push_left), self.push_left)
        active = (self.push_left > 0).float()
        self.xfrc[:, self.root, :3] = active @ self.push_f
        self.push_left = (self.push_left - 1).clamp(min=0)

    def _ll_step(self, cmd: torch.Tensor):
        a = self.ll(self._ll_obs(cmd))
        self.last_a = a
        target = a * self.scale + self.default
        for _ in range(g1.DECIMATION):
            tau = self.kp * (target - self.qpos[:, self.q_idx]) - self.kd * self.qvel[:, self.v_idx]
            self.ctrl[:, self.act_idx] = torch.maximum(torch.minimum(tau, self.effort), -self.effort)
            self._phys_step()

    # ---- state ----------------------------------------------------------------------------------------------------------------
    def pelvis(self) -> torch.Tensor:
        return self.qpos[:, self.qa:self.qa + 3]

    def yaw(self) -> torch.Tensor:
        return quat_yaw(self.qpos[:, self.qa + 3:self.qa + 7])

    def upright(self) -> torch.Tensor:
        q = self.qpos[:, self.qa + 3:self.qa + 7]
        return 1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2)

    def obs(self) -> torch.Tensor:
        p, yaw = self.pelvis(), self.yaw()
        look = self.tt.look_at(p[:, :2])
        c, s = torch.cos(yaw), torch.sin(yaw)
        dir_b = torch.stack([c * look[:, 0] + s * look[:, 1], -s * look[:, 0] + c * look[:, 1]], -1)  # (forward, left) components
        d = self.tt.field_near(p[:, :2]).clamp(max=90.0) / 45.0
        wall, void = self.tt.rays(p[:, :2], yaw)
        v = self.qvel[:, self.va:self.va + 2]
        v_b = torch.stack([c * v[:, 0] + s * v[:, 1], -s * v[:, 0] + c * v[:, 1]], -1)
        grav, angv = self._grav_ang()
        plates = (self.got[:, : self.tt.n_p].float().mean(1, keepdim=True) if self.tt.n_p else torch.zeros(self.n, 1, device=self.dev))
        return torch.cat([dir_b, d[:, None], wall, void, v_b, angv[:, 2:3], grav, self.cmd / self.cmd_hi, plates], -1)

    # ---- reset ----------------------------------------------------------------------------------------------------------------
    def reset(self, idx: torch.Tensor):
        k = idx.numel()
        if k == 0:
            return
        tt, g = self.tt, self.gen
        cells = tt.start_cells
        far = tt.start_d > 1.0
        pool = torch.nonzero(far).squeeze(1)
        pick = pool[torch.randint(0, pool.numel(), (k,), device=self.dev, generator=g)]
        rc = cells[pick]
        xy = torch.stack([(rc[:, 1] + 0.5) * tt.res, -(rc[:, 0] + 0.5) * tt.res], -1) + (torch.rand(k, 2, device=self.dev, generator=g) - 0.5) * 0.15
        spawn = (torch.rand(k, device=self.dev, generator=g) < self.p_spawn) | (idx < self.n_spawn_forced)
        sp = torch.as_tensor(self.task.spawn_xy, dtype=torch.float32, device=self.dev)
        xy = torch.where(spawn[:, None], sp[None] + (torch.rand(k, 2, device=self.dev, generator=g) - 0.5) * 0.2, xy)
        look = tt.look_at(xy)
        yaw0 = torch.atan2(look[:, 1], look[:, 0])
        yaw0 = torch.where(look.abs().sum(1) < 1e-6, torch.zeros_like(yaw0), yaw0)
        yaw = yaw0 + torch.randn(k, device=self.dev, generator=g) * torch.where(spawn, 0.15, 0.5)
        z = tt.ground_at(xy) + STAND_Z
        q = self.q_default_full[None].repeat(k, 1)
        q[:, self.qa:self.qa + 2] = xy
        q[:, self.qa + 2] = z
        q[:, self.qa + 3] = torch.cos(yaw / 2)
        q[:, self.qa + 4:self.qa + 6] = 0.0
        q[:, self.qa + 6] = torch.sin(yaw / 2)
        self.qpos[idx] = q
        self.qvel[idx] = 0.0
        self.ctrl[idx] = 0.0
        self.warm[idx] = 0.0
        self.xfrc[idx] = 0.0
        self.push_left[idx] = 0
        self.push_done[idx] = xy[:, 0:1] >= self.push_x[None]  # a start beyond a trigger never gets that push
        self.phase[idx] = 0.0
        self.last_a[idx] = 0.0
        self.cmd[idx] = 0.0
        self.got[idx] = False
        self.t[idx] = 0
        self.ep_return[idx] = 0.0
        self.ep_len[idx] = 0
        d = tt.field_near(xy)
        self.prev_field[idx] = d
        self.start_d[idx] = d
        self.start_x[idx] = xy[:, 0]
        self.from_spawn[idx] = spawn
        self.max_t[idx] = ((15.0 + d.clamp(max=90.0) / 0.3) / (g1.CTRL_DT * NAV_REPEAT)).long()  # 15 s plus a 0.3 m/s walk from the start

    # ---- step -----------------------------------------------------------------------------------------------------------------
    def step(self, action: torch.Tensor):
        """action (n, 3) in [-1, 1]. Returns obs, reward, done, info; finished envs are reset (obs is of the new episode, info['final_obs']
        of the one that ended)."""
        a = action.clamp(-1, 1)
        target = self.cmd_lo + (a + 1) * 0.5 * (self.cmd_hi - self.cmd_lo)
        new_cmd = CMD_SMOOTH * self.cmd + (1 - CMD_SMOOTH) * target
        new_cmd = torch.maximum(torch.minimum(new_cmd, self.cmd_clip_hi), self.cmd_clip_lo)
        smooth = ((new_cmd - self.cmd) / self.cmd_hi).pow(2).sum(1)
        self.cmd = new_cmd
        self._apply_pushes()
        for _ in range(NAV_REPEAT):
            self._ll_step(self.cmd)
        self.check_overflow()
        self.t += 1
        self.ep_len += 1

        p = self.pelvis()
        tt = self.tt
        field = tt.field_near(p[:, :2])
        valid = (field < navtask.UNREACHABLE) & (self.prev_field < navtask.UNREACHABLE)
        prog = torch.where(valid, (self.prev_field - field).clamp(-1.0, 1.0), torch.zeros_like(field))
        self.prev_field = field
        on = tt.on_plates(p) & ~self.got
        self.got |= on
        goal = tt.reached_goal(p)
        lava = tt.in_hazard(p)
        surf = tt.ground_at(p[:, :2])
        fell = (self.upright() < FALL_UPRIGHT) | (p[:, 2] < surf + FALL_BELOW_SURFACE) | (p[:, 2] < self.floor_z - VOID_DROP)
        fail = (lava | fell) & ~goal
        timeout = (self.t >= self.max_t) & ~goal & ~fail
        reward = R_PROGRESS * prog + R_PLATE * on.sum(1) + R_GOAL * goal + R_FAIL * fail + R_TIME + R_SMOOTH * smooth
        done = goal | fail | timeout
        self.ep_return += reward
        info = {"success": goal, "lava": lava & fail, "fell": fell & fail & ~lava, "timeout": timeout, "plates": self.got.sum(1), "ep_len": self.ep_len.clone(),
                "ep_return": self.ep_return.clone(), "from_spawn": self.from_spawn.clone(), "start_d": self.start_d.clone(), "done": done, "fail": fail,
                "x": p[:, 0].clone(), "start_x": self.start_x.clone()}
        ids = torch.nonzero(done).squeeze(1)
        if ids.numel():
            info["final_obs"] = self.obs()
            self.reset(ids)
        return self.obs(), reward, done, info

    # ---- ghosts ---------------------------------------------------------------------------------------------------------------
    def body_state(self, env_ids) -> list:
        """MjData-like views (xpos, xmat, xquat as numpy) of the given envs, for g1.Rig.payload (positions of the last physics substep)."""
        ids = torch.as_tensor(env_ids, device=self.dev)
        xpos = self.wp.to_torch(self.wd.xpos)[ids].cpu().numpy()
        xmat = self.wp.to_torch(self.wd.xmat)[ids].cpu().numpy()
        xquat = self.wp.to_torch(self.wd.xquat)[ids].cpu().numpy()
        return [SimpleNamespace(xpos=xpos[i], xmat=xmat[i].reshape(xmat.shape[1], 9), xquat=xquat[i]) for i in range(len(ids))]
