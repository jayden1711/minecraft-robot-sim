"""Unitree G1 (29 DoF) walking in a region scene, plus the ghost rig that shows it in Minecraft. MuJoCo is the physics.

Low-level policy: the published `unitree_rl_mjlab` velocity policy v0 (Apache-2.0), run FROZEN through onnxruntime. Its gains,
default pose, action scale, observation layout and command ranges are copied from the repo's own deploy.yaml; the G1 model is
that repo's own `g1.xml` (Apache-2.0, derived from Unitree's BSD-3 description). Nothing from there is committed here: the
files are read from `MCROBOT_G1_ROOT` (default: the robot-arena-sim checkout's third_party/unitree_rl_mjlab). See THIRD_PARTY.md.
Physics settings follow mjlab: 5 ms timestep, 4 substeps per 20 ms policy step, explicit PD torques.
"""

from __future__ import annotations

import json
import math
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

G1_ROOT = Path(os.environ.get("MCROBOT_G1_ROOT", Path.home() / "projects/robot-arena-sim/third_party/unitree_rl_mjlab"))
XML = G1_ROOT / "src/assets/robots/unitree_g1/xmls/g1.xml"
ONNX = G1_ROOT / "deploy/robots/g1/config/policy/velocity/v0/exported/policy.onnx"

PHYS_DT = 0.005
DECIMATION = 4
CTRL_DT = PHYS_DT * DECIMATION
CMD_RANGE = ((-0.5, 1.0), (-0.5, 0.5), (-1.0, 1.0))  # vx, vy, yaw rate (deploy.yaml)
GAIT_PERIOD = 0.6

JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint", "left_ankle_pitch_joint",
    "left_ankle_roll_joint", "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint", "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint", "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
    "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
KP = np.array([40.2, 99.1, 40.2, 99.1, 28.5, 28.5, 40.2, 99.1, 40.2, 99.1, 28.5, 28.5, 40.2, 28.5, 28.5,
               14.3, 14.3, 14.3, 14.3, 14.3, 16.8, 16.8, 14.3, 14.3, 14.3, 14.3, 14.3, 16.8, 16.8])
KD = np.array([2.6, 6.3, 2.6, 6.3, 1.8, 1.8, 2.6, 6.3, 2.6, 6.3, 1.8, 1.8, 2.6, 1.8, 1.8,
               0.9, 0.9, 0.9, 0.9, 0.9, 1.1, 1.1, 0.9, 0.9, 0.9, 0.9, 0.9, 1.1, 1.1])
DEFAULT = np.array([-0.1, 0, 0, 0.3, -0.2, 0, -0.1, 0, 0, 0.3, -0.2, 0, 0, 0, 0,
                    0.35, 0.18, 0, 0.87, 0, 0, 0, 0.35, -0.18, 0, 0.87, 0, 0, 0])
ACTION_SCALE = np.array([0.55, 0.35, 0.55, 0.35, 0.44, 0.44, 0.55, 0.35, 0.55, 0.35, 0.44, 0.44, 0.55, 0.44, 0.44,
                         0.44, 0.44, 0.44, 0.44, 0.44, 0.07, 0.07, 0.44, 0.44, 0.44, 0.44, 0.44, 0.07, 0.07])
ARMATURE = KP / (2 * np.pi * 10.0) ** 2  # mjlab: stiffness = armature * (2 pi 10 Hz)^2


def _effort():
    lim = []
    for j in JOINTS:
        if "hip_roll" in j or "knee" in j:
            lim.append(139.0)
        elif "hip_pitch" in j or "hip_yaw" in j or j == "waist_yaw_joint":
            lim.append(88.0)
        elif "ankle" in j or j in ("waist_roll_joint", "waist_pitch_joint"):
            lim.append(50.0)
        elif "wrist_pitch" in j or "wrist_yaw" in j:
            lim.append(5.0)
        else:
            lim.append(25.0)
    return np.array(lim)


EFFORT = _effort()


def region_boxes(xml_path: str | Path):
    """Collider boxes (col_*, colp_*) of an exported region MJCF: list of (name, pos, half, friction, priority, condim)."""
    out = []
    for g in ET.parse(xml_path).getroot().iter("geom"):
        n = g.get("name", "")
        if n.startswith(("col_", "colp_")):
            out.append((n, [float(v) for v in g.get("pos").split()], [float(v) for v in g.get("size").split()],
                        float(g.get("friction", "1").split()[0]), int(g.get("priority", "0")), int(g.get("condim", "3"))))
    return out


def build_model(region_xml: str | Path | None = None, extra_boxes=()) -> mujoco.MjModel:
    """G1 (torque motors, mjlab armature) plus the region's collider boxes (friction 1) and any extra (pos, half) boxes."""
    spec = mujoco.MjSpec.from_file(str(XML))
    meshdir = XML.parent / spec.meshdir
    for m in spec.meshes:
        m.file = str((meshdir / Path(m.file).name).resolve())
    spec.meshdir = ""
    for coll in (spec.actuators, spec.keys):
        for x in list(coll):
            spec.delete(x)
    spec.option.timestep = PHYS_DT
    for i, name in enumerate(JOINTS):
        spec.joint(name).armature = float(ARMATURE[i])
        a = spec.add_actuator(name=name, target=name, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        a.gainprm[0] = 1.0
        a.ctrllimited = 1
        a.ctrlrange = [-float(EFFORT[i]), float(EFFORT[i])]
    wb = spec.worldbody
    boxes = list(region_boxes(region_xml)) if region_xml else []
    boxes += [(f"extra_{i}", p, h, 1.0, 0, 3) for i, (p, h) in enumerate(extra_boxes)]
    for n, p, h, mu, prio, condim in boxes:
        g = wb.add_geom(name=n, type=mujoco.mjtGeom.mjGEOM_BOX, pos=p, size=h)
        g.friction = [mu, 0.005, 0.0001]
        if prio:  # slippery blocks: priority above the robot's geoms, else the robot's friction wins (see mjcf.to_mjcf)
            g.priority = prio
            g.condim = condim
    return spec.compile()


class G1Walker:
    """One G1 in a model from build_model. step(cmd) advances one 20 ms policy step."""

    def __init__(self, model: mujoco.MjModel, onnx_path: str | Path = ONNX):
        import onnxruntime as rt

        self.m = model
        self.d = mujoco.MjData(model)
        so = rt.SessionOptions()
        so.intra_op_num_threads = so.inter_op_num_threads = 1
        self.sess = rt.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
        self.in_name = self.sess.get_inputs()[0].name
        jid = [model.joint(n).id for n in JOINTS]
        self.q_idx = np.array([model.jnt_qposadr[i] for i in jid])
        self.v_idx = np.array([model.jnt_dofadr[i] for i in jid])
        self.act_ids = np.array([model.actuator(n).id for n in JOINTS])
        self.root = model.body("pelvis").id
        free = model.joint("floating_base_joint").id
        self.qadr, self.vadr = model.jnt_qposadr[free], model.jnt_dofadr[free]
        self.foot_ids = [model.body(n).id for n in ("left_ankle_roll_link", "right_ankle_roll_link")]

    def reset(self, xy, yaw: float, z: float = 0.80):
        d = self.d
        mujoco.mj_resetData(self.m, d)
        d.qpos[self.qadr:self.qadr + 3] = [xy[0], xy[1], z]
        d.qpos[self.qadr + 3:self.qadr + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        d.qpos[self.q_idx] = DEFAULT
        self.target = DEFAULT.copy()
        self.last_action = np.zeros(29, np.float32)
        self.phase = 0.0
        mujoco.mj_forward(self.m, d)

    def obs(self, cmd) -> np.ndarray:
        d = self.d
        self.phase = (self.phase + CTRL_DT / GAIT_PERIOD) % 1.0
        ph = [math.sin(2 * math.pi * self.phase), math.cos(2 * math.pi * self.phase)]
        if np.linalg.norm(cmd) < 0.1:
            ph = [0.0, 0.0]
        R = d.xmat[self.root].reshape(3, 3)
        grav = R.T @ np.array([0.0, 0.0, -1.0])
        return np.hstack([d.qvel[self.vadr + 3:self.vadr + 6], grav, cmd, ph,
                          d.qpos[self.q_idx] - DEFAULT, d.qvel[self.v_idx], self.last_action]).astype(np.float32)

    def step(self, cmd) -> None:
        cmd = np.array([np.clip(cmd[i], *CMD_RANGE[i]) for i in range(3)], np.float32)
        a = self.sess.run(None, {self.in_name: self.obs(cmd)[None]})[0][0]
        self.last_action = a.astype(np.float32)
        self.target = a * ACTION_SCALE + DEFAULT
        d = self.d
        for _ in range(DECIMATION):
            tau = KP * (self.target - d.qpos[self.q_idx]) - KD * d.qvel[self.v_idx]
            d.ctrl[self.act_ids] = np.clip(tau, -EFFORT, EFFORT)
            mujoco.mj_step(self.m, d)

    def set_push(self, force) -> None:
        """External force (N, world frame) on the pelvis for the next steps; (0, 0, 0) clears it. Used by the arena's push zone."""
        self.d.xfrc_applied[self.root, :3] = force

    # state helpers (MuJoCo frame: x east, y north, z up)
    def pelvis_pos(self) -> np.ndarray:
        return self.d.qpos[self.qadr:self.qadr + 3].copy()

    def yaw(self) -> float:
        w, x, y, z = self.d.qpos[self.qadr + 3:self.qadr + 7]
        return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    def upright(self) -> float:
        """cos of the tilt of the pelvis z axis from vertical (1 = upright)."""
        return float(self.d.xmat[self.root].reshape(3, 3)[2, 2])

    def fallen(self) -> bool:
        return self.upright() < 0.5 or self.d.qpos[self.qadr + 2] < 0.4


# ---- ghost rig ---------------------------------------------------------------------------------------------
# Block-display boxes from the visual meshes of selected links: (body, block). Boxes are the AABB of the link's meshes in
# the link frame, so they rotate with the link. Colour by role; the head is distinct so the heading is readable.
GRAY, LGRAY, BLUE, BLACK, WHITE = ("minecraft:gray_concrete", "minecraft:light_gray_concrete", "minecraft:light_blue_concrete",
                                   "minecraft:black_concrete", "minecraft:white_concrete")
RIG = [  # (body, mesh names or None for all visual meshes of the body, block)
    ("pelvis", None, GRAY),
    ("torso_link", ("torso_link",), LGRAY),
    ("torso_link", ("head_link",), WHITE),
    ("left_hip_roll_link", None, GRAY), ("right_hip_roll_link", None, GRAY),
    ("left_hip_yaw_link", None, BLUE), ("right_hip_yaw_link", None, BLUE),
    ("left_knee_link", None, LGRAY), ("right_knee_link", None, LGRAY),
    ("left_ankle_roll_link", None, BLACK), ("right_ankle_roll_link", None, BLACK),
    ("left_shoulder_yaw_link", None, BLUE), ("right_shoulder_yaw_link", None, BLUE),
    ("left_elbow_link", None, LGRAY), ("right_elbow_link", None, LGRAY),
    ("left_wrist_yaw_link", None, BLACK), ("right_wrist_yaw_link", None, BLACK),
]


def link_aabb(model: mujoco.MjModel, body: str, meshes=None):
    """(min, max) of the body's visual-mesh vertices (optionally only the named meshes) in the body frame, MuJoCo axes."""
    b = model.body(body).id
    pts = []
    for g in range(model.ngeom):
        if model.geom_bodyid[g] != b or model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or model.geom_contype[g]:
            continue
        mid = model.geom_dataid[g]
        if meshes is not None and model.mesh(mid).name not in meshes:
            continue
        v = model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid] + model.mesh_vertnum[mid]]
        Rg = np.zeros(9)
        mujoco.mju_quat2Mat(Rg, model.geom_quat[g])
        pts.append(v @ Rg.reshape(3, 3).T + model.geom_pos[g])
    if not pts:
        raise ValueError(f"no visual mesh for {body} {meshes}")
    p = np.vstack(pts)
    return p.min(0), p.max(0)


class Rig:
    """Turns the walker's link poses into Minecraft block_display transformations (see GhostManager.java).

    Coordinates: MuJoCo (x, y, z) in metres relative to the region's minimum corner maps to Minecraft
    origin + (x, z, -y) / meters_per_block. A part is a unit cube [0,1]^3 in Minecraft local axes: scale = box size,
    left_rotation = link rotation, translation = position of the cube's minimum corner relative to the ghost's anchor.
    """

    def __init__(self, model: mujoco.MjModel, origin, anchor_mj, meters_per_block: float = 1.0, parts=RIG, style: str = "blocks"):
        self.m = model
        self.origin = np.asarray(origin, float)
        self.mpb = meters_per_block
        self.anchor_mc = self.mc_pos(np.asarray(anchor_mj, float))
        self.style = style
        if style == "models":
            spec = load_rig_models()
            self.model_scale = float(spec["model_scale_m"])
            self.bid = [model.body(p["body"]).id for p in spec["parts"]]
            self.mnames = [p["model"] for p in spec["parts"]]
            self.mcentre = [np.asarray(p["centre_link"], float) for p in spec["parts"]]
            return
        if style != "blocks":
            raise ValueError(f"unknown rig style {style!r}")
        self.bid = [model.body(n).id for n, _, _ in parts]
        # local box in link-frame MuJoCo axes -> cube min corner and size in "Minecraft-local" axes (x, z, -y)
        self.cmin, self.size, self.local = [], [], []
        for n, meshes, _ in parts:
            lo, hi = link_aabb(model, n, meshes)
            self.local.append((lo, hi))
            self.cmin.append(np.array([lo[0], lo[2], -hi[1]]))
            self.size.append(np.array([hi[0] - lo[0], hi[2] - lo[2], hi[1] - lo[1]]))
        self.blocks = [blk for _, _, blk in parts]

    def mc_pos(self, p_mj: np.ndarray) -> np.ndarray:
        return self.origin + np.array([p_mj[0], p_mj[2], -p_mj[1]]) / self.mpb

    def payload(self, d: mujoco.MjData, ghost: int = 0, ticks: int = 2, spawn: bool = False) -> dict:
        if self.style == "models":
            return self._payload_models(d, ghost, ticks, spawn)
        parts = []
        for k, b in enumerate(self.bid):
            R = d.xmat[b].reshape(3, 3)
            q = d.xquat[b]  # w x y z (MuJoCo)
            lrot = [q[1], q[3], -q[2], q[0]]  # Minecraft quaternion x y z w for the rotation M R M^T
            # the cube's minimum corner is the link-local MuJoCo point (x0, y1, z0), because Minecraft z = -y
            corner = d.xpos[b] + R @ np.array([self.cmin[k][0], -self.cmin[k][2], self.cmin[k][1]])
            t = self.mc_pos(corner) - self.anchor_mc
            s = self.size[k] / self.mpb
            parts.append({"b": self.blocks[k], "t": [round(float(v), 4) for v in t], "r": [round(float(v), 5) for v in lrot], "s": [round(float(v), 4) for v in s]})
        return {"op": "parts", "ghost": ghost, "ticks": ticks, "spawn": spawn, "style": "blocks",
                "anchor": [float(v) for v in self.anchor_mc], "parts": parts}

    def _payload_models(self, d: mujoco.MjData, ghost: int, ticks: int, spawn: bool) -> dict:
        """item_display parts: the entity sits at the model centre (link point centre_link), so t is that point minus the anchor."""
        s = round(self.model_scale / self.mpb, 5)
        parts = []
        for k, b in enumerate(self.bid):
            R = d.xmat[b].reshape(3, 3)
            q = d.xquat[b]
            lrot = [q[1], q[3], -q[2], q[0]]
            t = self.mc_pos(d.xpos[b] + R @ self.mcentre[k]) - self.anchor_mc
            parts.append({"m": self.mnames[k], "t": [round(float(v), 4) for v in t], "r": [round(float(v), 5) for v in lrot], "s": [s, s, s]})
        return {"op": "parts", "ghost": ghost, "ticks": ticks, "spawn": spawn, "style": "models",
                "anchor": [float(v) for v in self.anchor_mc], "parts": parts}


def load_rig_models(path: str | Path | None = None) -> dict:
    """rig_models.json written by scripts/build_pack.py (env MCROBOT_RIG_MODELS overrides the default under results/m1/pack/)."""
    p = Path(path or os.environ.get("MCROBOT_RIG_MODELS", Path(__file__).resolve().parents[2] / "results/m1/pack/rig_models.json"))
    if not p.exists():
        raise FileNotFoundError(f"rig models not found at {p}: run scripts/build_pack.py (or set MCROBOT_RIG_MODELS)")
    return json.loads(p.read_text())
