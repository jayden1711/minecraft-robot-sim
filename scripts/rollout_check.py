"""Plain-MuJoCo rollout of an exported MJCF: needs only mujoco + numpy (no mcrobot, no Minecraft, no sockets).

Drops a seeded ball with random velocity into the scene and prints a sha256 of the whole qpos trajectory.
usage: rollout_check.py model.xml [seed] [steps]
"""

import hashlib
import sys

import mujoco
import numpy as np


def rollout(xml_text: str, seed: int, steps: int) -> tuple[str, np.ndarray]:
    probe = '<body name="probe" pos="1.5 -1.5 2.5"><freejoint/><geom type="sphere" size="0.15" mass="1"/></body></worldbody>'
    m = mujoco.MjModel.from_xml_string(xml_text.replace("</worldbody>", probe))
    d = mujoco.MjData(m)
    rng = np.random.default_rng(seed)
    d.qvel[:3] = rng.uniform(-2, 2, 3)
    traj = np.empty((steps, m.nq))
    for i in range(steps):
        mujoco.mj_step(m, d)
        traj[i] = d.qpos
    return hashlib.sha256(traj.tobytes()).hexdigest(), d.qpos.copy()


if __name__ == "__main__":
    xml = open(sys.argv[1]).read()
    h, q = rollout(xml, int(sys.argv[2]) if len(sys.argv) > 2 else 0, int(sys.argv[3]) if len(sys.argv) > 3 else 1500)
    print(h, *[f"{v:.4f}" for v in q[:3]])
