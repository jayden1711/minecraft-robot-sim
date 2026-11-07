"""Scheduled external pushes on the pelvis (the arena's PUSH ZONE). Simulation only: a force on the pelvis body in MuJoCo, applied when the
pelvis first passes a trigger x; the in-game pistons animate from the cue events this schedule reports. No randomness: the same pelvis
track gives the same forces and cues, so a lap is repeatable and the navigation training can use identical pushes.

A push is a dict (the arena's route.json "pushes" entries): {"cue": int, "x_m": trigger x (MuJoCo m), "dir": [dx, dy] unit vector in the
MuJoCo frame (x east, y north), "force_n": float, "duration_s": float}.
"""

from __future__ import annotations

import numpy as np


class PushSchedule:
    def __init__(self, pushes):
        self.pushes = sorted((dict(p) for p in pushes), key=lambda p: p["x_m"])
        self.reset()

    def reset(self) -> None:
        self._fired = [False] * len(self.pushes)
        self._until = [-1.0] * len(self.pushes)

    def update(self, t: float, pelvis_xy) -> tuple[np.ndarray, list[int]]:
        """(force on the pelvis in N, MuJoCo world frame; cues newly fired at this call). Call once per policy step with the sim time."""
        force = np.zeros(3)
        fired = []
        for i, p in enumerate(self.pushes):
            if not self._fired[i] and pelvis_xy[0] >= p["x_m"]:
                self._fired[i] = True
                self._until[i] = t + p["duration_s"]
                fired.append(int(p["cue"]))
            if self._fired[i] and t < self._until[i]:
                force += p["force_n"] * np.array([p["dir"][0], p["dir"][1], 0.0])
        return force, fired
