"""Task spec from marker blocks (see PLAN.md 'Reward spec from blocks'). Pure numpy; evaluated per env in training."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mjcf import Scene
from .region import Kind

GOAL_MARGIN = 0.35  # m, horizontal and below: robot base within this of the gold block's box counts as reached
HAZARD_MARGIN = 0.10  # m, horizontal margin around red wool / lava boxes
ABOVE_MARGIN = 0.5  # m, how far above the block top still counts (standing on top of it)
PLATE_RADIUS = 0.5  # m, horizontal distance to a plate center


@dataclass
class TaskSpec:
    goals: np.ndarray  # (n, 2, 3) boxes as [lo, hi] in MuJoCo frame
    hazards: np.ndarray
    checkpoints: np.ndarray
    spawns: np.ndarray  # (n, 3) top-center of command blocks

    @classmethod
    def from_scene(cls, scene: Scene) -> "TaskSpec":
        def boxes(kind):
            ms = [m for m in scene.markers if m.kind == kind]
            return np.array([[m.center - m.half, m.center + m.half] for m in ms]).reshape(-1, 2, 3)

        sp = boxes(Kind.SPAWN)
        spawns = np.stack([(sp[:, 0, 0] + sp[:, 1, 0]) / 2, (sp[:, 0, 1] + sp[:, 1, 1]) / 2, sp[:, 1, 2]], 1) if len(sp) else np.zeros((0, 3))
        return cls(boxes(Kind.GOAL), boxes(Kind.HAZARD), boxes(Kind.CHECKPOINT), spawns)

    @staticmethod
    def _inside(p: np.ndarray, b: np.ndarray, margin_xy: float, margin_below: float, margin_above: float) -> np.ndarray:
        """p (3,), b (n,2,3) -> bool (n,)"""
        lo, hi = b[:, 0], b[:, 1]
        ok_xy = ((p[:2] >= lo[:, :2] - margin_xy) & (p[:2] <= hi[:, :2] + margin_xy)).all(1)
        ok_z = (p[2] >= lo[:, 2] - margin_below) & (p[2] <= hi[:, 2] + margin_above)
        return ok_xy & ok_z

    def reached_goal(self, p: np.ndarray) -> bool:
        return bool(len(self.goals) and self._inside(p, self.goals, GOAL_MARGIN, GOAL_MARGIN, ABOVE_MARGIN).any())

    def in_hazard(self, p: np.ndarray) -> bool:
        return bool(len(self.hazards) and self._inside(p, self.hazards, HAZARD_MARGIN, 0.0, ABOVE_MARGIN).any())

    def checkpoint_hit(self, p: np.ndarray) -> np.ndarray:
        """bool (n_checkpoints,): which plates the robot is currently on."""
        if not len(self.checkpoints):
            return np.zeros(0, bool)
        c = (self.checkpoints[:, 0] + self.checkpoints[:, 1]) / 2
        near_xy = np.linalg.norm(p[:2] - c[:, :2], axis=1) <= PLATE_RADIUS
        z_ok = (p[2] >= self.checkpoints[:, 0, 2] - 0.1) & (p[2] <= self.checkpoints[:, 1, 2] + ABOVE_MARGIN)
        return near_xy & z_ok
