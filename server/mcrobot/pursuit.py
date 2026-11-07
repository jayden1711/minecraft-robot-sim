"""Scripted command generator for the G1 demo and the arena check: turn in place towards the next corner, then walk the leg with
pure pursuit. No learning; it only produces (vx, vy, yaw rate) for the frozen walking policy."""

from __future__ import annotations

import math

import numpy as np


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def drop_collinear(points):
    """Waypoints without the intermediate ones that lie on a straight line (labels in a route must not make the G1 stop and turn)."""
    pts = [np.asarray(p, float)[:2] for p in points]
    keep = [0]
    for i in range(1, len(pts) - 1):
        a, b, c = pts[keep[-1]], pts[i], pts[i + 1]
        if abs((b - a)[0] * (c - b)[1] - (b - a)[1] * (c - b)[0]) > 1e-6:
            keep.append(i)
    keep.append(len(pts) - 1)
    return [pts[i] for i in keep]


def route_progress(wp, k: int, p) -> float:
    """Fraction (0..1) of the polyline wp walked when the robot is at p and the controller is on leg k (wp[k-1] -> wp[k])."""
    seg = [float(np.linalg.norm(np.asarray(b[:2], float) - np.asarray(a[:2], float))) for a, b in zip(wp, wp[1:])]
    total = sum(seg) or 1.0
    k = min(max(k, 1), len(wp) - 1)
    a, b = np.asarray(wp[k - 1][:2], float), np.asarray(wp[k][:2], float)
    u = (b - a) / (seg[k - 1] or 1.0)
    s = float(np.clip(np.dot(np.asarray(p[:2], float) - a, u), 0.0, seg[k - 1]))
    return (sum(seg[: k - 1]) + s) / total


class Pursuit:
    def __init__(self, waypoints, v: float = 0.45, reach: float = 0.25, blend: bool = False):
        """blend: at a waypoint whose next leg is within 0.35 rad of the current heading, carry on walking instead of the one-step stop
        that precedes every turn (needed for a densely sampled curved route such as the slalom; off by default)."""
        self.wp = list(waypoints)
        self.k = 1
        self.v = v
        self.reach = reach
        self.blend = blend
        self.mode = "turn"

    @property
    def done(self) -> bool:
        return self.k >= len(self.wp)

    def command(self, p, yaw):
        a, b = np.asarray(self.wp[self.k - 1][:2], float), np.asarray(self.wp[self.k][:2], float)
        leg = b - a
        L = float(np.linalg.norm(leg))
        u = leg / L
        s = float(np.dot(p[:2] - a, u))
        if L - s < self.reach:
            self.k += 1
            if self.blend and not self.done:
                nxt = np.asarray(self.wp[self.k][:2], float) - b
                if abs(wrap(math.atan2(nxt[1], nxt[0]) - yaw)) < 0.35:
                    self.mode = "walk"
                    return self.command(p, yaw)
            self.mode = "turn"
            return np.zeros(3), False
        target = a + u * min(s + 0.5, L)
        err = wrap(math.atan2(*(target - p[:2])[::-1]) - yaw)
        if self.mode == "turn":
            if abs(wrap(math.atan2(u[1], u[0]) - yaw)) < 0.35:
                self.mode = "walk"
            return np.array([0.0, 0.0, float(np.clip(1.8 * wrap(math.atan2(u[1], u[0]) - yaw), -1.0, 1.0))]), True
        vx = self.v * float(np.clip((L - s) / 0.8, 0.5, 1.0)) * (1.0 if abs(err) < 0.5 else 0.4)
        return np.array([vx, 0.0, float(np.clip(2.0 * err, -1.0, 1.0))]), True
