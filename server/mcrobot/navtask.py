"""Navigation task geometry from an exported region MJCF, rasterised once and queried in batch with torch (CPU or GPU).

From the collider boxes and marker sites: a ground grid (kind per cell: 0 void or pit, 1 floor, 2 wall; surface height), a geodesic
distance field to the gold block and a lookahead direction field (both used for reward shaping and as the policy's compass, never
as a map), the lava boxes, the pressure plates and the spawn. The policy's only view of the world is `rays`: distances along
N_RAYS directions to the nearest wall and to the nearest void or lava, sampled from the grid. MuJoCo metres, x east, y north (row = -y).
Goal, plate and hazard rules follow tasks.py (the goal rule is relative to the robot's base height).
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import numpy as np
import torch

from . import nav
from .tasks import ABOVE_MARGIN, G1_BASE_HEIGHT, GOAL_MARGIN, HAZARD_MARGIN, PLATE_RADIUS

ROBOT_RADIUS = 0.30  # m, clearance kept from walls and voids when building the free space for the geodesic fields
WALL_ABOVE, VOID_BELOW = 0.25, 0.30  # m: a cell whose top is this far above/below the floor is a wall/void
HEADROOM = 0.9  # m: boxes starting higher than this above the floor (the finish gate crossbar) do not block walking
N_RAYS, RAY_RANGE, RAY_STEP = 16, 3.0, 0.25
MAX_PLATES = 8
UNREACHABLE = 99.0
LOOKAHEAD_CELLS = 4
VOID, FLOOR, WALL = 0, 1, 2


@dataclass
class NavTask:
    res: float
    floor_z: float
    spawn_xy: np.ndarray
    rows: int
    cols: int
    kind: np.ndarray  # (rows, cols) uint8
    ground: np.ndarray  # (rows, cols) float, walking-surface height (floor_z where there is none)
    free: np.ndarray  # (rows, cols) bool, floor cells at least ROBOT_RADIUS from walls and voids
    field: np.ndarray  # (rows, cols) geodesic distance to the gold block, UNREACHABLE where there is no path
    look: np.ndarray  # (rows, cols, 2) unit vector towards the point LOOKAHEAD_CELLS further down the field
    hazards: np.ndarray  # (H, 6) xlo xhi ylo yhi zlo zhi
    plates: np.ndarray  # (P, 2)
    goal: np.ndarray  # (2, 3) lo, hi of the union of the gold blocks
    goal_reachable: bool

    @property
    def n_plates(self) -> int:
        return len(self.plates)

    @classmethod
    def from_xml(cls, xml_path, res: float = 0.25, robot_radius: float = ROBOT_RADIUS) -> "NavTask":
        lay = nav.load_layout(xml_path)
        sp_c, sp_h = lay.markers["spawn"][0]
        floor_z = float(sp_c[2] + sp_h[2])
        cols, rows = int(math.ceil(lay.sx * lay.cell / res)), int(math.ceil(lay.sy * lay.cell / res))
        xs = (np.arange(cols) + 0.5) * res
        ys = -(np.arange(rows) + 0.5) * res
        top = np.full((rows, cols), -np.inf)
        for p, h in lay.boxes:
            if p[2] - h[2] > floor_z + HEADROOM:
                continue
            c0, c1 = np.searchsorted(xs, p[0] - h[0] + 1e-9), np.searchsorted(xs, p[0] + h[0] - 1e-9, side="right")
            r0, r1 = np.searchsorted(-ys, -(p[1] + h[1]) + 1e-9), np.searchsorted(-ys, -(p[1] - h[1]) - 1e-9, side="right")
            if c1 > c0 and r1 > r0:
                sub = top[r0:r1, c0:c1]
                np.maximum(sub, p[2] + h[2], out=sub)
        kind = np.full((rows, cols), VOID, np.uint8)
        kind[(top > floor_z - VOID_BELOW) & (top <= floor_z + WALL_ABOVE)] = FLOOR
        kind[top > floor_z + WALL_ABOVE] = WALL
        ground = np.where(kind == FLOOR, top, floor_z)
        hz = [[c[0] - h[0], c[0] + h[0], c[1] - h[1], c[1] + h[1], c[2] - h[2], c[2] + h[2]] for c, h in lay.markers.get("hazard", [])]
        hazards = np.array(hz, float).reshape(-1, 6)
        walk = kind == FLOOR
        for h in hazards:  # lava cells are never walkable, even where a block of the pit bottom looks like floor
            c0, c1 = np.searchsorted(xs, h[0] - HAZARD_MARGIN), np.searchsorted(xs, h[1] + HAZARD_MARGIN, side="right")
            r0, r1 = np.searchsorted(-ys, -(h[3] + HAZARD_MARGIN)), np.searchsorted(-ys, -(h[2] - HAZARD_MARGIN), side="right")
            walk[r0:r1, c0:c1] = False
        free = _erode(walk, int(robot_radius // res))  # k cells: a 1 m bridge (4 cells) keeps a 0.5 m band
        plates = np.array([c[:2] for c, _ in lay.markers.get("checkpoint", [])][:MAX_PLATES], float).reshape(-1, 2)
        goals = lay.markers.get("goal", [])
        if goals:
            lo = np.min([c - h for c, h in goals], 0)
            hi = np.max([c + h for c, h in goals], 0)
            goal = np.array([lo, hi])
            field = _dijkstra(free, goal, res, xs, ys)
        else:
            goal = np.zeros((2, 3))
            field = np.full((rows, cols), np.inf)
        reachable = bool(np.isfinite(field[free]).any() and _cell(sp_c[:2], res, rows, cols) is not None and np.isfinite(field[_cell(sp_c[:2], res, rows, cols)]))
        field = np.where(np.isfinite(field), field, UNREACHABLE).astype(np.float32)
        return cls(res, floor_z, np.array([sp_c[0], sp_c[1]]), rows, cols, kind, ground.astype(np.float32), free, field, _lookahead(free, field, res),
                   hazards, plates, goal, reachable)

    def to(self, device) -> "TorchTask":
        return TorchTask(self, device)


def _cell(xy, res, rows, cols):
    c, r = int(xy[0] // res), int(-xy[1] // res)
    return (r, c) if 0 <= r < rows and 0 <= c < cols else None


def _erode(walk: np.ndarray, k: int) -> np.ndarray:
    """Cells of `walk` whose (2k+1)^2 neighbourhood is all walk (out-of-grid counts as not walk)."""
    out = walk.copy()
    pad = np.pad(walk, k, constant_values=False)
    rows, cols = walk.shape
    for dr in range(2 * k + 1):
        for dc in range(2 * k + 1):
            out &= pad[dr:dr + rows, dc:dc + cols]
    return out


def _dijkstra(free: np.ndarray, goal: np.ndarray, res: float, xs, ys) -> np.ndarray:
    """Geodesic distance (m) from every free cell to the goal box (8-connected, no corner cutting). Seeds: free cells within GOAL_MARGIN of the box."""
    rows, cols = free.shape
    dist = np.full((rows, cols), np.inf)
    gx = (xs >= goal[0, 0] - GOAL_MARGIN) & (xs <= goal[1, 0] + GOAL_MARGIN)
    gy = (ys >= goal[0, 1] - GOAL_MARGIN) & (ys <= goal[1, 1] + GOAL_MARGIN)
    seeds = np.argwhere(np.outer(gy, gx) & free)
    pq = []
    for r, c in seeds:
        dist[r, c] = 0.0
        pq.append((0.0, int(r), int(c)))
    heapq.heapify(pq)
    steps = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0), (1, 1, 1.4142), (1, -1, 1.4142), (-1, 1, 1.4142), (-1, -1, 1.4142)]
    while pq:
        d, r, c = heapq.heappop(pq)
        if d > dist[r, c]:
            continue
        for dr, dc, w in steps:
            rr, cc = r + dr, c + dc
            if 0 <= rr < rows and 0 <= cc < cols and free[rr, cc]:
                if dr and dc and not (free[r, cc] and free[rr, c]):
                    continue
                nd = d + w * res
                if nd < dist[rr, cc]:
                    dist[rr, cc] = nd
                    heapq.heappush(pq, (nd, rr, cc))
    return dist


def _lookahead(free: np.ndarray, field: np.ndarray, res: float) -> np.ndarray:
    """Per cell: unit vector (world x, y) towards the cell reached by walking LOOKAHEAD_CELLS steps down the geodesic field."""
    rows, cols = free.shape
    f = np.where(free, field, np.inf)
    pad = np.pad(f, 1, constant_values=np.inf)
    best = np.zeros((rows, cols), np.int64)
    bval = np.full((rows, cols), np.inf)
    offs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for k, (dr, dc) in enumerate(offs):
        v = pad[1 + dr:1 + dr + rows, 1 + dc:1 + dc + cols]
        better = v < bval
        bval = np.where(better, v, bval)
        best = np.where(better, k, best)
    dr_t = np.array([o[0] for o in offs])[best]
    dc_t = np.array([o[1] for o in offs])[best]
    rr, cc = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    r2, c2 = rr.copy(), cc.copy()
    for _ in range(LOOKAHEAD_CELLS):  # follow the best-neighbour pointers
        nr = np.clip(r2 + dr_t[r2, c2], 0, rows - 1)
        nc = np.clip(c2 + dc_t[r2, c2], 0, cols - 1)
        r2, c2 = nr, nc
    vx = (c2 - cc).astype(np.float32)
    vy = -(r2 - rr).astype(np.float32)  # row grows towards -y
    n = np.maximum(np.hypot(vx, vy), 1e-6)
    out = np.stack([vx / n, vy / n], -1)
    out[~free] = 0.0
    return out.astype(np.float32)


class TorchTask:
    """Per-step quantities for a batch of envs. p: (N, 3) pelvis position, xy: (N, 2), yaw: (N,)."""

    def __init__(self, t: NavTask, device):
        self.t, self.dev, self.res = t, device, t.res
        f = lambda a, dt=torch.float32: torch.as_tensor(np.asarray(a), dtype=dt, device=device)  # noqa: E731
        self.kind = f(t.kind, torch.long)
        self.ground = f(t.ground)
        self.field = f(t.field)
        self.look = f(t.look)
        self.free = f(t.free, torch.bool)
        self.rows, self.cols = t.rows, t.cols
        self.haz = f(t.hazards)
        self.plates = f(t.plates)
        self.goal = f(t.goal)
        self.floor_z = t.floor_z
        self.has_goal = bool(t.goal[1, 0] > t.goal[0, 0])
        self.n_p = len(t.plates)
        ang = torch.linspace(-math.pi, math.pi, N_RAYS + 1, device=device)[:-1]
        steps = torch.arange(1, int(RAY_RANGE / RAY_STEP) + 1, device=device, dtype=torch.float32) * RAY_STEP
        self.ray_ang, self.ray_t = ang, steps
        spawn_cells = torch.nonzero(self.free & (self.field < UNREACHABLE)).float()  # (K, 2) row, col
        self.start_cells = spawn_cells
        self.start_d = self.field[self.free & (self.field < UNREACHABLE)]

    def cell_idx(self, xy: torch.Tensor):
        c = (xy[..., 0] / self.res).floor().long()
        r = (-xy[..., 1] / self.res).floor().long()
        inside = (c >= 0) & (c < self.cols) & (r >= 0) & (r < self.rows)
        return r.clamp(0, self.rows - 1), c.clamp(0, self.cols - 1), inside

    def field_at(self, xy: torch.Tensor) -> torch.Tensor:
        """Geodesic distance (m) to the gold block; UNREACHABLE off free space (the free grid is eroded, so use field_near for the pelvis)."""
        r, c, ok = self.cell_idx(xy)
        return torch.where(ok, self.field[r, c], torch.full_like(self.field[r, c], UNREACHABLE))

    def field_near(self, xy: torch.Tensor) -> torch.Tensor:
        """Field value at the best of the 3x3 cells around xy (the eroded free grid may miss a cell the robot is standing on)."""
        r, c, ok = self.cell_idx(xy)
        best = None
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                v = self.field[(r + dr).clamp(0, self.rows - 1), (c + dc).clamp(0, self.cols - 1)]
                best = v if best is None else torch.minimum(best, v)
        return torch.where(ok, best, torch.full_like(best, UNREACHABLE))

    def look_at(self, xy: torch.Tensor) -> torch.Tensor:
        """Unit vector (N, 2) towards the goal along the shortest walkable path; (0, 0) where the field is undefined (searches 3x3 neighbours)."""
        r, c, ok = self.cell_idx(xy)
        out = self.look[r, c]
        for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)):
            alt = self.look[(r + dr).clamp(0, self.rows - 1), (c + dc).clamp(0, self.cols - 1)]
            out = torch.where((out.abs().sum(-1, keepdim=True) < 1e-6), alt, out)
        return out * ok[:, None]

    def ground_at(self, xy: torch.Tensor) -> torch.Tensor:
        r, c, ok = self.cell_idx(xy)
        return torch.where(ok & (self.kind[r, c] == FLOOR), self.ground[r, c], torch.full_like(self.ground[r, c], self.floor_z))

    # ---- sensing --------------------------------------------------------------------------------------------------------------
    def rays(self, xy: torch.Tensor, yaw: torch.Tensor):
        """(wall, void), each (N, N_RAYS) in [0, 1]: distance along each body-frame direction to the first wall / the first void or
        lava cell, over RAY_RANGE metres, 1 = nothing seen."""
        a = yaw[:, None] + self.ray_ang[None]  # (N, K)
        pts = xy[:, None, None, :] + self.ray_t[None, None, :, None] * torch.stack([torch.cos(a), torch.sin(a)], -1)[:, :, None, :]  # (N, K, S, 2)
        r, c, ok = self.cell_idx(pts)
        k = torch.where(ok, self.kind[r, c], torch.full_like(r, WALL))
        s = self.ray_t.numel()

        def first(hit):
            idx = torch.where(hit.any(-1), hit.float().argmax(-1) + 1, torch.full(hit.shape[:2], s + 1, device=hit.device))
            return (idx.float() - 1).clamp(max=s) / s  # steps before the first hit, normalised; none = 1

        wall, void = first(k == WALL), first(k == VOID)
        return wall, torch.where(void < wall, void, torch.ones_like(void))  # a void behind a wall is hidden

    # ---- reward spec (tasks.py rules; the pelvis stands G1_BASE_HEIGHT above the surface) ------------------------------------------
    def in_hazard(self, p: torch.Tensor) -> torch.Tensor:
        """In a lava box's footprint (+margin) and no higher than ABOVE_MARGIN over its top: the G1 fell through the lava block."""
        if self.haz.shape[0] == 0:
            return torch.zeros(p.shape[0], dtype=torch.bool, device=self.dev)
        h = self.haz
        ok_xy = (p[:, None, 0] >= h[:, 0] - HAZARD_MARGIN) & (p[:, None, 0] <= h[:, 1] + HAZARD_MARGIN) & \
                (p[:, None, 1] >= h[:, 2] - HAZARD_MARGIN) & (p[:, None, 1] <= h[:, 3] + HAZARD_MARGIN)
        ok_z = (p[:, None, 2] >= h[:, 4]) & (p[:, None, 2] <= h[:, 5] + ABOVE_MARGIN)
        return (ok_xy & ok_z).any(1)

    def reached_goal(self, p: torch.Tensor, base_height: float = G1_BASE_HEIGHT) -> torch.Tensor:
        if not self.has_goal:
            return torch.zeros(p.shape[0], dtype=torch.bool, device=self.dev)
        g = self.goal
        ok_xy = (p[:, 0] >= g[0, 0] - GOAL_MARGIN) & (p[:, 0] <= g[1, 0] + GOAL_MARGIN) & (p[:, 1] >= g[0, 1] - GOAL_MARGIN) & (p[:, 1] <= g[1, 1] + GOAL_MARGIN)
        ok_z = (p[:, 2] >= g[0, 2] - GOAL_MARGIN) & (p[:, 2] <= g[1, 2] + ABOVE_MARGIN + base_height)
        return ok_xy & ok_z

    def on_plates(self, p: torch.Tensor, base_height: float = G1_BASE_HEIGHT) -> torch.Tensor:
        """(N, MAX_PLATES) bool: the plate is within PLATE_RADIUS horizontally and the pelvis is at standing height over its top."""
        out = torch.zeros(p.shape[0], MAX_PLATES, dtype=torch.bool, device=self.dev)
        if self.n_p:
            d = torch.linalg.norm(p[:, None, :2] - self.plates[None], dim=2)
            out[:, : self.n_p] = d <= PLATE_RADIUS
        return out
