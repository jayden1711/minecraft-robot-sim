"""Greedy merge of voxel cells into larger axis-aligned boxes."""

from __future__ import annotations

import itertools

import numpy as np

Box = tuple[int, int, int, int, int, int, int]  # x0, y0, z0, x1, y1, z1 (exclusive), label


def _greedy(labels: np.ndarray) -> list[Box]:
    """Scan order y, z, x. Grow each unclaimed cell along x, then z, then y while every cell has the same label."""
    sx, sy, sz = labels.shape
    used = np.zeros(labels.shape, bool)
    out: list[Box] = []
    for y, z, x in np.argwhere(labels.transpose(1, 2, 0) != 0):
        if used[x, y, z]:
            continue
        lab = labels[x, y, z]
        x1 = x + 1
        while x1 < sx and labels[x1, y, z] == lab and not used[x1, y, z]:
            x1 += 1
        z1 = z + 1
        while z1 < sz and ((labels[x:x1, y, z1] == lab) & ~used[x:x1, y, z1]).all():
            z1 += 1
        y1 = y + 1
        while y1 < sy and ((labels[x:x1, y1, z:z1] == lab) & ~used[x:x1, y1, z:z1]).all():
            y1 += 1
        used[x:x1, y:y1, z:z1] = True
        out.append((int(x), int(y), int(z), int(x1), int(y1), int(z1), int(lab)))
    return out


def merge_boxes(labels: np.ndarray, try_orders: bool = True) -> list[Box]:
    """Cover all non-zero cells of `labels` (shape sx,sy,sz) with disjoint boxes of constant label.

    With try_orders, all six axis priorities are tried and the result with the fewest boxes is kept.
    Output is sorted, so identical input gives identical output.
    """
    labels = np.asarray(labels)
    if labels.ndim != 3:
        raise ValueError("labels must be 3-D")
    perms = list(itertools.permutations(range(3))) if try_orders else [(0, 1, 2)]
    best: list[Box] | None = None
    for perm in perms:
        boxes = _greedy(np.ascontiguousarray(labels.transpose(perm)))
        inv = np.argsort(perm)
        mapped: list[Box] = []
        for b in boxes:
            lo = [b[i] for i in range(3)]
            hi = [b[3 + i] for i in range(3)]
            mapped.append(
                (lo[inv[0]], lo[inv[1]], lo[inv[2]], hi[inv[0]], hi[inv[1]], hi[inv[2]], b[6])
            )
        if best is None or len(mapped) < len(best):
            best = mapped
    assert best is not None
    return sorted(best)


def rasterize(boxes: list[Box], shape: tuple[int, int, int]) -> np.ndarray:
    """Inverse of merge_boxes (for tests): label grid, with -1 where boxes overlap."""
    out = np.zeros(shape, np.int64)
    for x0, y0, z0, x1, y1, z1, lab in boxes:
        sl = out[x0:x1, y0:y1, z0:z1]
        sl[sl != 0] = -1
        sl[sl == 0] = lab
    return out
