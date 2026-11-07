import numpy as np
import pytest

from mcrobot.boxes import merge_boxes, rasterize


def covers_exactly(labels):
    boxes = merge_boxes(labels)
    r = rasterize(boxes, labels.shape)
    assert (r >= 0).all(), "boxes overlap"
    assert (r == labels).all(), "boxes do not cover exactly the labelled cells"
    return boxes


def test_single_cell():
    g = np.zeros((3, 3, 3), np.uint8)
    g[1, 1, 1] = 1
    assert covers_exactly(g) == [(1, 1, 1, 2, 2, 2, 1)]


def test_flat_floor_is_one_box():
    g = np.zeros((10, 1, 12), np.uint8)
    g[:] = 1
    assert len(covers_exactly(g)) == 1


def test_full_cube_is_one_box():
    assert len(covers_exactly(np.ones((4, 5, 6), np.uint8))) == 1


def test_labels_never_merge():
    g = np.ones((4, 1, 1), np.uint8)
    g[2:] = 2
    boxes = covers_exactly(g)
    assert sorted(b[6] for b in boxes) == [1, 2]


@pytest.mark.parametrize("seed", range(25))
def test_random_grids_exact_cover(seed):
    rng = np.random.default_rng(seed)
    g = (rng.random((7, 5, 8)) < rng.uniform(0.2, 0.9)).astype(np.uint8)
    g *= rng.integers(1, 3, g.shape, dtype=np.uint8)
    covers_exactly(g)


def test_best_of_orders_not_worse():
    rng = np.random.default_rng(0)
    g = (rng.random((8, 6, 9)) < 0.6).astype(np.uint8)
    assert len(merge_boxes(g, True)) <= len(merge_boxes(g, False))


def test_tall_column_needs_order_search():
    g = np.zeros((1, 1, 1), np.uint8)
    col = np.ones((1, 20, 1), np.uint8)
    assert len(merge_boxes(col, True)) == 1 and len(merge_boxes(g)) == 0


def test_deterministic():
    rng = np.random.default_rng(3)
    g = (rng.random((6, 6, 6)) < 0.5).astype(np.uint8)
    assert merge_boxes(g) == merge_boxes(g.copy())


def test_maze_merges_well():
    from mcrobot.synthetic import demo_maze

    r = demo_maze()
    n_cells = int(r.collider_labels().sum())
    n_boxes = len(merge_boxes(r.collider_labels()))
    assert n_boxes < n_cells / 4
