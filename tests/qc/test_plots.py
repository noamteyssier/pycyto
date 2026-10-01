"""``pycyto.qc.plots`` building blocks."""

import numpy as np

from pycyto.qc.plots import rank_curve


def test_rank_curve():
    rng = np.random.default_rng(0)
    umis = np.sort(rng.integers(1, 10_000, 5_000))[::-1]
    is_cell = umis > 2_000
    curve = rank_curve(umis, is_cell, n_points=50)
    ranks = [r for r, _, _ in curve]
    assert ranks[0] == 1 and ranks[-1] == len(umis) and ranks == sorted(set(ranks))
    prev = 0
    for rank, u, frac in curve:  # each point summarizes barcodes (prev, rank]
        assert u == umis[rank - 1]
        assert frac == round(float(is_cell[prev:rank].mean()), 3)
        prev = rank
    assert rank_curve(np.array([], dtype=int), np.array([], dtype=bool)) == []
