"""``pycyto.qc.metrics`` building blocks and the h5ad scan behind the per-cell metrics."""

import os

import anndata as ad
import numpy as np
import polars as pl
import pytest

from pycyto.qc.metrics import rank_curve
from pycyto.qc.parse import FilteredCounts


class TestFilteredCounts:
    @pytest.mark.parametrize("probe", ["A-A01", "A-A02"])  # CSR and CSC
    def test_matches_dense(self, cyto_dir, probe):
        root, _ = cyto_dir
        path = os.path.join(root, "counts", f"{probe}.filt.h5ad")
        adata = ad.read_h5ad(path)
        dense = adata.X.toarray()
        fc = FilteredCounts.from_h5ad(path)
        fc_small = FilteredCounts.from_h5ad(path, chunk_rows=7)  # force many chunks
        np.testing.assert_array_equal(fc.cells["n_genes"].to_numpy(), (dense > 0).sum(1))
        np.testing.assert_allclose(fc.feature_totals, dense.sum(0))
        assert fc.cells.equals(fc_small.cells)
        np.testing.assert_allclose(fc.feature_totals, fc_small.feature_totals)
        assert fc.n_features == adata.n_vars
        # probe suffix stripped from obs names
        barcodes = fc.cells["barcode"].cast(pl.String)
        assert barcodes.to_list() == [n.split("-", 1)[0] for n in adata.obs_names]
        assert barcodes.str.len_chars().eq(16).all()


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
