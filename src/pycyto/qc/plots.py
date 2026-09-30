"""Chart inputs for the QC report, per probe barcode and pooled over the run.

Everything here is a transformation of a parsed :class:`~pycyto.qc.parse.CytoRun`; the
report's JavaScript does the drawing.
"""

from typing import Self

import numpy as np
import polars as pl
from pydantic import BaseModel

from .parse import CytoRun

_LOG_EDGES = np.round(np.arange(0, 6.10, 0.05), 2)
LOG_BINS = _LOG_EDGES[:-1]
"""log10 bin edges (width 0.05) for the histograms embedded in the report."""


def log_hist(values: np.ndarray) -> list[int] | None:
    """Histogram of ``log10(values)`` over :data:`LOG_BINS`.

    Parameters
    ----------
    values : np.ndarray
        Positive counts; values below 1 are clamped to 1.

    Returns
    -------
    list[int] or None
        One count per bin, or None when ``values`` is empty.
    """
    if len(values) == 0:
        return None
    counts, _ = np.histogram(np.log10(np.maximum(values, 1)), bins=_LOG_EDGES)
    return counts.tolist()


def rank_curve(umis_desc: np.ndarray, is_cell_desc: np.ndarray, n_points: int = 300) -> list:
    """Barcode-rank curve downsampled to ~n_points log-spaced ranks.

    Parameters
    ----------
    umis_desc : np.ndarray
        UMI count per barcode, sorted descending.
    is_cell_desc : np.ndarray
        Whether each barcode is a called cell, in the same order.
    n_points : int
        Target number of points on the curve.

    Returns
    -------
    list
        Points ``[rank, umis, fraction of barcodes in the segment that are cells]``,
        where each point summarizes the barcodes ranked ``(previous rank, rank]``.
    """
    n = len(umis_desc)
    if n == 0:
        return []
    ranks = np.unique(np.clip(np.round(np.logspace(0, np.log10(n), n_points)), 1, n)).astype(int)
    prev = np.concatenate([[0], ranks[:-1]])
    csum = np.concatenate([[0], np.cumsum(is_cell_desc)])
    frac = (csum[ranks] - csum[prev]) / (ranks - prev)
    return [[int(r), int(u), round(float(f), 3)] for r, u, f in zip(ranks, umis_desc[ranks - 1], frac)]


def _cell_hists(cells: pl.DataFrame) -> dict[str, list[int] | None]:
    """``umi_hist`` / ``gene_hist`` for a slice of :attr:`CytoRun.cells`."""
    return {
        "umi_hist": log_hist(cells["n_umis"].to_numpy()),
        "gene_hist": log_hist(cells["n_genes"].to_numpy()),
    }


class PooledPlots(BaseModel):
    """Run-wide plot inputs, pooled over every probe barcode.

    Attributes
    ----------
    umi_hist, gene_hist : list[int] or None
        :func:`log_hist` of UMIs and genes per called cell across the run; None when
        the run has no cells.
    """

    umi_hist: list[int] | None
    gene_hist: list[int] | None

    @classmethod
    def compute(cls, run: CytoRun) -> Self:
        """Pooled plot inputs for a run.

        Parameters
        ----------
        run : CytoRun

        Returns
        -------
        PooledPlots
        """
        return cls(**_cell_hists(run.cells))


class ProbePlots(BaseModel):
    """Plot inputs for one probe barcode.

    Attributes
    ----------
    curve : list[tuple[int, int, float]]
        Barcode-rank curve from :func:`rank_curve`: ``(rank, umis, cell fraction)`` points.
    umi_hist, gene_hist : list[int] or None
        :func:`log_hist` of UMIs and genes per called cell; None when the probe has no cells.
    """

    curve: list[tuple[int, int, float]]
    umi_hist: list[int] | None
    gene_hist: list[int] | None

    @classmethod
    def compute(cls, run: CytoRun, probe: str) -> Self:
        """Plot inputs for one probe barcode.

        Parameters
        ----------
        run : CytoRun
        probe : str
            A member of ``run.stats.probes``.

        Returns
        -------
        ProbePlots
        """
        cells = run.cells.filter(pl.col("probe") == probe)
        ranked = (
            run.stats.reads.entries[probe]
            .with_columns(is_cell=pl.col("barcode").is_in(cells["barcode"].implode()))
            .sort("n_umis", descending=True)
        )
        curve = rank_curve(ranked["n_umis"].to_numpy(), ranked["is_cell"].to_numpy())
        return cls(curve=curve, **_cell_hists(cells))
