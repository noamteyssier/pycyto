"""Chart inputs for the QC report, per probe barcode and pooled over the run.

Everything here is a transformation of a parsed :class:`~pycyto.qc.parse.CytoRun`; the
report's JavaScript does the drawing.
"""

from typing import Self

import numpy as np
import polars as pl
from pandera.typing.polars import DataFrame
from pydantic import BaseModel

from ..config import FlexBarcode
from .parse import CellTable, CytoRun

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


RankPoint = tuple[int, int, float]
"""One barcode-rank curve point: ``(rank, umis, fraction of the segment that are cells)``."""


def rank_curve(umis_desc: np.ndarray, is_cell_desc: np.ndarray, n_points: int = 300) -> list[RankPoint]:
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
    list[RankPoint]
        Each point summarizes the barcodes ranked ``(previous rank, rank]``.
    """
    n = len(umis_desc)
    if n == 0:
        return []
    ranks = np.unique(np.clip(np.round(np.logspace(0, np.log10(n), n_points)), 1, n)).astype(int)
    prev = np.concatenate([[0], ranks[:-1]])
    csum = np.concatenate([[0], np.cumsum(is_cell_desc)])
    frac = (csum[ranks] - csum[prev]) / (ranks - prev)
    return [(int(r), int(u), round(float(f), 3)) for r, u, f in zip(ranks, umis_desc[ranks - 1], frac)]


class CellHists(BaseModel):
    """Per-cell histograms over some set of called cells.

    Attributes
    ----------
    umi_hist, gene_hist : list[int] or None
        :func:`log_hist` of UMIs and genes per cell; None when there are no cells.
    """

    umi_hist: list[int] | None
    gene_hist: list[int] | None

    @classmethod
    def from_cells(cls, cells: DataFrame[CellTable]) -> Self:
        """Histograms for a slice of :attr:`CytoRun.cells`.

        Parameters
        ----------
        cells : DataFrame[CellTable]
            All of :attr:`CytoRun.cells`, or a subset of its rows.

        Returns
        -------
        CellHists
        """
        return cls(
            umi_hist=log_hist(cells["n_umis"].to_numpy()),
            gene_hist=log_hist(cells["n_genes"].to_numpy()),
        )


class ProbePlots(BaseModel):
    """Plot inputs for one probe barcode.

    Attributes
    ----------
    curve : list[RankPoint]
        Barcode-rank curve over every cell barcode under the probe, from :func:`rank_curve`.
    hists : CellHists
        Histograms over the probe's called cells.
    """

    curve: list[RankPoint]
    hists: CellHists


class Plots(BaseModel):
    """Every chart input in the report.

    Attributes
    ----------
    log_bins : list[float]
        :data:`LOG_BINS`, so the report can label histogram axes.
    pooled : CellHists
        Histograms over every called cell in the run.
    probes : dict[FlexBarcode, ProbePlots]
        Per-probe-barcode plots, in :attr:`CytoStats.probes` order.
    """

    log_bins: list[float]
    pooled: CellHists
    probes: dict[FlexBarcode, ProbePlots]

    @classmethod
    def compute(cls, run: CytoRun) -> Self:
        """Plot inputs for a run.

        Parameters
        ----------
        run : CytoRun

        Returns
        -------
        Plots
        """
        return cls(
            log_bins=LOG_BINS.tolist(),
            pooled=CellHists.from_cells(run.cells),
            probes={probe: cls._probe(run, probe) for probe in run.stats.probes},
        )

    @staticmethod
    def _probe(run: CytoRun, probe: str) -> ProbePlots:
        cells = run.cells.filter(pl.col("probe") == probe)
        ranked = (
            run.stats.reads.entries[probe]
            .with_columns(is_cell=pl.col("barcode").is_in(cells["barcode"].implode()))
            .sort("n_umis", descending=True)
        )
        curve = rank_curve(ranked["n_umis"].to_numpy(), ranked["is_cell"].to_numpy())
        return ProbePlots(curve=curve, hists=CellHists.from_cells(cells))
