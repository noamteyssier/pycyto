"""Chart inputs every cyto workflow reports.

Everything here is a transformation of a parsed :class:`~pycyto.qc.parse.CytoRun`; the
report's JavaScript does the drawing. Workflow modules subclass :class:`Plots` and
:class:`ProbePlots` to add their own charts.
"""

from typing import Any, Self

import numpy as np
import polars as pl
from pydantic import BaseModel

from ..config import FlexBarcode
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


def probe_rank_curve(run: CytoRun, probe: str, cell_barcodes: pl.Series | None = None) -> list[RankPoint]:
    """Barcode-rank curve over every cell barcode under one probe barcode.

    Parameters
    ----------
    run : CytoRun
    probe : str
        A member of ``run.stats.probes``.
    cell_barcodes : pl.Series or None
        Barcodes that are called cells, for the curve's cell-fraction shading. None
        marks every barcode as background.

    Returns
    -------
    list[RankPoint]
    """
    is_cell = pl.lit(False) if cell_barcodes is None else pl.col("barcode").is_in(cell_barcodes.implode())
    ranked = run.stats.reads.entries[probe].with_columns(is_cell=is_cell).sort("n_umis", descending=True)
    return rank_curve(ranked["n_umis"].to_numpy(), ranked["is_cell"].to_numpy())


class ProbePlots(BaseModel):
    """Plot inputs for one probe barcode.

    Attributes
    ----------
    curve : list[RankPoint]
        Barcode-rank curve from :func:`probe_rank_curve`.
    """

    curve: list[RankPoint]


class Plots(BaseModel):
    """Every chart input in the report.

    Subclasses redeclare ``probes`` with their :class:`ProbePlots` subclass, add a
    ``pooled`` field for run-wide charts, and extend ``_fields`` / ``_probe``.

    Attributes
    ----------
    log_bins : list[float]
        :data:`LOG_BINS`, so the report can label histogram axes.
    probes : dict[FlexBarcode, ProbePlots]
        Per-probe-barcode plots, in :attr:`CytoStats.probes` order.
    """

    log_bins: list[float]
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
        return cls(**cls._fields(run))

    @classmethod
    def _fields(cls, run: CytoRun) -> dict[str, Any]:
        """Constructor kwargs; subclasses extend with ``super()._fields(run) | {...}``."""
        return {"log_bins": LOG_BINS.tolist(), "probes": {probe: cls._probe(run, probe) for probe in run.stats.probes}}

    @classmethod
    def _probe(cls, run: CytoRun, probe: str) -> ProbePlots:
        """Plot inputs for one probe barcode; subclasses return their :class:`ProbePlots` subclass."""
        return ProbePlots(curve=probe_rank_curve(run, probe))
