"""Chart inputs: the shared :class:`Plots` / :class:`ProbePlots` and each workflow's subclass.

Everything here is a transformation of a parsed :class:`~pycyto.qc.parse.CytoRun`; the
report's JavaScript does the drawing. Workflow subclasses add their own charts, overriding
``compute`` to build the base model and extend it.
"""

from typing import Self

import numpy as np
import polars as pl
from pandera.typing.polars import DataFrame
from pydantic import BaseModel

from ..config import FlexBarcode
from .metrics import safe_div
from .parse import BarcodeReadStats, CellTable, CrisprCytoRun, CytoRun, GexCytoRun

_LOG_EDGES = np.round(np.arange(0, 6.10, 0.05), 2)
LOG_BINS = _LOG_EDGES[:-1]
"""log10 bin edges (width 0.05) for the histograms embedded in the report."""

TOP_GUIDES = 10
"""Most abundant guides listed in the CRISPR report."""


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
    is_cell = pl.lit(False) if cell_barcodes is None else pl.col(BarcodeReadStats.barcode).is_in(cell_barcodes.implode())
    ranked = run.stats.reads.entries[probe].with_columns(is_cell=is_cell).sort(BarcodeReadStats.n_umis, descending=True)
    return rank_curve(ranked[BarcodeReadStats.n_umis].to_numpy(), ranked["is_cell"].to_numpy())


# ============================================================================
# Shared
# ============================================================================
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

    Subclasses add a ``pooled`` field for run-wide charts and build ``probes`` from their
    own :class:`ProbePlots` subclass.

    Attributes
    ----------
    log_bins : list[float]
        :data:`LOG_BINS`, so the report can label histogram axes.
    probes : dict[FlexBarcode, ProbePlots]
        Per-probe-barcode plots, in :attr:`CytoStats.probes` order. Holds the workflow's
        :class:`ProbePlots` subclass; :meth:`pycyto.qc.Report.payload` dumps its extra fields.
    """

    log_bins: list[float] = LOG_BINS.tolist()
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
        return cls(probes={probe: ProbePlots(curve=probe_rank_curve(run, probe)) for probe in run.stats.probes})


# ============================================================================
# GEX
# ============================================================================
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
        """Histograms for a slice of :attr:`~pycyto.qc.parse.GexCytoRun.cells`.

        Parameters
        ----------
        cells : DataFrame[CellTable]
            All of :attr:`~pycyto.qc.parse.GexCytoRun.cells`, or a subset of its rows.

        Returns
        -------
        CellHists
        """
        return cls(
            umi_hist=log_hist(cells[CellTable.n_umis].to_numpy()),
            gene_hist=log_hist(cells[CellTable.n_genes].to_numpy()),
        )


class GexProbePlots(ProbePlots):
    """Plot inputs for one probe barcode of a GEX run.

    Attributes
    ----------
    hists : CellHists
        Histograms over the probe's called cells.
    """

    hists: CellHists


class GexPlots(Plots):
    """Chart inputs for a GEX run.

    Attributes
    ----------
    pooled : CellHists
        Histograms over every called cell in the run.
    """

    pooled: CellHists

    @classmethod
    def compute(cls, run: GexCytoRun) -> Self:
        def probe_plots(probe: str) -> GexProbePlots:
            cells = run.cells_by_probe[probe]
            return GexProbePlots(
                curve=probe_rank_curve(run, probe, cells[CellTable.barcode]), hists=CellHists.from_cells(cells)
            )

        return cls(
            pooled=CellHists.from_cells(run.cells),
            probes={probe: probe_plots(probe) for probe in run.stats.probes},
        )


# ============================================================================
# CRISPR
# ============================================================================
class TopGuide(BaseModel):
    """One row of the report's most-abundant-guides table.

    Attributes
    ----------
    guide : str
    umis : int
        UMIs over every probe barcode.
    frac : float or None
        Share of all guide UMIs in the run.
    """

    guide: str
    umis: int
    frac: float | None


class GuidePooled(BaseModel):
    """Run-wide guide coverage charts.

    Attributes
    ----------
    guide_hist : list[int] or None
        :func:`log_hist` of UMIs per detected guide.
    top_guides : list[TopGuide]
        The :data:`TOP_GUIDES` most abundant guides, largest first.
    """

    guide_hist: list[int] | None
    top_guides: list[TopGuide]


class CrisprProbePlots(ProbePlots):
    """Plot inputs for one probe barcode of a CRISPR run.

    Attributes
    ----------
    guide_hist : list[int] or None
        :func:`log_hist` of UMIs per detected guide under this probe barcode.
    """

    guide_hist: list[int] | None


class CrisprPlots(Plots):
    """Chart inputs for a CRISPR run.

    Attributes
    ----------
    pooled : GuidePooled
    """

    pooled: GuidePooled

    @classmethod
    def compute(cls, run: CrisprCytoRun) -> Self:
        totals = run.guide_totals
        order = np.argsort(-totals, kind="stable")[:TOP_GUIDES]
        total = totals.sum()

        def probe_plots(probe: str) -> CrisprProbePlots:
            umis = run.guide_umis[probe]
            return CrisprProbePlots(curve=probe_rank_curve(run, probe), guide_hist=log_hist(umis[umis > 0]))

        return cls(
            pooled=GuidePooled(
                guide_hist=log_hist(totals[totals > 0]),
                top_guides=[
                    TopGuide(guide=run.guide_names[i], umis=int(totals[i]), frac=safe_div(float(totals[i]), float(total)))
                    for i in order
                    if totals[i] > 0
                ],
            ),
            probes={probe: probe_plots(probe) for probe in run.stats.probes},
        )
