"""GEX (``cyto workflow gex``) metrics, plots and alerts.

Cells are exactly the barcodes in cyto's ``counts/<probe>.filt.h5ad``; probe barcodes
without that file have no cells.
"""

from typing import Any, Self

import numpy as np
import polars as pl
from pandera.typing.polars import DataFrame
from pydantic import BaseModel

from ..config import FlexBarcode
from .alerts import THRESH, Adder, examples, level_above, level_below
from .metrics import ProbeMetrics, SummaryMetrics, safe_div
from .parse import CellTable, GexRun
from .plots import Plots, ProbePlots, log_hist, probe_rank_curve
from .workflow import Workflow


class GexProbeMetrics(ProbeMetrics):
    """Per-probe-barcode metrics for a GEX run.

    Attributes
    ----------
    cells : int
        Called cells.
    reads_in_cells : int
        Mapped reads belonging to called cells.
    frac_reads_in_cells : float or None
        ``reads_in_cells / mapped_reads``.
    median_umis_per_cell : float or None
        Median UMIs over called cells; None when the probe has no cells.
    """

    cells: int
    reads_in_cells: int
    frac_reads_in_cells: float | None
    median_umis_per_cell: float | None

    @classmethod
    def _fields(cls, run: GexRun, probe: str) -> dict[str, Any]:
        base = super()._fields(run, probe)
        in_cells = run.cells.filter(pl.col("probe") == probe)
        return base | {
            "cells": in_cells.height,
            "reads_in_cells": in_cells["n_reads"].sum(),
            "frac_reads_in_cells": safe_div(in_cells["n_reads"].sum(), base["mapped_reads"]),
            "median_umis_per_cell": in_cells["n_umis"].median(),
        }


class GexSummaryMetrics(SummaryMetrics):
    """Run-level metrics for a GEX run.

    Attributes
    ----------
    genes_in_reference : int
        Features in the count matrices, or in the ``gex`` library when no probe has cells.
    estimated_cells : int
        Called cells over all probe barcodes.
    probe_barcodes_with_cells : int
        Probe barcodes with at least one called cell.
    probe_barcodes_without_cells : int
        Probe barcodes with reads but no called cells.
    cells_median_per_probe : float or None
        Median called cells among probe barcodes with cells.
    cells_cv_per_probe : float or None
        Coefficient of variation (sample std / mean) of called cells across probe barcodes
        with cells; None with fewer than three such probe barcodes.
    mean_reads_per_cell, mean_mapped_reads_per_cell : float or None
        ``total_reads`` and ``mapped_reads`` divided by ``estimated_cells``.
    median_umis_per_cell, median_genes_per_cell : float or None
        Medians over all called cells.
    total_genes_detected : int or None
        Features with a nonzero total in any probe's called cells.
    frac_reads_in_cells : float or None
        Mapped reads in called cells / all mapped reads.
    background_probe_read_frac : float or None
        Mapped reads in probe barcodes without cells / all mapped reads.
    """

    genes_in_reference: int
    estimated_cells: int
    probe_barcodes_with_cells: int
    probe_barcodes_without_cells: int
    cells_median_per_probe: float | None
    cells_cv_per_probe: float | None
    mean_reads_per_cell: float | None
    mean_mapped_reads_per_cell: float | None
    median_umis_per_cell: float | None
    median_genes_per_cell: float | None
    total_genes_detected: int | None
    frac_reads_in_cells: float | None
    background_probe_read_frac: float | None

    @classmethod
    def _fields(cls, run: GexRun, probes: list[GexProbeMetrics]) -> dict[str, Any]:
        cells = run.cells
        mapping = run.stats.mapping
        counts = list(run.counts.values())
        called = [p for p in probes if p.cells > 0]
        cells_per_probe = np.array([p.cells for p in called], dtype=float)
        mapped = sum(p.mapped_reads for p in probes)
        n_cells = cells.height
        return super()._fields(run, probes) | {
            "genes_in_reference": counts[0].n_features if counts else run.stats.library.entries["gex"].total_aggr,
            "estimated_cells": n_cells,
            "probe_barcodes_with_cells": len(called),
            "probe_barcodes_without_cells": len(probes) - len(called),
            "cells_median_per_probe": float(np.median(cells_per_probe)) if called else None,
            "cells_cv_per_probe": float(cells_per_probe.std(ddof=1) / cells_per_probe.mean()) if len(called) >= 3 else None,
            "mean_reads_per_cell": safe_div(mapping.total_reads, n_cells),
            "mean_mapped_reads_per_cell": safe_div(mapping.mapped_reads, n_cells),
            "median_umis_per_cell": cells["n_umis"].median(),
            "median_genes_per_cell": cells["n_genes"].median(),
            "total_genes_detected": int(np.logical_or.reduce([c.features_detected for c in counts]).sum()) if counts else None,
            "frac_reads_in_cells": safe_div(cells["n_reads"].sum(), mapped),
            "background_probe_read_frac": safe_div(sum(p.mapped_reads for p in probes if p.cells == 0), mapped),
        }


class CellHists(BaseModel):
    """Per-cell histograms over some set of called cells.

    Attributes
    ----------
    umi_hist, gene_hist : list[int] or None
        :func:`~pycyto.qc.plots.log_hist` of UMIs and genes per cell; None when there are no cells.
    """

    umi_hist: list[int] | None
    gene_hist: list[int] | None

    @classmethod
    def from_cells(cls, cells: DataFrame[CellTable]) -> Self:
        """Histograms for a slice of :attr:`GexRun.cells`.

        Parameters
        ----------
        cells : DataFrame[CellTable]
            All of :attr:`GexRun.cells`, or a subset of its rows.

        Returns
        -------
        CellHists
        """
        return cls(
            umi_hist=log_hist(cells["n_umis"].to_numpy()),
            gene_hist=log_hist(cells["n_genes"].to_numpy()),
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
    probes : dict[FlexBarcode, GexProbePlots]
    """

    pooled: CellHists
    probes: dict[FlexBarcode, GexProbePlots]

    @classmethod
    def _fields(cls, run: GexRun) -> dict[str, Any]:
        return super()._fields(run) | {"pooled": CellHists.from_cells(run.cells)}

    @classmethod
    def _probe(cls, run: GexRun, probe: str) -> GexProbePlots:
        cells = run.cells.filter(pl.col("probe") == probe)
        return GexProbePlots(curve=probe_rank_curve(run, probe, cells["barcode"]), hists=CellHists.from_cells(cells))


def alert_rules(summary: GexSummaryMetrics, probes: list[GexProbeMetrics], add: Adder) -> None:
    """GEX alert rules; see :func:`pycyto.qc.alerts.build_alerts`."""
    fr = summary.frac_reads_in_cells
    add(
        level_below(fr, THRESH.frac_reads_in_cells),
        "Low fraction of reads in cells",
        f"{fr or 0:.1%} of mapped reads are in cell barcodes "
        f"(expected ≥ {THRESH.frac_reads_in_cells.warn:.0%}).",
    )
    bg = summary.background_probe_read_frac
    add(
        level_above(bg, THRESH.background_probe_read_frac),
        "Reads in probe barcodes without cells",
        f"{bg or 0:.1%} of mapped reads went to {summary.probe_barcodes_without_cells} probe "
        "barcodes with no cells. Check for unexpected probe barcodes or barcode hopping.",
    )

    called = [p for p in probes if p.cells > 0]
    low = [p for p in called if p.median_umis_per_cell is not None and p.median_umis_per_cell < THRESH.median_umis_per_cell]
    add(
        "warn" if low else None,
        "Low median UMIs per cell",
        f"{len(low)} probe barcode(s) with cells have a median below "
        f"{THRESH.median_umis_per_cell:g} UMIs per cell: "
        + examples(low, lambda p: f"{p.probe} ({p.median_umis_per_cell:.0f})")
        + ".",
    )
    min_frac = THRESH.frac_reads_in_cells.error
    low_probes = {p.probe for p in low}
    low_frac = [
        p
        for p in called
        if p.frac_reads_in_cells is not None and p.frac_reads_in_cells < min_frac and p.probe not in low_probes
    ]
    add(
        "warn" if low_frac else None,
        "Probe barcodes with low reads in cells",
        f"{len(low_frac)} probe barcode(s) have <{min_frac:.0%} of reads in cells: "
        + examples(low_frac, lambda p: f"{p.probe} ({p.frac_reads_in_cells:.0%})")
        + ".",
    )
    cv = summary.cells_cv_per_probe
    add(
        level_above(cv, THRESH.cells_cv),
        "Uneven cell counts across probe barcodes",
        f"Coefficient of variation of cells per probe barcode is {cv or 0:.2f}.",
    )


WORKFLOW = Workflow(
    run=GexRun, probe_metrics=GexProbeMetrics, summary=GexSummaryMetrics, plots=GexPlots, alert_rules=alert_rules
)
