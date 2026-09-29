"""Alert cutoffs for every workflow, grouped the same way as :mod:`pycyto.qc.alerts`.

Shared by the alert rules and the per-probe-barcode ``flag`` in :mod:`pycyto.qc.metrics`.
"""

from typing import Literal, NamedTuple

from pydantic import BaseModel, ConfigDict

Level = Literal["ok", "warn", "error"]
"""Severity of an alert or a per-probe-barcode flag."""


class Band(NamedTuple):
    """A two-level threshold: cross ``warn`` for a warning, ``error`` for an error."""

    warn: float
    error: float


class GexThresholds(BaseModel):
    """Cutoffs for :class:`~pycyto.qc.alerts.GexAlerts` and :attr:`~pycyto.qc.metrics.GexProbeMetrics.flag`.

    Attributes
    ----------
    frac_reads_in_cells : Band
        Fraction of mapped reads in cells, run-wide; below -> warn / error. The error
        level is also the per-probe-barcode cutoff.
    background_probe_read_frac : float
        Fraction of mapped reads in probe barcodes without cells; above -> warn.
    median_umis_per_cell : float
        Per-probe-barcode median UMIs per cell; below -> warn.
    cells_cv : float
        Coefficient of variation of cells across probe barcodes with cells; above -> warn.
    """

    model_config = ConfigDict(frozen=True)

    frac_reads_in_cells: Band = Band(warn=0.70, error=0.50)
    background_probe_read_frac: float = 0.05
    median_umis_per_cell: float = 500
    cells_cv: float = 0.5


class CrisprThresholds(BaseModel):
    """Cutoffs for :class:`~pycyto.qc.alerts.CrisprAlerts`.

    Attributes
    ----------
    frac_guides_detected : Band
        Fraction of library guides with at least one UMI; below -> warn / error.
    guide_skew_ratio : float
        90th / 10th percentile of UMIs per guide; above -> warn.
    """

    model_config = ConfigDict(frozen=True)

    frac_guides_detected: Band = Band(warn=0.90, error=0.75)
    guide_skew_ratio: float = 10


class Thresholds(BaseModel):
    """Alert cutoffs: the shared ones, plus one block per workflow.

    Attributes
    ----------
    mapped_frac : Band
        Fraction of reads mapped; below -> warn / error.
    failed_umi_qual_of_total : float
        Fraction of all reads failing UMI quality; above -> warn.
    gex : GexThresholds
    crispr : CrisprThresholds
    """

    model_config = ConfigDict(frozen=True)

    mapped_frac: Band = Band(warn=0.70, error=0.50)
    failed_umi_qual_of_total: float = 0.10
    gex: GexThresholds = GexThresholds()
    crispr: CrisprThresholds = CrisprThresholds()


THRESH = Thresholds()
