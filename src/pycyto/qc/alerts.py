"""Cell Ranger-style alerts, derived from the computed metrics.

The two rules every workflow shares live here; each workflow module adds its own
through the ``rules`` callback of :func:`build_alerts`.
"""

from collections.abc import Callable
from typing import Literal, NamedTuple

from pydantic import BaseModel, ConfigDict

from .metrics import ProbeMetrics, SummaryMetrics

Level = Literal["ok", "warn", "error"]


class Band(NamedTuple):
    """A two-level threshold: cross ``warn`` for a warning, ``error`` for an error."""

    warn: float
    error: float


class Thresholds(BaseModel):
    """Alert thresholds for every workflow.

    Attributes
    ----------
    mapped_frac : Band
        Fraction of reads mapped; below -> warn / error.
    failed_umi_qual_of_total : float
        Fraction of all reads failing UMI quality; above -> warn.
    frac_reads_in_cells : Band
        GEX. Fraction of mapped reads in cells, run-wide; below -> warn / error. The
        error level is also the per-probe-barcode cutoff.
    background_probe_read_frac : float
        GEX. Fraction of mapped reads in probe barcodes without cells; above -> warn.
    median_umis_per_cell : float
        GEX. Per-probe-barcode median UMIs per cell; below -> warn.
    cells_cv : float
        GEX. Coefficient of variation of cells across probe barcodes with cells; above -> warn.
    frac_guides_detected : Band
        CRISPR. Fraction of library guides with at least one UMI; below -> warn / error.
    guide_skew_ratio : float
        CRISPR. 90th / 10th percentile of UMIs per guide; above -> warn.
    """

    model_config = ConfigDict(frozen=True)

    mapped_frac: Band = Band(warn=0.70, error=0.50)
    failed_umi_qual_of_total: float = 0.10
    frac_reads_in_cells: Band = Band(warn=0.70, error=0.50)
    background_probe_read_frac: float = 0.05
    median_umis_per_cell: float = 500
    cells_cv: float = 0.5
    frac_guides_detected: Band = Band(warn=0.90, error=0.75)
    guide_skew_ratio: float = 10


THRESH = Thresholds()


class Alert(BaseModel):
    """One alert shown in the report's banner.

    Attributes
    ----------
    level : {"ok", "warn", "error"}
    title : str
        Short headline.
    detail : str
        One or two sentences with the offending numbers.
    """

    level: Level
    title: str
    detail: str


Adder = Callable[[Level | None, str, str], None]
"""``add(level, title, detail)``: records an alert when ``level`` is not None."""

Rules = Callable[[SummaryMetrics, list[ProbeMetrics], Adder], None]
"""A workflow's alert rules: inspect the metrics and call ``add`` for each alert."""


def level_below(value: float | None, band: Band) -> Level | None:
    """``error`` / ``warn`` when ``value`` falls below the band's cutoffs."""
    if value is None:
        return None
    return "error" if value < band.error else "warn" if value < band.warn else None


def level_above(value: float | None, warn: float) -> Level | None:
    """``warn`` when ``value`` exceeds ``warn``."""
    return "warn" if value is not None and value > warn else None


def examples(probes: list[ProbeMetrics], fmt: Callable[[ProbeMetrics], str], limit: int = 10) -> str:
    """Comma-separated ``fmt(probe)`` for the first ``limit`` probes, with an ellipsis if cut."""
    return ", ".join(fmt(p) for p in probes[:limit]) + (" …" if len(probes) > limit else "")


def build_alerts(summary: SummaryMetrics, probes: list[ProbeMetrics], rules: Rules) -> list[Alert]:
    """Alerts for a run: the shared rules, then the workflow's.

    Parameters
    ----------
    summary : SummaryMetrics
    probes : list[ProbeMetrics]
        One per probe barcode.
    rules : Rules
        The workflow's rules, e.g. ``gex.alerts``.

    Returns
    -------
    list[Alert]
        Every triggered alert, or a single ``ok`` alert when none triggered.
    """
    alerts: list[Alert] = []

    def add(level: Level | None, title: str, detail: str) -> None:
        if level:
            alerts.append(Alert(level=level, title=title, detail=detail))

    mf = summary.mapped_reads_frac
    add(
        level_below(mf, THRESH.mapped_frac),
        "Low fraction of reads mapped",
        f"{mf or 0:.1%} of reads mapped (expected ≥ {THRESH.mapped_frac.warn:.0%}). "
        f"The biggest unmapped category is “{summary.top_unmapped_reason}”.",
    )
    fu = summary.failed_umi_qual_of_total
    add(
        level_above(fu, THRESH.failed_umi_qual_of_total),
        "Many reads failed UMI quality",
        f"{fu or 0:.1%} of all reads failed the UMI quality filter. "
        "This can point to low base quality in R1.",
    )
    rules(summary, probes, add)

    return alerts or [Alert(level="ok", title="No issues detected", detail="All checked metrics are within expected ranges.")]
