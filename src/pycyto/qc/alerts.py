"""Cell Ranger-style alerts, derived from the computed metrics."""

from collections.abc import Callable
from typing import Literal, NamedTuple

import numpy as np
from pydantic import BaseModel, ConfigDict

from .metrics import ProbeMetrics, SummaryMetrics

Level = Literal["ok", "warn", "error"]


class Band(NamedTuple):
    """A two-level threshold: cross ``warn`` for a warning, ``error`` for an error."""

    warn: float
    error: float


class Thresholds(BaseModel):
    """Alert thresholds.

    Attributes
    ----------
    mapped_frac : Band
        Fraction of reads mapped; below -> warn / error.
    failed_umi_qual_of_total : float
        Fraction of all reads failing UMI quality; above -> warn.
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

    mapped_frac: Band = Band(warn=0.70, error=0.50)
    failed_umi_qual_of_total: float = 0.10
    frac_reads_in_cells: Band = Band(warn=0.70, error=0.50)
    background_probe_read_frac: float = 0.05
    median_umis_per_cell: float = 500
    cells_cv: float = 0.5


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


def _below(value: float | None, warn: float, error: float) -> Level | None:
    if value is None:
        return None
    return "error" if value < error else "warn" if value < warn else None


def _above(value: float | None, warn: float) -> Level | None:
    return "warn" if value is not None and value > warn else None


def _examples(probes: list[ProbeMetrics], fmt: Callable[[ProbeMetrics], str], limit: int = 10) -> str:
    return ", ".join(fmt(p) for p in probes[:limit]) + (" …" if len(probes) > limit else "")


def build_alerts(summary: SummaryMetrics, probes: list[ProbeMetrics]) -> list[Alert]:
    """Alerts for a run.

    Parameters
    ----------
    summary : SummaryMetrics
    probes : list[ProbeMetrics]
        One per probe barcode.

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
        _below(mf, *THRESH.mapped_frac),
        "Low fraction of reads mapped",
        f"{mf or 0:.1%} of reads mapped (expected ≥ {THRESH.mapped_frac.warn:.0%}). "
        f"The biggest unmapped category is “{summary.top_unmapped_reason}”.",
    )
    fu = summary.failed_umi_qual_of_total
    add(
        _above(fu, THRESH.failed_umi_qual_of_total),
        "Many reads failed UMI quality",
        f"{fu or 0:.1%} of all reads failed the UMI quality filter. "
        "This can point to low base quality in R1.",
    )
    fr = summary.frac_reads_in_cells
    add(
        _below(fr, *THRESH.frac_reads_in_cells),
        "Low fraction of reads in cells",
        f"{fr or 0:.1%} of mapped reads are in cell barcodes "
        f"(expected ≥ {THRESH.frac_reads_in_cells.warn:.0%}).",
    )
    bg = summary.background_probe_read_frac
    add(
        _above(bg, THRESH.background_probe_read_frac),
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
        + _examples(low, lambda p: f"{p.probe} ({p.median_umis_per_cell:.0f})")
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
        + _examples(low_frac, lambda p: f"{p.probe} ({p.frac_reads_in_cells:.0%})")
        + ".",
    )
    if len(called) >= 3:
        n_cells = np.array([p.cells for p in called], dtype=float)
        cv = float(n_cells.std(ddof=1) / n_cells.mean())
        add(
            _above(cv, THRESH.cells_cv),
            "Uneven cell counts across probe barcodes",
            f"Coefficient of variation of cells per probe barcode is {cv:.2f}.",
        )

    return alerts or [Alert(level="ok", title="No issues detected", detail="All checked metrics are within expected ranges.")]
