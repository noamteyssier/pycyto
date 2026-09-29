"""Cell Ranger-style alerts, derived from the computed metrics.

:class:`Alerts` holds the rules every workflow shares, one field per rule;
:class:`GexAlerts` / :class:`CrisprAlerts` add their own. ``Alerts.triggered()`` is the
list the report shows. Cutoffs live in :mod:`pycyto.qc.thresholds`, grouped the same way.
"""

from collections.abc import Callable
from typing import Self

from pydantic import BaseModel

from .metrics import CrisprMetrics, GexMetrics, GexProbeMetrics, Metrics, ProbeMetrics
from .thresholds import THRESH, Band, Level


# ============================================================================
# Alerts and the helpers rules are written with
# ============================================================================
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

    @classmethod
    def when(cls, level: Level | None, title: str, detail: Callable[[], str]) -> Self | None:
        """The alert, or None when ``level`` is None (the rule did not trigger).

        Parameters
        ----------
        level : Level or None
            From :func:`level_below` / :func:`level_above`, or a rule's own test.
        title : str
        detail : Callable[[], str]
            Builds the detail text; only called when the rule triggered, so it may assume
            the values it formats are not None.

        Returns
        -------
        Alert or None
        """
        return cls(level=level, title=title, detail=detail()) if level else None


NO_ISSUES = Alert(level="ok", title="No issues detected", detail="All checked metrics are within expected ranges.")
"""What the report shows when no rule triggered."""


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


# ============================================================================
# Shared
# ============================================================================
class Alerts(BaseModel):
    """The alert rules every workflow runs, one field per rule.

    A field is the :class:`Alert` when its rule triggered, None otherwise. Workflow
    subclasses add their rules as further fields and override ``compute`` to build the
    base model and extend it.

    Attributes
    ----------
    low_mapped_frac : Alert or None
        Fraction of reads mapped below :attr:`Thresholds.mapped_frac`.
    failed_umi_qual : Alert or None
        Fraction of all reads failing UMI quality above :attr:`Thresholds.failed_umi_qual_of_total`.
    """

    low_mapped_frac: Alert | None
    failed_umi_qual: Alert | None

    @classmethod
    def compute(cls, metrics: Metrics) -> Self:
        """Run every rule against a run's metrics.

        Parameters
        ----------
        metrics : Metrics

        Returns
        -------
        Alerts
        """
        summary = metrics.summary
        mapped_frac = summary.mapped_reads_frac
        failed_umi_qual = summary.failed_umi_qual_of_total
        return cls(
            low_mapped_frac=Alert.when(
                level_below(mapped_frac, THRESH.mapped_frac),
                "Low fraction of reads mapped",
                lambda: f"{mapped_frac:.1%} of reads mapped (expected ≥ {THRESH.mapped_frac.warn:.0%}). "
                f"The biggest unmapped category is “{summary.top_unmapped_reason}”.",
            ),
            failed_umi_qual=Alert.when(
                level_above(failed_umi_qual, THRESH.failed_umi_qual_of_total),
                "Many reads failed UMI quality",
                lambda: f"{failed_umi_qual:.1%} of all reads failed the UMI quality filter. "
                "This can point to low base quality in R1.",
            ),
        )

    def triggered(self) -> list[Alert]:
        """Every alert that fired, in field order, or :data:`NO_ISSUES` when none did.

        This is the list the report's banner shows.
        """
        return [a for name in type(self).model_fields if (a := getattr(self, name))] or [NO_ISSUES]


# ============================================================================
# GEX
# ============================================================================
class GexAlerts(Alerts):
    """Alert rules for a GEX run: cells, reads in cells, background probe barcodes.

    Attributes
    ----------
    low_reads_in_cells : Alert or None
        Run-wide fraction of mapped reads in cells below :attr:`GexThresholds.frac_reads_in_cells`.
    background_probe_reads : Alert or None
        Fraction of mapped reads in probe barcodes without cells above
        :attr:`GexThresholds.background_probe_read_frac`.
    low_median_umis : Alert or None
        Probe barcodes whose median UMIs per cell is below :attr:`GexThresholds.median_umis_per_cell`.
    low_probe_reads_in_cells : Alert or None
        Probe barcodes with cells whose fraction of reads in cells is below the error level of
        :attr:`GexThresholds.frac_reads_in_cells`, excluding those already in ``low_median_umis``.
    uneven_cells : Alert or None
        Coefficient of variation of cells per probe barcode above :attr:`GexThresholds.cells_cv`.
    """

    low_reads_in_cells: Alert | None
    background_probe_reads: Alert | None
    low_median_umis: Alert | None
    low_probe_reads_in_cells: Alert | None
    uneven_cells: Alert | None

    @classmethod
    def compute(cls, metrics: GexMetrics) -> Self:
        base = Alerts.compute(metrics)
        summary = metrics.summary
        thresh = THRESH.gex
        frac_reads_in_cells = summary.frac_reads_in_cells
        background_frac = summary.background_probe_read_frac
        cells_cv = summary.cells_cv_per_probe
        called = [p for p in metrics.probes if p.cells > 0]
        low_umis = cls._below_median_umis(called)
        low_frac = cls._below_frac_reads_in_cells(called, exclude={p.probe for p in low_umis})
        return cls(
            **base.model_dump(),
            low_reads_in_cells=Alert.when(
                level_below(frac_reads_in_cells, thresh.frac_reads_in_cells),
                "Low fraction of reads in cells",
                lambda: f"{frac_reads_in_cells:.1%} of mapped reads are in cell barcodes "
                f"(expected ≥ {thresh.frac_reads_in_cells.warn:.0%}).",
            ),
            background_probe_reads=Alert.when(
                level_above(background_frac, thresh.background_probe_read_frac),
                "Reads in probe barcodes without cells",
                lambda: f"{background_frac:.1%} of mapped reads went to {summary.probe_barcodes_without_cells} probe "
                "barcodes with no cells. Check for unexpected probe barcodes or barcode hopping.",
            ),
            low_median_umis=Alert.when(
                "warn" if low_umis else None,
                "Low median UMIs per cell",
                lambda: f"{len(low_umis)} probe barcode(s) with cells have a median below "
                f"{thresh.median_umis_per_cell:g} UMIs per cell: "
                + examples(low_umis, lambda p: f"{p.probe} ({p.median_umis_per_cell:.0f})")
                + ".",
            ),
            low_probe_reads_in_cells=Alert.when(
                "warn" if low_frac else None,
                "Probe barcodes with low reads in cells",
                lambda: f"{len(low_frac)} probe barcode(s) have <{thresh.frac_reads_in_cells.error:.0%} of reads in cells: "
                + examples(low_frac, lambda p: f"{p.probe} ({p.frac_reads_in_cells:.0%})")
                + ".",
            ),
            uneven_cells=Alert.when(
                level_above(cells_cv, thresh.cells_cv),
                "Uneven cell counts across probe barcodes",
                lambda: f"Coefficient of variation of cells per probe barcode is {cells_cv:.2f}.",
            ),
        )

    @staticmethod
    def _below_median_umis(called: list[GexProbeMetrics]) -> list[GexProbeMetrics]:
        """Probe barcodes with cells whose median UMIs per cell is below the cutoff."""
        cutoff = THRESH.gex.median_umis_per_cell
        return [p for p in called if p.median_umis_per_cell is not None and p.median_umis_per_cell < cutoff]

    @staticmethod
    def _below_frac_reads_in_cells(called: list[GexProbeMetrics], exclude: set[str]) -> list[GexProbeMetrics]:
        """Probe barcodes with cells whose fraction of reads in cells is below the error level.

        ``exclude`` lists probe barcodes already reported by another rule.
        """
        cutoff = THRESH.gex.frac_reads_in_cells.error
        return [
            p
            for p in called
            if p.frac_reads_in_cells is not None and p.frac_reads_in_cells < cutoff and p.probe not in exclude
        ]


# ============================================================================
# CRISPR
# ============================================================================
class CrisprAlerts(Alerts):
    """Alert rules for a CRISPR run: guide library coverage and evenness.

    Attributes
    ----------
    guides_missing : Alert or None
        Fraction of library guides detected below :attr:`CrisprThresholds.frac_guides_detected`.
    uneven_guides : Alert or None
        90th / 10th percentile ratio of UMIs per guide above :attr:`CrisprThresholds.guide_skew_ratio`.
    """

    guides_missing: Alert | None
    uneven_guides: Alert | None

    @classmethod
    def compute(cls, metrics: CrisprMetrics) -> Self:
        base = Alerts.compute(metrics)
        summary = metrics.summary
        thresh = THRESH.crispr
        frac_detected = summary.frac_guides_detected
        skew = summary.guide_skew_ratio
        return cls(
            **base.model_dump(),
            guides_missing=Alert.when(
                level_below(frac_detected, thresh.frac_guides_detected),
                "Guides missing from the library",
                lambda: f"Only {frac_detected:.1%} of the {summary.guides_in_library:,} guides in the library have "
                f"any UMIs (expected ≥ {thresh.frac_guides_detected.warn:.0%}). Check library "
                "complexity and guide capture.",
            ),
            uneven_guides=Alert.when(
                level_above(skew, thresh.guide_skew_ratio),
                "Uneven guide coverage",
                lambda: f"The 90th/10th percentile ratio of UMIs per guide is {skew:.1f} "
                f"(expected ≤ {thresh.guide_skew_ratio:g}). A few guides may dominate the library.",
            ),
        )
