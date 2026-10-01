"""CRISPR (``cyto workflow crispr``) metrics, plots and alerts.

A CRISPR run has no cell calls of its own, so the report describes guide capture per probe
barcode and how evenly the guide library is covered across the run. Guide assignment
(``assignments/``) is not reported yet.
"""

from typing import Any

import numpy as np
from pydantic import BaseModel

from ..config import FlexBarcode
from .alerts import THRESH, Adder, level_above, level_below
from .metrics import ProbeMetrics, SummaryMetrics, safe_div
from .parse import CrisprRun
from .plots import Plots, ProbePlots, log_hist, probe_rank_curve
from .workflow import Workflow

TOP_GUIDES = 10
"""Most abundant guides listed in the report."""


class CrisprSummaryMetrics(SummaryMetrics):
    """Run-level metrics for a CRISPR run.

    Attributes
    ----------
    guides_in_library : int
        Guides in the count matrices.
    guides_detected : int
        Guides with at least one UMI over the whole run.
    frac_guides_detected : float
        ``guides_detected / guides_in_library``.
    guide_umis : int
        UMIs over every guide and probe barcode.
    median_umis_per_guide : float
        Median over every guide in the library, including guides with no UMIs.
    guide_skew_ratio : float or None
        90th / 10th percentile of UMIs per guide, a standard library-evenness measure;
        None when the 10th percentile is zero.
    mean_reads_per_probe : float or None
        Mapped reads / probe barcodes with reads.
    """

    guides_in_library: int
    guides_detected: int
    frac_guides_detected: float
    guide_umis: int
    median_umis_per_guide: float
    guide_skew_ratio: float | None
    mean_reads_per_probe: float | None

    @classmethod
    def _fields(cls, run: CrisprRun, probes: list[ProbeMetrics]) -> dict[str, Any]:
        totals = run.guide_totals
        p10, p90 = np.percentile(totals, [10, 90])
        return super()._fields(run, probes) | {
            "guides_in_library": len(totals),
            "guides_detected": int((totals > 0).sum()),
            "frac_guides_detected": float((totals > 0).mean()),
            "guide_umis": int(totals.sum()),
            "median_umis_per_guide": float(np.median(totals)),
            "guide_skew_ratio": safe_div(float(p90), float(p10)),
            "mean_reads_per_probe": safe_div(run.stats.mapping.mapped_reads, len(probes)),
        }


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
        :func:`~pycyto.qc.plots.log_hist` of UMIs per detected guide.
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
        :func:`~pycyto.qc.plots.log_hist` of UMIs per detected guide under this probe barcode.
    """

    guide_hist: list[int] | None


class CrisprPlots(Plots):
    """Chart inputs for a CRISPR run.

    Attributes
    ----------
    pooled : GuidePooled
    probes : dict[FlexBarcode, CrisprProbePlots]
    """

    pooled: GuidePooled
    probes: dict[FlexBarcode, CrisprProbePlots]

    @classmethod
    def _fields(cls, run: CrisprRun) -> dict[str, Any]:
        totals = run.guide_totals
        order = np.argsort(-totals, kind="stable")[:TOP_GUIDES]
        total = totals.sum()
        pooled = GuidePooled(
            guide_hist=log_hist(totals[totals > 0]),
            top_guides=[
                TopGuide(guide=run.guide_names[i], umis=int(totals[i]), frac=safe_div(float(totals[i]), float(total)))
                for i in order
                if totals[i] > 0
            ],
        )
        return super()._fields(run) | {"pooled": pooled}

    @classmethod
    def _probe(cls, run: CrisprRun, probe: str) -> CrisprProbePlots:
        umis = run.guide_umis[probe]
        return CrisprProbePlots(curve=probe_rank_curve(run, probe), guide_hist=log_hist(umis[umis > 0]))


def alert_rules(summary: CrisprSummaryMetrics, probes: list[ProbeMetrics], add: Adder) -> None:
    """CRISPR alert rules; see :func:`pycyto.qc.alerts.build_alerts`."""
    fd = summary.frac_guides_detected
    add(
        level_below(fd, THRESH.frac_guides_detected),
        "Guides missing from the library",
        f"Only {fd or 0:.1%} of the {summary.guides_in_library or 0:,} guides in the library have "
        f"any UMIs (expected ≥ {THRESH.frac_guides_detected.warn:.0%}). Check library "
        "complexity and guide capture.",
    )
    skew = summary.guide_skew_ratio
    add(
        level_above(skew, THRESH.guide_skew_ratio),
        "Uneven guide coverage",
        f"The 90th/10th percentile ratio of UMIs per guide is {skew or 0:.1f} "
        f"(expected ≤ {THRESH.guide_skew_ratio:g}). A few guides may dominate the library.",
    )


WORKFLOW = Workflow(
    run=CrisprRun, probe_metrics=ProbeMetrics, summary=CrisprSummaryMetrics, plots=CrisprPlots, alert_rules=alert_rules
)
