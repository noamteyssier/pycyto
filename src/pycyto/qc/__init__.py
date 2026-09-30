"""Cell Ranger-style QC reports for a single cyto output directory (``cyto workflow gex``).

Entry point: :func:`build_report` (CLI: ``pycyto qc``).
"""

import datetime as dt
import logging
import os
from importlib.metadata import version

from .metrics import ProbeMetrics, SummaryMetrics
from .parse import CytoRun
from .plots import LOG_BINS, PooledPlots, ProbePlots
from .render import render_html, write_csvs

__all__ = ["build_report", "collect"]

logger = logging.getLogger("pycyto.qc")


def collect(cyto_outdir: str, title: str | None = None) -> dict:
    """Compute every metric and plot input for the report; returns the report payload."""
    run = CytoRun.read(cyto_outdir)
    logger.info(f"Computing QC for a cyto {run.stats.workflow} run with {len(run.stats.probes)} probe barcodes")
    probes = [ProbeMetrics.compute(run, probe) for probe in run.stats.probes]
    return {
        "workflow": run.stats.workflow,
        "title": title or os.path.basename(run.path),
        "generated": dt.datetime.now().isoformat(sep=" ", timespec="seconds"),
        "version": version("pycyto"),
        "summary": SummaryMetrics.compute(run, probes).model_dump(),
        "probes": [p.model_dump() for p in probes],
        "plots": {probe: ProbePlots.compute(run, probe).model_dump() for probe in run.stats.probes},
        "pooled": PooledPlots.compute(run).model_dump(),
        "log_bins": LOG_BINS.tolist(),
    }


def build_report(
    cyto_outdir: str,
    output: str | None = None,
    title: str | None = None,
    write_csv: bool = True,
) -> str:
    """Write the HTML report (and optionally the metric CSVs). Returns the HTML path."""
    payload = collect(cyto_outdir, title=title)
    output = output or os.path.join(cyto_outdir, "qc_report.html")
    with open(output, "w", encoding="utf-8") as fh:
        fh.write(render_html(payload))
    logger.info(f"Wrote QC report: {output}")
    if write_csv:
        paths = write_csvs(payload, os.path.splitext(output)[0])
        logger.info(f"Wrote metrics: {', '.join(paths)}")
    return output
