"""Cell Ranger-style QC reports for a single ``cyto workflow gex`` or ``crispr`` output directory.

Entry point: :func:`build_report` (CLI: ``pycyto qc``). The workflow is detected from
``stats/mapping_lib.json``; its :class:`~pycyto.qc.workflow.Workflow` (``gex.WORKFLOW``,
``crispr.WORKFLOW``) supplies the run, metric, plot and alert classes, and everything
else is shared.
"""

import datetime as dt
import logging
import os
from importlib.metadata import version

from . import crispr, gex
from .alerts import build_alerts
from .parse import CytoStats
from .render import render_html, write_csvs
from .workflow import Workflow

__all__ = ["WORKFLOWS", "build_report", "collect"]

logger = logging.getLogger("pycyto.qc")

WORKFLOWS: dict[str, Workflow] = {"gex": gex.WORKFLOW, "crispr": crispr.WORKFLOW}


def collect(cyto_outdir: str, title: str | None = None) -> dict:
    """Compute every metric and plot input for the report; returns the report payload."""
    stats = CytoStats.read(cyto_outdir)
    wf = WORKFLOWS[stats.workflow]
    logger.info(f"Computing QC for a cyto {stats.workflow} run with {len(stats.probes)} probe barcodes")
    run = wf.run.from_stats(cyto_outdir, stats)
    probes = [wf.probe_metrics.compute(run, probe) for probe in stats.probes]
    summary = wf.summary.compute(run, probes)
    return {
        "workflow": stats.workflow,
        "title": title or os.path.basename(run.path),
        "generated": dt.datetime.now().isoformat(sep=" ", timespec="seconds"),
        "version": version("pycyto"),
        "summary": summary.model_dump(),
        "alerts": [a.model_dump() for a in build_alerts(summary, probes, wf.alert_rules)],
        "probes": [p.model_dump() for p in probes],
        "plots": wf.plots.compute(run).model_dump(),
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
    for alert in payload["alerts"]:
        if alert["level"] != "ok":
            logger.warning(f"{alert['title']}: {alert['detail']}")
    return output
