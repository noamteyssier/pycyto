"""Cell Ranger-style QC reports for a single ``cyto workflow gex`` or ``crispr`` output directory.

Entry point: :func:`build_report` (CLI: ``pycyto qc``). The workflow is detected from
``stats/mapping_lib.json``; :func:`collect` picks that workflow's run, metric and plot
models and alert rules (:mod:`pycyto.qc.parse`, :mod:`pycyto.qc.metrics`,
:mod:`pycyto.qc.plots`, :mod:`pycyto.qc.alerts`).
"""

import datetime as dt
import logging
import os
from importlib.metadata import version

from .alerts import CrisprAlerts, GexAlerts
from .metrics import CrisprMetrics, GexMetrics
from .parse import CrisprCytoRun, GexCytoRun, detect_workflow
from .plots import CrisprPlots, GexPlots
from .render import render_html, write_csvs

__all__ = ["build_report", "collect"]

logger = logging.getLogger("pycyto.qc")


def collect(cyto_outdir: str, title: str | None = None) -> dict:
    """Compute every metric and plot input for the report; returns the report payload."""
    workflow = detect_workflow(cyto_outdir)
    logger.info(f"Computing QC for a cyto {workflow} run")
    match workflow:
        case "gex":
            run = GexCytoRun.read(cyto_outdir)
            metrics = GexMetrics.compute(run)
            plots = GexPlots.compute(run)
            alerts = GexAlerts.compute(metrics)
        case "crispr":
            run = CrisprCytoRun.read(cyto_outdir)
            metrics = CrisprMetrics.compute(run)
            plots = CrisprPlots.compute(run)
            alerts = CrisprAlerts.compute(metrics)
    # serialize_as_any: nested fields are typed with the shared base models (e.g.
    # ``Plots.probes``) but hold the workflow's subclasses; dump all of their fields
    dump = lambda m: m.model_dump(serialize_as_any=True)
    return {
        "workflow": workflow,
        "title": title or os.path.basename(run.path),
        "generated": dt.datetime.now().isoformat(sep=" ", timespec="seconds"),
        "version": version("pycyto"),
        "summary": dump(metrics.summary),
        "alerts": [a.model_dump() for a in alerts.triggered()],
        "probes": [dump(p) for p in metrics.probes],
        "plots": dump(plots),
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
