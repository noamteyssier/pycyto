"""The QC report payload: everything ``report.js`` reads, as one model.

:meth:`Report.compute` is the whole pipeline for a run: detect the workflow, read the
run, compute metrics, plots and alerts. :meth:`Report.write` renders the HTML (and the
metric CSVs); :meth:`Report.payload` is the dict embedded in it.
"""

import datetime as dt
import logging
import os
from importlib.metadata import version
from typing import Self

from pydantic import BaseModel
from pydantic_extra_types.semantic_version import SemanticVersion

from .alerts import Alert, CrisprAlerts, GexAlerts
from .metrics import CrisprMetrics, GexMetrics, ProbeMetrics, SummaryMetrics
from .parse import (
    CrisprCytoRun,
    GexCytoRun,
    InputRun,
    Library,
    UnmappedReason,
    Workflow,
    detect_workflow,
)
from .plots import CrisprPlots, GexPlots, Plots
from .render import render_html, write_csvs

logger = logging.getLogger("pycyto.qc")


class Report(BaseModel):
    """One QC report. The field names are the payload keys ``report.js`` reads.

    ``summary``, ``probes`` and ``plots`` are typed with the shared base models but hold
    the workflow's subclasses; :meth:`payload` dumps the subclass's fields.

    Attributes
    ----------
    workflow : Workflow
    title : str
        Shown in the header; defaults to the run directory's name.
    generated : str
        Local timestamp of this report.
    version : SemanticVersion
        pycyto version that wrote the report.
    summary : SummaryMetrics
        Run-level metrics (Summary tab, ``*_metrics_summary.csv``).
    alerts : list[Alert]
        Triggered alerts, or the single ``ok`` alert.
    unmapped : list[UnmappedReason]
        Why reads failed to map, largest first (Mapping & run tab).
    library : list[Library]
        Reference libraries mapped against (Mapping & run tab).
    run : list[InputRun]
        Mapping time per input file (Mapping & run tab).
    timings : dict[str, float]
        Seconds per pipeline module (Mapping & run tab).
    probes : list[ProbeMetrics]
        One row per probe barcode (Probe barcodes tab, ``*_probe_metrics.csv``).
    plots : Plots
        Chart inputs: ``log_bins``, per-probe ``probes`` and run-wide ``pooled``.
    """

    workflow: Workflow
    title: str
    generated: str
    version: SemanticVersion
    summary: SummaryMetrics
    alerts: list[Alert]
    unmapped: list[UnmappedReason]
    library: list[Library]
    run: list[InputRun]
    timings: dict[str, float]
    probes: list[ProbeMetrics]
    plots: Plots

    def payload(self) -> dict:
        """The report as the JSON-ready dict embedded in the HTML.

        ``polymorphic_serialization`` dumps every nested model by its runtime class, so the
        workflow subclasses' fields are included; JSON mode turns ``version`` into a string.
        """
        return self.model_dump(mode="json", polymorphic_serialization=True)

    def write(self, output: str | None = None, csv: bool = True) -> str:
        """Write the self-contained HTML report, and optionally the metric CSVs next to it.

        Parameters
        ----------
        output : str, optional
            HTML path; defaults to ``qc_report.html`` inside the run directory.
        csv : bool
            Also write ``<output stem>_metrics_summary.csv`` and ``<output stem>_probe_metrics.csv``.

        Returns
        -------
        str
            The HTML path.
        """
        payload = self.payload()
        output = output or os.path.join(self.summary.cyto_outdir, "qc_report.html")
        with open(output, "w", encoding="utf-8") as fh:
            fh.write(render_html(payload))
        logger.info(f"Wrote QC report: {output}")
        if csv:
            paths = write_csvs(payload, os.path.splitext(output)[0])
            logger.info(f"Wrote metrics: {', '.join(paths)}")
        return output

    @classmethod
    def compute(cls, cyto_outdir: str, title: str | None = None) -> Self:
        """Run the whole QC pipeline on a cyto output directory.

        Parameters
        ----------
        cyto_outdir : str
            A completed ``cyto workflow gex`` or ``crispr`` output directory.
        title : str, optional
            Report title; defaults to the directory name.

        Returns
        -------
        Report
        """
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
        return cls(
            workflow=workflow,
            title=title or os.path.basename(run.path),
            generated=dt.datetime.now().isoformat(sep=" ", timespec="seconds"),
            version=version("pycyto"),
            summary=metrics.summary,
            alerts=alerts.triggered(),
            unmapped=run.stats.mapping.unmapped,
            library=list(run.stats.library.entries.values()),
            run=run.stats.inputs,
            timings=run.timings.by_module,
            probes=metrics.probes,
            plots=plots,
        )
