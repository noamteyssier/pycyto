"""The per-workflow pieces of the QC report, bundled so :func:`pycyto.qc.collect` can dispatch.

Each workflow module (:mod:`pycyto.qc.gex`, :mod:`pycyto.qc.crispr`) exposes one
:class:`Workflow` instance named ``WORKFLOW``.
"""

from dataclasses import dataclass

from .alerts import Rules
from .metrics import ProbeMetrics, SummaryMetrics
from .parse import CytoRun
from .plots import Plots


@dataclass(frozen=True)
class Workflow:
    """Everything that differs between cyto workflows.

    Attributes
    ----------
    run : type[CytoRun]
        Run model; its ``from_stats`` reads the workflow's count files.
    probe_metrics : type[ProbeMetrics]
        Per-probe-barcode metrics model.
    summary : type[SummaryMetrics]
        Run-level metrics model.
    plots : type[Plots]
        Chart inputs model.
    alert_rules : Rules
        Workflow-specific alert rules, run after the shared ones.
    """

    run: type[CytoRun]
    probe_metrics: type[ProbeMetrics]
    summary: type[SummaryMetrics]
    plots: type[Plots]
    alert_rules: Rules
