"""Cell Ranger-style QC reports for a single ``cyto workflow gex`` or ``crispr`` output directory.

Entry point: :class:`Report` (CLI: ``pycyto qc``). ``Report.compute(cyto_outdir)`` detects
the workflow from ``stats/mapping_lib.json``, reads the run and computes its metrics,
plots and alerts (:mod:`pycyto.qc.parse`, :mod:`pycyto.qc.metrics`,
:mod:`pycyto.qc.plots`, :mod:`pycyto.qc.alerts`); ``Report.write()`` renders the HTML
and CSVs.
"""

from .report import Report

__all__ = ["Report"]
