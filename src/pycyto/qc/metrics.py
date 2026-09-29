"""QC metrics every cyto workflow reports.

Everything here is a transformation of a parsed :class:`~pycyto.qc.parse.CytoRun`.
Workflow modules (:mod:`pycyto.qc.gex`, :mod:`pycyto.qc.crispr`) subclass these models
and extend ``_fields`` with their own metrics. Chart inputs live in :mod:`pycyto.qc.plots`.
"""

from typing import Any, Self

from pydantic import BaseModel

from .parse import CytoRun


def safe_div(a, b) -> float | None:
    """``a / b``, or None when either is missing or ``b`` is zero."""
    return a / b if a is not None and b else None


class ProbeMetrics(BaseModel):
    """One row of the report's probe table: the metrics every workflow has per probe barcode.

    Attributes
    ----------
    probe : str
        Probe barcode.
    mapped_reads : int
        Mapped reads under this probe barcode, over all cell barcodes.
    umis : int
        Deduplicated UMIs under this probe barcode, over all cell barcodes.
    """

    probe: str
    mapped_reads: int
    umis: int

    @classmethod
    def compute(cls, run: CytoRun, probe: str) -> Self:
        """Metrics for one probe barcode.

        Parameters
        ----------
        run : CytoRun
        probe : str
            A member of ``run.stats.probes``.

        Returns
        -------
        ProbeMetrics
        """
        return cls(**cls._fields(run, probe))

    @classmethod
    def _fields(cls, run: CytoRun, probe: str) -> dict[str, Any]:
        """Constructor kwargs; subclasses extend with ``super()._fields(...) | {...}``."""
        reads = run.stats.reads.entries[probe]
        return {"probe": probe, "mapped_reads": reads["n_reads"].sum(), "umis": reads["n_umis"].sum()}


class SummaryMetrics(BaseModel):
    """Run-level metrics every workflow reports: reads, mapping, saturation.

    Behind the report's Summary tab and ``*_metrics_summary.csv``. Fields are ``None``
    when undefined.

    Attributes
    ----------
    cyto_outdir : str
        Absolute path to the run.
    total_reads, mapped_reads, mapped_reads_frac
        Straight from ``mapping_map.json``.
    top_unmapped_reason : str or None
        Label of the largest :class:`~pycyto.qc.parse.UnmappedReason`.
    failed_umi_qual_of_total : float or None
        Reads failing the UMI quality filter / ``total_reads``.
    probe_barcodes_in_library : int
        Probe barcodes in the reference, from ``mapping_lib.json``.
    probe_barcodes_with_reads : int
        Probe barcodes with a reads table.
    seq_saturation : float or None
        ``1 - UMIs / mapped reads`` over all barcodes.
    umi_corrected_frac : float or None
        Corrected UMIs / total UMIs over all probe barcodes.
    """

    cyto_outdir: str
    total_reads: int
    mapped_reads: int
    mapped_reads_frac: float
    top_unmapped_reason: str | None
    failed_umi_qual_of_total: float | None
    probe_barcodes_in_library: int
    probe_barcodes_with_reads: int
    seq_saturation: float | None
    umi_corrected_frac: float | None

    @classmethod
    def compute(cls, run: CytoRun, probes: list[ProbeMetrics]) -> Self:
        """Run-level metrics.

        Parameters
        ----------
        run : CytoRun
        probes : list[ProbeMetrics]
            One per probe barcode in ``run.stats.probes``.

        Returns
        -------
        SummaryMetrics
        """
        return cls(**cls._fields(run, probes))

    @classmethod
    def _fields(cls, run: CytoRun, probes: list[ProbeMetrics]) -> dict[str, Any]:
        """Constructor kwargs; subclasses extend with ``super()._fields(...) | {...}``."""
        mapping = run.stats.mapping
        umi = run.stats.umi.entries.values()
        mapped = sum(p.mapped_reads for p in probes)
        return {
            "cyto_outdir": run.path,
            "total_reads": mapping.total_reads,
            "mapped_reads": mapping.mapped_reads,
            "mapped_reads_frac": mapping.mapped_reads_frac,
            "top_unmapped_reason": mapping.unmapped[0].label if mapping.unmapped else None,
            "failed_umi_qual_of_total": safe_div(mapping.unmapped_reads("failed_umi_qual"), mapping.total_reads),
            "probe_barcodes_in_library": run.stats.library.entries["probe"].total_elem,
            "probe_barcodes_with_reads": len(probes),
            "seq_saturation": 1 - sum(p.umis for p in probes) / mapped if mapped else None,
            "umi_corrected_frac": safe_div(sum(u.corrected for u in umi), sum(u.total for u in umi)),
        }
