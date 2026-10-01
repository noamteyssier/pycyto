"""QC metrics: per-probe :class:`ProbeMetrics`, run-level :class:`SummaryMetrics`, and each workflow's subclass.

Everything here is a transformation of a parsed :class:`~pycyto.qc.parse.CytoRun`.
:class:`Metrics` (:class:`GexMetrics`, :class:`CrisprMetrics`) bundles the two for a run;
``collect`` calls its ``compute``. Workflow subclasses of the metric models add their own
fields and override ``compute`` to build the base model and extend it:
``cls(**Base.compute(...).model_dump(), extra=...)``. Chart inputs live in
:mod:`pycyto.qc.plots`, alert rules in :mod:`pycyto.qc.alerts`.
"""

from typing import Self

import numpy as np
from pydantic import BaseModel

from .parse import BarcodeReadStats, CellTable, CrisprCytoRun, CytoRun, GexCytoRun


def safe_div(a, b) -> float | None:
    """``a / b``, or None when either is missing or ``b`` is zero."""
    return a / b if a is not None and b else None


# ============================================================================
# Shared
# ============================================================================
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
        reads = run.stats.reads.entries[probe]
        return cls(probe=probe, mapped_reads=reads[BarcodeReadStats.n_reads].sum(), umis=reads[BarcodeReadStats.n_umis].sum())


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
        mapping = run.stats.mapping
        umi = run.stats.umi.entries.values()
        mapped = sum(p.mapped_reads for p in probes)
        return cls(
            cyto_outdir=run.path,
            total_reads=mapping.total_reads,
            mapped_reads=mapping.mapped_reads,
            mapped_reads_frac=mapping.mapped_reads_frac,
            top_unmapped_reason=mapping.unmapped[0].label if mapping.unmapped else None,
            failed_umi_qual_of_total=safe_div(mapping.unmapped_reads("failed_umi_qual"), mapping.total_reads),
            probe_barcodes_in_library=run.stats.library.entries["probe"].total_elem,
            probe_barcodes_with_reads=len(probes),
            seq_saturation=1 - sum(p.umis for p in probes) / mapped if mapped else None,
            umi_corrected_frac=safe_div(sum(u.corrected for u in umi), sum(u.total for u in umi)),
        )


class Metrics(BaseModel):
    """All metrics for a run: one :class:`ProbeMetrics` per probe barcode plus the :class:`SummaryMetrics`.

    Subclasses redeclare the fields with their workflow's metric models and override
    ``compute`` to build them.

    Attributes
    ----------
    probes : list[ProbeMetrics]
        In :attr:`CytoStats.probes` order; the rows of the report's probe table.
    summary : SummaryMetrics
    """

    probes: list[ProbeMetrics]
    summary: SummaryMetrics

    @classmethod
    def compute(cls, run: CytoRun) -> Self:
        """Every metric for a run.

        Parameters
        ----------
        run : CytoRun

        Returns
        -------
        Metrics
        """
        probes = [ProbeMetrics.compute(run, probe) for probe in run.stats.probes]
        return cls(probes=probes, summary=SummaryMetrics.compute(run, probes))


# ============================================================================
# GEX
# ============================================================================
class GexProbeMetrics(ProbeMetrics):
    """Per-probe-barcode metrics for a GEX run.

    Attributes
    ----------
    cells : int
        Called cells.
    reads_in_cells : int
        Mapped reads belonging to called cells.
    frac_reads_in_cells : float or None
        ``reads_in_cells / mapped_reads``.
    median_umis_per_cell : float or None
        Median UMIs over called cells; None when the probe has no cells.
    """

    cells: int
    reads_in_cells: int
    frac_reads_in_cells: float | None
    median_umis_per_cell: float | None

    @classmethod
    def compute(cls, run: GexCytoRun, probe: str) -> Self:
        base = ProbeMetrics.compute(run, probe)
        in_cells = run.cells_by_probe[probe]
        reads_in_cells = in_cells[CellTable.n_reads].sum()
        return cls(
            **base.model_dump(),
            cells=in_cells.height,
            reads_in_cells=reads_in_cells,
            frac_reads_in_cells=safe_div(reads_in_cells, base.mapped_reads),
            median_umis_per_cell=in_cells[CellTable.n_umis].median(),
        )


class GexSummaryMetrics(SummaryMetrics):
    """Run-level metrics for a GEX run.

    Attributes
    ----------
    genes_in_reference : int
        Genes in the ``gex`` library (``mapping_lib.json``), which is also the feature
        axis of every count matrix.
    estimated_cells : int
        Called cells over all probe barcodes.
    probe_barcodes_with_cells : int
        Probe barcodes with at least one called cell.
    probe_barcodes_without_cells : int
        Probe barcodes with reads but no called cells.
    cells_median_per_probe : float or None
        Median called cells among probe barcodes with cells.
    cells_cv_per_probe : float or None
        Coefficient of variation (sample std / mean) of called cells across probe barcodes
        with cells; None with fewer than three such probe barcodes.
    mean_reads_per_cell, mean_mapped_reads_per_cell : float or None
        ``total_reads`` and ``mapped_reads`` divided by ``estimated_cells``.
    median_umis_per_cell, median_genes_per_cell : float or None
        Medians over all called cells.
    total_genes_detected : int or None
        Features with a nonzero total in any probe's called cells.
    frac_reads_in_cells : float or None
        Mapped reads in called cells / all mapped reads.
    background_probe_read_frac : float or None
        Mapped reads in probe barcodes without cells / all mapped reads.
    """

    genes_in_reference: int
    estimated_cells: int
    probe_barcodes_with_cells: int
    probe_barcodes_without_cells: int
    cells_median_per_probe: float | None
    cells_cv_per_probe: float | None
    mean_reads_per_cell: float | None
    mean_mapped_reads_per_cell: float | None
    median_umis_per_cell: float | None
    median_genes_per_cell: float | None
    total_genes_detected: int | None
    frac_reads_in_cells: float | None
    background_probe_read_frac: float | None

    @classmethod
    def compute(cls, run: GexCytoRun, probes: list[GexProbeMetrics]) -> Self:
        base = SummaryMetrics.compute(run, probes)
        cells = run.cells
        counts = list(run.counts.values())
        called = [p for p in probes if p.cells > 0]
        cells_per_probe = np.array([p.cells for p in called], dtype=float)
        mapped = sum(p.mapped_reads for p in probes)
        n_cells = cells.height
        return cls(
            **base.model_dump(),
            genes_in_reference=run.stats.library.entries["gex"].total_aggr,
            estimated_cells=n_cells,
            probe_barcodes_with_cells=len(called),
            probe_barcodes_without_cells=len(probes) - len(called),
            cells_median_per_probe=float(np.median(cells_per_probe)) if called else None,
            cells_cv_per_probe=float(cells_per_probe.std(ddof=1) / cells_per_probe.mean()) if len(called) >= 3 else None,
            mean_reads_per_cell=safe_div(base.total_reads, n_cells),
            mean_mapped_reads_per_cell=safe_div(base.mapped_reads, n_cells),
            median_umis_per_cell=cells[CellTable.n_umis].median(),
            median_genes_per_cell=cells[CellTable.n_genes].median(),
            total_genes_detected=int((np.sum([c.feature_totals for c in counts], axis=0) > 0).sum()) if counts else None,
            frac_reads_in_cells=safe_div(cells[CellTable.n_reads].sum(), mapped),
            background_probe_read_frac=safe_div(sum(p.mapped_reads for p in probes if p.cells == 0), mapped),
        )


class GexMetrics(Metrics):
    """All metrics for a GEX run.

    Attributes
    ----------
    probes : list[GexProbeMetrics]
    summary : GexSummaryMetrics
    """

    probes: list[GexProbeMetrics]
    summary: GexSummaryMetrics

    @classmethod
    def compute(cls, run: GexCytoRun) -> Self:
        probes = [GexProbeMetrics.compute(run, probe) for probe in run.stats.probes]
        return cls(probes=probes, summary=GexSummaryMetrics.compute(run, probes))


# ============================================================================
# CRISPR (per-probe metrics are the shared ProbeMetrics)
# ============================================================================
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
    def compute(cls, run: CrisprCytoRun, probes: list[ProbeMetrics]) -> Self:
        base = SummaryMetrics.compute(run, probes)
        totals = run.guide_totals
        p10, p90 = np.percentile(totals, [10, 90])
        return cls(
            **base.model_dump(),
            guides_in_library=len(totals),
            guides_detected=int((totals > 0).sum()),
            frac_guides_detected=float((totals > 0).mean()),
            guide_umis=int(totals.sum()),
            median_umis_per_guide=float(np.median(totals)),
            guide_skew_ratio=safe_div(float(p90), float(p10)),
            mean_reads_per_probe=safe_div(base.mapped_reads, len(probes)),
        )


class CrisprMetrics(Metrics):
    """All metrics for a CRISPR run.

    Attributes
    ----------
    probes : list[ProbeMetrics]
        The shared per-probe metrics; CRISPR adds none of its own yet.
    summary : CrisprSummaryMetrics
    """

    summary: CrisprSummaryMetrics

    @classmethod
    def compute(cls, run: CrisprCytoRun) -> Self:
        probes = [ProbeMetrics.compute(run, probe) for probe in run.stats.probes]
        return cls(probes=probes, summary=CrisprSummaryMetrics.compute(run, probes))
