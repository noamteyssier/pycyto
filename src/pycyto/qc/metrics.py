"""QC metrics: per-probe :class:`ProbeMetrics`, run-level :class:`SummaryMetrics`, and each workflow's subclass.

Everything here is a transformation of a parsed :class:`~pycyto.qc.parse.CytoRun`.
:class:`Metrics` (:class:`GexMetrics`, :class:`CrisprMetrics`) bundles the two for a run;
``collect`` calls its ``compute``. Workflow subclasses of the metric models add their own
fields and override ``compute`` to build the base model and extend it:
``cls(**Base.compute(...).model_dump(), extra=...)``. Chart inputs live in
:mod:`pycyto.qc.plots`, alert rules in :mod:`pycyto.qc.alerts`, cutoffs in
:mod:`pycyto.qc.thresholds`.
"""

from typing import Self

import numpy as np
from pydantic import BaseModel, computed_field

from ..config import FLEX_V2_BARCODE_RE
from .parse import BarcodeReadStats, CellTable, CrisprCytoRun, CytoRun, GexCytoRun
from .thresholds import THRESH, Level


def safe_div(a, b) -> float | None:
    """``a / b``, or None when either is missing or ``b`` is zero."""
    return a / b if a is not None and b else None


# ============================================================================
# Shared
# ============================================================================
class Well(BaseModel):
    """A Flex-V2 probe barcode's position on its 96-well plate, for the report's plate map.

    Attributes
    ----------
    set : str
        Probe set, ``A``-``D``; one plate each.
    row : str
        ``A``-``H``.
    col : int
        1-12.
    """

    set: str
    row: str
    col: int

    @classmethod
    def from_barcode(cls, barcode: str) -> Self | None:
        """The well for a Flex-V2 barcode (``A-B07`` -> set A, row B, column 7); None for other formats."""
        m = FLEX_V2_BARCODE_RE.match(barcode)
        return cls(set=m[1], row=m[2], col=int(m[3])) if m else None


class ProbeMetrics(BaseModel):
    """One row of the report's probe table: the metrics every workflow has per probe barcode.

    Attributes
    ----------
    Workflow subclasses add a ``flag`` (:data:`~pycyto.qc.thresholds.Level` or None): the
    per-probe-barcode status shown as a dot in the report.

    Attributes
    ----------
    probe : str
        Probe barcode.
    well : Well or None
        Plate position for Flex-V2 barcodes; None for other formats (the report then draws
        bars instead of a plate map).
    n_barcodes : int
        Cell barcodes seen under this probe barcode (rows of its reads table).
    mapped_reads : int
        Mapped reads under this probe barcode, over all cell barcodes.
    umis : int
        Deduplicated UMIs under this probe barcode, over all cell barcodes.
    frac_of_mapped_reads : float or None
        ``mapped_reads`` / mapped reads over every probe barcode.
    seq_saturation : float or None
        ``1 - umis / mapped_reads``.
    umi_corrected_frac : float
        Fraction of UMIs collapsed by error correction, from ``stats/umi``.
    """

    probe: str
    well: Well | None
    n_barcodes: int
    mapped_reads: int
    umis: int
    frac_of_mapped_reads: float | None
    seq_saturation: float | None
    umi_corrected_frac: float

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
        mapped = int(reads[BarcodeReadStats.n_reads].sum())
        umis = int(reads[BarcodeReadStats.n_umis].sum())
        return cls(
            probe=probe,
            well=Well.from_barcode(probe),
            n_barcodes=reads.height,
            mapped_reads=mapped,
            umis=umis,
            frac_of_mapped_reads=safe_div(mapped, run.stats.reads.mapped_reads),
            seq_saturation=1 - umis / mapped if mapped else None,
            umi_corrected_frac=run.stats.umi.entries[probe].fraction_corrected,
        )


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
    whitelist_size : int
        Cell barcodes in the whitelist library.
    seq_saturation : float or None
        ``1 - UMIs / mapped reads`` over all barcodes.
    umi_corrected_frac : float or None
        Corrected UMIs / total UMIs over all probe barcodes.
    mapping_sec : float
        Wall time of the mapping step, from ``.timings.tsv``.
    n_inputs : int
        Input FASTQ files (pairs) mapped.
    """

    cyto_outdir: str
    total_reads: int
    mapped_reads: int
    mapped_reads_frac: float
    top_unmapped_reason: str | None
    failed_umi_qual_of_total: float | None
    probe_barcodes_in_library: int
    probe_barcodes_with_reads: int
    whitelist_size: int
    seq_saturation: float | None
    umi_corrected_frac: float | None
    mapping_sec: float
    n_inputs: int

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
            whitelist_size=run.stats.library.entries["whitelist"].total_elem,
            seq_saturation=1 - sum(p.umis for p in probes) / mapped if mapped else None,
            umi_corrected_frac=safe_div(sum(u.corrected for u in umi), sum(u.total for u in umi)),
            mapping_sec=run.stats.timings["Mapping"],
            n_inputs=len(run.stats.inputs),
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
    has_filtered_h5ad : bool
        Whether cyto wrote ``counts/<probe>.filt.h5ad``; without it the probe has no cells.
    cells : int
        Called cells.
    reads_in_cells : int
        Mapped reads belonging to called cells.
    frac_reads_in_cells, frac_umis_in_cells : float or None
        Reads / UMIs in called cells as a fraction of the probe's ``mapped_reads`` / ``umis``.
    mean_reads_per_cell : float or None
        ``mapped_reads / cells``.
    median_umis_per_cell, median_genes_per_cell : float or None
        Medians over called cells; None when the probe has no cells.
    total_genes_detected : int or None
        Features with a nonzero total over the probe's called cells; None without a filtered h5ad.
    flag : Level or None
        Computed. None without cells; ``warn`` when :attr:`low_median_umis` or
        :attr:`low_frac_reads_in_cells`; ``ok`` otherwise.
    """

    has_filtered_h5ad: bool
    cells: int
    reads_in_cells: int
    frac_reads_in_cells: float | None
    frac_umis_in_cells: float | None
    mean_reads_per_cell: float | None
    median_umis_per_cell: float | None
    median_genes_per_cell: float | None
    total_genes_detected: int | None

    @property
    def low_median_umis(self) -> bool:
        """Has cells, and their median UMIs per cell is below :attr:`GexThresholds.median_umis_per_cell`."""
        return self.cells > 0 and self.median_umis_per_cell < THRESH.gex.median_umis_per_cell

    @property
    def low_frac_reads_in_cells(self) -> bool:
        """Has cells, and the fraction of reads in them is below the error level of :attr:`GexThresholds.frac_reads_in_cells`."""
        frac = self.frac_reads_in_cells
        return self.cells > 0 and frac is not None and frac < THRESH.gex.frac_reads_in_cells.error

    @computed_field
    @property
    def flag(self) -> Level | None:
        if self.cells == 0:
            return None
        return "warn" if self.low_median_umis or self.low_frac_reads_in_cells else "ok"

    @classmethod
    def compute(cls, run: GexCytoRun, probe: str) -> Self:
        base = ProbeMetrics.compute(run, probe)
        counts = run.counts.get(probe)
        in_cells = run.cells_by_probe[probe]
        cells = in_cells.height
        reads_in_cells = int(in_cells[CellTable.n_reads].sum())
        return cls(
            **base.model_dump(),
            has_filtered_h5ad=counts is not None,
            cells=cells,
            reads_in_cells=reads_in_cells,
            frac_reads_in_cells=safe_div(reads_in_cells, base.mapped_reads),
            frac_umis_in_cells=safe_div(in_cells[CellTable.n_umis].sum(), base.umis),
            mean_reads_per_cell=safe_div(base.mapped_reads, cells),
            median_umis_per_cell=in_cells[CellTable.n_umis].median(),
            median_genes_per_cell=in_cells[CellTable.n_genes].median(),
            total_genes_detected=int((counts.feature_totals > 0).sum()) if counts else None,
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
# CRISPR
# ============================================================================
class CrisprProbeMetrics(ProbeMetrics):
    """Per-probe-barcode metrics for a CRISPR run.

    Attributes
    ----------
    guides_detected : int
        Guides with at least one UMI under this probe barcode.
    flag : None
        A CRISPR run has no cell calls or assignments, so there is no per-probe verdict.
    """

    guides_detected: int
    flag: None = None

    @classmethod
    def compute(cls, run: CrisprCytoRun, probe: str) -> Self:
        base = ProbeMetrics.compute(run, probe)
        return cls(**base.model_dump(), guides_detected=int((run.guide_umis[probe] > 0).sum()))


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
    probes : list[CrisprProbeMetrics]
    summary : CrisprSummaryMetrics
    """

    probes: list[CrisprProbeMetrics]
    summary: CrisprSummaryMetrics

    @classmethod
    def compute(cls, run: CrisprCytoRun) -> Self:
        probes = [CrisprProbeMetrics.compute(run, probe) for probe in run.stats.probes]
        return cls(probes=probes, summary=CrisprSummaryMetrics.compute(run, probes))
