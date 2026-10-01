"""Per-probe and run-level QC metrics for a ``cyto workflow gex`` run.

Everything here is a transformation of a parsed :class:`~pycyto.qc.parse.CytoRun`.
Chart inputs live in :mod:`pycyto.qc.plots`.
"""

from typing import Self

import numpy as np
import polars as pl
from pydantic import BaseModel

from .parse import CytoRun


def _div(a, b) -> float | None:
    return a / b if a is not None and b else None


class ProbeMetrics(BaseModel):
    """One row of the report's probe table.

    Attributes
    ----------
    probe : str
        Probe barcode.
    mapped_reads : int
        Mapped reads under this probe barcode, over all cell barcodes.
    umis : int
        Deduplicated UMIs under this probe barcode, over all cell barcodes.
    cells : int
        Called cells.
    reads_in_cells : int
        Mapped reads belonging to called cells.
    frac_reads_in_cells : float or None
        ``reads_in_cells / mapped_reads``.
    median_umis_per_cell : float or None
        Median UMIs over called cells; None when the probe has no cells.
    """

    probe: str
    mapped_reads: int
    umis: int
    cells: int
    reads_in_cells: int
    frac_reads_in_cells: float | None
    median_umis_per_cell: float | None

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
        in_cells = run.cells.filter(pl.col("probe") == probe)
        mapped = reads["n_reads"].sum()
        return cls(
            probe=probe,
            mapped_reads=mapped,
            umis=reads["n_umis"].sum(),
            cells=in_cells.height,
            reads_in_cells=in_cells["n_reads"].sum(),
            frac_reads_in_cells=_div(in_cells["n_reads"].sum(), mapped),
            median_umis_per_cell=in_cells["n_umis"].median(),
        )


class SummaryMetrics(BaseModel):
    """Run-level metrics behind the report's Summary tab and ``*_metrics_summary.csv``.

    Fields are ``None`` when undefined, e.g. per-cell medians on a run with no cells.

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
    genes_in_reference : int
        Features in the count matrices, or in the ``gex`` library when no probe has cells.
    probe_barcodes_with_reads : int
        Probe barcodes with a reads table.
    seq_saturation : float or None
        ``1 - UMIs / mapped reads`` over all barcodes.
    umi_corrected_frac : float or None
        Corrected UMIs / total UMIs over all probe barcodes.
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

    cyto_outdir: str
    total_reads: int
    mapped_reads: int
    mapped_reads_frac: float
    top_unmapped_reason: str | None
    failed_umi_qual_of_total: float | None
    probe_barcodes_in_library: int
    genes_in_reference: int
    probe_barcodes_with_reads: int
    seq_saturation: float | None
    umi_corrected_frac: float | None
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
        cells = run.cells
        mapping, lib = run.stats.mapping, run.stats.library.entries
        umi = run.stats.umi.entries.values()
        counts = list(run.counts.values())
        called = [p for p in probes if p.cells > 0]
        cells_per_probe = np.array([p.cells for p in called], dtype=float)
        mapped = sum(p.mapped_reads for p in probes)
        n_cells = cells.height
        return cls(
            cyto_outdir=run.path,
            total_reads=mapping.total_reads,
            mapped_reads=mapping.mapped_reads,
            mapped_reads_frac=mapping.mapped_reads_frac,
            top_unmapped_reason=mapping.unmapped[0].label if mapping.unmapped else None,
            failed_umi_qual_of_total=_div(mapping.unmapped_reads("failed_umi_qual"), mapping.total_reads),
            probe_barcodes_in_library=lib["probe"].total_elem,
            genes_in_reference=counts[0].n_features if counts else lib["gex"].total_aggr,
            probe_barcodes_with_reads=len(probes),
            seq_saturation=1 - sum(p.umis for p in probes) / mapped if mapped else None,
            umi_corrected_frac=_div(sum(u.corrected for u in umi), sum(u.total for u in umi)),
            estimated_cells=n_cells,
            probe_barcodes_with_cells=len(called),
            probe_barcodes_without_cells=len(probes) - len(called),
            cells_median_per_probe=float(np.median(cells_per_probe)) if called else None,
            cells_cv_per_probe=float(cells_per_probe.std(ddof=1) / cells_per_probe.mean()) if len(called) >= 3 else None,
            mean_reads_per_cell=_div(mapping.total_reads, n_cells),
            mean_mapped_reads_per_cell=_div(mapping.mapped_reads, n_cells),
            median_umis_per_cell=cells["n_umis"].median(),
            median_genes_per_cell=cells["n_genes"].median(),
            total_genes_detected=int(np.logical_or.reduce([c.features_detected for c in counts]).sum()) if counts else None,
            frac_reads_in_cells=_div(cells["n_reads"].sum(), mapped),
            background_probe_read_frac=_div(sum(p.mapped_reads for p in probes if p.cells == 0), mapped),
        )
