"""Per-probe and run-level QC metrics for a ``cyto workflow gex`` run.

Everything here is a transformation of a parsed :class:`~pycyto.qc.parse.CytoRun`.
"""

import logging
from typing import Self

import numpy as np
import polars as pl
from pydantic import BaseModel

from .parse import CytoRun

logger = logging.getLogger("pycyto.qc")

_CELLS_SCHEMA = {
    "probe": pl.String,
    "barcode": pl.Categorical,
    "n_umis": pl.Int64,
    "n_reads": pl.Int64,
    "n_genes": pl.Int64,
}


def _div(a, b) -> float | None:
    return a / b if a is not None and b else None


def cell_table(run: CytoRun) -> pl.DataFrame:
    """Every called cell in the run, one row each.

    A cell is a barcode in a probe's :class:`~pycyto.qc.parse.FilteredCounts` that also
    appears in that probe's reads table. Called cells missing from the reads table are
    dropped with a warning.

    Parameters
    ----------
    run : CytoRun

    Returns
    -------
    pl.DataFrame
        Columns ``probe, barcode, n_umis, n_reads, n_genes``.
    """
    tables = []
    for probe, counts in run.counts.items():
        cells = counts.cells.join(run.stats.reads.entries[probe], on="barcode", how="inner")
        if (missing := counts.cells.height - cells.height) > 0:
            logger.warning(f"[{probe}] - {missing} filtered barcodes missing from reads stats")
        tables.append(cells.with_columns(probe=pl.lit(probe)))
    return pl.concat([pl.DataFrame(schema=_CELLS_SCHEMA), *tables], how="diagonal")


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
    """

    probe: str
    mapped_reads: int
    umis: int
    cells: int
    reads_in_cells: int

    @classmethod
    def compute(cls, run: CytoRun, cells: pl.DataFrame, probe: str) -> Self:
        """Metrics for one probe barcode.

        Parameters
        ----------
        run : CytoRun
        cells : pl.DataFrame
            Output of :func:`cell_table` for ``run``.
        probe : str
            A member of ``run.stats.probes``.

        Returns
        -------
        ProbeMetrics
        """
        reads = run.stats.reads.entries[probe]
        in_cells = cells.filter(pl.col("probe") == probe)
        return cls(
            probe=probe,
            mapped_reads=reads["n_reads"].sum(),
            umis=reads["n_umis"].sum(),
            cells=in_cells.height,
            reads_in_cells=in_cells["n_reads"].sum(),
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
    cells_median_per_probe : float or None
        Median called cells among probe barcodes with cells.
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
    probe_barcodes_in_library: int
    genes_in_reference: int
    probe_barcodes_with_reads: int
    seq_saturation: float | None
    umi_corrected_frac: float | None
    estimated_cells: int
    probe_barcodes_with_cells: int
    cells_median_per_probe: float | None
    mean_reads_per_cell: float | None
    mean_mapped_reads_per_cell: float | None
    median_umis_per_cell: float | None
    median_genes_per_cell: float | None
    total_genes_detected: int | None
    frac_reads_in_cells: float | None
    background_probe_read_frac: float | None

    @classmethod
    def compute(cls, run: CytoRun, cells: pl.DataFrame, probes: list[ProbeMetrics]) -> Self:
        """Run-level metrics.

        Parameters
        ----------
        run : CytoRun
        cells : pl.DataFrame
            Output of :func:`cell_table` for ``run``.
        probes : list[ProbeMetrics]
            One per probe barcode in ``run.stats.probes``.

        Returns
        -------
        SummaryMetrics
        """
        mapping, lib = run.stats.mapping, run.stats.library.entries
        umi = run.stats.umi.entries.values()
        counts = list(run.counts.values())
        called = [p for p in probes if p.cells > 0]
        mapped = sum(p.mapped_reads for p in probes)
        n_cells = cells.height
        return cls(
            cyto_outdir=run.path,
            total_reads=mapping.total_reads,
            mapped_reads=mapping.mapped_reads,
            mapped_reads_frac=mapping.mapped_reads_frac,
            probe_barcodes_in_library=lib["probe"].total_elem,
            genes_in_reference=counts[0].n_features if counts else lib["gex"].total_aggr,
            probe_barcodes_with_reads=len(probes),
            seq_saturation=1 - sum(p.umis for p in probes) / mapped if mapped else None,
            umi_corrected_frac=_div(sum(u.corrected for u in umi), sum(u.total for u in umi)),
            estimated_cells=n_cells,
            probe_barcodes_with_cells=len(called),
            cells_median_per_probe=float(np.median([p.cells for p in called])) if called else None,
            mean_reads_per_cell=_div(mapping.total_reads, n_cells),
            mean_mapped_reads_per_cell=_div(mapping.mapped_reads, n_cells),
            median_umis_per_cell=cells["n_umis"].median(),
            median_genes_per_cell=cells["n_genes"].median(),
            total_genes_detected=int(np.logical_or.reduce([c.features_detected for c in counts]).sum()) if counts else None,
            frac_reads_in_cells=_div(cells["n_reads"].sum(), mapped),
            background_probe_read_frac=_div(sum(p.mapped_reads for p in probes if p.cells == 0), mapped),
        )
