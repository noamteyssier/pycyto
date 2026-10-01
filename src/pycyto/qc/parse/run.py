"""A whole cyto output directory: the shared ``stats/`` and ``.timings.tsv`` plus each workflow's count files."""

import logging
import os
from abc import abstractmethod
from functools import cached_property
from typing import Self

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame
from pydantic import BaseModel, model_validator

from ...config import FlexBarcode
from .counts import CellTable, FilteredCounts, scan_h5ad
from .stats import CytoStats

logger = logging.getLogger("pycyto.qc")


class TimingRow(pa.DataFrameModel):
    """Schema for cyto's ``.timings.tsv``: one row per pipeline module per probe barcode.

    Attributes
    ----------
    ibu_name : str
        Probe barcode the step ran for, or ``All-Barcodes`` for run-wide steps (``Mapping``).
    module : str
        Pipeline module, e.g. ``Mapping``, ``Counting``, ``ConversionH5ad``.
    elapsed : float
        Wall seconds. Non-negative.
    """

    ibu_name: str
    module: str
    elapsed: float = pa.Field(ge=0)


class Timings(BaseModel):
    """The ``.timings.tsv`` file at the run root, summed by pipeline module.

    Attributes
    ----------
    by_module : dict[str, float]
        Module name -> total seconds over every probe barcode. Whatever modules cyto
        wrote; the report draws them all. Must include ``Mapping``.
    """

    by_module: dict[str, float]

    @model_validator(mode="after")
    def _has_mapping(self) -> Self:
        if "Mapping" not in self.by_module:
            raise ValueError(f"no Mapping step in .timings.tsv (modules: {sorted(self.by_module)})")
        return self

    @property
    def mapping(self) -> float:
        """Seconds spent in the ``Mapping`` step, the one module the report relies on."""
        return self.by_module["Mapping"]

    @classmethod
    def read(cls, cyto_outdir: str) -> Self:
        """Read and validate ``<cyto_outdir>/.timings.tsv``.

        Parameters
        ----------
        cyto_outdir : str
            A cyto output directory.

        Returns
        -------
        Timings

        Raises
        ------
        pandera.errors.SchemaError
            If a column is missing or an elapsed time is negative.
        pydantic.ValidationError
            If there is no ``Mapping`` row.
        """
        dtypes = {name: col.type for name, col in TimingRow.to_schema().dtypes.items()}
        rows = TimingRow.validate(pl.read_csv(os.path.join(cyto_outdir, ".timings.tsv"), separator="\t", schema_overrides=dtypes))
        return cls(by_module=dict(rows.group_by(TimingRow.module).agg(pl.col(TimingRow.elapsed).sum()).iter_rows()))


class CytoRun(BaseModel):
    """A cyto output directory: the shared ``stats/`` and ``.timings.tsv`` plus whatever the workflow writes.

    :class:`GexCytoRun` and :class:`CrisprCytoRun` add the workflow's count files; this base
    holds what every workflow has.

    Attributes
    ----------
    path : str
        Absolute path to the directory.
    stats : CytoStats
        Everything under ``stats/``.
    timings : Timings
        Seconds per pipeline module, from ``.timings.tsv`` at the root.
    """

    path: str
    stats: CytoStats
    timings: Timings

    @classmethod
    @abstractmethod
    def read(cls, cyto_outdir: str) -> Self:
        """Load and validate a cyto output directory: ``stats/``, ``.timings.tsv`` and the workflow's count files.

        Parameters
        ----------
        cyto_outdir : str
            A cyto output directory.

        Returns
        -------
        CytoRun
        """


class GexCytoRun(CytoRun):
    """A ``cyto workflow gex`` output directory.

    Attributes
    ----------
    counts : dict[FlexBarcode, FilteredCounts]
        Filtered count summaries for the probe barcodes that have a
        ``counts/<probe>.filt.h5ad``, in :attr:`CytoStats.probes` order. Probe barcodes
        without one have no called cells.
    """

    counts: dict[FlexBarcode, FilteredCounts]

    @classmethod
    def read(cls, cyto_outdir: str) -> Self:
        stats = CytoStats.read(cyto_outdir)
        paths = {p: os.path.join(cyto_outdir, "counts", f"{p}.filt.h5ad") for p in stats.probes}
        return cls(
            path=os.path.abspath(cyto_outdir),
            stats=stats,
            timings=Timings.read(cyto_outdir),
            counts={p: FilteredCounts.from_h5ad(f) for p, f in paths.items() if os.path.exists(f)},
        )

    @cached_property
    def cells(self) -> DataFrame[CellTable]:
        """Every called cell in the run, one row each.

        A cell is a barcode in a probe's :class:`FilteredCounts` that also appears in that
        probe's reads table. Called cells missing from the reads table are dropped with a
        warning. Computed and validated once on first access.

        Returns
        -------
        DataFrame[CellTable]
        """
        dtypes = {name: col.type for name, col in CellTable.to_schema().dtypes.items()}
        tables = []
        for probe, counts in self.counts.items():
            cells = counts.cells.join(self.stats.reads.entries[probe], on=CellTable.barcode, how="inner")
            if (missing := counts.cells.height - cells.height) > 0:
                logger.warning(f"[{probe}] - {missing} filtered barcodes missing from reads stats")
            tables.append(cells.with_columns(pl.lit(probe).alias(CellTable.probe)))
        return CellTable.validate(pl.concat([pl.DataFrame(schema=dtypes), *tables], how="diagonal"))

    @cached_property
    def cells_by_probe(self) -> dict[str, pl.DataFrame]:
        """:attr:`cells` split by probe barcode (same columns); every probe in :attr:`CytoStats.probes` has an entry, possibly empty."""
        parts = self.cells.partition_by(CellTable.probe, as_dict=True)
        return {p: parts.get((p,), self.cells.clear()) for p in self.stats.probes}


class GuideUmis(pa.DataFrameModel):
    """Schema for :attr:`CrisprCytoRun.guide_umis`: UMIs per guide per probe barcode.

    One row per (probe barcode, guide); every probe barcode has a row for every guide,
    since all ``counts/<probe>.h5ad`` files share one guide library.

    Attributes
    ----------
    probe : pl.Categorical
        Probe barcode.
    guide : pl.Categorical
        Guide name, from the h5ad ``var`` index. Categorical: ~20k names repeated once per
        probe barcode.
    umis : float
        UMIs for the guide summed over every cell barcode under the probe barcode.
        Float because that is how cyto's matrices are stored; values are whole numbers.
    """

    probe: pl.Categorical
    guide: pl.Categorical
    umis: float = pa.Field(ge=0)

    # The two checks together say every probe barcode has every guide exactly once. They
    # replace ``Config.unique = ["probe", "guide"]``, which costs ~0.5 GB on a real run.
    @pa.dataframe_check
    def guides_unique_within_probe(cls, data: pa.PolarsData) -> pl.LazyFrame:
        """No guide appears twice under one probe barcode."""
        per_probe = data.lazyframe.group_by("probe").agg((pl.col("guide").n_unique() == pl.len()).alias("ok"))
        return per_probe.select(pl.col("ok").all())

    @pa.dataframe_check
    def every_probe_has_every_guide(cls, data: pa.PolarsData) -> pl.LazyFrame:
        """The count files share one guide library: rows == probe barcodes x guides."""
        return data.lazyframe.select(pl.len() == pl.col("probe").n_unique() * pl.col("guide").n_unique())


class CrisprCytoRun(CytoRun):
    """A ``cyto workflow crispr`` output directory.

    A CRISPR run has no cell calls: ``counts/<probe>.h5ad`` holds guide UMIs for every
    barcode. Only the per-guide totals are kept, not the matrices. Guide assignments
    (``assignments/``) are not read.

    Attributes
    ----------
    guide_umis : DataFrame[GuideUmis]
        UMIs per guide per probe barcode, in :attr:`CytoStats.probes` order and, within a
        probe barcode, in the h5ad ``var`` order.
    """

    guide_umis: DataFrame[GuideUmis]

    @classmethod
    def read(cls, cyto_outdir: str) -> Self:
        stats = CytoStats.read(cyto_outdir)
        tables = []
        for probe in stats.probes:
            scan = scan_h5ad(os.path.join(cyto_outdir, "counts", f"{probe}.h5ad"))
            tables.append(
                pl.DataFrame({GuideUmis.guide: pl.Series(scan.names, dtype=pl.Categorical), GuideUmis.umis: scan.totals})
                .with_columns(pl.lit(probe, dtype=pl.Categorical).alias(GuideUmis.probe))
            )
        return cls(
            path=os.path.abspath(cyto_outdir),
            stats=stats,
            timings=Timings.read(cyto_outdir),
            guide_umis=pl.concat(tables).select(GuideUmis.probe, GuideUmis.guide, GuideUmis.umis),
        )

    @cached_property
    def guide_totals(self) -> pl.DataFrame:
        """UMIs per guide summed over every probe barcode: columns ``guide``, ``umis``, in ``var`` order."""
        return self.guide_umis.group_by(GuideUmis.guide, maintain_order=True).agg(pl.col(GuideUmis.umis).sum())

    @cached_property
    def guide_umis_by_probe(self) -> dict[str, pl.DataFrame]:
        """:attr:`guide_umis` split by probe barcode (same columns), one entry per probe barcode."""
        parts = self.guide_umis.partition_by(GuideUmis.probe, as_dict=True)
        return {p: parts[(p,)] for p in self.stats.probes}
