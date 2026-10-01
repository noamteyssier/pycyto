"""A whole cyto output directory: the shared ``stats/`` plus each workflow's count files."""

import logging
import os
from abc import abstractmethod
from functools import cached_property
from typing import Self

import numpy as np
import polars as pl
from pandera.typing.polars import DataFrame
from pydantic import BaseModel, ConfigDict

from ...config import FlexBarcode
from .counts import CellTable, FilteredCounts, scan_h5ad
from .stats import CytoStats

logger = logging.getLogger("pycyto.qc")


class CytoRun(BaseModel):
    """A cyto output directory: the shared ``stats/`` plus whatever the workflow writes.

    :class:`GexCytoRun` and :class:`CrisprCytoRun` add the workflow's count files; this base
    holds what every workflow has.

    Attributes
    ----------
    path : str
        Absolute path to the directory.
    stats : CytoStats
        Everything under ``stats/``.
    """

    path: str
    stats: CytoStats

    @classmethod
    @abstractmethod
    def read(cls, cyto_outdir: str) -> Self:
        """Load and validate a cyto output directory: ``stats/`` plus the workflow's count files.

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


class CrisprCytoRun(CytoRun):
    """A ``cyto workflow crispr`` output directory.

    A CRISPR run has no cell calls: ``counts/<probe>.h5ad`` holds guide UMIs for every
    barcode. Only the per-guide totals are kept, not the matrices. Guide assignments
    (``assignments/``) are not read.

    Attributes
    ----------
    guide_names : list[str]
        Guides in the library, in ``var`` order. Every probe's count file shares this list.
    guide_umis : dict[FlexBarcode, np.ndarray]
        UMIs per guide summed over every barcode, per probe barcode, in
        :attr:`CytoStats.probes` order.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    guide_names: list[str]
    guide_umis: dict[FlexBarcode, np.ndarray]

    @classmethod
    def read(cls, cyto_outdir: str) -> Self:
        stats = CytoStats.read(cyto_outdir)
        names: list[str] = []
        umis = {}
        for probe in stats.probes:
            scan = scan_h5ad(os.path.join(cyto_outdir, "counts", f"{probe}.h5ad"))
            umis[probe], names = scan.totals, scan.names
        return cls(path=os.path.abspath(cyto_outdir), stats=stats, guide_names=names, guide_umis=umis)

    @cached_property
    def guide_totals(self) -> np.ndarray:
        """UMIs per guide summed over every probe barcode, in :attr:`guide_names` order."""
        return np.sum(list(self.guide_umis.values()), axis=0)
