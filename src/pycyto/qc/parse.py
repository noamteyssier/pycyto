"""Readers for the structured outputs cyto writes next to its counts.

They assume a completed cyto run: every file is present and well-formed.
"""

import logging
import os
from functools import cached_property
from pathlib import Path
from typing import Self

import anndata as ad
import numpy as np
import pandera.polars as pa
import polars as pl
import scipy.sparse as sp
from pandera.typing.polars import DataFrame
from pydantic import BaseModel, ConfigDict, TypeAdapter, field_validator

from ..config import FLEX_V1_BARCODES, FLEX_V2_BARCODES, FlexBarcode

logger = logging.getLogger("pycyto.qc")

# Rank of each known barcode within its format, so `ReadStats._probe_sort_key` groups by prefix
# (V1) or plate position (V2). FLEX_V1_BARCODES is generated prefix-interleaved
# (BC001, CR001, AB001, BC002, …); sorting it gives the intuitive AB001..AB016,
# BC001..BC016, CR001..CR016 grouping.
_FLEX_V1_RANK = {bc: i for i, bc in enumerate(sorted(FLEX_V1_BARCODES))}
_FLEX_V2_RANK = {bc: i for i, bc in enumerate(FLEX_V2_BARCODES)}

# unmapped-read categories in ``stats/mapping_map.json`` -> human-readable labels
UNMAPPED_LABELS = {
    "missing_feature": "No gene probe match",
    "missing_probe": "No probe barcode match",
    "failed_umi_qual": "UMI failed quality",
    "missing_whitelist": "Cell barcode not in whitelist",
    "umi_truncated": "UMI truncated",
}


class BarcodeReadStats(pa.DataFrameModel):
    """Schema for one ``stats/reads/<probe>.reads.tsv.zst`` table.

    Each row is a cell barcode seen under the probe barcode, with its mapped read and
    deduplicated UMI counts. Validation enforces unique barcodes, non-negative counts,
    and that every UMI is backed by at least one read.

    Attributes
    ----------
    barcode : pl.Categorical
        Cell barcode sequence. Unique within a table. Categorical so that the same barcode
        across many probe tables shares one dictionary entry.
    n_umis : int
        Deduplicated UMIs for this cell barcode. Non-negative.
    n_reads : int
        Mapped reads for this cell barcode. Non-negative and at least ``n_umis``.
    """

    barcode: pl.Categorical = pa.Field(unique=True)
    n_umis: int = pa.Field(ge=0)
    n_reads: int = pa.Field(ge=0)

    @pa.dataframe_check
    def reads_cover_umis(cls, data: pa.PolarsData) -> pl.LazyFrame:
        """Every UMI is backed by at least one read."""
        return data.lazyframe.select(pl.col("n_reads") >= pl.col("n_umis"))


class ReadStats(BaseModel):
    """The ``stats/reads/`` directory: one :class:`BarcodeReadStats` table per probe barcode.

    All tables are held in memory. A loaded probe barcode is roughly 600k rows at about
    30 bytes each, so a fully loaded 384-plex run is on the order of 7 GB.

    Attributes
    ----------
    entries : dict[FlexBarcode, DataFrame[BarcodeReadStats]]
        Reads tables keyed by probe barcode, validated against :class:`BarcodeReadStats`
        on assignment. Always ordered Flex-V2 first in plate order, then Flex-V1 grouped
        by prefix (``AB``, ``BC``, ``CR``).
    """

    entries: dict[FlexBarcode, DataFrame[BarcodeReadStats]]

    @classmethod
    def from_dir(cls, reads_dir: str) -> Self:
        """Read and validate every reads table in a directory.

        Parameters
        ----------
        reads_dir : str
            Path to ``stats/reads``.

        Returns
        -------
        ReadStats

        Raises
        ------
        pydantic.ValidationError
            If a file's probe name is not a known Flex barcode or a table violates
            :class:`BarcodeReadStats`.
        """
        suffix = ".reads.tsv.zst"
        dtypes = {name: col.type for name, col in BarcodeReadStats.to_schema().dtypes.items()}
        return cls(
            entries={
                f.removesuffix(suffix): pl.read_csv(os.path.join(reads_dir, f), separator="\t", schema_overrides=dtypes)
                for f in os.listdir(reads_dir)
                if f.endswith(suffix)
            }
        )

    @field_validator("entries")
    @classmethod
    def _sort_probes(cls, entries: dict[str, DataFrame[BarcodeReadStats]]) -> dict[str, DataFrame[BarcodeReadStats]]:
        return {p: entries[p] for p in sorted(entries, key=cls._probe_sort_key)}

    @staticmethod
    def _probe_sort_key(probe: str) -> tuple[int, int]:
        """Flex-V2 barcodes in plate order, then Flex-V1 barcodes by prefix."""
        v2_rank = _FLEX_V2_RANK.get(probe.replace("_", "-"))
        return (0, v2_rank) if v2_rank is not None else (1, _FLEX_V1_RANK[probe])

    @property
    def probes(self) -> list[str]:
        """Probe barcodes with a reads table, in :attr:`entries` order."""
        return list(self.entries)


class ProbeUmiStats(BaseModel):
    """One ``stats/umi/<probe>.umi.json`` file: UMI error-correction counts for a probe barcode.

    Attributes
    ----------
    total : int
        UMIs observed under this probe barcode before correction.
    corrected : int
        UMIs that were collapsed onto a neighbour by error correction.
    """

    total: int
    corrected: int

    @classmethod
    def from_json(cls, path: str) -> Self:
        """Read and validate a ``<probe>.umi.json`` file.

        Parameters
        ----------
        path : str
            Path to the JSON file.

        Returns
        -------
        ProbeUmiStats
        """
        return cls.model_validate_json(Path(path).read_bytes())


class UmiStats(BaseModel):
    """The ``stats/umi/`` directory: one :class:`ProbeUmiStats` per probe barcode.

    Attributes
    ----------
    entries : dict[FlexBarcode, ProbeUmiStats]
        Per-probe UMI stats keyed by probe barcode. Keys must be known Flex barcodes.
    """

    entries: dict[FlexBarcode, ProbeUmiStats]

    @classmethod
    def from_dir(cls, umi_dir: str, probes: list[str]) -> Self:
        """Read the UMI stats file for each probe barcode.

        Parameters
        ----------
        umi_dir : str
            Path to ``stats/umi``.
        probes : list[str]
            Probe barcodes to read, typically :attr:`ReadStats.probes`.

        Returns
        -------
        UmiStats

        Raises
        ------
        pydantic.ValidationError
            If a probe name is not a known Flex barcode or a file is malformed.
        """
        return cls(entries={p: ProbeUmiStats.from_json(os.path.join(umi_dir, f"{p}.umi.json")) for p in probes})


class MappingStats(BaseModel):
    """The ``stats/mapping_map.json`` file: run-level read mapping stats.

    Only the fields the QC report uses are declared. Others in the file are dropped.

    Attributes
    ----------
    total_reads : int
        Reads in the input FASTQs.
    mapped_reads : int
        Reads with a valid cell barcode, probe barcode, UMI and feature match.
    mapped_reads_frac : float
        ``mapped_reads / total_reads``.
    unmapped : dict[str, int or float]
        Why reads failed to map: ``<reason>`` -> read count and ``<reason>_frac`` ->
        fraction of unmapped reads, for the reasons in :data:`UNMAPPED_LABELS`. A read
        can fail more than one check.
    """

    total_reads: int
    mapped_reads: int
    mapped_reads_frac: float
    unmapped: dict[str, int | float]

    @classmethod
    def from_json(cls, path: str) -> Self:
        """Read and validate ``mapping_map.json``.

        Parameters
        ----------
        path : str
            Path to the JSON file.

        Returns
        -------
        MappingStats
        """
        return cls.model_validate_json(Path(path).read_bytes())


class Library(BaseModel):
    """One entry of ``stats/mapping_lib.json``: a reference library cyto mapped against.

    Only the fields the QC report uses are declared. Others in the file are dropped.

    Attributes
    ----------
    name : str
        Library name, e.g. ``probe``, ``whitelist``, ``gex``.
    total_elem : int
        Sequences in the library, including any expanded variants.
    total_aggr : int
        Distinct features the sequences aggregate to (e.g. genes for ``gex``).
    """

    name: str
    total_elem: int
    total_aggr: int


class Libraries(BaseModel):
    """The ``stats/mapping_lib.json`` file: every reference library, keyed by name.

    Attributes
    ----------
    entries : dict[str, Library]
        Libraries keyed by :attr:`Library.name`.
    """

    entries: dict[str, Library]

    @classmethod
    def from_json(cls, path: str) -> Self:
        """Read and validate ``mapping_lib.json``.

        Parameters
        ----------
        path : str
            Path to the JSON file, which holds a list of library objects.

        Returns
        -------
        Libraries
        """
        libraries = TypeAdapter(list[Library]).validate_json(Path(path).read_bytes())
        return cls(entries={lib.name: lib for lib in libraries})


class CellCounts(pa.DataFrameModel):
    """Schema for the per-cell table derived from a ``counts/<probe>.filt.h5ad``.

    Attributes
    ----------
    barcode : pl.Categorical
        Cell barcode with cyto's ``-<probe>`` suffix stripped. Unique within a table.
    n_genes : int
        Features with a nonzero count in the cell.
    """

    barcode: pl.Categorical = pa.Field(unique=True)
    n_genes: int = pa.Field(ge=0)


class FilteredCounts(BaseModel):
    """One ``counts/<probe>.filt.h5ad`` file: cyto's called cells for a probe barcode.

    Only per-cell and per-feature summaries are kept, not the count matrix.

    Attributes
    ----------
    cells : DataFrame[CellCounts]
        One row per called cell.
    feature_totals : np.ndarray
        Summed counts per feature over all cells, in ``var`` order.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    cells: DataFrame[CellCounts]
    feature_totals: np.ndarray

    @property
    def n_features(self) -> int:
        """Features in the count matrix."""
        return len(self.feature_totals)

    @property
    def features_detected(self) -> np.ndarray:
        """Boolean mask of features with a nonzero total over all cells."""
        return self.feature_totals > 0

    @classmethod
    def from_h5ad(cls, path: str, chunk_rows: int = 10_000) -> Self:
        """Scan a filtered count h5ad.

        ``X`` is read in row chunks from a backed AnnData so memory stays bounded by
        ``chunk_rows`` rather than the matrix size.

        Parameters
        ----------
        path : str
            Path to a ``<probe>.filt.h5ad`` file.
        chunk_rows : int
            Rows of ``X`` to load per chunk.

        Returns
        -------
        FilteredCounts
        """
        adata = ad.read_h5ad(path, backed="r")
        try:
            n_genes = np.zeros(adata.n_obs, dtype=np.int64)
            totals = np.zeros(adata.n_vars, dtype=np.float64)
            for start in range(0, adata.n_obs, chunk_rows):
                block = sp.csr_matrix(adata.X[start : start + chunk_rows])
                block.eliminate_zeros()
                n_genes[start : start + block.shape[0]] = np.diff(block.indptr)
                totals += np.asarray(block.sum(axis=0)).ravel()
            # cyto names cells ``<barcode>-<probe>`` (e.g. ``ACGT...-A-A02``)
            barcodes = [name.split("-", 1)[0] for name in adata.obs_names]
        finally:
            adata.file.close()
        cells = pl.DataFrame({"barcode": barcodes, "n_genes": n_genes}).cast({"barcode": pl.Categorical})
        return cls(cells=cells, feature_totals=totals)


class CytoStats(BaseModel):
    """Everything cyto writes under ``<cyto_outdir>/stats``.

    All files are read and validated up front by :meth:`read`.

    Attributes
    ----------
    mapping : MappingStats
        Run-level read mapping stats.
    library : Libraries
        Reference libraries mapped against.
    reads : ReadStats
        Per-probe barcode tables.
    umi : UmiStats
        Per-probe UMI correction stats.
    """

    mapping: MappingStats
    library: Libraries
    reads: ReadStats
    umi: UmiStats

    @classmethod
    def read(cls, cyto_outdir: str) -> Self:
        """Load the stats for a cyto output directory.

        Parameters
        ----------
        cyto_outdir : str
            A ``cyto workflow gex`` output directory containing ``stats/``.

        Returns
        -------
        CytoStats

        Raises
        ------
        pydantic.ValidationError
            If any JSON file is malformed or a probe name is not a known Flex barcode.
        """
        root = os.path.join(cyto_outdir, "stats")
        reads = ReadStats.from_dir(os.path.join(root, "reads"))
        return cls(
            mapping=MappingStats.from_json(os.path.join(root, "mapping_map.json")),
            library=Libraries.from_json(os.path.join(root, "mapping_lib.json")),
            reads=reads,
            umi=UmiStats.from_dir(os.path.join(root, "umi"), reads.probes),
        )

    @property
    def probes(self) -> list[str]:
        """Probe barcodes with a reads file, in :attr:`ReadStats.probes` order."""
        return self.reads.probes

    @property
    def workflow(self) -> str:
        """The cyto workflow that produced this directory.

        Returns
        -------
        str
            Currently always ``"gex"``.

        Raises
        ------
        ValueError
            If no ``gex`` library is present in ``mapping_lib.json``.
        """
        if "gex" in self.library.entries:
            return "gex"
        raise ValueError(
            f"Can't tell which cyto workflow produced this directory (libraries: {sorted(self.library.entries)}); "
            "expected a `cyto workflow gex` run"
        )


class CellTable(pa.DataFrameModel):
    """Schema for :attr:`CytoRun.cells`: every called cell in a run, one row each.

    Attributes
    ----------
    probe : str
        Probe barcode the cell was called under.
    barcode : pl.Categorical
        Cell barcode. Unique together with ``probe``.
    n_umis : int
        Deduplicated UMIs, from the probe's reads table.
    n_reads : int
        Mapped reads, from the probe's reads table.
    n_genes : int
        Features with a nonzero count, from the probe's filtered h5ad.
    """

    probe: str
    barcode: pl.Categorical
    n_umis: int = pa.Field(ge=0)
    n_reads: int = pa.Field(ge=0)
    n_genes: int = pa.Field(ge=0)

    class Config:
        unique = ["probe", "barcode"]  # noqa: RUF012  (pandera reads a plain list here)


class CytoRun(BaseModel):
    """A ``cyto workflow gex`` output directory.

    Attributes
    ----------
    path : str
        Absolute path to the directory.
    stats : CytoStats
        Everything under ``stats/``.
    counts : dict[FlexBarcode, FilteredCounts]
        Filtered count summaries for the probe barcodes that have a
        ``counts/<probe>.filt.h5ad``, in :attr:`CytoStats.probes` order. Probe barcodes
        without one have no called cells.
    """

    path: str
    stats: CytoStats
    counts: dict[FlexBarcode, FilteredCounts]

    @classmethod
    def read(cls, cyto_outdir: str) -> Self:
        """Load and validate a cyto output directory.

        Parameters
        ----------
        cyto_outdir : str
            A ``cyto workflow gex`` output directory.

        Returns
        -------
        CytoRun
        """
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
            cells = counts.cells.join(self.stats.reads.entries[probe], on="barcode", how="inner")
            if (missing := counts.cells.height - cells.height) > 0:
                logger.warning(f"[{probe}] - {missing} filtered barcodes missing from reads stats")
            tables.append(cells.with_columns(probe=pl.lit(probe)))
        return CellTable.validate(pl.concat([pl.DataFrame(schema=dtypes), *tables], how="diagonal"))
