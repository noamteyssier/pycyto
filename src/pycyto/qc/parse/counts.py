"""Models derived from cyto's count h5ads under ``<cyto_outdir>/counts``."""

from typing import NamedTuple, Self

import anndata as ad
import numpy as np
import pandera.polars as pa
import polars as pl
import scipy.sparse as sp
from pandera.typing.polars import DataFrame
from pydantic import BaseModel, ConfigDict


class H5adScan(NamedTuple):
    """Per-barcode and per-feature summaries of one count h5ad; see :func:`scan_h5ad`.

    Attributes
    ----------
    barcodes : list[str]
        Cell barcodes in ``obs`` order, cyto's ``-<probe>`` suffix stripped.
    n_features : np.ndarray
        Features with a nonzero count, per barcode.
    totals : np.ndarray
        Summed counts per feature over all barcodes, in ``var`` order.
    names : list[str]
        Feature names in ``var`` order.
    """

    barcodes: list[str]
    n_features: np.ndarray
    totals: np.ndarray
    names: list[str]


def scan_h5ad(path: str, chunk_rows: int = 10_000) -> H5adScan:
    """Scan a cyto count h5ad without loading the matrix.

    ``X`` is read in row chunks from a backed AnnData so memory stays bounded by
    ``chunk_rows`` rather than the matrix size.

    Parameters
    ----------
    path : str
        Path to a cyto count h5ad.
    chunk_rows : int
        Rows of ``X`` to load per chunk.

    Returns
    -------
    H5adScan
    """
    adata = ad.read_h5ad(path, backed="r")
    try:
        n_features = np.zeros(adata.n_obs, dtype=np.int64)
        totals = np.zeros(adata.n_vars, dtype=np.float64)
        for start in range(0, adata.n_obs, chunk_rows):
            block = sp.csr_matrix(adata.X[start : start + chunk_rows])
            block.eliminate_zeros()
            n_features[start : start + block.shape[0]] = np.diff(block.indptr)
            totals += np.asarray(block.sum(axis=0)).ravel()
        # cyto names cells ``<barcode>-<probe>`` (e.g. ``ACGT...-A-A02``)
        barcodes = [name.split("-", 1)[0] for name in adata.obs_names]
        names = adata.var_names.tolist()
    finally:
        adata.file.close()
    return H5adScan(barcodes, n_features, totals, names)


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

    @classmethod
    def from_h5ad(cls, path: str, chunk_rows: int = 10_000) -> Self:
        """Scan a filtered count h5ad; see :func:`scan_h5ad`.

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
        scan = scan_h5ad(path, chunk_rows)
        cells = pl.DataFrame({CellCounts.barcode: scan.barcodes, CellCounts.n_genes: scan.n_features}).cast(
            {CellCounts.barcode: pl.Categorical}
        )
        return cls(cells=cells, feature_totals=scan.totals)


class CellTable(pa.DataFrameModel):
    """Schema for :attr:`~pycyto.qc.parse.GexCytoRun.cells`: every called cell in a run, one row each.

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
