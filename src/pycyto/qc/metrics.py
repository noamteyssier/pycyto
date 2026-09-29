"""QC building blocks shared by every cyto workflow.

Workflow-specific metrics live in :mod:`pycyto.qc.gex`.
"""

import os
from typing import Any

import anndata as ad
import numpy as np
import polars as pl
import scipy.sparse as sp

from .parse import load_json, read_barcode_stats


def _div(a, b) -> float | None:
    return a / b if a is not None and b else None


def read_counts(path: str, chunk_rows: int = 10_000) -> tuple[pl.DataFrame, np.ndarray, list[str]]:
    """Scan a cyto count h5ad.

    Returns a ``barcode, n_features`` frame (features with nonzero counts per barcode),
    the per-feature count totals, and the feature names. ``X`` is read in row chunks
    from a backed AnnData so memory stays bounded.
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
    return pl.DataFrame({"barcode": barcodes, "n_features": n_features}), totals, names


def probe_basics(cyto_outdir: str, probe: str) -> tuple[pl.DataFrame, dict[str, Any], tuple[int, int]]:
    """Per-barcode reads stats plus the metrics every workflow reports for a probe barcode.

    Returns the ``barcode, n_umis, n_reads`` frame, the shared metrics and the
    ``(corrected, total)`` UMI counts from ``stats/umi/<probe>.umi.json``.
    """
    stats = os.path.join(cyto_outdir, "stats")
    df = read_barcode_stats(os.path.join(stats, "reads", f"{probe}.reads.tsv.zst"))
    umi_stats = load_json(os.path.join(stats, "umi", f"{probe}.umi.json"))
    mapped, umis = df["n_reads"].sum(), df["n_umis"].sum()
    rec = {"probe": probe, "mapped_reads": mapped, "umis": umis}
    return df, rec, (umi_stats["corrected"], umi_stats["total"])


def run_summary(results: list[dict], meta: dict, cyto_outdir: str) -> dict[str, Any]:
    """Run-level metrics every workflow reports: reads, mapping, saturation."""
    probes = pl.DataFrame([r["rec"] for r in results], infer_schema_length=None)
    mapping = meta["mapping"]
    lib = {d["name"]: d for d in meta["library"]}
    corrected, total_umis = np.sum([r["umi_counts"] for r in results], axis=0)
    probe_mapped = probes["mapped_reads"].sum()
    return {
        "cyto_outdir": os.path.abspath(cyto_outdir),
        "total_reads": mapping["total_reads"],
        "mapped_reads": mapping["mapped_reads"],
        "mapped_reads_frac": mapping["mapped_reads_frac"],
        "probe_barcodes_in_library": lib["probe"]["total_elem"],
        "probe_barcodes_with_reads": probes.height,
        "seq_saturation": 1 - probes["umis"].sum() / probe_mapped if probe_mapped else None,
        "umi_corrected_frac": _div(float(corrected), float(total_umis)),
    }
