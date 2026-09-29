"""GEX (``cyto workflow gex``) metrics.

Cells are exactly the barcodes in cyto's ``counts/<probe>.filt.h5ad``; probe barcodes
without that file have no cells.
"""

import logging
import os
from typing import Any

import numpy as np
import polars as pl

from .metrics import _div, probe_basics, read_counts, run_summary

logger = logging.getLogger("pycyto.qc")


def process_probe(cyto_outdir: str, probe: str) -> dict[str, Any]:
    """Metrics and per-cell arrays for one probe barcode."""
    df, rec, umi_counts = probe_basics(cyto_outdir, probe)

    filt = os.path.join(cyto_outdir, "counts", f"{probe}.filt.h5ad")
    detected = None
    if os.path.exists(filt):
        cells, totals, _ = read_counts(filt)
        detected = totals > 0
        df = df.join(cells.rename({"n_features": "n_genes"}), on="barcode", how="left")
        if (missing := cells.height - df["n_genes"].count()) > 0:
            logger.warning(f"[{probe}] - {missing} filtered barcodes missing from reads stats")
    else:
        df = df.with_columns(n_genes=pl.lit(None, dtype=pl.Int64))
    df = df.with_columns(is_cell=pl.col("n_genes").is_not_null())
    in_cells = df.filter("is_cell")

    rec |= {"cells": in_cells.height, "reads_in_cells": in_cells["n_reads"].sum()}
    cell_umis = in_cells["n_umis"].to_numpy()
    cell_genes = in_cells["n_genes"].to_numpy()
    return {
        "rec": rec,
        "cell_umis": cell_umis,
        "cell_genes": cell_genes,
        "detected": detected,
        "umi_counts": umi_counts,
    }


def summarize(results: list[dict], meta: dict, cyto_outdir: str) -> dict[str, Any]:
    """Run-level metrics (the numbers behind the report's Summary tab)."""
    probes = pl.DataFrame([r["rec"] for r in results], infer_schema_length=None)
    called = probes.filter(pl.col("cells") > 0)
    lib = {d["name"]: d for d in meta["library"]}
    cell_umis = np.concatenate([r["cell_umis"] for r in results])
    cell_genes = np.concatenate([r["cell_genes"] for r in results])
    detected = [r["detected"] for r in results if r["detected"] is not None]
    total_reads, mapped_reads = meta["mapping"]["total_reads"], meta["mapping"]["mapped_reads"]
    n_cells, probe_mapped = called["cells"].sum(), probes["mapped_reads"].sum()
    return run_summary(results, meta, cyto_outdir) | {
        "estimated_cells": n_cells,
        "probe_barcodes_with_cells": called.height,
        "mean_reads_per_cell": _div(total_reads, n_cells),
        "mean_mapped_reads_per_cell": _div(mapped_reads, n_cells),
        "median_umis_per_cell": float(np.median(cell_umis)) if len(cell_umis) else None,
        "median_genes_per_cell": float(np.median(cell_genes)) if len(cell_genes) else None,
        "total_genes_detected": int(np.logical_or.reduce(detected).sum()) if detected else None,
        "genes_in_reference": len(detected[0]) if detected else lib["gex"]["total_aggr"],
        "frac_reads_in_cells": _div(probes["reads_in_cells"].sum(), probe_mapped),
        "background_probe_read_frac": _div(
            probes.filter(pl.col("cells") == 0)["mapped_reads"].sum(), probe_mapped
        ),
        "cells_median_per_probe": called["cells"].median(),
    }
