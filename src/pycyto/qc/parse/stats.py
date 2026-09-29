"""Models for everything cyto writes under ``<cyto_outdir>/stats``."""

import os
from functools import cached_property
from pathlib import Path
from typing import Literal, Self

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame
from pydantic import (
    BaseModel,
    TypeAdapter,
    ValidationInfo,
    field_validator,
    model_validator,
)

from ...config import FLEX_V1_BARCODES, FLEX_V2_BARCODES, FlexBarcode

# Rank of each known barcode within its format, so `ReadStats._probe_sort_key` groups by prefix
# (V1) or plate position (V2). FLEX_V1_BARCODES is generated prefix-interleaved
# (BC001, CR001, AB001, BC002, …); sorting it gives the intuitive AB001..AB016,
# BC001..BC016, CR001..CR016 grouping.
_FLEX_V1_RANK = {bc: i for i, bc in enumerate(sorted(FLEX_V1_BARCODES))}
_FLEX_V2_RANK = {bc: i for i, bc in enumerate(FLEX_V2_BARCODES)}

Workflow = Literal["gex", "crispr"]
"""The cyto workflows the report supports, named as in ``mapping_lib.json``."""

FEATURE: dict[Workflow, str] = {"gex": "gene probe", "crispr": "guide"}
"""What a read failing ``missing_feature`` did not match, per workflow."""

# unmapped-read categories in ``stats/mapping_map.json`` -> human-readable labels
UNMAPPED_LABELS = {
    "missing_feature": "No feature match",  # replaced with the workflow's FEATURE when known
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
        return data.lazyframe.select(pl.col(cls.n_reads) >= pl.col(cls.n_umis))


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

    @cached_property
    def mapped_reads(self) -> int:
        """Mapped reads over every probe barcode: ``n_reads`` summed over all tables.

        Differs slightly from ``mapping_map.json``'s ``mapped_reads``; use this when
        normalising per-probe-barcode counts so the fractions sum to one.
        """
        return sum(int(t[BarcodeReadStats.n_reads].sum()) for t in self.entries.values())


class ProbeUmiStats(BaseModel):
    """One ``stats/umi/<probe>.umi.json`` file: UMI error-correction counts for a probe barcode.

    Attributes
    ----------
    total : int
        UMIs observed under this probe barcode before correction.
    corrected : int
        UMIs that were collapsed onto a neighbour by error correction.
    fraction_corrected : float
        ``corrected / total``, as reported by cyto.
    """

    total: int
    corrected: int
    fraction_corrected: float

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


class UnmappedReason(BaseModel):
    """One reason reads failed to map, from the ``unmapped`` block of ``mapping_map.json``.

    Attributes
    ----------
    reason : str
        cyto's key, e.g. ``missing_feature``.
    label : str
        Human-readable label from :data:`UNMAPPED_LABELS`.
    reads : int
        Reads that failed this check.
    frac_of_reads : float or None
        ``reads / total_reads``.
    frac_of_unmapped : float
        Fraction of unmapped reads that failed this check, as reported by cyto.
    """

    reason: str
    label: str
    reads: int
    frac_of_reads: float | None
    frac_of_unmapped: float


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
    unmapped : list[UnmappedReason]
        Why reads failed to map, largest first. A read can fail more than one check.
    """

    total_reads: int
    mapped_reads: int
    mapped_reads_frac: float
    unmapped: list[UnmappedReason]

    @model_validator(mode="before")
    @classmethod
    def _pair_unmapped(cls, data, info: ValidationInfo):
        """Turn cyto's flat ``{reason: reads, reason_frac: frac}`` block into reasons.

        Pass ``context={"feature": ...}`` (see :meth:`from_json`) to label ``missing_feature``
        with what the workflow maps reads against.
        """
        if not (isinstance(data, dict) and isinstance(raw := data.get("unmapped"), dict)):
            return data
        labels = dict(UNMAPPED_LABELS)
        if feature := (info.context or {}).get("feature"):
            labels["missing_feature"] = f"No {feature} match"
        total = data.get("total_reads")
        reasons = [
            {
                "reason": key,
                "label": labels.get(key, key.replace("_", " ")),
                "reads": reads,
                "frac_of_reads": reads / total if total else None,
                "frac_of_unmapped": raw[f"{key}_frac"],
            }
            for key, reads in raw.items()
            if not key.endswith("_frac")
        ]
        return {**data, "unmapped": sorted(reasons, key=lambda r: -r["reads"])}

    def unmapped_reads(self, reason: str) -> int | None:
        """Reads that failed one check, or None if cyto did not report that reason."""
        return next((r.reads for r in self.unmapped if r.reason == reason), None)

    @classmethod
    def from_json(cls, path: str, feature: str | None = None) -> Self:
        """Read and validate ``mapping_map.json``.

        Parameters
        ----------
        path : str
            Path to the JSON file.
        feature : str, optional
            What the workflow maps reads against (see :data:`FEATURE`), used to label the
            ``missing_feature`` unmapped reason.

        Returns
        -------
        MappingStats
        """
        return cls.model_validate_json(Path(path).read_bytes(), context={"feature": feature})


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

    @property
    def workflow(self) -> Workflow:
        """The cyto workflow, from which feature library was mapped against.

        Raises
        ------
        ValueError
            If neither a ``gex`` nor a ``crispr`` library is present.
        """
        for workflow in FEATURE:
            if workflow in self.entries:
                return workflow
        raise ValueError(
            f"Can't tell which cyto workflow produced this directory (libraries: {sorted(self.entries)}); "
            "expected a `cyto workflow gex` or `cyto workflow crispr` run"
        )


def detect_workflow(cyto_outdir: str) -> Workflow:
    """The cyto workflow that produced a directory, from ``stats/mapping_lib.json`` alone.

    Cheap: reads one small JSON file, so callers can pick the right :class:`CytoRun`
    subclass before loading everything else.

    Parameters
    ----------
    cyto_outdir : str
        A cyto output directory containing ``stats/``.

    Returns
    -------
    Workflow

    Raises
    ------
    ValueError
        If the workflow cannot be told from ``mapping_lib.json``.
    """
    return Libraries.from_json(os.path.join(cyto_outdir, "stats", "mapping_lib.json")).workflow


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
            A cyto output directory containing ``stats/``.

        Returns
        -------
        CytoStats

        Raises
        ------
        pydantic.ValidationError
            If any JSON file is malformed or a probe name is not a known Flex barcode.
        ValueError
            If the workflow cannot be told from ``mapping_lib.json``.
        """
        root = os.path.join(cyto_outdir, "stats")
        library = Libraries.from_json(os.path.join(root, "mapping_lib.json"))
        reads = ReadStats.from_dir(os.path.join(root, "reads"))
        return cls(
            mapping=MappingStats.from_json(os.path.join(root, "mapping_map.json"), feature=FEATURE[library.workflow]),
            library=library,
            reads=reads,
            umi=UmiStats.from_dir(os.path.join(root, "umi"), reads.probes),
        )

    @property
    def probes(self) -> list[str]:
        """Probe barcodes with a reads file, in :attr:`ReadStats.probes` order."""
        return self.reads.probes

    @property
    def workflow(self) -> Workflow:
        """The cyto workflow that produced this directory; see :attr:`Libraries.workflow`."""
        return self.library.workflow
