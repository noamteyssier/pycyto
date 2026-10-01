"""Validated models for the structured outputs cyto writes next to its counts.

They assume a completed cyto run: every file is present and well-formed.

* :mod:`.stats` - everything under ``stats/``, rooted at :class:`CytoStats`
* :mod:`.counts` - summaries of the ``counts/`` h5ads
* :mod:`.run` - :class:`CytoRun` and its workflow subclasses, tying the two together
"""

from .counts import CellCounts, CellTable, FilteredCounts, H5adScan, scan_h5ad
from .run import CrisprRun, CytoRun, GexRun
from .stats import (
    FEATURE,
    UNMAPPED_LABELS,
    BarcodeReadStats,
    CytoStats,
    Libraries,
    Library,
    MappingStats,
    ProbeUmiStats,
    ReadStats,
    UmiStats,
    UnmappedReason,
    Workflow,
)

__all__ = [
    "FEATURE",
    "UNMAPPED_LABELS",
    "BarcodeReadStats",
    "CellCounts",
    "CellTable",
    "CrisprRun",
    "CytoRun",
    "CytoStats",
    "FilteredCounts",
    "GexRun",
    "H5adScan",
    "Libraries",
    "Library",
    "MappingStats",
    "ProbeUmiStats",
    "ReadStats",
    "UmiStats",
    "UnmappedReason",
    "Workflow",
    "scan_h5ad",
]
