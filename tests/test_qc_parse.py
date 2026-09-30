"""``pycyto.qc.parse``: cyto metadata readers and workflow detection."""

import pytest
from pydantic import ValidationError

from pycyto.qc.parse import CytoStats, Libraries, ProbeUmiStats, ReadStats, UmiStats


class TestParsers:
    def test_probe_sort_key(self):
        probes = ["B-A01", "A-B01", "A-A10", "A-A02", "BC002", "BC001"]
        assert sorted(probes, key=ReadStats._probe_sort_key) == ["A-A02", "A-A10", "A-B01", "B-A01", "BC001", "BC002"]


def test_umi_stats_keys_must_be_flex_barcodes():
    umi = ProbeUmiStats(total=10, corrected=1)
    assert set(UmiStats(entries={"BC001": umi, "A-A01": umi, "A_A01": umi}).entries) == {"BC001", "A-A01", "A_A01"}
    with pytest.raises(ValidationError, match="BC017"):
        UmiStats(entries={"BC001": umi, "BC017": umi})


def test_workflow(cyto_dir):
    assert CytoStats.read(cyto_dir[0]).workflow == "gex"
    with pytest.raises(ValueError):
        _ = CytoStats.model_construct(library=Libraries.model_construct(entries={"probe": None})).workflow
