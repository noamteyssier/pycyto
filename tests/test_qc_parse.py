"""``pycyto.qc.parse``: cyto metadata readers and workflow detection."""

import pandera.polars as pa
import pytest
from pydantic import ValidationError

from pycyto.qc.parse import (
    CytoStats,
    Libraries,
    MappingStats,
    ProbeUmiStats,
    ReadStats,
    Timings,
    UmiStats,
    detect_workflow,
)


def test_mapping_stats_pairs_unmapped_reasons():
    raw = {
        "total_reads": 100,
        "mapped_reads": 60,
        "mapped_reads_frac": 0.6,
        "unmapped": {"missing_feature": 10, "missing_feature_frac": 0.25, "failed_umi_qual": 30, "failed_umi_qual_frac": 0.75},
    }
    m = MappingStats.model_validate(raw)
    assert [r.reason for r in m.unmapped] == ["failed_umi_qual", "missing_feature"]  # largest first
    top = m.unmapped[0]
    assert (top.label, top.reads, top.frac_of_reads, top.frac_of_unmapped) == ("UMI failed quality", 30, 0.3, 0.75)
    assert m.unmapped_reads("missing_feature") == 10 and m.unmapped_reads("not_a_reason") is None
    assert MappingStats.model_validate(m.model_dump()) == m  # structured input passes through


class TestParsers:
    def test_probe_sort_key(self):
        probes = ["B-A01", "A-B01", "A-A10", "A-A02", "BC002", "BC001"]
        assert sorted(probes, key=ReadStats._probe_sort_key) == ["A-A02", "A-A10", "A-B01", "B-A01", "BC001", "BC002"]


def test_umi_stats_keys_must_be_flex_barcodes():
    umi = ProbeUmiStats(total=10, corrected=1, fraction_corrected=0.1)
    assert set(UmiStats(entries={"BC001": umi, "A-A01": umi, "A_A01": umi}).entries) == {"BC001", "A-A01", "A_A01"}
    with pytest.raises(ValidationError, match="BC017"):
        UmiStats(entries={"BC001": umi, "BC017": umi})


def test_timings(tmp_path):
    path = tmp_path / ".timings.tsv"
    path.write_text("ibu_name\tmodule\telapsed\nAll-Barcodes\tMapping\t10.5\nA-A01\tCounting\t1\nA-A02\tCounting\t2\n")
    t = Timings.read(str(tmp_path))
    assert t.by_module == {"Mapping": 10.5, "Counting": 3.0} and t.mapping == 10.5
    path.write_text("ibu_name\tmodule\telapsed\nA-A01\tCounting\t1\n")
    with pytest.raises(ValidationError, match="no Mapping step"):
        Timings.read(str(tmp_path))
    path.write_text("ibu_name\tmodule\telapsed\nAll-Barcodes\tMapping\t-1\n")
    with pytest.raises(pa.errors.SchemaError):
        Timings.read(str(tmp_path))


def test_workflow(cyto_dir, crispr_dir):
    assert detect_workflow(cyto_dir[0]) == CytoStats.read(cyto_dir[0]).workflow == "gex"
    assert detect_workflow(crispr_dir[0]) == CytoStats.read(crispr_dir[0]).workflow == "crispr"
    with pytest.raises(ValueError):
        _ = CytoStats.model_construct(library=Libraries.model_construct(entries={"probe": None})).workflow
