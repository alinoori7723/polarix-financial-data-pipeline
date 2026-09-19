from __future__ import annotations

import pytest

from polarix.orchestration import pipeline_status as ps


def test_status_constants_present() -> None:
    for c in (
        "STATUS_PLANNED",
        "STATUS_SKIPPED_NO_MT5_SILVER",
        "STATUS_DATABENTO_DOWNLOAD_OK",
        "STATUS_DATABENTO_RANGE_UNAVAILABLE",
        "STATUS_CME_INGEST_OK",
        "STATUS_CME_QUALITY_PASS",
        "STATUS_ALIGNMENT_PASS",
        "STATUS_BAR_ALIGNMENT_PARTIAL",
        "STATUS_GOLD_FEATURES_FAIL",
        "STATUS_EDA_PASS",
        "STATUS_DISK_GUARD_STOP",
        "STATUS_MEMORY_GUARD_STOP",
        "STATUS_DAY_COMPLETE",
        "STATUS_DAY_FAILED",
    ):
        assert hasattr(ps, c), f"missing {c}"


@pytest.mark.parametrize(
    "decision,expected",
    [
        ("PASS", "ALIGNMENT_PASS"),
        ("PARTIAL", "ALIGNMENT_PARTIAL"),
        ("FAIL", "ALIGNMENT_FAIL"),
        ("UNKNOWN", "ALIGNMENT_FAIL"),
        ("", "ALIGNMENT_FAIL"),
        (None, "ALIGNMENT_FAIL"),
    ],
)
def test_map_decision_to_stage(decision, expected) -> None:
    assert ps.map_decision_to_stage("ALIGNMENT", decision) == expected
