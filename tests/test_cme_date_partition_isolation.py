from __future__ import annotations

import datetime as _dt
import json
import subprocess
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from polarix.ingestion.cme_reference_ingest import (
    CROSS_DATE_CME_SOURCE_REJECTED,
    LEGACY_ROOT_FILES_INCLUDED,
    MISSING_CME_RAW_FOR_DATE,
    CmeIngestError,
    IngestConfig,
    ingest,
)
from polarix.orchestration.pipeline_orchestrator import OrchestratorConfig, process_one_date

REPO_ROOT = Path(__file__).resolve().parents[1]


def _make_mbp1(
    path: Path,
    *,
    raw_symbol: str = "ESM6",
    action: str = "T",
    side: str = "B",
    price: float = 5000.0,
    size: int = 1,
    ts_event: int = 1700000000000000000,
    ts_recv: int = 1700000000000000010,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema(
        [
            ("ts_event", pa.int64()),
            ("ts_recv", pa.int64()),
            ("raw_symbol", pa.string()),
            ("action", pa.string()),
            ("side", pa.string()),
            ("price", pa.float64()),
            ("size", pa.int64()),
            ("bid_px_00", pa.float64()),
            ("ask_px_00", pa.float64()),
            ("bid_sz_00", pa.int64()),
            ("ask_sz_00", pa.int64()),
        ]
    )
    table = pa.Table.from_pydict(
        {
            "ts_event": [ts_event],
            "ts_recv": [ts_recv],
            "raw_symbol": [raw_symbol],
            "action": [action],
            "side": [side],
            "price": [price],
            "size": [size],
            "bid_px_00": [price - 0.25],
            "ask_px_00": [price + 0.25],
            "bid_sz_00": [5],
            "ask_sz_00": [5],
        },
        schema=schema,
    )
    pq.write_table(table, path, compression="zstd")
    return path


def _config(tmp_path: Path, date: str, **kw) -> IngestConfig:
    return IngestConfig(
        input_root=tmp_path / "raw" / "cme_sample",
        output_root=tmp_path / "normalized" / "cme_reference",
        reports_root=tmp_path / "reports",
        date=date,
        force=True,
        **kw,
    )


def test_1_ingest_reads_only_requested_date_partition(tmp_path: Path) -> None:
    root = tmp_path / "raw" / "cme_sample"
    f_target = _make_mbp1(root / "date=2026-05-19" / "today.parquet")
    _make_mbp1(root / "date=2026-05-18" / "yesterday.parquet")
    result = ingest(_config(tmp_path, "2026-05-19"))
    assert not result.missing_sample
    source_paths = [str(p) for p in result.source_files]
    assert str(f_target) in source_paths
    assert all(("date=2026-05-18" not in p for p in source_paths))
    assert all(("date=2026-05-19" in p for p in source_paths))


def test_2_other_date_partitions_are_ignored(tmp_path: Path) -> None:
    root = tmp_path / "raw" / "cme_sample"
    _make_mbp1(root / "date=2026-05-18" / "yesterday.parquet")
    result = ingest(_config(tmp_path, "2026-05-19"))
    assert result.missing_sample is True
    assert result.missing_reason == MISSING_CME_RAW_FOR_DATE
    assert result.source_files == []


def test_3_root_level_legacy_files_are_ignored_by_default(tmp_path: Path) -> None:
    root = tmp_path / "raw" / "cme_sample"
    _make_mbp1(root / "legacy.parquet")
    result = ingest(_config(tmp_path, "2026-05-19"))
    assert result.missing_sample is True
    assert result.missing_reason == MISSING_CME_RAW_FOR_DATE
    assert result.source_files == []
    assert result.legacy_files_included == []


def test_4_include_legacy_root_files_opt_in(tmp_path: Path) -> None:
    root = tmp_path / "raw" / "cme_sample"
    legacy = _make_mbp1(root / "legacy.parquet", raw_symbol="ESM6")
    today = _make_mbp1(root / "date=2026-05-19" / "today.parquet", raw_symbol="ESM6")
    other = _make_mbp1(root / "date=2026-05-18" / "yesterday.parquet", raw_symbol="ESM6")
    result = ingest(_config(tmp_path, "2026-05-19", include_legacy_root_files=True))
    source_paths = [str(p) for p in result.source_files]
    assert str(today) in source_paths
    assert str(legacy) in source_paths
    assert str(other) not in source_paths
    assert [str(p) for p in result.legacy_files_included] == [str(legacy)]
    doc = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert LEGACY_ROOT_FILES_INCLUDED in doc["isolation_warnings"]
    assert doc["include_legacy_root_files"] is True


def test_5_empty_date_partition_fails_closed_with_missing(tmp_path: Path) -> None:
    root = tmp_path / "raw" / "cme_sample"
    (root / "date=2026-05-19").mkdir(parents=True)
    _make_mbp1(root / "legacy.parquet")
    _make_mbp1(root / "date=2026-05-18" / "other.parquet")
    result = ingest(_config(tmp_path, "2026-05-19"))
    assert result.missing_sample is True
    assert result.missing_reason == MISSING_CME_RAW_FOR_DATE
    assert result.legacy_files_included == []


def test_6_cross_date_partition_source_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "raw" / "cme_sample"
    other_partition = root / "date=2026-05-18"
    _make_mbp1(other_partition / "trade.parquet")
    cfg = IngestConfig(
        input_root=other_partition,
        output_root=tmp_path / "out",
        reports_root=tmp_path / "rep",
        date="2026-05-19",
        force=True,
        include_legacy_root_files=True,
    )
    with pytest.raises(CmeIngestError) as ei:
        ingest(cfg)
    assert CROSS_DATE_CME_SOURCE_REJECTED in str(ei.value)


def test_7_manifest_records_source_files_and_input_root(tmp_path: Path) -> None:
    root = tmp_path / "raw" / "cme_sample"
    _make_mbp1(root / "date=2026-05-19" / "today.parquet")
    result = ingest(_config(tmp_path, "2026-05-19"))
    assert result.manifest_path is not None
    doc = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    for key in (
        "source_date",
        "input_root",
        "source_files",
        "include_legacy_root_files",
        "legacy_files_included",
        "rejected_cross_date_files",
        "isolation_warnings",
    ):
        assert key in doc, f"manifest missing key {key!r}"
    assert doc["source_date"] == "2026-05-19"
    assert doc["input_root"].endswith("cme_sample")
    assert all(("date=2026-05-19" in s for s in doc["source_files"]))
    assert doc["include_legacy_root_files"] is False
    assert doc["legacy_files_included"] == []
    assert doc["rejected_cross_date_files"] == []
    assert doc["isolation_warnings"] == []


def test_8_orchestrator_cme_ingest_argv_points_at_date_partition(tmp_path: Path) -> None:
    data = tmp_path / "data"
    t0 = int(_dt.datetime.fromisoformat("2026-05-19T05:00:00+00:00").timestamp() * 1000)
    for sym in ("SPX500", "NDX100"):
        path = (
            data
            / "normalized"
            / "mt5_ticks"
            / f"symbol={sym}"
            / "date=2026-05-19"
            / "part-0001.parquet"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.table(
                {
                    "symbol": [sym] * 2,
                    "time_msc_utc_ms": pa.array([t0, t0 + 10000], type=pa.int64()),
                }
            ),
            path,
            compression="zstd",
        )
    raw = data / "raw" / "cme_sample" / "date=2026-05-19" / "today.parquet"
    raw.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"x": [1]}), raw)
    cfg = OrchestratorConfig(
        dates=["2026-05-19"],
        data_root=data,
        reports_root=tmp_path / "reports",
        repo_root=REPO_ROOT,
        python_exe=Path("python"),
        dry_run=False,
        min_free_disk_gb=0.0,
        min_available_memory_gb=0.0,
    )
    invoked: list[list[str]] = []

    def runner(argv, timeout, cwd):
        invoked.append(list(argv))
        return subprocess.CompletedProcess(argv, returncode=0, stdout="", stderr="")

    process_one_date(cfg, "2026-05-19", runner=runner)
    ingest_argvs = [
        argv for argv in invoked if any(("ingest_cme_reference_sample.py" in a for a in argv))
    ]
    assert ingest_argvs, "CME_INGEST was not invoked"
    argv = ingest_argvs[0]
    assert "--input-root" in argv
    idx = argv.index("--input-root")
    assert argv[idx + 1].endswith(str(Path("raw") / "cme_sample"))
    assert "--date" in argv
    assert argv[argv.index("--date") + 1] == "2026-05-19"
    assert "--include-legacy-root-files" not in argv


def test_9_no_trading_functions_introduced() -> None:
    files = (
        "src/polarix/ingestion/cme_reference_ingest.py",
        "src/polarix/orchestration/pipeline_orchestrator.py",
        "src/polarix/quality/cme_reference_quality.py",
        "scripts/ingest_cme_reference_sample.py",
    )
    forbidden = (
        "order_send",
        "order_check",
        "position_close",
        "positions_close",
        "position_modify",
        "TradeRequest",
        "MqlTradeRequest",
    )
    for path in files:
        text = (REPO_ROOT / path).read_text(encoding="utf-8")
        for tok in forbidden:
            assert tok not in text, f"{path}: forbidden {tok!r}"


def test_discovery_legacy_flag_propagates_to_manifest_warnings(tmp_path: Path) -> None:
    root = tmp_path / "raw" / "cme_sample"
    _make_mbp1(root / "legacy.parquet")
    _make_mbp1(root / "date=2026-05-19" / "today.parquet")
    result = ingest(_config(tmp_path, "2026-05-19", include_legacy_root_files=True))
    doc = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert LEGACY_ROOT_FILES_INCLUDED in doc["isolation_warnings"]


def test_discovery_no_date_path_keeps_legacy_recursive_behaviour(tmp_path: Path) -> None:
    root = tmp_path / "raw" / "cme_sample"
    _make_mbp1(root / "legacy.parquet")
    _make_mbp1(root / "date=2026-05-18" / "yesterday.parquet")
    cfg = IngestConfig(
        input_root=root,
        output_root=tmp_path / "out",
        reports_root=tmp_path / "rep",
        date=None,
        dry_run=True,
        force=True,
    )
    result = ingest(cfg)
    assert result.manifest_path is None
    assert any(("legacy.parquet" in str(p) for p in result.source_files))


def test_cross_date_detector_only_fires_on_other_date_segments(tmp_path: Path) -> None:
    root = tmp_path / "raw" / "cme_sample"
    target = root / "date=2026-05-19" / "databento_20260518_x.parquet"
    _make_mbp1(target)
    result = ingest(_config(tmp_path, "2026-05-19"))
    assert not result.missing_sample
    assert result.rejected_cross_date_files == []


def test_cli_returns_missing_for_date_when_partition_empty(tmp_path: Path) -> None:
    from polarix.ingestion.cme_reference_ingest import MISSING_CME_RAW_FOR_DATE

    root = tmp_path / "raw" / "cme_sample"
    root.mkdir(parents=True, exist_ok=True)
    result = ingest(_config(tmp_path, "2026-05-19"))
    assert result.missing_reason == MISSING_CME_RAW_FOR_DATE
