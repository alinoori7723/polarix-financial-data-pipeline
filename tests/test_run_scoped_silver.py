from __future__ import annotations

import datetime as _dt
import io
import json
from contextlib import redirect_stdout
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from polarix.normalization import silver_paths
from polarix.normalization.normalization import NormalizationConfig, NormalizationError, normalize
from polarix.orchestration.run_metadata import RunMetadata
from polarix.quality.telemetry_quality import QualityConfig, build_quality_report

REPO_ROOT = Path(__file__).resolve().parents[1]
DATE = "2026-05-22"
RUN_A = "live_run_20260522T023634Z"
RUN_B = "live_run_20260522T124444Z"
OFFSET_MIN = 120
OFFSET_MS = OFFSET_MIN * 60 * 1000
RUN_A_START = "2026-05-22T02:36:34Z"
RUN_A_END = "2026-05-22T04:00:00Z"
RUN_B_START = "2026-05-22T12:44:44Z"
RUN_B_END = "2026-05-22T16:50:00Z"
BRONZE_FIELDS = [
    ("symbol", pa.string()),
    ("time_msc_raw", pa.int64()),
    ("recv_time_utc_ms", pa.int64()),
    ("monotonic_ns", pa.int64()),
    ("bid", pa.float64()),
    ("ask", pa.float64()),
    ("last", pa.float64()),
    ("bid_scaled", pa.int64()),
    ("ask_scaled", pa.int64()),
    ("last_scaled", pa.int64()),
    ("volume", pa.int64()),
    ("flags", pa.int64()),
    ("spread_points", pa.int64()),
    ("suppressed_count", pa.int64()),
    ("first_suppressed_time_ms", pa.int64()),
    ("last_suppressed_time_ms", pa.int64()),
    ("suppressed_reason", pa.string()),
]
BRONZE_SCHEMA = pa.schema(BRONZE_FIELDS)


def _iso_ms(iso: str) -> int:
    s = iso[:-1] + "+00:00" if iso.endswith("Z") else iso
    return int(_dt.datetime.fromisoformat(s).timestamp() * 1000)


def _bronze_row(symbol: str, utc_ms: int) -> dict:
    return {
        "symbol": symbol,
        "time_msc_raw": utc_ms + OFFSET_MS,
        "recv_time_utc_ms": utc_ms,
        "monotonic_ns": 0,
        "bid": 100.0,
        "ask": 100.5,
        "last": 0.0,
        "bid_scaled": 10000,
        "ask_scaled": 10050,
        "last_scaled": 0,
        "volume": 0,
        "flags": 0,
        "spread_points": 5,
        "suppressed_count": 0,
        "first_suppressed_time_ms": None,
        "last_suppressed_time_ms": None,
        "suppressed_reason": None,
    }


def _write_bronze(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = {name: [] for name, _ in BRONZE_FIELDS}
    for r in rows:
        for name, _ in BRONZE_FIELDS:
            cols[name].append(r.get(name))
    pq.write_table(pa.Table.from_pydict(cols, schema=BRONZE_SCHEMA), path, compression="zstd")


def _seed_bronze_two_runs(raw_root: Path) -> dict[str, list[int]]:
    run_a_times = [_iso_ms("2026-05-22T02:40:00Z"), _iso_ms("2026-05-22T03:10:00Z")]
    run_b_times = [
        _iso_ms("2026-05-22T12:45:00Z"),
        _iso_ms("2026-05-22T14:00:00Z"),
        _iso_ms("2026-05-22T16:40:00Z"),
    ]
    for symbol in ("SPX500", "NDX100"):
        _write_bronze(
            raw_root / f"symbol={symbol}" / f"date={DATE}" / "hour=02" / "part-a.parquet",
            [_bronze_row(symbol, run_a_times[0])],
        )
        _write_bronze(
            raw_root / f"symbol={symbol}" / f"date={DATE}" / "hour=03" / "part-a.parquet",
            [_bronze_row(symbol, run_a_times[1])],
        )
        _write_bronze(
            raw_root / f"symbol={symbol}" / f"date={DATE}" / "hour=12" / "part-b.parquet",
            [_bronze_row(symbol, run_b_times[0])],
        )
        _write_bronze(
            raw_root / f"symbol={symbol}" / f"date={DATE}" / "hour=14" / "part-b.parquet",
            [_bronze_row(symbol, run_b_times[1])],
        )
        _write_bronze(
            raw_root / f"symbol={symbol}" / f"date={DATE}" / "hour=16" / "part-b.parquet",
            [_bronze_row(symbol, run_b_times[2])],
        )
    return {"run_a": run_a_times, "run_b": run_b_times}


def _metadata(run_id: str, start: str, end: str) -> RunMetadata:
    return RunMetadata(
        source_path=Path(f"/fake/logger_runs/{run_id}/logger_manifest.json"),
        source_kind="logger_manifest",
        run_id=run_id,
        verified_offset_min=OFFSET_MIN,
        timestamp_semantics_status="OFFSET_VERIFIED_FOR_SESSION",
        clock_status={},
        broker_metadata={
            "account_company": "TestBroker",
            "account_server": "Test-Server",
            "account_login_hash": "deadbeef",
        },
        run_started_at_utc=start,
        run_ended_at_utc=end,
        final_decision="PASS",
        symbols=("SPX500", "NDX100"),
        source_type="run_scoped",
        latest_data_file_mtime_utc=end,
    )


def _normalize_run(
    tmp_path: Path, raw_root: Path, run_id: str, start: str, end: str, *, force: bool = False
):
    cfg = NormalizationConfig(
        raw_root=raw_root,
        silver_root=tmp_path / "silver",
        date=DATE,
        metadata=_metadata(run_id, start, end),
        force=force,
        run_id=run_id,
        run_window_start_utc=start,
        run_window_end_utc=end,
    )
    return normalize(cfg)


def _silver_root(tmp_path: Path) -> Path:
    return (tmp_path / "silver").resolve()


def test_a_normalize_run_a_writes_only_run_scoped_partition(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    _seed_bronze_two_runs(raw_root)
    _normalize_run(tmp_path, raw_root, RUN_A, RUN_A_START, RUN_A_END)
    sroot = _silver_root(tmp_path)
    run_a_files = silver_paths._part_files(
        silver_paths.run_scoped_silver_dir(sroot, "SPX500", DATE, RUN_A)
    )
    assert run_a_files, "run_A run-scoped Silver part files must exist"
    assert silver_paths.legacy_part_files(sroot, "SPX500", DATE) == []
    assert (sroot / f"date={DATE}" / f"run_id={RUN_A}" / "normalization_manifest.json").exists()


def test_b_normalize_run_b_does_not_touch_run_a(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    _seed_bronze_two_runs(raw_root)
    _normalize_run(tmp_path, raw_root, RUN_A, RUN_A_START, RUN_A_END)
    sroot = _silver_root(tmp_path)
    run_a_dir = silver_paths.run_scoped_silver_dir(sroot, "SPX500", DATE, RUN_A)
    before = {p.name: p.read_bytes() for p in silver_paths._part_files(run_a_dir)}
    assert before
    _normalize_run(tmp_path, raw_root, RUN_B, RUN_B_START, RUN_B_END)
    after = {p.name: p.read_bytes() for p in silver_paths._part_files(run_a_dir)}
    assert after == before, "run_A partition must be byte-identical after run_B normalize"
    assert silver_paths._part_files(
        silver_paths.run_scoped_silver_dir(sroot, "SPX500", DATE, RUN_B)
    )


def test_c_force_run_b_rebuilds_only_run_b(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    _seed_bronze_two_runs(raw_root)
    _normalize_run(tmp_path, raw_root, RUN_A, RUN_A_START, RUN_A_END)
    _normalize_run(tmp_path, raw_root, RUN_B, RUN_B_START, RUN_B_END)
    sroot = _silver_root(tmp_path)
    run_a_dir = silver_paths.run_scoped_silver_dir(sroot, "SPX500", DATE, RUN_A)
    a_before = {p.name: p.read_bytes() for p in silver_paths._part_files(run_a_dir)}
    _normalize_run(tmp_path, raw_root, RUN_B, RUN_B_START, RUN_B_END, force=True)
    a_after = {p.name: p.read_bytes() for p in silver_paths._part_files(run_a_dir)}
    assert a_after == a_before, "--force on run_B must NOT touch run_A"
    run_b_dir = silver_paths.run_scoped_silver_dir(sroot, "SPX500", DATE, RUN_B)
    assert silver_paths._part_files(run_b_dir)


def test_d_run_b_silver_excludes_overnight_rows(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    times = _seed_bronze_two_runs(raw_root)
    result = _normalize_run(tmp_path, raw_root, RUN_B, RUN_B_START, RUN_B_END)
    sroot = _silver_root(tmp_path)
    run_b_dir = silver_paths.run_scoped_silver_dir(sroot, "SPX500", DATE, RUN_B)
    table = pa.concat_tables(
        [pq.ParquetFile(f).read() for f in silver_paths._part_files(run_b_dir)]
    )
    recv = sorted(table.column("recv_time_utc_ms").to_pylist())
    assert recv == sorted(times["run_b"])
    for a_ms in times["run_a"]:
        assert a_ms not in recv
    assert result.rows_excluded_by_window == 2 * 2


def test_d2_run_scoped_normalization_requires_a_window(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    _seed_bronze_two_runs(raw_root)
    cfg = NormalizationConfig(
        raw_root=raw_root,
        silver_root=tmp_path / "silver",
        date=DATE,
        metadata=_metadata(RUN_B, RUN_B_START, RUN_B_END),
        run_id=RUN_B,
        run_window_start_utc=None,
        run_window_end_utc=None,
    )
    with pytest.raises(NormalizationError):
        normalize(cfg)


def test_e_telemetry_quality_run_id_reads_only_that_run(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    times = _seed_bronze_two_runs(raw_root)
    _normalize_run(tmp_path, raw_root, RUN_A, RUN_A_START, RUN_A_END)
    _normalize_run(tmp_path, raw_root, RUN_B, RUN_B_START, RUN_B_END)
    sroot = _silver_root(tmp_path)
    cfg = QualityConfig(
        date=DATE,
        raw_root=raw_root,
        silver_root=sroot,
        reports_root=tmp_path / "reports",
        metadata=_metadata(RUN_B, RUN_B_START, RUN_B_END),
        run_id=RUN_B,
    )
    report = build_quality_report(cfg)
    assert report["requested_run_id"] == RUN_B
    assert report["selected_run_id"] == RUN_B
    assert report["silver_layout"] == silver_paths.LAYOUT_RUN_SCOPED
    assert report["silver_layout_version"] == silver_paths.SILVER_LAYOUT_VERSION_RUN_SCOPED
    assert report["per_symbol"]["SPX500"]["total_rows"] == len(times["run_b"])
    assert report["per_symbol"]["NDX100"]["total_rows"] == len(times["run_b"])


def test_e2_telemetry_quality_no_run_id_multi_run_fails_closed(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    _seed_bronze_two_runs(raw_root)
    _normalize_run(tmp_path, raw_root, RUN_A, RUN_A_START, RUN_A_END)
    _normalize_run(tmp_path, raw_root, RUN_B, RUN_B_START, RUN_B_END)
    cfg = QualityConfig(
        date=DATE,
        raw_root=raw_root,
        silver_root=_silver_root(tmp_path),
        reports_root=tmp_path / "reports",
        metadata=_metadata(RUN_B, RUN_B_START, RUN_B_END),
        run_id=None,
    )
    report = build_quality_report(cfg)
    assert report["decision"] == "FAIL"
    assert any(("multiple run-scoped Silver partitions" in w for w in report["fatal_warnings"]))


def _run_overlap_cli(argv: list[str]) -> tuple[int, dict]:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "check_databento_overlap_window_under_test",
        REPO_ROOT / "scripts" / "check_databento_overlap_window.py",
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = mod.main(argv)
    return (rc, json.loads(buf.getvalue()))


def test_f_overlap_planner_run_id_returns_run_b_window(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    _seed_bronze_two_runs(raw_root)
    _normalize_run(tmp_path, raw_root, RUN_A, RUN_A_START, RUN_A_END)
    _normalize_run(tmp_path, raw_root, RUN_B, RUN_B_START, RUN_B_END)
    rc, out = _run_overlap_cli(
        ["--date", DATE, "--mt5-root", str(_silver_root(tmp_path)), "--run-id", RUN_B]
    )
    assert rc == 0
    assert out["selected_run_id"] == RUN_B
    rec = out["cme_databento_recommendations"]["ES"]
    start = rec["recommended_start_utc"]
    end = rec["recommended_end_utc"]
    assert start is not None and end is not None
    assert "2026-05-22T12:" in start, f"expected NY-window start, got {start}"
    assert start > "2026-05-22T12:00", "recommended window must not start in the overnight run"
    assert "2026-05-22T16:" in end, f"expected NY-window end, got {end}"


def test_g_overlap_planner_no_run_id_multi_run_fails_closed(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    _seed_bronze_two_runs(raw_root)
    _normalize_run(tmp_path, raw_root, RUN_A, RUN_A_START, RUN_A_END)
    _normalize_run(tmp_path, raw_root, RUN_B, RUN_B_START, RUN_B_END)
    rc, out = _run_overlap_cli(["--date", DATE, "--mt5-root", str(_silver_root(tmp_path))])
    assert rc == 2
    assert out["error"] == "AMBIGUOUS_MULTIPLE_RUN_PARTITIONS"
    assert set(out["available_run_ids"]) == {RUN_A, RUN_B}
    assert "cme_databento_recommendations" not in out


def _make_run_scoped_metadata_dir(reports_root: Path, run_id: str, start: str, end: str) -> Path:
    d = reports_root / "logger_runs" / run_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "logger_manifest.json").write_text(
        json.dumps(
            {
                "timestamp_semantics": {
                    "verified_offset_min": OFFSET_MIN,
                    "status": "OFFSET_VERIFIED_FOR_SESSION",
                },
                "generated_at_utc": end,
                "symbols": ["SPX500", "NDX100"],
            }
        ),
        encoding="utf-8",
    )
    (d / "summary.json").write_text(
        json.dumps(
            {
                "started_at_utc": start,
                "ended_at_utc": end,
                "latest_data_file_mtime_utc": end,
                "final_decision": "PASS",
                "broker_metadata": {
                    "account_company": "TestBroker",
                    "account_server": "Test-Server",
                    "account_login_hash": "deadbeef",
                },
                "symbols": ["SPX500", "NDX100"],
            }
        ),
        encoding="utf-8",
    )
    return d


def _run_normalize_cli(argv: list[str]) -> tuple[int, str]:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "normalize_telemetry_under_test", REPO_ROOT / "scripts" / "normalize_telemetry.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = mod.main(argv)
    return (rc, buf.getvalue())


def test_h_normalize_no_run_id_multi_run_fails_closed(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    _seed_bronze_two_runs(raw_root)
    reports_root = tmp_path / "reports"
    _make_run_scoped_metadata_dir(reports_root, RUN_A, RUN_A_START, RUN_A_END)
    _make_run_scoped_metadata_dir(reports_root, RUN_B, RUN_B_START, RUN_B_END)
    rc, _out = _run_normalize_cli(
        [
            "--date",
            DATE,
            "--raw-root",
            str(raw_root),
            "--normalized-root",
            str(tmp_path / "silver"),
            "--reports-root",
            str(reports_root),
        ]
    )
    assert rc == 2, "normalize without --run-id must fail closed on 2 candidate runs"
    assert not (tmp_path / "silver").exists() or not list(
        (tmp_path / "silver").rglob("part-*.parquet")
    )


def test_h2_normalize_cli_run_id_writes_run_scoped(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    _seed_bronze_two_runs(raw_root)
    reports_root = tmp_path / "reports"
    _make_run_scoped_metadata_dir(reports_root, RUN_A, RUN_A_START, RUN_A_END)
    _make_run_scoped_metadata_dir(reports_root, RUN_B, RUN_B_START, RUN_B_END)
    rc, out = _run_normalize_cli(
        [
            "--date",
            DATE,
            "--raw-root",
            str(raw_root),
            "--normalized-root",
            str(tmp_path / "silver"),
            "--reports-root",
            str(reports_root),
            "--run-id",
            RUN_B,
        ]
    )
    assert rc == 0
    doc = json.loads(out)
    assert doc["selected_run_id"] == RUN_B
    assert doc["silver_layout"] == "run_scoped"
    assert doc["silver_layout_version"] == silver_paths.SILVER_LAYOUT_VERSION_RUN_SCOPED
    sroot = (tmp_path / "silver").resolve()
    assert silver_paths._part_files(
        silver_paths.run_scoped_silver_dir(sroot, "SPX500", DATE, RUN_B)
    )
    assert doc["rows_excluded_by_window"] == 4


PHASE_2F2_FILES = (
    "src/polarix/normalization/silver_paths.py",
    "src/polarix/normalization/normalization.py",
    "src/polarix/quality/telemetry_quality.py",
    "src/polarix/orchestration/run_metadata.py",
    "scripts/normalize_telemetry.py",
    "scripts/telemetry_quality_report.py",
    "scripts/check_databento_overlap_window.py",
)


def test_i_no_trading_or_forbidden_code_in_phase_2f2_files() -> None:
    forbidden = (
        "order_send",
        "order_check",
        "order_calc_margin",
        "order_calc_profit",
        "position_close",
        "positions_close",
        "position_modify",
        "TradeRequest",
        "MqlTradeRequest",
        "import MetaTrader5",
        "from MetaTrader5",
        "model.fit(",
        "train_test_split",
        "make_labels",
        "build_targets",
        "cumsum(",
        "cvd_cumulative",
        "cvd_running",
        "cvd_total",
    )
    for path in PHASE_2F2_FILES:
        text = (REPO_ROOT / path).read_text(encoding="utf-8")
        for tok in forbidden:
            assert tok not in text, f"{path}: forbidden token {tok!r}"


def test_i2_select_silver_classifies_legacy_and_missing(tmp_path: Path) -> None:
    sroot = tmp_path / "silver"
    sel = silver_paths.select_silver(sroot, "SPX500", DATE)
    assert sel.layout == silver_paths.LAYOUT_MISSING
    legacy_dir = silver_paths.legacy_silver_dir(sroot, "SPX500", DATE)
    legacy_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"x": [1]}), legacy_dir / "part-legacy.parquet")
    sel = silver_paths.select_silver(sroot, "SPX500", DATE)
    assert sel.layout == silver_paths.LAYOUT_LEGACY_DATE_LEVEL
    assert sel.ok is True
    assert any(("LEGACY" in w for w in sel.warnings))
    rs_dir = silver_paths.run_scoped_silver_dir(sroot, "SPX500", DATE, RUN_B)
    rs_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"x": [1]}), rs_dir / "part-rs.parquet")
    sel = silver_paths.select_silver(sroot, "SPX500", DATE, run_id=RUN_B)
    assert sel.layout == silver_paths.LAYOUT_RUN_SCOPED
    assert sel.ok is True
