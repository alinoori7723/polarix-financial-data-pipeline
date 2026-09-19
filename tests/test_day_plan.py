from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from polarix.orchestration.day_plan import (
    build_day_plan,
    build_multiday_plan,
    list_available_dates,
    mt5_window_ms,
    render_multiday_plan_text,
)


def _write_mt5(path: Path, ts_ms: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {"symbol": ["SPX500"] * len(ts_ms), "time_msc_utc_ms": pa.array(ts_ms, type=pa.int64())}
    )
    pq.write_table(table, path, compression="zstd")


def test_lists_available_mt5_dates(tmp_path: Path) -> None:
    data = tmp_path / "data"
    mt5_root = data / "normalized" / "mt5_ticks"
    for d in ("2026-05-18", "2026-05-19"):
        _write_mt5(
            mt5_root / "symbol=SPX500" / f"date={d}" / "part-0001.parquet",
            ts_ms=[1700000000000, 1700000000100],
        )
    dates = list_available_dates(mt5_root, ["SPX500", "NDX100"])
    assert dates == ["2026-05-18", "2026-05-19"]


def test_mt5_window_returns_min_max(tmp_path: Path) -> None:
    data = tmp_path / "data"
    mt5_root = data / "normalized" / "mt5_ticks"
    _write_mt5(
        mt5_root / "symbol=SPX500" / "date=2026-05-18" / "part-0001.parquet",
        ts_ms=[1700000000000, 1700000000500],
    )
    _write_mt5(
        mt5_root / "symbol=NDX100" / "date=2026-05-18" / "part-0001.parquet", ts_ms=[1700000000300]
    )
    win = mt5_window_ms(mt5_root, ["SPX500", "NDX100"], "2026-05-18")
    assert win == (1700000000000, 1700000000500)


def test_build_day_plan_recommends_window_with_roll(tmp_path: Path) -> None:
    data = tmp_path / "data"
    mt5_root = data / "normalized" / "mt5_ticks"
    import datetime as _dt

    t0 = int(
        _dt.datetime(2026, 5, 18, 5, 16, 46, 483000, tzinfo=_dt.timezone.utc).timestamp() * 1000
    )
    t1 = t0 + 4 * 60 * 60 * 1000
    _write_mt5(mt5_root / "symbol=SPX500" / "date=2026-05-18" / "part-0001.parquet", ts_ms=[t0, t1])
    plan = build_day_plan(
        "2026-05-18",
        data_root=data,
        reports_root=tmp_path / "reports",
        symbol_map={"ES": "SPX500", "NQ": "NDX100"},
        pre_roll_minutes=5,
        post_roll_minutes=5,
    )
    assert plan.mt5_symbols_present == ["SPX500"]
    assert plan.mt5_window_ms == (t0, t1)
    assert plan.recommended_cme_start_utc.endswith("Z")
    assert plan.recommended_cme_end_utc.endswith("Z")
    assert "05:11:46Z" in plan.recommended_cme_start_utc


def test_build_day_plan_reports_missing_cme_raw(tmp_path: Path) -> None:
    data = tmp_path / "data"
    mt5_root = data / "normalized" / "mt5_ticks"
    _write_mt5(
        mt5_root / "symbol=SPX500" / "date=2026-05-18" / "part-0001.parquet", ts_ms=[1700000000000]
    )
    plan = build_day_plan("2026-05-18", data_root=data, reports_root=tmp_path / "reports")
    assert plan.cme_raw_files == []
    assert plan.cme_normalized_files == []
    assert plan.gold_feature_files == []


def test_build_day_plan_reports_existing_cme_normalized(tmp_path: Path) -> None:
    data = tmp_path / "data"
    mt5_root = data / "normalized" / "mt5_ticks"
    _write_mt5(
        mt5_root / "symbol=SPX500" / "date=2026-05-18" / "part-0001.parquet", ts_ms=[1700000000000]
    )
    norm_dir = (
        data / "normalized" / "cme_reference" / "reference_trades" / "symbol=ES" / "date=2026-05-18"
    )
    norm_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"x": [1]}), norm_dir / "part-0001.parquet")
    plan = build_day_plan("2026-05-18", data_root=data, reports_root=tmp_path / "reports")
    assert len(plan.cme_normalized_files) == 1


def test_build_day_plan_does_not_call_databento(tmp_path: Path, monkeypatch) -> None:
    import socket

    def blocked(*a, **k):
        raise AssertionError("day plan must not open sockets")

    monkeypatch.setattr(socket, "create_connection", blocked)
    data = tmp_path / "data"
    mt5_root = data / "normalized" / "mt5_ticks"
    _write_mt5(
        mt5_root / "symbol=SPX500" / "date=2026-05-18" / "part-0001.parquet", ts_ms=[1700000000000]
    )
    build_day_plan("2026-05-18", data_root=data, reports_root=tmp_path / "reports")


def test_build_multiday_plan_summary(tmp_path: Path) -> None:
    data = tmp_path / "data"
    mt5_root = data / "normalized" / "mt5_ticks"
    for d in ("2026-05-18", "2026-05-19"):
        _write_mt5(
            mt5_root / "symbol=SPX500" / f"date={d}" / "part-0001.parquet",
            ts_ms=[1700000000000, 1700000001000],
        )
    plan = build_multiday_plan(data_root=data, reports_root=tmp_path / "reports")
    assert plan["summary"]["total_dates"] == 2
    assert plan["summary"]["dates_with_mt5_silver"] == 2
    assert plan["summary"]["dates_with_cme_raw"] == 0


def test_render_multiday_plan_text_includes_recommended_window(tmp_path: Path) -> None:
    data = tmp_path / "data"
    mt5_root = data / "normalized" / "mt5_ticks"
    _write_mt5(
        mt5_root / "symbol=SPX500" / "date=2026-05-18" / "part-0001.parquet", ts_ms=[1700000000000]
    )
    plan = build_multiday_plan(data_root=data, reports_root=tmp_path / "reports")
    txt = render_multiday_plan_text(plan)
    assert "Polarix Multi-Day Plan" in txt
    assert "recommended_cme_window" in txt
