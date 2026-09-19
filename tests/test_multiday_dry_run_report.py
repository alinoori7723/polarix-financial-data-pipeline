from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from polarix.orchestration.databento_cost_guard import (
    DOWNLOAD_BLOCKED_DRY_RUN,
    NOT_REQUESTED_DRY_RUN,
    PHYSICAL_LIMIT_NOT_REQUESTED_DRY_RUN,
)
from polarix.orchestration.multiday_dry_run_report import (
    DryRunConfig,
    build_multiday_dry_run_report,
    build_per_day_plan,
    render_dry_run_text,
)


def _write_mt5(path: Path, ts_ms: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {"symbol": ["SPX500"] * len(ts_ms), "time_msc_utc_ms": pa.array(ts_ms, type=pa.int64())}
    )
    pq.write_table(table, path, compression="zstd")


def _seed_mt5(tmp_path: Path, date: str) -> Path:
    import datetime as _dt

    t0 = int(_dt.datetime.fromisoformat(f"{date}T05:00:00+00:00").timestamp() * 1000)
    data = tmp_path / "data"
    mt5 = data / "normalized" / "mt5_ticks"
    _write_mt5(mt5 / "symbol=SPX500" / f"date={date}" / "part-0001.parquet", ts_ms=[t0, t0 + 60000])
    return data


def test_per_day_plan_contains_planned_steps(tmp_path: Path) -> None:
    data = _seed_mt5(tmp_path, "2026-05-18")
    cfg = DryRunConfig(
        dates=["2026-05-18"],
        data_root=data,
        reports_root=tmp_path / "reports",
        repo_root=Path(__file__).resolve().parents[1],
    )
    day = build_per_day_plan(cfg, "2026-05-18", python_exe="python")
    names = [s["name"] for s in day["planned_stages"]]
    assert "CME_INGEST" in names and "FEATURE_EDA" in names


def test_per_day_plan_contains_skip_reasons(tmp_path: Path) -> None:
    data = _seed_mt5(tmp_path, "2026-05-18")
    cfg = DryRunConfig(
        dates=["2026-05-18"],
        data_root=data,
        reports_root=tmp_path / "reports",
        repo_root=Path(__file__).resolve().parents[1],
    )
    day = build_per_day_plan(cfg, "2026-05-18", python_exe="python")
    non_running = [s for s in day["planned_stages"] if not s["would_run"]]
    assert non_running, "with no CME raw + no --download-cme, stages must not run"
    download = next((s for s in non_running if s["name"] == "CME_DOWNLOAD"))
    assert download["mark"] == "SKIP"
    assert download["reason"] == "DOWNLOAD_NOT_REQUESTED"
    downstream = [s for s in non_running if s["name"] != "CME_DOWNLOAD"]
    assert downstream
    for s in downstream:
        assert s["mark"] == "BLOCKED"
        assert s["reason"] == "BLOCKED_MISSING_CME_RAW"


def test_per_day_plan_includes_artifact_detection(tmp_path: Path) -> None:
    data = _seed_mt5(tmp_path, "2026-05-18")
    cfg = DryRunConfig(
        dates=["2026-05-18"],
        data_root=data,
        reports_root=tmp_path / "reports",
        repo_root=Path(__file__).resolve().parents[1],
    )
    day = build_per_day_plan(cfg, "2026-05-18", python_exe="python")
    for key in (
        "cme_raw_files_present",
        "cme_normalized_files_present",
        "alignment_report_present",
        "bar_alignment_report_present",
        "gold_features_present",
        "eda_report_present",
    ):
        assert key in day


def test_per_day_plan_recommended_databento_window(tmp_path: Path) -> None:
    data = _seed_mt5(tmp_path, "2026-05-18")
    cfg = DryRunConfig(
        dates=["2026-05-18"],
        data_root=data,
        reports_root=tmp_path / "reports",
        repo_root=Path(__file__).resolve().parents[1],
    )
    day = build_per_day_plan(cfg, "2026-05-18", python_exe="python")
    start, end = day["recommended_databento_window_utc"]
    assert start is not None and end is not None


def test_per_day_plan_contains_cost_and_physical_status(tmp_path: Path) -> None:
    data = _seed_mt5(tmp_path, "2026-05-18")
    cfg = DryRunConfig(
        dates=["2026-05-18"],
        data_root=data,
        reports_root=tmp_path / "reports",
        repo_root=Path(__file__).resolve().parents[1],
    )
    day = build_per_day_plan(cfg, "2026-05-18", python_exe="python")
    assert day["cost_estimate"]["status"] == NOT_REQUESTED_DRY_RUN
    assert day["physical_limit"]["status"] == PHYSICAL_LIMIT_NOT_REQUESTED_DRY_RUN


def test_per_day_plan_download_gate_decision_blocked_dry_run(tmp_path: Path) -> None:
    data = _seed_mt5(tmp_path, "2026-05-18")
    cfg = DryRunConfig(
        dates=["2026-05-18"],
        data_root=data,
        reports_root=tmp_path / "reports",
        repo_root=Path(__file__).resolve().parents[1],
        download_cme=True,
        allow_databento_download=True,
        acknowledge_cost_risk=True,
        api_key_present=True,
    )
    day = build_per_day_plan(cfg, "2026-05-18", python_exe="python")
    assert day["download_gate_decision"]["decision"] == DOWNLOAD_BLOCKED_DRY_RUN
    assert day["would_real_download_be_attempted_if_not_dry_run"] is True


def test_per_day_plan_run_selection_present(tmp_path: Path) -> None:
    data = _seed_mt5(tmp_path, "2026-05-18")
    cfg = DryRunConfig(
        dates=["2026-05-18"],
        data_root=data,
        reports_root=tmp_path / "reports",
        repo_root=Path(__file__).resolve().parents[1],
    )
    day = build_per_day_plan(cfg, "2026-05-18", python_exe="python")
    assert "run_selection" in day
    assert "selected_run_id" in day["run_selection"]
    assert day["run_selection"]["error_reason"] == "NO_RUN_SCOPED_DIR_FOR_DATE"


def test_per_day_plan_disk_and_memory_reported(tmp_path: Path) -> None:
    data = _seed_mt5(tmp_path, "2026-05-18")
    cfg = DryRunConfig(
        dates=["2026-05-18"],
        data_root=data,
        reports_root=tmp_path / "reports",
        repo_root=Path(__file__).resolve().parents[1],
    )
    day = build_per_day_plan(cfg, "2026-05-18", python_exe="python")
    assert "disk_free_gb_now" in day and day["disk_free_gb_now"] >= 0
    assert "available_memory_gb_now" in day and day["available_memory_gb_now"] >= 0


def test_multiday_dry_run_report_per_day(tmp_path: Path) -> None:
    data = _seed_mt5(tmp_path, "2026-05-18")
    _seed_mt5(tmp_path, "2026-05-19")
    cfg = DryRunConfig(
        dates=["2026-05-18", "2026-05-19"],
        data_root=data,
        reports_root=tmp_path / "reports",
        repo_root=Path(__file__).resolve().parents[1],
    )
    report = build_multiday_dry_run_report(cfg, python_exe="python")
    assert len(report["per_day"]) == 2
    assert report["dates_requested"] == ["2026-05-18", "2026-05-19"]
    assert report["this_report"].startswith("dry-run only")


def test_render_dry_run_text_contains_expected_sections(tmp_path: Path) -> None:
    data = _seed_mt5(tmp_path, "2026-05-18")
    cfg = DryRunConfig(
        dates=["2026-05-18"],
        data_root=data,
        reports_root=tmp_path / "reports",
        repo_root=Path(__file__).resolve().parents[1],
    )
    report = build_multiday_dry_run_report(cfg, python_exe="python")
    txt = render_dry_run_text(report)
    assert "Polarix Multi-Day Dry-Run Report" in txt
    assert "DATE 2026-05-18" in txt
    assert "planned_stages" in txt
    assert "download_gate_decision" in txt
    assert "cost_estimate_status" in txt
    assert "physical_limit_status" in txt
