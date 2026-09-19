from __future__ import annotations

import subprocess
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from polarix.orchestration import pipeline_status as ps
from polarix.orchestration.pipeline_orchestrator import (
    OrchestratorConfig,
    process_one_date,
    run_multiday,
)


def _make_mt5(path: Path, ts_ms: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {"symbol": ["SPX500"] * len(ts_ms), "time_msc_utc_ms": pa.array(ts_ms, type=pa.int64())}
    )
    pq.write_table(table, path, compression="zstd")


def _make_cme_raw(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"x": [1]}), path)


def _seed_day(tmp_path: Path, date: str, *, with_cme: bool = True) -> Path:
    data = tmp_path / "data"
    mt5 = data / "normalized" / "mt5_ticks"
    import datetime as _dt

    t0 = int(_dt.datetime.fromisoformat(f"{date}T05:00:00+00:00").timestamp() * 1000)
    _make_mt5(mt5 / "symbol=SPX500" / f"date={date}" / "part-0001.parquet", ts_ms=[t0, t0 + 10000])
    _make_mt5(mt5 / "symbol=NDX100" / f"date={date}" / "part-0001.parquet", ts_ms=[t0, t0 + 10000])
    if with_cme:
        raw = data / "raw" / "cme_sample" / f"date={date}"
        _make_cme_raw(raw / "ESNQ.parquet")
    return data


def _make_config(tmp_path: Path, dates: list[str], **overrides) -> OrchestratorConfig:
    data = tmp_path / "data"
    data.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[1]
    cfg = OrchestratorConfig(
        dates=dates,
        data_root=data,
        reports_root=tmp_path / "reports",
        repo_root=repo_root,
        python_exe=Path("python"),
        dry_run=True,
        min_free_disk_gb=0.0,
        min_available_memory_gb=0.0,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def test_disk_guard_stops_day(tmp_path: Path) -> None:
    _seed_day(tmp_path, "2026-05-18")
    cfg = _make_config(tmp_path, ["2026-05-18"], min_free_disk_gb=10000000.0)
    day = process_one_date(cfg, "2026-05-18")
    assert day.final_status == ps.STATUS_DISK_GUARD_STOP


def test_memory_guard_stops_day(tmp_path: Path, monkeypatch) -> None:
    _seed_day(tmp_path, "2026-05-18")
    from polarix.orchestration import resource_guards

    monkeypatch.setattr(resource_guards, "get_available_memory_gb", lambda: 0.0)
    cfg = _make_config(tmp_path, ["2026-05-18"], min_available_memory_gb=2.0)
    day = process_one_date(cfg, "2026-05-18")
    assert day.final_status == ps.STATUS_MEMORY_GUARD_STOP


def test_missing_mt5_silver_skipped_when_continue(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path, ["2026-05-18"], continue_on_missing_mt5=True)
    day = process_one_date(cfg, "2026-05-18")
    assert day.final_status == ps.STATUS_SKIPPED_NO_MT5_SILVER


def test_missing_mt5_silver_fails_day_when_strict(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path, ["2026-05-18"], continue_on_missing_mt5=False)
    day = process_one_date(cfg, "2026-05-18")
    assert day.final_status == ps.STATUS_DAY_FAILED


def test_missing_cme_raw_marked_range_unavailable(tmp_path: Path) -> None:
    _seed_day(tmp_path, "2026-05-18", with_cme=False)
    cfg = _make_config(tmp_path, ["2026-05-18"])
    day = process_one_date(cfg, "2026-05-18")
    assert day.final_status == ps.STATUS_DATABENTO_RANGE_UNAVAILABLE


def test_dry_run_lists_stage_argvs_without_running(tmp_path: Path) -> None:
    _seed_day(tmp_path, "2026-05-18")
    cfg = _make_config(tmp_path, ["2026-05-18"], dry_run=True)
    day = process_one_date(cfg, "2026-05-18")
    assert day.final_status == ps.STATUS_DAY_COMPLETE
    stage_names = [s["name"] for s in day.stages]
    assert "CME_INGEST" in stage_names
    assert "FEATURE_EDA" in stage_names
    assert all((s.get("dry_run") for s in day.stages))


def test_runner_called_per_stage_in_real_mode(tmp_path: Path) -> None:
    _seed_day(tmp_path, "2026-05-18")
    cfg = _make_config(tmp_path, ["2026-05-18"], dry_run=False, force=True)
    calls: list[list[str]] = []

    def runner(argv, timeout, cwd):
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, returncode=0, stdout="", stderr="")

    day = process_one_date(cfg, "2026-05-18", runner=runner)
    assert day.final_status == ps.STATUS_DAY_COMPLETE
    stage_scripts = [
        "ingest_cme_reference_sample.py",
        "cme_reference_quality_report.py",
        "alignment_quality_report.py",
        "bar_alignment_quality_report.py",
        "build_bar_features.py",
        "bar_feature_quality_report.py",
        "feature_eda_report.py",
    ]
    for script in stage_scripts:
        assert any((script in str(a) for a in calls)), f"{script} was not invoked"


def test_stage_failure_records_status_and_continues_by_default(tmp_path: Path) -> None:
    _seed_day(tmp_path, "2026-05-18")
    cfg = _make_config(tmp_path, ["2026-05-18"], dry_run=False, force=True)

    def runner(argv, timeout, cwd):
        if "ingest_cme_reference_sample.py" in argv[2]:
            return subprocess.CompletedProcess(argv, returncode=1, stdout="", stderr="boom")
        return subprocess.CompletedProcess(argv, returncode=0, stdout="", stderr="")

    day = process_one_date(cfg, "2026-05-18", runner=runner)
    statuses = [s["status"] for s in day.stages]
    assert ps.STAGE_FAIL in statuses
    assert day.final_status == ps.STATUS_DAY_FAILED
    assert ps.STAGE_SKIPPED_UPSTREAM_FAILED in statuses


def test_stage_failure_stops_day_when_strict(tmp_path: Path) -> None:
    _seed_day(tmp_path, "2026-05-18")
    cfg = _make_config(
        tmp_path, ["2026-05-18"], dry_run=False, force=True, continue_on_day_failure=False
    )

    def runner(argv, timeout, cwd):
        return subprocess.CompletedProcess(argv, returncode=1, stdout="", stderr="boom")

    day = process_one_date(cfg, "2026-05-18", runner=runner)
    assert day.final_status == ps.STATUS_DAY_FAILED


def test_databento_download_does_not_run_without_both_flags(tmp_path: Path) -> None:
    _seed_day(tmp_path, "2026-05-18", with_cme=False)
    cfg = _make_config(
        tmp_path, ["2026-05-18"], dry_run=False, download_cme=True, allow_databento_download=False
    )

    def download_runner(argv, timeout):
        raise AssertionError("download must not run when --allow-databento-download is false")

    def runner(argv, timeout, cwd):
        return subprocess.CompletedProcess(argv, returncode=0, stdout="", stderr="")

    day = process_one_date(cfg, "2026-05-18", runner=runner, download_runner=download_runner)
    assert day.download_outcome["status"] == "DATABENTO_DOWNLOAD_BLOCKED_BY_FLAGS"


def test_dry_run_never_invokes_download_runner_even_with_both_flags(tmp_path: Path) -> None:
    _seed_day(tmp_path, "2026-05-18", with_cme=False)
    cfg = _make_config(
        tmp_path, ["2026-05-18"], dry_run=True, download_cme=True, allow_databento_download=True
    )

    def download_runner(argv, timeout):
        raise AssertionError("dry-run must never call Databento")

    day = process_one_date(cfg, "2026-05-18", download_runner=download_runner)
    assert day.download_outcome["status"] == "DRY_RUN"


def test_run_multiday_writes_per_day_and_summary(tmp_path: Path) -> None:
    for d in ("2026-05-18", "2026-05-19"):
        _seed_day(tmp_path, d)
    cfg = _make_config(tmp_path, ["2026-05-18", "2026-05-19"], dry_run=False, force=True)

    def runner(argv, timeout, cwd):
        return subprocess.CompletedProcess(argv, returncode=0, stdout="", stderr="")

    result = run_multiday(cfg, runner=runner)
    assert len(result.per_day) == 2
    assert result.summary_json_path is not None and result.summary_json_path.exists()
    assert result.summary_txt_path is not None and result.summary_txt_path.exists()
    for p in result.per_day_json_paths:
        assert p.exists()


def test_orchestrator_does_not_aggregate_multi_day_dataframe(tmp_path: Path) -> None:
    for d in ("2026-05-18", "2026-05-19"):
        _seed_day(tmp_path, d)
    cfg = _make_config(tmp_path, ["2026-05-18", "2026-05-19"], dry_run=True)
    result = run_multiday(cfg)
    for day in result.per_day:
        d = day.to_dict()
        assert "stages" in d
        for forbidden in ("dataframe", "records", "rows", "all_rows", "bulk_data"):
            assert forbidden not in d, f"day report leaks {forbidden}"


PHASE_2F_FILES = (
    "src/polarix/orchestration/resource_guards.py",
    "src/polarix/orchestration/databento_download_plan.py",
    "src/polarix/orchestration/day_plan.py",
    "src/polarix/orchestration/pipeline_status.py",
    "src/polarix/orchestration/pipeline_orchestrator.py",
    "scripts/build_multiday_plan.py",
    "scripts/run_multiday_pipeline.py",
)


def _phase_2f_texts() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    return {p: (root / p).read_text(encoding="utf-8") for p in PHASE_2F_FILES}


def test_phase_2f_no_trading_functions() -> None:
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
    )
    for path, text in _phase_2f_texts().items():
        for token in forbidden:
            assert token not in text, f"{path}: forbidden {token!r}"


def test_phase_2f_no_mt5_sdk_imports() -> None:
    forbidden = (
        "import MetaTrader5",
        "from MetaTrader5",
        "import polarix.ingestion.mt5_readonly",
        "from polarix.ingestion.mt5_readonly",
        "mt5.order_send",
        "mt5.copy_ticks_from",
    )
    for path, text in _phase_2f_texts().items():
        for token in forbidden:
            assert token not in text, f"{path}: {token!r} present"


def test_phase_2f_no_model_training_imports() -> None:
    forbidden = (
        "import sklearn",
        "from sklearn",
        "import xgboost",
        "from xgboost",
        "import lightgbm",
        "from lightgbm",
        "import torch",
        "from torch",
        "import tensorflow",
        "from tensorflow",
        "model.fit(",
        "model_training",
        "train_test_split",
    )
    for path, text in _phase_2f_texts().items():
        for token in forbidden:
            assert token not in text, f"{path}: {token!r} present"


def test_phase_2f_no_cvd_cumulative_aggregation() -> None:
    forbidden = ("cumsum(", "cumulative_sum", "cvd_total", "cvd_running", "cvd_cumulative")
    for path, text in _phase_2f_texts().items():
        for token in forbidden:
            assert token not in text, f"{path}: {token!r} present"


def test_phase_2f_no_broad_multiday_databento_window_code() -> None:
    forbidden = ("days=7", "days=30", "weeks=", "TimeDelta(days=7", "timedelta(days=7")
    for path, text in _phase_2f_texts().items():
        for token in forbidden:
            assert token not in text, f"{path}: {token!r} present"


def test_phase_2f_no_pandas_corr_or_in_memory_eda_added() -> None:
    forbidden = ("pd.DataFrame.corr",)
    for path, text in _phase_2f_texts().items():
        for token in forbidden:
            assert token not in text, f"{path}: {token!r} present"


def test_phase_2f_no_synthetic_overlap_real_path() -> None:
    forbidden = (
        "synthetic_overlap",
        "fabricate_overlap",
        "manufacture_overlap",
        "shift_cme_to_mt5",
        "fake_overlap",
    )
    for path, text in _phase_2f_texts().items():
        for token in forbidden:
            assert token not in text, f"{path}: {token!r} present"


def test_phase_2f_no_automatic_data_deletion() -> None:
    import ast

    root = Path(__file__).resolve().parents[1]
    forbidden_fn_names = {"unlink", "rmtree", "remove"}
    for path in PHASE_2F_FILES:
        tree = ast.parse((root / path).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = None
                if isinstance(node.func, ast.Attribute):
                    name = node.func.attr
                elif isinstance(node.func, ast.Name):
                    name = node.func.id
                assert name not in forbidden_fn_names, (
                    f"{path}:{node.lineno}: forbidden deletion call {name!r}"
                )


def test_phase_2f_does_not_weaken_50ms_tick_contract() -> None:
    for path, text in _phase_2f_texts().items():
        assert "DEFAULT_ALIGNMENT_TOLERANCE_MS" not in text, (
            f"{path}: must not redefine tick-level tolerance"
        )
    from polarix.alignment.alignment_contract import DEFAULT_ALIGNMENT_TOLERANCE_MS as TC
    from polarix.alignment.alignment_quality import DEFAULT_ALIGNMENT_TOLERANCE_MS as TQ

    assert TC == 50
    assert TQ == 50


def test_keyboard_interrupt_terminates_active_children_and_exits_nonzero(tmp_path: Path) -> None:
    import subprocess as _sub

    from polarix.orchestration.subprocess_guard import ChildProcessRegistry

    _seed_day(tmp_path, "2026-05-18")
    cfg = _make_config(tmp_path, ["2026-05-18"], dry_run=False, force=True)
    registry = ChildProcessRegistry()

    class FakeRunningProc:
        pid = 9999
        args = ["python", "-u", "scripts/feature_eda_report.py"]
        terminate_called = False
        kill_called = False
        rc: int | None = None

        def poll(self):
            return self.rc

        def terminate(self):
            self.terminate_called = True
            self.rc = -15

        def kill(self):
            self.kill_called = True
            self.rc = -9

        def wait(self, timeout=None):
            return self.rc

    long_child = FakeRunningProc()
    triggered = {"once": False}

    def runner(argv, timeout, cwd):
        if not triggered["once"]:
            registry.add("LONG_RUNNER", long_child, argv)
            triggered["once"] = True
            raise KeyboardInterrupt
        return _sub.CompletedProcess(argv, returncode=0, stdout="", stderr="")

    result = run_multiday(cfg, runner=runner, registry=registry)
    assert result.interrupted is True
    assert long_child.terminate_called or long_child.kill_called
    assert any((r.get("outcome") in ("TERMINATED", "KILLED") for r in result.child_cleanup_records))


def test_dry_run_writes_detailed_report(tmp_path: Path) -> None:
    _seed_day(tmp_path, "2026-05-18")
    cfg = _make_config(tmp_path, ["2026-05-18"], dry_run=True)
    result = run_multiday(cfg)
    assert result.dry_run_report is not None
    assert result.dry_run_report_json_path is not None
    assert result.dry_run_report_txt_path is not None
    assert result.dry_run_report_json_path.exists()
    assert result.dry_run_report_txt_path.exists()
    [day] = result.dry_run_report["per_day"]
    assert "planned_stages" in day and day["planned_stages"]
    assert day["download_gate_decision"]["decision"] == "DOWNLOAD_BLOCKED_DRY_RUN"
    assert "per_day" in result.dry_run_report


def test_global_disk_guard_stops_entire_multiday_run(tmp_path: Path) -> None:
    import subprocess as _sub

    _seed_day(tmp_path, "2026-05-18")
    _seed_day(tmp_path, "2026-05-19")
    cfg = _make_config(
        tmp_path,
        ["2026-05-18", "2026-05-19"],
        dry_run=False,
        force=True,
        min_free_disk_gb=10000000.0,
    )

    def runner(argv, timeout, cwd):
        return _sub.CompletedProcess(argv, returncode=0, stdout="", stderr="")

    result = run_multiday(cfg, runner=runner)
    assert len(result.per_day) == 1
    assert result.summary["global_guard_stopped"] is True


def test_continue_on_day_failure_processes_next_date(tmp_path: Path) -> None:
    import subprocess as _sub

    _seed_day(tmp_path, "2026-05-18")
    _seed_day(tmp_path, "2026-05-19")
    cfg = _make_config(tmp_path, ["2026-05-18", "2026-05-19"], dry_run=False, force=True)

    def runner(argv, timeout, cwd):
        if "2026-05-18" in str(argv):
            return _sub.CompletedProcess(argv, returncode=1, stdout="", stderr="boom")
        return _sub.CompletedProcess(argv, returncode=0, stdout="", stderr="")

    result = run_multiday(cfg, runner=runner)
    assert len(result.per_day) == 2
    assert result.summary["global_guard_stopped"] is False


PHASE_2G_FILES = (
    "src/polarix/orchestration/databento_cost_guard.py",
    "src/polarix/orchestration/run_selection.py",
    "src/polarix/orchestration/multiday_dry_run_report.py",
    "src/polarix/orchestration/subprocess_guard.py",
)


def _phase_2g_texts() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    return {p: (root / p).read_text(encoding="utf-8") for p in PHASE_2G_FILES}


def test_phase_2g_no_trading_functions() -> None:
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
    )
    for path, text in _phase_2g_texts().items():
        for token in forbidden:
            assert token not in text, f"{path}: forbidden {token!r}"


def test_phase_2g_no_mt5_sdk_imports() -> None:
    forbidden = (
        "import MetaTrader5",
        "from MetaTrader5",
        "import polarix.ingestion.mt5_readonly",
        "from polarix.ingestion.mt5_readonly",
        "mt5.order_send",
        "mt5.copy_ticks_from",
    )
    for path, text in _phase_2g_texts().items():
        for token in forbidden:
            assert token not in text, f"{path}: {token!r} present"


def test_phase_2g_no_model_training_imports() -> None:
    forbidden = (
        "import sklearn",
        "from sklearn",
        "import xgboost",
        "from xgboost",
        "import lightgbm",
        "from lightgbm",
        "import torch",
        "from torch",
        "import tensorflow",
        "from tensorflow",
        "model.fit(",
        "model_training",
        "train_test_split",
    )
    for path, text in _phase_2g_texts().items():
        for token in forbidden:
            assert token not in text, f"{path}: {token!r} present"


def test_phase_2g_no_cvd_cumulative_aggregation() -> None:
    forbidden = ("cumsum(", "cumulative_sum", "cvd_total", "cvd_running", "cvd_cumulative")
    for path, text in _phase_2g_texts().items():
        for token in forbidden:
            assert token not in text, f"{path}: {token!r} present"


def test_phase_2g_no_broad_multiday_databento_window_code() -> None:
    forbidden = ("days=7", "days=30", "weeks=", "TimeDelta(days=7", "timedelta(days=7")
    for path, text in _phase_2g_texts().items():
        for token in forbidden:
            assert token not in text, f"{path}: {token!r} present"


def test_phase_2g_no_fake_cost_estimate_constants() -> None:
    text = _phase_2g_texts()["src/polarix/orchestration/databento_cost_guard.py"]
    forbidden = (
        "estimated_cost_usd=0.50",
        "estimated_cost_usd=1.99",
        "estimated_cost_usd=2.50",
        "estimated_size_gb=0.5,",
        "estimated_size_gb=1.5,",
    )
    for token in forbidden:
        assert token not in text, f"databento_cost_guard.py: fake estimate token {token!r}"


def test_phase_2g_no_fake_physical_limit_constants() -> None:
    text = _phase_2g_texts()["src/polarix/orchestration/databento_cost_guard.py"]
    forbidden = (
        "selected_limit_type='record_limit', selected_limit_value=1000)",
        'selected_limit_type="record_limit", selected_limit_value=1000)',
    )
    for token in forbidden:
        assert token not in text, f"databento_cost_guard.py: fake limit {token!r}"


def test_phase_2g_no_synthetic_overlap_real_path() -> None:
    forbidden = (
        "synthetic_overlap",
        "fabricate_overlap",
        "manufacture_overlap",
        "shift_cme_to_mt5",
        "fake_overlap",
    )
    for path, text in _phase_2g_texts().items():
        for token in forbidden:
            assert token not in text, f"{path}: {token!r} present"


def test_phase_2g_does_not_weaken_50ms_tick_contract() -> None:
    for path, text in _phase_2g_texts().items():
        assert "DEFAULT_ALIGNMENT_TOLERANCE_MS" not in text, (
            f"{path}: must not redefine tick-level tolerance"
        )
