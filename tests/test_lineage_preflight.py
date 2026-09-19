from __future__ import annotations

import importlib.util
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Callable

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CLI_PATH = REPO_ROOT / "scripts" / "run_multiday_pipeline.py"


def _load_main(module_name: str) -> Callable[[list[str]], int]:
    spec = importlib.util.spec_from_file_location(module_name, CLI_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod.main


def _write_mt5(path: Path, ts_ms: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {"symbol": ["SPX500"] * len(ts_ms), "time_msc_utc_ms": pa.array(ts_ms, type=pa.int64())}
    )
    pq.write_table(table, path, compression="zstd")


def _seed_mt5_day(tmp_path: Path, date: str, hours: float = 4.0) -> Path:
    import datetime as _dt

    data = tmp_path / "data"
    mt5 = data / "normalized" / "mt5_ticks"
    t0 = int(_dt.datetime.fromisoformat(f"{date}T05:00:00+00:00").timestamp() * 1000)
    t1 = t0 + int(hours * 3600 * 1000)
    _write_mt5(mt5 / "symbol=SPX500" / f"date={date}" / "part-0001.parquet", ts_ms=[t0, t1])
    return data


def _seed_downstream_artifacts(
    reports_root: Path, date: str, data_root: Path, symbols: list[str] = ("ES", "NQ")
) -> None:
    reports_root.mkdir(parents=True, exist_ok=True)
    (reports_root / f"alignment_quality_{date}.json").write_text("{}", encoding="utf-8")
    (reports_root / f"bar_alignment_quality_{date}.json").write_text("{}", encoding="utf-8")
    (reports_root / f"feature_eda_{date}.json").write_text("{}", encoding="utf-8")
    norm_dir = (
        data_root
        / "normalized"
        / "cme_reference"
        / "reference_trades"
        / "symbol=ES"
        / f"date={date}"
    )
    norm_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"x": [1]}), norm_dir / "part-0001.parquet")
    gold_dir = (
        data_root
        / "features"
        / "bar_features"
        / "symbol_pair=ES_SPX500"
        / f"date={date}"
        / "bucket=15s"
    )
    gold_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"x": [1]}), gold_dir / "part-0001.parquet")


def test_lineage_status_marks_rebuild_when_downstream_without_cme_raw(tmp_path: Path) -> None:
    from polarix.orchestration.multiday_dry_run_report import (
        LINEAGE_DOWNSTREAM_WITHOUT_RAW_CME,
        LINEAGE_LEGACY_ARTIFACTS_PRESENT,
        LINEAGE_MISSING_CME_RAW,
        LINEAGE_REBUILD_REQUIRED_AFTER_CME_DOWNLOAD,
        LINEAGE_UNTRUSTED_WARNING,
        DryRunConfig,
        build_per_day_plan,
    )

    data = _seed_mt5_day(tmp_path, "2026-05-18", hours=4.0)
    _seed_downstream_artifacts(tmp_path / "reports", "2026-05-18", data)
    cfg = DryRunConfig(
        dates=["2026-05-18"], data_root=data, reports_root=tmp_path / "reports", repo_root=REPO_ROOT
    )
    day = build_per_day_plan(cfg, "2026-05-18", python_exe="python")
    assert day["cme_raw_files_present"] is False
    assert day["downstream_artifacts_present"] is True
    assert LINEAGE_MISSING_CME_RAW in day["lineage_status"]
    assert LINEAGE_DOWNSTREAM_WITHOUT_RAW_CME in day["lineage_status"]
    assert LINEAGE_LEGACY_ARTIFACTS_PRESENT in day["lineage_status"]
    assert LINEAGE_REBUILD_REQUIRED_AFTER_CME_DOWNLOAD in day["lineage_status"]
    assert LINEAGE_UNTRUSTED_WARNING in day["lineage_warnings"]


def test_lineage_status_clean_when_cme_raw_present(tmp_path: Path) -> None:
    from polarix.orchestration.multiday_dry_run_report import (
        LINEAGE_CLEAN_READY,
        DryRunConfig,
        build_per_day_plan,
    )

    data = _seed_mt5_day(tmp_path, "2026-05-18", hours=4.0)
    raw_dir = data / "raw" / "cme_sample" / "date=2026-05-18"
    raw_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"x": [1]}), raw_dir / "sample.parquet")
    cfg = DryRunConfig(
        dates=["2026-05-18"], data_root=data, reports_root=tmp_path / "reports", repo_root=REPO_ROOT
    )
    day = build_per_day_plan(cfg, "2026-05-18", python_exe="python")
    assert day["cme_raw_files_present"] is True
    assert LINEAGE_CLEAN_READY in day["lineage_status"]


def test_planned_stages_include_cme_download_before_cme_ingest(tmp_path: Path) -> None:
    from polarix.orchestration.multiday_dry_run_report import DryRunConfig, build_per_day_plan

    data = _seed_mt5_day(tmp_path, "2026-05-18", hours=4.0)
    cfg = DryRunConfig(
        dates=["2026-05-18"], data_root=data, reports_root=tmp_path / "reports", repo_root=REPO_ROOT
    )
    day = build_per_day_plan(cfg, "2026-05-18", python_exe="python")
    names = [s["name"] for s in day["planned_stages"]]
    assert names[0] == "CME_DOWNLOAD"
    assert "CME_INGEST" in names
    assert names.index("CME_DOWNLOAD") < names.index("CME_INGEST")


def test_no_download_cme_flag_marks_stage_skip_not_requested(tmp_path: Path) -> None:
    from polarix.orchestration.multiday_dry_run_report import DryRunConfig, build_per_day_plan

    data = _seed_mt5_day(tmp_path, "2026-05-18", hours=4.0)
    cfg = DryRunConfig(
        dates=["2026-05-18"],
        data_root=data,
        reports_root=tmp_path / "reports",
        repo_root=REPO_ROOT,
        download_cme=False,
    )
    day = build_per_day_plan(cfg, "2026-05-18", python_exe="python")
    stages = {s["name"]: s for s in day["planned_stages"]}
    assert stages["CME_DOWNLOAD"]["mark"] == "SKIP"
    assert stages["CME_DOWNLOAD"]["reason"] == "DOWNLOAD_NOT_REQUESTED"
    for name in (
        "CME_INGEST",
        "CME_QUALITY",
        "ALIGNMENT_QUALITY",
        "BAR_ALIGNMENT",
        "BUILD_BAR_FEATURES",
        "BAR_FEATURE_QUALITY",
        "FEATURE_EDA",
    ):
        assert stages[name]["mark"] == "BLOCKED"
        assert stages[name]["reason"] == "BLOCKED_MISSING_CME_RAW"


def test_download_cme_dry_run_marks_download_plan(tmp_path: Path) -> None:
    from polarix.orchestration.multiday_dry_run_report import DryRunConfig, build_per_day_plan

    data = _seed_mt5_day(tmp_path, "2026-05-18", hours=4.0)
    cfg = DryRunConfig(
        dates=["2026-05-18"],
        data_root=data,
        reports_root=tmp_path / "reports",
        repo_root=REPO_ROOT,
        download_cme=True,
        allow_databento_download=True,
        acknowledge_cost_risk=True,
        api_key_present=True,
    )
    day = build_per_day_plan(cfg, "2026-05-18", python_exe="python")
    stages = {s["name"]: s for s in day["planned_stages"]}
    assert stages["CME_DOWNLOAD"]["mark"] == "PLAN"
    assert stages["CME_DOWNLOAD"]["reason"] == "DOWNLOAD_WOULD_BE_ATTEMPTED_IF_REAL"
    for name in ("CME_INGEST", "CME_QUALITY", "ALIGNMENT_QUALITY"):
        assert stages[name]["mark"] == "RUN_AFTER_DOWNLOAD"


def test_high_volume_window_warning_fires_above_threshold(tmp_path: Path) -> None:
    from polarix.orchestration.multiday_dry_run_report import (
        WARNING_HIGH_DATA_VOLUME_WINDOW,
        DryRunConfig,
        build_per_day_plan,
    )

    data = _seed_mt5_day(tmp_path, "2026-05-18", hours=18.0)
    cfg = DryRunConfig(
        dates=["2026-05-18"],
        data_root=data,
        reports_root=tmp_path / "reports",
        repo_root=REPO_ROOT,
        high_volume_window_hours=6.0,
    )
    day = build_per_day_plan(cfg, "2026-05-18", python_exe="python")
    assert day["high_volume_window_warning"] == WARNING_HIGH_DATA_VOLUME_WINDOW
    assert day["recommended_databento_window_hours"] > 6.0


def test_high_volume_window_silent_when_below_threshold(tmp_path: Path) -> None:
    from polarix.orchestration.multiday_dry_run_report import DryRunConfig, build_per_day_plan

    data = _seed_mt5_day(tmp_path, "2026-05-18", hours=2.0)
    cfg = DryRunConfig(
        dates=["2026-05-18"],
        data_root=data,
        reports_root=tmp_path / "reports",
        repo_root=REPO_ROOT,
        high_volume_window_hours=6.0,
    )
    day = build_per_day_plan(cfg, "2026-05-18", python_exe="python")
    assert day["high_volume_window_warning"] is None


def test_console_output_includes_per_day_summary(tmp_path: Path) -> None:
    data = _seed_mt5_day(tmp_path, "2026-05-18", hours=4.0)
    _seed_downstream_artifacts(tmp_path / "reports", "2026-05-18", data)
    main = _load_main("phase_2g1_cli_console")
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(
            [
                "--dates",
                "2026-05-18",
                "--data-root",
                str(data),
                "--reports-root",
                str(tmp_path / "reports"),
                "--dry-run",
                "--min-free-disk-gb",
                "0",
                "--min-available-memory-gb",
                "0",
            ]
        )
    assert rc == 0
    parsed = json.loads(buf.getvalue())
    assert "per_day_summary" in parsed
    assert parsed["per_day_summary"], "console must surface per-day reasoning"
    day_summary = parsed["per_day_summary"][0]
    assert day_summary["date"] == "2026-05-18"
    assert "lineage_status" in day_summary
    assert "next_action" in day_summary
    assert "DOWNSTREAM_WITHOUT_RAW_CME" in day_summary["lineage_status"]


def test_download_preflight_never_calls_get_range(tmp_path: Path, monkeypatch) -> None:
    main = _load_main("phase_2g1_cli_preflight_never_calls")
    sentinel = {"get_range_called": False}

    class FakeTimeseries:
        def get_range(self, *a, **k):
            sentinel["get_range_called"] = True
            raise AssertionError("preflight must not invoke timeseries.get_range")

    class FakeMetadata:
        def get_cost(self, *a, **k):
            sentinel["get_range_called"] = True
            raise AssertionError("preflight must not invoke metadata.get_cost")

        def get_billable_size(self, *a, **k):
            sentinel["get_range_called"] = True
            raise AssertionError("preflight must not invoke metadata.get_billable_size")

    class FakeClient:
        metadata = FakeMetadata()
        timeseries = FakeTimeseries()

    cli_mod = sys.modules["phase_2g1_cli_preflight_never_calls"]

    def _patched_cost(*, client_factory=None, api_key_present=None):
        client = (client_factory or (lambda: FakeClient()))()
        if hasattr(client.metadata, "get_cost") and hasattr(client.metadata, "get_billable_size"):
            from polarix.orchestration.databento_cost_guard import (
                ESTIMATE_CAPABILITY_AVAILABLE,
                CostEstimate,
            )

            return CostEstimate(
                status=ESTIMATE_CAPABILITY_AVAILABLE,
                source="capability_probe",
                message="capability available; not called",
            )
        from polarix.orchestration.databento_cost_guard import (
            COST_ESTIMATE_UNAVAILABLE,
            CostEstimate,
        )

        return CostEstimate(status=COST_ESTIMATE_UNAVAILABLE)

    def _patched_phys(**kw):
        from polarix.orchestration.databento_cost_guard import (
            PHYSICAL_LIMIT_SUPPORTED,
            PhysicalLimitCapability,
        )

        return PhysicalLimitCapability(
            status=PHYSICAL_LIMIT_SUPPORTED,
            supported_limit_types=("record_limit",),
            selected_limit_type=None,
            selected_limit_value=None,
            sdk_call_target="timeseries.get_range",
        )

    monkeypatch.setattr(cli_mod, "detect_databento_cost_estimate_capability", _patched_cost)
    monkeypatch.setattr(cli_mod, "detect_databento_physical_limit_capability", _patched_phys)
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(
            [
                "--dates",
                "2026-05-18",
                "--data-root",
                str(tmp_path / "data"),
                "--reports-root",
                str(tmp_path / "reports"),
                "--download-preflight",
                "--download-cme",
                "--allow-databento-download",
                "--acknowledge-cost-risk",
                "--min-free-disk-gb",
                "0",
                "--min-available-memory-gb",
                "0",
            ]
        )
    assert rc in (0, 1)
    assert sentinel["get_range_called"] is False


def test_download_preflight_reports_capability_statuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DATABENTO_API_KEY", raising=False)
    main = _load_main("phase_2g1_cli_preflight_status")
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(
            [
                "--dates",
                "2026-05-18",
                "--data-root",
                str(tmp_path / "data"),
                "--reports-root",
                str(tmp_path / "reports"),
                "--download-preflight",
                "--download-cme",
                "--allow-databento-download",
                "--acknowledge-cost-risk",
                "--min-free-disk-gb",
                "0",
                "--min-available-memory-gb",
                "0",
            ]
        )
    assert rc == 1
    out = json.loads(buf.getvalue())
    assert out["mode"] == "download_preflight"
    assert "cost_estimate_capability_status" in out
    assert "physical_limit_capability_status" in out
    assert out["real_download_would_be_blocked"] is True
    assert "MISSING_API_KEY" in out["blocked_reasons"]


def test_download_preflight_with_fake_api_key_never_hits_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATABENTO_API_KEY", "fake-test-key-not-real")
    _seed_mt5_day(tmp_path, "2026-05-18", hours=4.0)
    main = _load_main("phase_2g1_cli_preflight_fake_key")
    sentinel = {"net_called": False}

    class _FakeTimeseries:
        def get_range(self, *a, **k):
            sentinel["net_called"] = True
            raise AssertionError("preflight must not invoke timeseries.get_range")

    class _FakeMetadata:
        def get_cost(self, *a, **k):
            sentinel["net_called"] = True
            raise AssertionError("preflight must not invoke metadata.get_cost")

        def get_billable_size(self, *a, **k):
            sentinel["net_called"] = True
            raise AssertionError("preflight must not invoke metadata.get_billable_size")

    class _FakeClient:
        metadata = _FakeMetadata()
        timeseries = _FakeTimeseries()

    cli_mod = sys.modules["phase_2g1_cli_preflight_fake_key"]

    def _patched_cost(*, client_factory=None, api_key_present=None):
        from polarix.orchestration.databento_cost_guard import (
            ESTIMATE_CAPABILITY_AVAILABLE,
            CostEstimate,
        )

        return CostEstimate(
            status=ESTIMATE_CAPABILITY_AVAILABLE,
            source="capability_probe",
            message="capability available; not called",
        )

    def _patched_phys(**kw):
        from polarix.orchestration.databento_cost_guard import (
            PHYSICAL_LIMIT_SUPPORTED,
            PhysicalLimitCapability,
        )

        return PhysicalLimitCapability(
            status=PHYSICAL_LIMIT_SUPPORTED,
            supported_limit_types=("record_limit",),
            selected_limit_type=None,
            selected_limit_value=None,
            sdk_call_target="timeseries.get_range",
        )

    monkeypatch.setattr(cli_mod, "detect_databento_cost_estimate_capability", _patched_cost)
    monkeypatch.setattr(cli_mod, "detect_databento_physical_limit_capability", _patched_phys)
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(
            [
                "--dates",
                "2026-05-18",
                "--data-root",
                str(tmp_path / "data"),
                "--reports-root",
                str(tmp_path / "reports"),
                "--download-preflight",
                "--download-cme",
                "--allow-databento-download",
                "--acknowledge-cost-risk",
                "--min-free-disk-gb",
                "0",
                "--min-available-memory-gb",
                "0",
            ]
        )
    assert rc in (0, 1)
    assert sentinel["net_called"] is False
    out = json.loads(buf.getvalue())
    assert out["mode"] == "download_preflight"
    assert "MISSING_API_KEY" not in out.get("blocked_reasons", [])


def test_download_preflight_blocks_without_acknowledge_cost_risk(tmp_path: Path) -> None:
    main = _load_main("phase_2g1_cli_preflight_no_ack")
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(
            [
                "--dates",
                "2026-05-18",
                "--data-root",
                str(tmp_path / "data"),
                "--reports-root",
                str(tmp_path / "reports"),
                "--download-preflight",
                "--download-cme",
                "--allow-databento-download",
                "--min-free-disk-gb",
                "0",
                "--min-available-memory-gb",
                "0",
            ]
        )
    out = json.loads(buf.getvalue())
    assert "MISSING_FLAG_ACKNOWLEDGE_COST_RISK" in out["blocked_reasons"]
    assert rc == 1


PHASE_2G1_FILES = (
    "src/polarix/orchestration/multiday_dry_run_report.py",
    "src/polarix/orchestration/databento_cost_guard.py",
    "scripts/run_multiday_pipeline.py",
)


def _phase_2g1_texts() -> dict[str, str]:
    return {p: (REPO_ROOT / p).read_text(encoding="utf-8") for p in PHASE_2G1_FILES}


def test_phase_2g1_no_trading_functions() -> None:
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
    for path, text in _phase_2g1_texts().items():
        for token in forbidden:
            assert token not in text, f"{path}: forbidden {token!r}"


def test_phase_2g1_preflight_does_not_call_timeseries_in_source() -> None:
    import ast

    cli_text = _phase_2g1_texts()["scripts/run_multiday_pipeline.py"]
    tree = ast.parse(cli_text)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "get_range":
                pytest.fail(
                    f"scripts/run_multiday_pipeline.py:{node.lineno}: must not call any '.get_range(...)' from the CLI"
                )


def test_phase_2g1_planned_stages_use_explicit_reasons() -> None:
    text = _phase_2g1_texts()["src/polarix/orchestration/multiday_dry_run_report.py"]
    assert "BLOCKED_MISSING_CME_RAW" in text
    assert "DOWNLOAD_NOT_REQUESTED" in text
    assert "DOWNLOAD_WOULD_BE_ATTEMPTED_IF_REAL" in text


def test_databento_api_key_is_scrubbed_by_default() -> None:
    import os

    assert os.environ.get("DATABENTO_API_KEY") is None, (
        "DATABENTO_API_KEY leaked into a test that did not opt in via monkeypatch.setenv; check tests/conftest.py _isolate_databento_api_key fixture"
    )


def test_conftest_declares_databento_isolation_fixture() -> None:
    import ast

    conftest = (REPO_ROOT / "tests" / "conftest.py").read_text(encoding="utf-8")
    assert "_isolate_databento_api_key" in conftest, (
        "tests/conftest.py must define the _isolate_databento_api_key autouse fixture; do not remove it"
    )
    assert 'monkeypatch.delenv("DATABENTO_API_KEY"' in conftest, (
        "tests/conftest.py must call monkeypatch.delenv on DATABENTO_API_KEY for every test"
    )
    tree = ast.parse(conftest)
    found = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for dec in node.decorator_list:
            if isinstance(dec, ast.Call) and getattr(dec.func, "attr", None) == "fixture":
                for kw in dec.keywords:
                    if (
                        kw.arg == "autouse"
                        and isinstance(kw.value, ast.Constant)
                        and (kw.value.value is True)
                    ):
                        found = True
    assert found, "tests/conftest.py must define an @pytest.fixture(autouse=True)"
