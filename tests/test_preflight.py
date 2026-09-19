from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from polarix.common.clock_health import STATUS_COARSE_OK, STATUS_UNSAFE, ClockHealth
from polarix.common.config import load_config
from polarix.orchestration import preflight

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "logger.example.json"


@pytest.fixture()
def cfg(tmp_path: Path):
    base = load_config(CONFIG_PATH)
    redirected = type(base)(
        environment=base.environment,
        broker_profile=base.broker_profile,
        current_observed_broker=base.current_observed_broker,
        data_root=tmp_path / "data",
        reports_root=tmp_path / "reports",
        logs_root=tmp_path / "logs",
        symbols=base.symbols,
        plausible_broker_utc_offsets_minutes=base.plausible_broker_utc_offsets_minutes,
        max_host_clock_offset_ms=base.max_host_clock_offset_ms,
        max_future_jitter_ms=base.max_future_jitter_ms,
        max_live_tick_age_ms=base.max_live_tick_age_ms,
        required_fresh_ticks_for_offset=base.required_fresh_ticks_for_offset,
        flush_max_rows=base.flush_max_rows,
        flush_max_seconds=base.flush_max_seconds,
        closed_market_idle_backoff_seconds_max=base.closed_market_idle_backoff_seconds_max,
        parquet_compression=base.parquet_compression,
        price_scale_default=base.price_scale_default,
        no_trading_functions_allowed=base.no_trading_functions_allowed,
        fail_if_terminal_trade_allowed=base.fail_if_terminal_trade_allowed,
        max_failed_attempts_before_human_review=base.max_failed_attempts_before_human_review,
        raw=base.raw,
    )
    redirected.data_root.mkdir(parents=True, exist_ok=True)
    redirected.reports_root.mkdir(parents=True, exist_ok=True)
    redirected.logs_root.mkdir(parents=True, exist_ok=True)
    redirected.raw_dataset_dir.mkdir(parents=True, exist_ok=True)
    return redirected


class _FakeMT5Module:
    def __init__(
        self,
        *,
        terminal_trade_allowed: bool = False,
        account_trade_allowed: bool = True,
        terminal_connected: bool = True,
        initialize_ok: bool = True,
        symbol_select_ok: bool = True,
        symbol_select_overrides: dict[str, bool] | None = None,
    ) -> None:
        self.terminal_trade_allowed = terminal_trade_allowed
        self.account_trade_allowed = account_trade_allowed
        self.terminal_connected = terminal_connected
        self.initialize_ok = initialize_ok
        self.symbol_select_ok = symbol_select_ok
        self.symbol_select_overrides = symbol_select_overrides or {}
        self.shutdown_called = False

    def initialize(self) -> bool:
        return self.initialize_ok

    def shutdown(self) -> None:
        self.shutdown_called = True

    def last_error(self):
        return (0, "fake_ok")

    def terminal_info(self):
        return SimpleNamespace(
            company="FundedNext",
            name="MetaTrader 5",
            build=4200,
            connected=self.terminal_connected,
            trade_allowed=self.terminal_trade_allowed,
            dlls_allowed=False,
        )

    def account_info(self):
        return SimpleNamespace(
            login=999000,
            server="FundedNext-Server",
            company="FundedNext",
            currency="USD",
            leverage=100,
            trade_allowed=self.account_trade_allowed,
            trade_expert=False,
        )

    def symbol_select(self, name: str, _enable: bool):
        if name in self.symbol_select_overrides:
            return self.symbol_select_overrides[name]
        return self.symbol_select_ok


def test_preflight_mt5_surface_fails_closed_when_terminal_trade_allowed():
    fake = _FakeMT5Module(terminal_trade_allowed=True, account_trade_allowed=False)
    res = preflight.check_mt5_surface(symbols=("SPX500", "NDX100"), mt5_module=fake)
    assert res.ok is False
    assert res.error is not None
    assert "trade_allowed" in res.error.lower()
    assert fake.shutdown_called is True


def test_preflight_mt5_surface_passes_when_only_account_trade_allowed():
    fake = _FakeMT5Module(terminal_trade_allowed=False, account_trade_allowed=True)
    res = preflight.check_mt5_surface(symbols=("SPX500", "NDX100"), mt5_module=fake)
    assert res.ok is True
    notes = res.detail.get("notes") or []
    assert any(("account_info.trade_allowed=True" in n for n in notes))


def test_preflight_disk_below_threshold_fails(tmp_path):
    res = preflight.check_disk(path=str(tmp_path), min_free_gb=10000000.0)
    assert res.ok is False
    assert res.error and "disk free" in res.error


def test_preflight_disk_above_threshold_passes(tmp_path):
    res = preflight.check_disk(path=str(tmp_path), min_free_gb=0.0)
    assert res.ok is True


def test_preflight_clock_unsafe_fails(monkeypatch):
    from polarix.orchestration import preflight as pf

    def fake_check(threshold_ms: int, **_kwargs):
        return ClockHealth(
            status=STATUS_UNSAFE, offset_ms=999.0, threshold_ms=threshold_ms, source="fake"
        )

    monkeypatch.setattr(pf, "check_clock_health", fake_check)
    res = pf.check_clock(threshold_ms=50)
    assert res.ok is False
    assert "not safe" in (res.error or "").lower()


def test_preflight_clock_ok_passes(monkeypatch):
    from polarix.orchestration import preflight as pf

    def fake_check(threshold_ms: int, **_kwargs):
        return ClockHealth(
            status=STATUS_COARSE_OK, offset_ms=1.0, threshold_ms=threshold_ms, source="fake"
        )

    monkeypatch.setattr(pf, "check_clock_health", fake_check)
    res = pf.check_clock(threshold_ms=50)
    assert res.ok is True


def test_preflight_config_missing_fails(tmp_path):
    missing = tmp_path / "no_such.json"
    res = preflight.check_config_present(missing)
    assert res.ok is False
    assert "not found" in (res.error or "")


def test_preflight_python_import_logger_passes():
    res = preflight.check_python_can_import_logger()
    assert res.ok is True


def test_preflight_no_trading_invariant_passes_on_current_source():
    res = preflight.check_no_trading_invariant(REPO_ROOT)
    assert res.ok is True, res


def test_preflight_directories_writable_in_tmp(tmp_path):
    res = preflight.check_directories_writable([tmp_path / "a", tmp_path / "b"])
    assert res.ok is True


def test_run_preflight_overall_ok_with_fake_mt5(cfg, monkeypatch):
    fake = _FakeMT5Module()
    from polarix.orchestration import preflight as pf

    monkeypatch.setattr(
        pf,
        "check_clock_health",
        lambda threshold_ms, **kw: ClockHealth(
            status=STATUS_COARSE_OK, offset_ms=1.0, threshold_ms=threshold_ms, source="fake"
        ),
    )
    report = pf.run_preflight(
        cfg=cfg,
        config_path=CONFIG_PATH,
        repo_root=REPO_ROOT,
        min_free_disk_gb=0.0,
        min_available_memory_gb=0.0,
        mt5_module=fake,
        disk_path=str(cfg.data_root),
    )
    assert report.overall_ok is True
    assert any((c.name == "mt5_surface" and c.ok for c in report.checks))


def test_run_preflight_fails_when_terminal_trade_allowed(cfg, monkeypatch):
    fake = _FakeMT5Module(terminal_trade_allowed=True)
    from polarix.orchestration import preflight as pf

    monkeypatch.setattr(
        pf,
        "check_clock_health",
        lambda threshold_ms, **kw: ClockHealth(
            status=STATUS_COARSE_OK, offset_ms=1.0, threshold_ms=threshold_ms, source="fake"
        ),
    )
    report = pf.run_preflight(
        cfg=cfg,
        config_path=CONFIG_PATH,
        repo_root=REPO_ROOT,
        min_free_disk_gb=0.0,
        min_available_memory_gb=0.0,
        mt5_module=fake,
        disk_path=str(cfg.data_root),
    )
    assert report.overall_ok is False
    mt5_check = [c for c in report.checks if c.name == "mt5_surface"][0]
    assert mt5_check.ok is False


def test_write_preflight_failed_report_creates_file(cfg, monkeypatch):
    fake = _FakeMT5Module(terminal_trade_allowed=True)
    from polarix.orchestration import preflight as pf

    monkeypatch.setattr(
        pf,
        "check_clock_health",
        lambda threshold_ms, **kw: ClockHealth(
            status=STATUS_COARSE_OK, offset_ms=1.0, threshold_ms=threshold_ms, source="fake"
        ),
    )
    report = pf.run_preflight(
        cfg=cfg,
        config_path=CONFIG_PATH,
        repo_root=REPO_ROOT,
        min_free_disk_gb=0.0,
        min_available_memory_gb=0.0,
        mt5_module=fake,
        disk_path=str(cfg.data_root),
    )
    assert report.overall_ok is False
    out = pf.write_preflight_failed_report(cfg.reports_root, report, "TESTSTAMP")
    assert out.exists()
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["overall_ok"] is False
