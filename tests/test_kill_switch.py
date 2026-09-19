from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import polarix.ingestion.mt5_readonly as ro_mod
from polarix.ingestion.mt5_readonly import KillSwitchTripped, ReadOnlyMT5


class _FakeMT5:
    def __init__(self, trade_allowed_terminal: bool, trade_allowed_account: bool = True):
        self._trade_allowed_terminal = trade_allowed_terminal
        self._trade_allowed_account = trade_allowed_account

    def initialize(self) -> bool:
        return True

    def shutdown(self) -> None:
        return None

    def last_error(self):
        return (0, "ok")

    def terminal_info(self):
        return SimpleNamespace(
            name="FakeTerminal",
            company="FakeBroker",
            path="C:/fake",
            build=4200,
            trade_allowed=self._trade_allowed_terminal,
            dlls_allowed=False,
        )

    def account_info(self):
        return SimpleNamespace(
            login=123456,
            server="Fake-Server",
            company="FakeBroker",
            currency="USD",
            leverage=100,
            trade_allowed=self._trade_allowed_account,
            trade_expert=False,
        )

    def symbol_select(self, _name, _enable):
        return True

    def symbol_info(self, _name):
        return None

    def symbol_info_tick(self, _name):
        return None


def _install_fake(monkeypatch, *, terminal_trade_allowed: bool, account_trade_allowed: bool = True):
    fake = _FakeMT5(
        trade_allowed_terminal=terminal_trade_allowed, trade_allowed_account=account_trade_allowed
    )
    monkeypatch.setattr(ro_mod, "mt5", fake)
    monkeypatch.setattr(ro_mod, "HAS_MT5", True)
    return fake


def test_kill_switch_trips_when_terminal_trade_allowed(monkeypatch):
    _install_fake(monkeypatch, terminal_trade_allowed=True)
    ro = ReadOnlyMT5(fail_if_terminal_trade_allowed=True)
    ro.initialize()
    with pytest.raises(KillSwitchTripped):
        ro.check_kill_switch()


def test_kill_switch_does_not_trip_when_only_account_trade_allowed(monkeypatch):
    _install_fake(monkeypatch, terminal_trade_allowed=False, account_trade_allowed=True)
    ro = ReadOnlyMT5(fail_if_terminal_trade_allowed=True)
    ro.initialize()
    ro.check_kill_switch()
    assert ro.account_info().trade_allowed is True
    assert ro.terminal_info().trade_allowed is False


def test_main_returns_nonzero_when_kill_switch_trips(monkeypatch, tmp_path):
    _install_fake(monkeypatch, terminal_trade_allowed=True)
    from polarix.common.config import load_config

    cfg = load_config(Path(__file__).resolve().parents[1] / "config" / "logger.example.json")
    redirected = type(cfg)(
        environment=cfg.environment,
        broker_profile=cfg.broker_profile,
        current_observed_broker=cfg.current_observed_broker,
        data_root=tmp_path / "data",
        reports_root=tmp_path / "reports",
        logs_root=tmp_path / "logs",
        symbols=cfg.symbols,
        plausible_broker_utc_offsets_minutes=cfg.plausible_broker_utc_offsets_minutes,
        max_host_clock_offset_ms=cfg.max_host_clock_offset_ms,
        max_future_jitter_ms=cfg.max_future_jitter_ms,
        max_live_tick_age_ms=cfg.max_live_tick_age_ms,
        required_fresh_ticks_for_offset=cfg.required_fresh_ticks_for_offset,
        flush_max_rows=cfg.flush_max_rows,
        flush_max_seconds=cfg.flush_max_seconds,
        closed_market_idle_backoff_seconds_max=cfg.closed_market_idle_backoff_seconds_max,
        parquet_compression=cfg.parquet_compression,
        price_scale_default=cfg.price_scale_default,
        no_trading_functions_allowed=cfg.no_trading_functions_allowed,
        fail_if_terminal_trade_allowed=cfg.fail_if_terminal_trade_allowed,
        max_failed_attempts_before_human_review=cfg.max_failed_attempts_before_human_review,
        raw=cfg.raw,
    )
    from polarix.ingestion.main import run_logger

    rc = run_logger(redirected, run_once=True)
    assert rc == 1
