from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import polarix.ingestion.mt5_readonly as ro_mod
from polarix.common.config import load_config
from polarix.ingestion.main import run_logger
from polarix.ingestion.mt5_readonly import KillSwitchTripped, ReadOnlyMT5
from polarix.ingestion.parquet_writer import ParquetWriter
from polarix.ingestion.tick_filter import FilteredEvent


class FlippingFakeMT5:
    def __init__(self, flip_after_calls: int) -> None:
        self.flip_after_calls = flip_after_calls
        self.terminal_calls = 0
        self.tick_call = 0
        self.account_trade_allowed = True

    def initialize(self) -> bool:
        return True

    def shutdown(self) -> None:
        return None

    def last_error(self):
        return (0, "ok")

    def terminal_info(self):
        self.terminal_calls += 1
        trade_allowed = self.terminal_calls > self.flip_after_calls
        return SimpleNamespace(
            name="FakeTerminal",
            company="FakeBroker",
            path="C:/fake",
            build=4242,
            trade_allowed=trade_allowed,
            dlls_allowed=False,
        )

    def account_info(self):
        return SimpleNamespace(
            login=999999,
            server="Fake-Server",
            company="FakeBroker",
            currency="USD",
            leverage=100,
            trade_allowed=self.account_trade_allowed,
            trade_expert=False,
        )

    def symbol_select(self, _name, _enable):
        return True

    def symbol_info(self, _name):
        return SimpleNamespace(
            name=_name,
            description="",
            digits=1,
            point=0.1,
            trade_mode=0,
            spread=2,
            spread_float=True,
        )

    def symbol_info_tick(self, symbol):
        self.tick_call += 1
        return SimpleNamespace(
            time=1,
            time_msc=self.tick_call * 1000,
            bid=5000.0,
            ask=5000.1,
            last=5000.0,
            volume=1,
            flags=2,
        )


def _redirected_cfg(tmp_path: Path):
    cfg = load_config(Path(__file__).resolve().parents[1] / "config" / "logger.example.json")
    return type(cfg)(
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


def test_runtime_kill_after_logger_started(monkeypatch, tmp_path: Path):
    fake = FlippingFakeMT5(flip_after_calls=2)
    monkeypatch.setattr(ro_mod, "mt5", fake)
    monkeypatch.setattr(ro_mod, "HAS_MT5", True)
    cfg = _redirected_cfg(tmp_path)
    rc = run_logger(cfg, run_once=False)
    assert rc == 1
    assert fake.terminal_calls >= 2
    crit_files = list((tmp_path / "reports").glob("logger_critical-*.json"))
    assert crit_files, "CRITICAL health record was not emitted"
    import json

    payload = json.loads(crit_files[0].read_text(encoding="utf-8"))
    assert payload["severity"] == "CRITICAL"
    assert payload["event"] == "KILL_SWITCH_TRIPPED"
    assert "reason" in payload


def test_kill_switch_checked_before_parquet_flush(tmp_path: Path):
    calls = {"hook": 0}

    def hook():
        calls["hook"] += 1
        raise KillSwitchTripped("simulated mid-flush kill")

    pw = ParquetWriter(
        raw_dataset_dir=tmp_path, flush_max_rows=2, flush_max_seconds=60, before_flush_hook=hook
    )
    ev = FilteredEvent(
        symbol="SPX500",
        time_msc_raw=1700000000000,
        recv_time_utc_ms=1700000000000,
        monotonic_ns=1700000000000000000,
        bid=5000.0,
        ask=5000.1,
        last=5000.0,
        bid_scaled=500000,
        ask_scaled=500010,
        last_scaled=500000,
        volume=1,
        flags=2,
        spread_points=1,
        suppressed_count=0,
        first_suppressed_time_ms=None,
        last_suppressed_time_ms=None,
        suppressed_reason=None,
    )
    pw.add(ev)
    with pytest.raises(KillSwitchTripped):
        pw.add(ev)
    assert calls["hook"] >= 1
    assert list(tmp_path.rglob("part-*.parquet")) == []
    assert len(pw._buffer) == 2
    with pytest.raises(KillSwitchTripped):
        pw.flush()
    assert calls["hook"] >= 2
    assert list(tmp_path.rglob("part-*.parquet")) == []
    assert len(pw._buffer) == 2


def test_account_only_trade_allowed_does_not_kill(monkeypatch, tmp_path: Path):
    fake = FlippingFakeMT5(flip_after_calls=10**9)
    fake.account_trade_allowed = True
    monkeypatch.setattr(ro_mod, "mt5", fake)
    monkeypatch.setattr(ro_mod, "HAS_MT5", True)
    cfg = _redirected_cfg(tmp_path)
    rc = run_logger(cfg, run_once=True)
    assert rc == 0
    assert not list((tmp_path / "reports").glob("logger_critical-*.json"))


def test_terminal_trade_allowed_kills_at_startup(monkeypatch, tmp_path: Path):
    fake = FlippingFakeMT5(flip_after_calls=0)
    monkeypatch.setattr(ro_mod, "mt5", fake)
    monkeypatch.setattr(ro_mod, "HAS_MT5", True)
    cfg = _redirected_cfg(tmp_path)
    rc = run_logger(cfg, run_once=True)
    assert rc == 1
    assert list((tmp_path / "reports").glob("logger_critical-*.json"))


def test_check_kill_switch_does_not_raise_when_account_only(monkeypatch):
    fake = FlippingFakeMT5(flip_after_calls=10**9)
    fake.account_trade_allowed = True
    monkeypatch.setattr(ro_mod, "mt5", fake)
    monkeypatch.setattr(ro_mod, "HAS_MT5", True)
    ro = ReadOnlyMT5(fail_if_terminal_trade_allowed=True)
    ro.initialize()
    for _ in range(5):
        ro.check_kill_switch()
    assert ro.terminal_info().trade_allowed is False
    assert ro.account_info().trade_allowed is True
