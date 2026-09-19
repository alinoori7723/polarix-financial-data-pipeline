from __future__ import annotations

import json
from pathlib import Path

import pytest

from polarix.common.config import ConfigError, load_config


def _good_config() -> dict:
    return {
        "environment": "sandbox",
        "broker_profile": "dynamic",
        "current_observed_broker": "FundedNext",
        "data_root": ".polarix/data",
        "reports_root": ".polarix/reports",
        "logs_root": ".polarix/logs",
        "symbols": ["SPX500", "NDX100"],
        "plausible_broker_utc_offsets_minutes": [-300, -240, 0, 60, 120, 180],
        "max_host_clock_offset_ms": 50,
        "max_future_jitter_ms": 250,
        "max_live_tick_age_ms": 5000,
        "required_fresh_ticks_for_offset": 20,
        "flush_max_rows": 5000,
        "flush_max_seconds": 60,
        "closed_market_idle_backoff_seconds_max": 30,
        "parquet_compression": "zstd",
        "price_scale_default": 100,
        "no_trading_functions_allowed": True,
        "fail_if_terminal_trade_allowed": True,
        "max_failed_attempts_before_human_review": 3,
    }


def _write(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "logger.config.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def test_loads_example_config_file():
    cfg = load_config((Path(__file__).resolve().parents[1] / "config" / "logger.example.json"))
    assert cfg.environment == "sandbox"
    assert cfg.parquet_compression == "zstd"
    assert cfg.fail_if_terminal_trade_allowed is True
    assert cfg.no_trading_functions_allowed is True
    assert "SPX500" in cfg.symbols and "NDX100" in cfg.symbols
    assert 0 in cfg.plausible_broker_utc_offsets_minutes


def test_rejects_non_sandbox(tmp_path: Path):
    bad = _good_config()
    bad["environment"] = "production"
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, bad))


def test_rejects_non_zstd(tmp_path: Path):
    bad = _good_config()
    bad["parquet_compression"] = "snappy"
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, bad))


def test_rejects_disabled_kill_switch(tmp_path: Path):
    bad = _good_config()
    bad["fail_if_terminal_trade_allowed"] = False
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, bad))


def test_rejects_trading_functions_allowed(tmp_path: Path):
    bad = _good_config()
    bad["no_trading_functions_allowed"] = False
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, bad))


def test_rejects_offsets_out_of_range(tmp_path: Path):
    bad = _good_config()
    bad["plausible_broker_utc_offsets_minutes"] = [-2000, 0]
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, bad))
