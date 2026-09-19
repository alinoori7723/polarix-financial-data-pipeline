from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path("config/logger.config.json")


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class LoggerConfig:
    environment: str
    broker_profile: str
    current_observed_broker: str
    data_root: Path
    reports_root: Path
    logs_root: Path
    symbols: tuple[str, ...]
    plausible_broker_utc_offsets_minutes: tuple[int, ...]
    max_host_clock_offset_ms: int
    max_future_jitter_ms: int
    max_live_tick_age_ms: int
    required_fresh_ticks_for_offset: int
    flush_max_rows: int
    flush_max_seconds: int
    closed_market_idle_backoff_seconds_max: int
    parquet_compression: str
    price_scale_default: int
    no_trading_functions_allowed: bool
    fail_if_terminal_trade_allowed: bool
    max_failed_attempts_before_human_review: int
    raw: dict[str, Any] = field(repr=False, default_factory=dict)

    @property
    def dataset_name(self) -> str:
        return "mt5_ticks"

    @property
    def raw_dataset_dir(self) -> Path:
        return self.data_root / "raw" / self.dataset_name

    @property
    def compacted_dataset_dir(self) -> Path:
        return self.data_root / "compacted" / self.dataset_name


def _require(d: dict[str, Any], key: str, expected_type: type | tuple[type, ...]) -> Any:
    if key not in d:
        raise ConfigError(f"missing config key: {key}")
    value = d[key]
    if not isinstance(value, expected_type):
        raise ConfigError(
            f"config key {key} has type {type(value).__name__}, expected {expected_type}"
        )
    return value


def load_config(path: Path | str | None = None) -> LoggerConfig:
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise ConfigError(f"config file not found: {cfg_path}")
    raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError("config root must be a JSON object")
    symbols_raw = _require(raw, "symbols", list)
    if not symbols_raw or not all((isinstance(s, str) and s for s in symbols_raw)):
        raise ConfigError("symbols must be a non-empty list of strings")
    offsets_raw = _require(raw, "plausible_broker_utc_offsets_minutes", list)
    if not all((isinstance(o, int) for o in offsets_raw)):
        raise ConfigError("plausible_broker_utc_offsets_minutes must be all ints")
    if not all((-720 <= o <= 840 for o in offsets_raw)):
        raise ConfigError("offsets out of plausible range [-720, +840] minutes")
    parquet_compression = _require(raw, "parquet_compression", str).lower()
    if parquet_compression != "zstd":
        raise ConfigError("parquet_compression must be 'zstd'")
    fail_if_terminal_trade_allowed = _require(raw, "fail_if_terminal_trade_allowed", bool)
    if not fail_if_terminal_trade_allowed:
        raise ConfigError("fail_if_terminal_trade_allowed must be true")
    no_trading_functions_allowed = _require(raw, "no_trading_functions_allowed", bool)
    if not no_trading_functions_allowed:
        raise ConfigError("no_trading_functions_allowed must be true")
    cfg = LoggerConfig(
        environment=_require(raw, "environment", str),
        broker_profile=_require(raw, "broker_profile", str),
        current_observed_broker=_require(raw, "current_observed_broker", str),
        data_root=Path(_require(raw, "data_root", str)),
        reports_root=Path(_require(raw, "reports_root", str)),
        logs_root=Path(_require(raw, "logs_root", str)),
        symbols=tuple(symbols_raw),
        plausible_broker_utc_offsets_minutes=tuple(offsets_raw),
        max_host_clock_offset_ms=int(_require(raw, "max_host_clock_offset_ms", int)),
        max_future_jitter_ms=int(_require(raw, "max_future_jitter_ms", int)),
        max_live_tick_age_ms=int(_require(raw, "max_live_tick_age_ms", int)),
        required_fresh_ticks_for_offset=int(_require(raw, "required_fresh_ticks_for_offset", int)),
        flush_max_rows=int(_require(raw, "flush_max_rows", int)),
        flush_max_seconds=int(_require(raw, "flush_max_seconds", int)),
        closed_market_idle_backoff_seconds_max=int(
            _require(raw, "closed_market_idle_backoff_seconds_max", int)
        ),
        parquet_compression=parquet_compression,
        price_scale_default=int(_require(raw, "price_scale_default", int)),
        no_trading_functions_allowed=no_trading_functions_allowed,
        fail_if_terminal_trade_allowed=fail_if_terminal_trade_allowed,
        max_failed_attempts_before_human_review=int(
            _require(raw, "max_failed_attempts_before_human_review", int)
        ),
        raw=raw,
    )
    if cfg.environment != "sandbox":
        raise ConfigError(f"refusing to run outside sandbox environment, got {cfg.environment!r}")
    return cfg


def ensure_runtime_dirs(cfg: LoggerConfig) -> None:
    for d in (
        cfg.data_root,
        cfg.reports_root,
        cfg.logs_root,
        cfg.raw_dataset_dir,
        cfg.compacted_dataset_dir,
    ):
        d.mkdir(parents=True, exist_ok=True)
