from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

from polarix import __version__
from polarix.common.clock_health import check_clock_health
from polarix.common.config import LoggerConfig, ensure_runtime_dirs, load_config
from polarix.common.health import write_critical_health, write_health_snapshot, write_manifest
from polarix.ingestion.mt5_readonly import KillSwitchTripped, MT5UnavailableError, ReadOnlyMT5
from polarix.ingestion.parquet_writer import ParquetWriter
from polarix.ingestion.tick_filter import TickFilter
from polarix.normalization.timestamp_semantics import FreshnessGate, TimestampSemantics

_IDLE_BACKOFF_BASE_S = 0.1
_IDLE_BACKOFF_FACTOR = 1.6
_ACTIVE_POLL_INTERVAL_S = 0.025
_HEALTH_SNAPSHOT_EVERY_S = 30
_STALENESS_EVALUATION_EVERY_S = 5
STOP_SIGNAL_FILENAME = "stop.signal"


class GracefulExit(Exception):
    pass


def _stop_signal_path(cfg: LoggerConfig) -> Path:
    return cfg.logs_root / STOP_SIGNAL_FILENAME


def _stop_signal_present(cfg: LoggerConfig) -> bool:
    try:
        return _stop_signal_path(cfg).exists()
    except OSError:
        return False


def _install_signal_handlers() -> None:

    def _handler(signum, _frame):
        raise GracefulExit(f"signal {signum}")

    signal.signal(signal.SIGINT, _handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handler)


def _build_runtime(
    cfg: LoggerConfig, mt5_ro: ReadOnlyMT5
) -> tuple[TickFilter, ParquetWriter, TimestampSemantics]:
    tf = TickFilter(price_scale=cfg.price_scale_default)

    def _pre_flush() -> None:
        mt5_ro.check_kill_switch()
        if _stop_signal_present(cfg):
            raise GracefulExit("stop.signal")

    pw = ParquetWriter(
        raw_dataset_dir=cfg.raw_dataset_dir,
        flush_max_rows=cfg.flush_max_rows,
        flush_max_seconds=cfg.flush_max_seconds,
        compression=cfg.parquet_compression,
        before_flush_hook=_pre_flush,
    )
    ts = TimestampSemantics(
        plausible_offsets_minutes=cfg.plausible_broker_utc_offsets_minutes,
        gate=FreshnessGate(
            max_future_jitter_ms=cfg.max_future_jitter_ms,
            max_live_tick_age_ms=cfg.max_live_tick_age_ms,
        ),
        required_fresh_ticks=cfg.required_fresh_ticks_for_offset,
    )
    return (tf, pw, ts)


def _emit_health(
    cfg: LoggerConfig,
    mt5_ro: ReadOnlyMT5,
    tf: TickFilter,
    pw: ParquetWriter,
    ts: TimestampSemantics,
    clock_status: dict,
) -> None:
    terminal = mt5_ro.terminal_info()
    account = mt5_ro.account_info()
    payload = {
        "logger_version": __version__,
        "environment": cfg.environment,
        "broker": {
            "terminal_company": terminal.company,
            "terminal_name": terminal.name,
            "terminal_build": terminal.build,
            "trade_allowed": terminal.trade_allowed,
            "dlls_allowed": terminal.dlls_allowed,
        },
        "account": {
            "login_hash": account.login_hash,
            "server": account.server,
            "company": account.company,
            "currency": account.currency,
            "leverage": account.leverage,
            "trade_allowed": account.trade_allowed,
            "trade_expert": account.trade_expert,
        },
        "clock_health": clock_status,
        "timestamp_semantics": ts.manifest(),
        "tick_filter_metrics": tf.metrics(),
        "writer_metrics": {
            "files_written": pw.metrics.files_written,
            "rows_written": pw.metrics.rows_written,
            "bytes_written": pw.metrics.bytes_written,
        },
        "symbols": list(cfg.symbols),
        "no_trading_functions_allowed": cfg.no_trading_functions_allowed,
        "fail_if_terminal_trade_allowed": cfg.fail_if_terminal_trade_allowed,
    }
    write_health_snapshot(cfg.reports_root, payload)


def _emit_critical_health(
    cfg: LoggerConfig,
    mt5_ro: ReadOnlyMT5,
    tf: TickFilter,
    pw: ParquetWriter,
    ts: TimestampSemantics,
    clock_status: dict,
    reason: str,
) -> None:
    terminal_dict: dict | None = None
    account_dict: dict | None = None
    try:
        terminal = mt5_ro.terminal_info()
        terminal_dict = {
            "terminal_company": terminal.company,
            "terminal_name": terminal.name,
            "terminal_build": terminal.build,
            "trade_allowed": terminal.trade_allowed,
            "dlls_allowed": terminal.dlls_allowed,
        }
    except Exception as exc:
        terminal_dict = {"error": f"{type(exc).__name__}: {exc}"}
    try:
        account = mt5_ro.account_info()
        account_dict = {
            "login_hash": account.login_hash,
            "server": account.server,
            "company": account.company,
            "currency": account.currency,
            "leverage": account.leverage,
            "trade_allowed": account.trade_allowed,
            "trade_expert": account.trade_expert,
        }
    except Exception as exc:
        account_dict = {"error": f"{type(exc).__name__}: {exc}"}
    payload = {
        "severity": "CRITICAL",
        "event": "KILL_SWITCH_TRIPPED",
        "reason": reason,
        "logger_version": __version__,
        "environment": cfg.environment,
        "broker": terminal_dict,
        "account": account_dict,
        "clock_health": clock_status,
        "timestamp_semantics": ts.manifest(),
        "tick_filter_metrics": tf.metrics(),
        "writer_metrics": {
            "files_written": pw.metrics.files_written,
            "rows_written": pw.metrics.rows_written,
            "bytes_written": pw.metrics.bytes_written,
            "flush_failures": pw.metrics.flush_failures,
            "last_failure_reason": pw.metrics.last_failure_reason,
        },
    }
    write_critical_health(cfg.reports_root, payload)


def _emit_manifest(cfg: LoggerConfig, ts: TimestampSemantics) -> None:
    payload = {
        "dataset": cfg.dataset_name,
        "data_root": cfg.data_root,
        "raw_dataset_dir": cfg.raw_dataset_dir,
        "compacted_dataset_dir": cfg.compacted_dataset_dir,
        "symbols": list(cfg.symbols),
        "compression": cfg.parquet_compression,
        "timestamp_semantics": ts.manifest(),
    }
    write_manifest(cfg.reports_root, payload)


def run_logger(cfg: LoggerConfig, run_once: bool = False) -> int:
    ensure_runtime_dirs(cfg)
    _install_signal_handlers()
    mt5_ro = ReadOnlyMT5(fail_if_terminal_trade_allowed=cfg.fail_if_terminal_trade_allowed)
    try:
        mt5_ro.initialize()
    except MT5UnavailableError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    tf, pw, ts = _build_runtime(cfg, mt5_ro)
    clock = check_clock_health(threshold_ms=cfg.max_host_clock_offset_ms)
    clock_status = {
        "status": clock.status,
        "offset_ms": clock.offset_ms,
        "threshold_ms": clock.threshold_ms,
        "source": clock.source,
        "error": clock.error,
    }
    try:
        mt5_ro.check_kill_switch()
        select = mt5_ro.select_symbols(cfg.symbols)
        missing = [s for s, ok in select.items() if not ok]
        if missing:
            print(f"WARNING: failed to select symbols: {missing}", file=sys.stderr)
        last_seen_time_msc: dict[str, int] = {}
        idle_backoff_s = _IDLE_BACKOFF_BASE_S
        last_health_ts = 0.0
        last_staleness_eval = 0.0
        while True:
            mt5_ro.check_kill_switch()
            if _stop_signal_present(cfg):
                raise GracefulExit("stop.signal")
            any_progress = False
            session_lost = False
            for symbol in cfg.symbols:
                try:
                    tick = mt5_ro.symbol_info_tick(symbol)
                except MT5UnavailableError:
                    session_lost = True
                    break
                if tick is None:
                    continue
                if last_seen_time_msc.get(symbol) == tick.time_msc:
                    continue
                last_seen_time_msc[symbol] = tick.time_msc
                recv_ms = mt5_ro.now_recv_ms()
                mono = mt5_ro.now_monotonic_ns()
                ts.observe(time_msc_raw=tick.time_msc, recv_time_utc_ms=recv_ms)
                try:
                    sinfo = mt5_ro.symbol_info(symbol)
                except MT5UnavailableError:
                    session_lost = True
                    break
                spread_points = sinfo.spread if sinfo is not None else 0
                event = tf.consider(
                    symbol=symbol,
                    time_msc_raw=tick.time_msc,
                    recv_time_utc_ms=recv_ms,
                    monotonic_ns=mono,
                    bid=tick.bid,
                    ask=tick.ask,
                    last=tick.last,
                    volume=tick.volume,
                    flags=tick.flags,
                    spread_points=spread_points,
                )
                if event is not None:
                    pw.add(event)
                    any_progress = True
            if session_lost:
                try:
                    mt5_ro.shutdown()
                    mt5_ro.initialize()
                    mt5_ro.check_kill_switch()
                    ts.reset_for_reconnect()
                    mt5_ro.select_symbols(cfg.symbols)
                except (MT5UnavailableError, KillSwitchTripped):
                    raise
            now = time.monotonic()
            if now - last_staleness_eval >= _STALENESS_EVALUATION_EVERY_S:
                ts.evaluate_staleness(now_recv_ms=mt5_ro.now_recv_ms())
                last_staleness_eval = now
            if now - last_health_ts >= _HEALTH_SNAPSHOT_EVERY_S:
                _emit_health(cfg, mt5_ro, tf, pw, ts, clock_status)
                _emit_manifest(cfg, ts)
                last_health_ts = now
            if run_once:
                break
            if any_progress:
                idle_backoff_s = _IDLE_BACKOFF_BASE_S
                time.sleep(_ACTIVE_POLL_INTERVAL_S)
            else:
                time.sleep(idle_backoff_s)
                idle_backoff_s = min(
                    cfg.closed_market_idle_backoff_seconds_max,
                    idle_backoff_s * _IDLE_BACKOFF_FACTOR,
                )
        return 0
    except KillSwitchTripped as exc:
        try:
            _emit_critical_health(cfg, mt5_ro, tf, pw, ts, clock_status, reason=str(exc))
        except Exception:
            pass
        print(f"KILL SWITCH: {exc}", file=sys.stderr)
        return 1
    except GracefulExit as exc:
        print(f"exit: {exc}", file=sys.stderr)
        return 0
    finally:
        try:
            pw.close()
        finally:
            try:
                _emit_health(cfg, mt5_ro, tf, pw, ts, clock_status)
                _emit_manifest(cfg, ts)
            except Exception:
                pass
            mt5_ro.shutdown()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="polarix-logger")
    parser.add_argument("--config", default=None, help="path to logger.config.json")
    parser.add_argument(
        "--once",
        action="store_true",
        help="run a single poll iteration and exit (useful for smoke checks)",
    )
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    return run_logger(cfg, run_once=args.once)


if __name__ == "__main__":
    raise SystemExit(main())
