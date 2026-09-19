from __future__ import annotations

import datetime as _dt
import glob
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from polarix.normalization import silver_paths
from polarix.normalization.normalization import (
    DEFAULT_JOIN_SAFE_THRESHOLD_MS,
    LATENCY_OUTLIER_HIGH_MS,
    LATENCY_OUTLIER_LOW_MS,
)
from polarix.orchestration.run_metadata import RunMetadata

REQUIRED_SYMBOLS = ("SPX500", "NDX100")
DEFAULT_JOIN_SAFE_LOW_PCT = 50.0
DEFAULT_LATENCY_OUTLIER_HIGH_PCT = 5.0
DEFAULT_NDX_SPREAD_SPIKE_PRICE = 50.0


@dataclass
class QualityThresholds:
    join_safe_low_pct: float = DEFAULT_JOIN_SAFE_LOW_PCT
    latency_outlier_high_pct: float = DEFAULT_LATENCY_OUTLIER_HIGH_PCT
    ndx_spread_spike_price: float = DEFAULT_NDX_SPREAD_SPIKE_PRICE


@dataclass
class QualityConfig:
    date: str
    raw_root: Path
    silver_root: Path
    reports_root: Path
    metadata: RunMetadata
    join_safe_threshold_ms: int = DEFAULT_JOIN_SAFE_THRESHOLD_MS
    thresholds: QualityThresholds = field(default_factory=QualityThresholds)
    required_symbols: tuple[str, ...] = REQUIRED_SYMBOLS
    run_id: Optional[str] = None

    def __post_init__(self) -> None:
        self.raw_root = Path(self.raw_root).resolve()
        self.silver_root = Path(self.silver_root).resolve()
        self.reports_root = Path(self.reports_root).resolve()


def _bronze_files(raw_root: Path, symbol: str, date: str) -> list[Path]:
    pattern = os.path.join(
        str(raw_root), f"symbol={symbol}", f"date={date}", "hour=*", "part-*.parquet"
    )
    return sorted((Path(p) for p in glob.glob(pattern)))


def _silver_files(silver_root: Path, symbol: str, date: str) -> list[Path]:
    pattern = os.path.join(str(silver_root), f"symbol={symbol}", f"date={date}", "part-*.parquet")
    return sorted((Path(p) for p in glob.glob(pattern)))


def _percentile(values: pa.ChunkedArray | pa.Array, pct: float) -> Optional[float]:
    if len(values) == 0:
        return None
    arr = values.to_numpy(zero_copy_only=False)
    if len(arr) == 0:
        return None
    arr = sorted(
        (float(v) for v in arr if v is not None and (not (isinstance(v, float) and math.isnan(v))))
    )
    if not arr:
        return None
    if len(arr) == 1:
        return arr[0]
    k = (len(arr) - 1) * pct
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return arr[int(k)]
    return arr[f] * (c - k) + arr[c] * (k - f)


def _read_silver_table(silver_files: list[Path]) -> Optional[pa.Table]:
    if not silver_files:
        return None
    chunks = []
    for f in silver_files:
        pf = pq.ParquetFile(f)
        chunks.append(pf.read())
    return pa.concat_tables(chunks) if chunks else None


def _row_count_by_hour(table: pa.Table) -> dict[int, int]:
    hours = table.column("source_hour").to_pylist()
    out: dict[int, int] = {}
    for h in hours:
        out[int(h)] = out.get(int(h), 0) + 1
    return out


def _summary_stats(values: pa.ChunkedArray | pa.Array) -> dict[str, Optional[float]]:
    if len(values) == 0:
        return {k: None for k in ("min", "mean", "p50", "p95", "p99", "max")}
    try:
        mn = pc.min(values).as_py()
    except Exception:
        mn = None
    try:
        mx = pc.max(values).as_py()
    except Exception:
        mx = None
    try:
        mean = pc.mean(values).as_py()
    except Exception:
        mean = None
    return {
        "min": float(mn) if mn is not None else None,
        "mean": float(mean) if mean is not None else None,
        "p50": _percentile(values, 0.5),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "max": float(mx) if mx is not None else None,
    }


def _verify_silver_readable(silver_files: list[Path]) -> tuple[bool, Optional[str]]:
    if not silver_files:
        return (False, "no silver files")
    sample = silver_files[0]
    try:
        import polars as pl

        _ = pl.read_parquet(str(sample))
    except Exception as exc:
        return (False, f"polars read failed: {exc}")
    try:
        import duckdb

        con = duckdb.connect(":memory:")
        con.execute("SELECT COUNT(*) FROM read_parquet(?)", [str(sample)]).fetchone()
        con.close()
    except Exception as exc:
        return (False, f"duckdb read failed: {exc}")
    return (True, None)


def build_quality_report(config: QualityConfig) -> dict:
    md = config.metadata
    warnings: list[str] = []
    fatal_warnings: list[str] = []
    per_symbol: dict[str, dict] = {}
    all_symbols = list(set(list(config.required_symbols) + list(md.symbols or ())))
    all_symbols.sort()
    silver_readable_overall = True
    silver_read_error: Optional[str] = None
    silver_layout_notes: list[str] = []
    selected_run_ids: set[str] = set()
    observed_layouts: set[str] = set()
    any_ambiguous = False
    for symbol in all_symbols:
        bronze_files = _bronze_files(config.raw_root, symbol, config.date)
        selection = silver_paths.select_silver(
            config.silver_root, symbol, config.date, config.run_id
        )
        silver_files = selection.files
        observed_layouts.add(selection.layout)
        if selection.ambiguous:
            any_ambiguous = True
        if selection.selected_run_id:
            selected_run_ids.add(selection.selected_run_id)
        for note in selection.warnings:
            silver_layout_notes.append(note)
        if selection.ambiguous:
            fatal_warnings.append(
                selection.error or f"{symbol}: multiple run partitions; pass --run-id"
            )
        sym_entry: dict = {
            "bronze_file_count": len(bronze_files),
            "silver_file_count": len(silver_files),
            "silver_files": [str(p) for p in silver_files],
            "silver_layout": selection.layout,
            "silver_layout_version": selection.layout_version,
            "selected_run_id": selection.selected_run_id,
            "available_run_ids": list(selection.available_run_ids),
            "row_count_by_hour": {},
            "total_rows": 0,
            "bid_ask_violation_count": 0,
            "scaled_bid_ask_violation_count": 0,
            "negative_spread_count": 0,
            "join_safe_count": 0,
            "join_safe_pct": None,
            "latency_outlier_count": 0,
            "latency_outlier_pct": None,
            "rows_with_suppression": 0,
            "suppressed_total": 0,
            "spread_summary": {},
            "residual_summary": {},
            "silver_readable": False,
            "silver_read_error": None,
        }
        if silver_files:
            ok, err = _verify_silver_readable(silver_files)
            sym_entry["silver_readable"] = ok
            sym_entry["silver_read_error"] = err
            if not ok:
                silver_readable_overall = False
                silver_read_error = err
        table = _read_silver_table(silver_files)
        if table is not None and table.num_rows > 0:
            total = table.num_rows
            sym_entry["total_rows"] = total
            sym_entry["row_count_by_hour"] = _row_count_by_hour(table)
            bid_ask_bad = int(
                pc.sum(pc.cast(pc.invert(table.column("is_bid_ask_valid")), pa.int64())).as_py()
                or 0
            )
            sym_entry["bid_ask_violation_count"] = bid_ask_bad
            if bid_ask_bad > 0:
                fatal_warnings.append(f"{symbol}: {bid_ask_bad} rows where ask < bid")
            scaled_bad = int(
                pc.sum(
                    pc.cast(pc.invert(table.column("is_scaled_bid_ask_valid")), pa.int64())
                ).as_py()
                or 0
            )
            sym_entry["scaled_bid_ask_violation_count"] = scaled_bad
            if scaled_bad > 0:
                fatal_warnings.append(f"{symbol}: {scaled_bad} rows where ask_scaled < bid_scaled")
            neg_spread = int(
                pc.sum(pc.cast(pc.invert(table.column("is_spread_valid")), pa.int64())).as_py() or 0
            )
            sym_entry["negative_spread_count"] = neg_spread
            if neg_spread > 0:
                fatal_warnings.append(f"{symbol}: {neg_spread} rows with negative spread")
            join_safe = int(pc.sum(pc.cast(table.column("is_join_safe"), pa.int64())).as_py() or 0)
            sym_entry["join_safe_count"] = join_safe
            sym_entry["join_safe_pct"] = join_safe / total * 100.0 if total else None
            lat_outliers = int(
                pc.sum(pc.cast(table.column("is_latency_outlier"), pa.int64())).as_py() or 0
            )
            sym_entry["latency_outlier_count"] = lat_outliers
            sym_entry["latency_outlier_pct"] = lat_outliers / total * 100.0 if total else None
            sup_counts = table.column("suppressed_count").to_numpy(zero_copy_only=False)
            sup_total = int(sum((int(v) for v in sup_counts if v is not None)))
            sup_rows = int(sum((1 for v in sup_counts if v is not None and int(v) > 0)))
            sym_entry["suppressed_total"] = sup_total
            sym_entry["rows_with_suppression"] = sup_rows
            sym_entry["spread_summary"] = _summary_stats(table.column("spread_price"))
            sym_entry["residual_summary"] = _summary_stats(table.column("residual_ms"))
            if (
                symbol == "NDX100"
                and sym_entry["spread_summary"].get("max") is not None
                and (sym_entry["spread_summary"]["max"] > config.thresholds.ndx_spread_spike_price)
            ):
                warnings.append(
                    f"NDX100 max spread {sym_entry['spread_summary']['max']:.4f} exceeds threshold {config.thresholds.ndx_spread_spike_price}"
                )
            if (
                sym_entry["join_safe_pct"] is not None
                and sym_entry["join_safe_pct"] < config.thresholds.join_safe_low_pct
            ):
                warnings.append(
                    f"{symbol}: join-safe percentage {sym_entry['join_safe_pct']:.2f}% is below threshold {config.thresholds.join_safe_low_pct}%"
                )
            if (
                sym_entry["latency_outlier_pct"] is not None
                and sym_entry["latency_outlier_pct"] > config.thresholds.latency_outlier_high_pct
            ):
                warnings.append(
                    f"{symbol}: latency-outlier percentage {sym_entry['latency_outlier_pct']:.2f}% exceeds threshold {config.thresholds.latency_outlier_high_pct}%"
                )
        per_symbol[symbol] = sym_entry
    required_present = all(
        (per_symbol.get(s, {}).get("total_rows", 0) > 0 for s in config.required_symbols)
    )
    required_silver_written = all(
        (per_symbol.get(s, {}).get("silver_file_count", 0) > 0 for s in config.required_symbols)
    )
    if (
        not required_present
        or not required_silver_written
        or (not silver_readable_overall)
        or fatal_warnings
    ):
        decision = "FAIL"
    elif warnings:
        decision = "PARTIAL"
    else:
        decision = "PASS"
    if any_ambiguous:
        overall_layout = "ambiguous_multiple_runs"
    elif silver_paths.LAYOUT_RUN_SCOPED in observed_layouts:
        overall_layout = silver_paths.LAYOUT_RUN_SCOPED
    elif silver_paths.LAYOUT_LEGACY_DATE_LEVEL in observed_layouts:
        overall_layout = silver_paths.LAYOUT_LEGACY_DATE_LEVEL
    else:
        overall_layout = silver_paths.LAYOUT_MISSING
    overall_layout_version = (
        silver_paths.SILVER_LAYOUT_VERSION_RUN_SCOPED
        if overall_layout == silver_paths.LAYOUT_RUN_SCOPED
        else silver_paths.SILVER_LAYOUT_VERSION_LEGACY
    )
    report_selected_run_id = config.run_id or (
        next(iter(selected_run_ids)) if len(selected_run_ids) == 1 else None
    )
    return {
        "source_date": config.date,
        "requested_run_id": config.run_id,
        "selected_run_id": report_selected_run_id,
        "silver_layout": overall_layout,
        "silver_layout_version": overall_layout_version,
        "silver_layout_notes": silver_layout_notes,
        "metadata_source_path": str(md.source_path),
        "metadata_source_kind": md.source_kind,
        "verified_offset_min": md.verified_offset_min,
        "verified_offset_ms": md.verified_offset_ms,
        "timestamp_semantics_status": md.timestamp_semantics_status,
        "clock_status": md.clock_status,
        "broker_metadata": md.broker_metadata,
        "symbols_included": all_symbols,
        "required_symbols": list(config.required_symbols),
        "join_safe_threshold_ms": config.join_safe_threshold_ms,
        "latency_outlier_high_ms": LATENCY_OUTLIER_HIGH_MS,
        "latency_outlier_low_ms": LATENCY_OUTLIER_LOW_MS,
        "thresholds": {
            "join_safe_low_pct": config.thresholds.join_safe_low_pct,
            "latency_outlier_high_pct": config.thresholds.latency_outlier_high_pct,
            "ndx_spread_spike_price": config.thresholds.ndx_spread_spike_price,
        },
        "per_symbol": per_symbol,
        "silver_readable": silver_readable_overall,
        "silver_read_error": silver_read_error,
        "warnings": warnings,
        "fatal_warnings": fatal_warnings,
        "decision": decision,
        "generated_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }


def render_text_report(report: dict) -> str:
    lines: list[str] = []
    lines.append("=" * 80)
    lines.append(f"Polarix Telemetry Quality Report  --  source_date={report['source_date']}")
    lines.append("=" * 80)
    lines.append(f"Decision               : {report['decision']}")
    lines.append(f"Generated (UTC)        : {report['generated_at_utc']}")
    lines.append(f"Requested run_id       : {report.get('requested_run_id')}")
    lines.append(f"Selected run_id        : {report.get('selected_run_id')}")
    lines.append(f"Silver layout          : {report.get('silver_layout')}")
    lines.append(f"Silver layout version  : {report.get('silver_layout_version')}")
    lines.append(f"Metadata source        : {report['metadata_source_path']}")
    lines.append(f"Metadata kind          : {report['metadata_source_kind']}")
    lines.append(f"Timestamp semantics    : {report['timestamp_semantics_status']}")
    lines.append(
        f"Verified offset        : {report['verified_offset_min']} min ({report['verified_offset_ms']} ms)"
    )
    lines.append(f"Join-safe threshold ms : {report['join_safe_threshold_ms']}")
    lines.append(
        f"Latency outlier band   : <{report['latency_outlier_low_ms']}ms or >{report['latency_outlier_high_ms']}ms"
    )
    clk = report.get("clock_status") or {}
    if clk:
        lines.append(
            f"Clock status           : {clk.get('status')} offset={clk.get('offset_ms')}ms source={clk.get('source')}"
        )
    bk = report.get("broker_metadata") or {}
    if bk:
        lines.append(
            f"Broker                 : {bk.get('account_company')} / {bk.get('account_server')} / login_hash={bk.get('account_login_hash')}"
        )
    lines.append("")
    lines.append(f"Required symbols       : {', '.join(report['required_symbols'])}")
    lines.append(f"Symbols included       : {', '.join(report['symbols_included'])}")
    lines.append("")
    for symbol, entry in report["per_symbol"].items():
        lines.append("-" * 80)
        lines.append(f"SYMBOL: {symbol}")
        lines.append(f"  bronze_files      : {entry['bronze_file_count']}")
        lines.append(f"  silver_files      : {entry['silver_file_count']}")
        lines.append(f"  total_rows        : {entry['total_rows']}")
        lines.append(f"  rows_by_hour      : {entry['row_count_by_hour']}")
        lines.append(f"  ask<bid rows      : {entry['bid_ask_violation_count']}")
        lines.append(f"  ask_s<bid_s rows  : {entry['scaled_bid_ask_violation_count']}")
        lines.append(f"  neg_spread rows   : {entry['negative_spread_count']}")
        js_pct = entry.get("join_safe_pct")
        lo_pct = entry.get("latency_outlier_pct")
        lines.append(
            f"  join-safe         : {entry['join_safe_count']} ({('-' if js_pct is None else f'{js_pct:.2f}%')})"
        )
        lines.append(
            f"  latency outliers  : {entry['latency_outlier_count']} ({('-' if lo_pct is None else f'{lo_pct:.2f}%')})"
        )
        lines.append(
            f"  suppression total : {entry['suppressed_total']}  rows_w/suppression: {entry['rows_with_suppression']}"
        )
        ss = entry.get("spread_summary") or {}
        rs = entry.get("residual_summary") or {}
        lines.append(
            "  spread (price)    : min={min} mean={mean} p50={p50} p95={p95} p99={p99} max={max}".format(
                **{k: ss.get(k) for k in ("min", "mean", "p50", "p95", "p99", "max")}
            )
        )
        lines.append(
            "  residual_ms       : min={min} mean={mean} p50={p50} p95={p95} p99={p99} max={max}".format(
                **{k: rs.get(k) for k in ("min", "mean", "p50", "p95", "p99", "max")}
            )
        )
        lines.append(
            f"  silver readable   : {entry['silver_readable']} (err={entry['silver_read_error']})"
        )
    lines.append("-" * 80)
    if report.get("silver_layout_notes"):
        lines.append("Silver layout notes:")
        for w in report["silver_layout_notes"]:
            lines.append(f"  - {w}")
    if report["warnings"]:
        lines.append("Warnings:")
        for w in report["warnings"]:
            lines.append(f"  - {w}")
    if report["fatal_warnings"]:
        lines.append("Fatal warnings:")
        for w in report["fatal_warnings"]:
            lines.append(f"  - {w}")
    lines.append("=" * 80)
    return "\n".join(lines) + "\n"


def write_reports(config: QualityConfig, report: dict) -> tuple[Path, Path]:
    config.reports_root.mkdir(parents=True, exist_ok=True)
    json_path = config.reports_root / f"telemetry_quality_{config.date}.json"
    txt_path = config.reports_root / f"telemetry_quality_{config.date}.txt"
    json_tmp = json_path.with_suffix(".json.tmp")
    json_tmp.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(json_tmp, json_path)
    txt_tmp = txt_path.with_suffix(".txt.tmp")
    txt_tmp.write_text(render_text_report(report), encoding="utf-8")
    os.replace(txt_tmp, txt_path)
    return (json_path, txt_path)
