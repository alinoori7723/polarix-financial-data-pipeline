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

from polarix.ingestion import cme_downloader as cme_dl
from polarix.ingestion.cme_databento_schema import (
    AGGRESSOR_UNKNOWN,
    DATASET,
    SCHEMA,
    VENDOR,
    CmeSchemaError,
    detect_binding,
)
from polarix.ingestion.cme_reference_ingest import (
    DEFAULT_SYMBOLS,
    REFERENCE_TRADES_SUBDIR,
    TRADE_ACTION,
    IngestResult,
)


@dataclass
class QualityThresholds:
    high_unknown_side_pct: float = 25.0


@dataclass
class CmeQualityConfig:
    date: str
    input_root: Path
    output_root: Path
    reports_root: Path
    symbols_requested: tuple[str, ...] = DEFAULT_SYMBOLS
    thresholds: QualityThresholds = field(default_factory=QualityThresholds)

    def __post_init__(self) -> None:
        self.input_root = Path(self.input_root).resolve()
        self.output_root = Path(self.output_root).resolve()
        self.reports_root = Path(self.reports_root).resolve()


def _percentile(values, pct: float) -> Optional[float]:
    arr = [
        float(v) for v in values if v is not None and (not (isinstance(v, float) and math.isnan(v)))
    ]
    if not arr:
        return None
    arr.sort()
    if len(arr) == 1:
        return arr[0]
    k = (len(arr) - 1) * pct
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return arr[int(k)]
    return arr[f] * (c - k) + arr[c] * (k - f)


def _summary_stats(values) -> dict:
    arr = [v for v in values if v is not None]
    if not arr:
        return {k: None for k in ("min", "mean", "p50", "p95", "p99", "max")}
    return {
        "min": float(min(arr)),
        "mean": float(sum(arr) / len(arr)),
        "p50": _percentile(arr, 0.5),
        "p95": _percentile(arr, 0.95),
        "p99": _percentile(arr, 0.99),
        "max": float(max(arr)),
    }


def _read_normalized(output_root: Path, symbol: str, date: str) -> Optional[pa.Table]:
    pattern = os.path.join(
        str(output_root),
        REFERENCE_TRADES_SUBDIR,
        f"symbol={symbol}",
        f"date={date}",
        "part-*.parquet",
    )
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    chunks = [pq.ParquetFile(f).read() for f in files]
    return pa.concat_tables(chunks)


def _silver_files_exist_for(output_root: Path, symbol: str, date: str) -> bool:
    pattern = os.path.join(
        str(output_root),
        REFERENCE_TRADES_SUBDIR,
        f"symbol={symbol}",
        f"date={date}",
        "part-*.parquet",
    )
    return bool(glob.glob(pattern))


def _verify_readable(output_root: Path, symbol: str, date: str) -> tuple[bool, Optional[str]]:
    pattern = os.path.join(
        str(output_root),
        REFERENCE_TRADES_SUBDIR,
        f"symbol={symbol}",
        f"date={date}",
        "part-*.parquet",
    )
    files = sorted(glob.glob(pattern))
    if not files:
        return (False, "no normalized files")
    try:
        import polars as pl

        _ = pl.read_parquet(files[0])
    except Exception as exc:
        return (False, f"polars read failed: {exc}")
    try:
        import duckdb

        con = duckdb.connect(":memory:")
        con.execute("SELECT COUNT(*) FROM read_parquet(?)", [str(files[0])]).fetchone()
        con.close()
    except Exception as exc:
        return (False, f"duckdb read failed: {exc}")
    return (True, None)


def _read_sample_files(input_root: Path) -> list[Path]:
    if not input_root.exists():
        return []
    return sorted(
        (
            Path(p)
            for p in glob.glob(os.path.join(str(input_root), "**", "*.parquet"), recursive=True)
        )
    )


def build_quality_report(
    config: CmeQualityConfig, *, ingest_result: Optional[IngestResult] = None
) -> dict:
    warnings: list[str] = []
    fatal_warnings: list[str] = []
    sample_files = _read_sample_files(config.input_root)
    if not sample_files:
        return {
            "date": config.date,
            "input_root": str(config.input_root),
            "output_root": str(config.output_root),
            "vendor": VENDOR,
            "dataset": DATASET,
            "schema": SCHEMA,
            "symbols_requested": list(config.symbols_requested),
            "decision": "FAIL",
            "error": "MISSING_SAMPLE_DATA",
            "fatal_warnings": [f"no Databento sample Parquet found under {config.input_root}"],
            "warnings": [],
            "source_files": [],
            "per_symbol": {},
            "generated_at_utc": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
        }
    first = pq.ParquetFile(sample_files[0]).read()
    try:
        binding = detect_binding(list(first.column_names))
        schema_ok = True
        schema_error = None
    except CmeSchemaError as exc:
        schema_ok = False
        schema_error = str(exc)
        fatal_warnings.append(f"schema rejection: {exc}")
        binding = None
    total_rows_read = sum((pq.ParquetFile(p).read().num_rows for p in sample_files))
    rows_by_action: dict[str, int] = {}
    rows_by_side_raw: dict[str, int] = {}
    trade_rows_extracted = 0
    if schema_ok:
        assert binding is not None
        for p in sample_files:
            t = pq.ParquetFile(p).read()
            actions = t.column(binding.action).cast(pa.string()).to_pylist()
            sides = t.column(binding.side).cast(pa.string()).to_pylist()
            for a in actions:
                key = a if a is not None else "<null>"
                rows_by_action[key] = rows_by_action.get(key, 0) + 1
            for s in sides:
                key = s if s is not None else "<null>"
                rows_by_side_raw[key] = rows_by_side_raw.get(key, 0) + 1
            trade_rows_extracted += sum((1 for a in actions if a == TRADE_ACTION))
    per_symbol: dict[str, dict] = {}
    valid_trades_total = 0
    for symbol in config.symbols_requested:
        ok, err = _verify_readable(config.output_root, symbol, config.date)
        table = _read_normalized(config.output_root, symbol, config.date)
        sym_entry: dict = {
            "silver_files_exist": _silver_files_exist_for(config.output_root, symbol, config.date),
            "readable": ok,
            "read_error": err,
            "total_rows": 0,
            "valid_reference_trades": 0,
            "unknown_side_count": 0,
            "unknown_side_pct": None,
            "missing_bbo_count": 0,
            "invalid_price_count": 0,
            "invalid_size_count": 0,
            "ts_event_min": None,
            "ts_event_max": None,
            "ts_recv_min": None,
            "ts_recv_max": None,
            "recv_minus_event_summary": {},
            "spread_summary": {},
        }
        if table is not None and table.num_rows > 0:
            total = table.num_rows
            sym_entry["total_rows"] = total
            valid = int(
                pc.sum(pc.cast(table.column("is_reference_trade_valid"), pa.int64())).as_py() or 0
            )
            sym_entry["valid_reference_trades"] = valid
            valid_trades_total += valid
            unk = int(
                pc.sum(
                    pc.cast(
                        pc.equal(
                            table.column("aggressor_side"),
                            pa.scalar(AGGRESSOR_UNKNOWN, pa.string()),
                        ),
                        pa.int64(),
                    )
                ).as_py()
                or 0
            )
            sym_entry["unknown_side_count"] = unk
            sym_entry["unknown_side_pct"] = unk / total * 100.0
            sym_entry["missing_bbo_count"] = int(
                pc.sum(pc.cast(pc.invert(table.column("is_bbo_available")), pa.int64())).as_py()
                or 0
            )
            sym_entry["invalid_price_count"] = int(
                pc.sum(pc.cast(pc.invert(table.column("is_price_valid")), pa.int64())).as_py() or 0
            )
            sym_entry["invalid_size_count"] = int(
                pc.sum(pc.cast(pc.invert(table.column("is_size_valid")), pa.int64())).as_py() or 0
            )
            event_ns = table.column("event_time_utc_ns").to_pylist()
            recv_ns = table.column("receive_time_utc_ns").to_pylist()
            if event_ns:
                sym_entry["ts_event_min"] = int(min((v for v in event_ns if v is not None)))
                sym_entry["ts_event_max"] = int(max((v for v in event_ns if v is not None)))
            if recv_ns:
                sym_entry["ts_recv_min"] = int(min((v for v in recv_ns if v is not None)))
                sym_entry["ts_recv_max"] = int(max((v for v in recv_ns if v is not None)))
            recv_minus_event = [
                r - e for r, e in zip(recv_ns, event_ns) if r is not None and e is not None
            ]
            sym_entry["recv_minus_event_summary"] = _summary_stats(recv_minus_event)
            if int(pc.sum(pc.cast(table.column("is_bbo_available"), pa.int64())).as_py() or 0) > 0:
                spreads = [v for v in table.column("spread").to_pylist() if v is not None]
                sym_entry["spread_summary"] = _summary_stats(spreads)
            if (
                sym_entry["unknown_side_pct"] is not None
                and sym_entry["unknown_side_pct"] > config.thresholds.high_unknown_side_pct
            ):
                warnings.append(
                    f"{symbol}: unknown_side_pct={sym_entry['unknown_side_pct']:.2f}% exceeds threshold {config.thresholds.high_unknown_side_pct}%"
                )
            if sym_entry["missing_bbo_count"] > 0:
                warnings.append(f"{symbol}: BBO missing on {sym_entry['missing_bbo_count']} row(s)")
        per_symbol[symbol] = sym_entry
    cme_raw_completeness_status: Optional[str] = None
    try:
        date_partition_dir = config.input_root / f"date={config.date}"
        raw_docs: list[dict] = []
        if date_partition_dir.exists():
            raw_docs.extend(cme_dl.find_metadata_for_raw_dir(date_partition_dir))
        if not raw_docs:
            raw_docs.extend(cme_dl.find_metadata_for_raw_dir(config.input_root))
        if raw_docs:
            cme_raw_completeness_status = cme_dl.directory_completeness_status(raw_docs)
    except Exception as exc:
        warnings.append(f"failed to read CME raw metadata sidecar(s): {exc}")
    if cme_raw_completeness_status == cme_dl.DATA_COMPLETENESS_TRUNCATED_BY_LIMIT:
        warnings.append(
            "CME raw sample is TRUNCATED_BY_LIMIT (capped by --max-download-records); quality metrics describe a partial window, not the full session"
        )
    elif cme_raw_completeness_status in (
        cme_dl.DATA_COMPLETENESS_UNKNOWN_NO_RECORD_LIMIT,
        cme_dl.DATA_COMPLETENESS_UNKNOWN_ROW_COUNT,
    ):
        warnings.append(
            f"CME raw sample completeness is {cme_raw_completeness_status}; truncation cannot be ruled out"
        )
    symbols_with_rows = [s for s, e in per_symbol.items() if e["total_rows"] > 0]
    if not schema_ok:
        decision = "FAIL"
    elif trade_rows_extracted == 0:
        decision = "FAIL"
        fatal_warnings.append("no trade rows (action=='T') in sample")
    elif not symbols_with_rows:
        decision = "FAIL"
        fatal_warnings.append("no ES or NQ rows extracted from sample")
    elif any(
        (
            not per_symbol[s]["readable"] and per_symbol[s]["silver_files_exist"]
            for s in symbols_with_rows
        )
    ):
        decision = "FAIL"
        fatal_warnings.append("normalized output cannot be read by polars/duckdb")
    elif cme_raw_completeness_status == cme_dl.DATA_COMPLETENESS_TRUNCATED_BY_LIMIT:
        decision = "PARTIAL"
    elif len(symbols_with_rows) < len([s for s in config.symbols_requested]) or warnings:
        decision = "PARTIAL"
    else:
        decision = "PASS"
    return {
        "date": config.date,
        "input_root": str(config.input_root),
        "output_root": str(config.output_root),
        "vendor": VENDOR,
        "dataset": DATASET,
        "schema": SCHEMA,
        "symbols_requested": list(config.symbols_requested),
        "symbols_found": symbols_with_rows,
        "source_files": [str(p) for p in sample_files],
        "schema_ok": schema_ok,
        "schema_error": schema_error,
        "total_rows_read": total_rows_read,
        "trade_rows_extracted": trade_rows_extracted,
        "valid_reference_trades": valid_trades_total,
        "rows_by_action": rows_by_action,
        "rows_by_side_raw": rows_by_side_raw,
        "per_symbol": per_symbol,
        "decision": decision,
        "warnings": warnings,
        "fatal_warnings": fatal_warnings,
        "cme_raw_completeness_status": cme_raw_completeness_status,
        "generated_at_utc": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
    }


def render_text_report(report: dict) -> str:
    lines = []
    lines.append("=" * 80)
    lines.append(f"Polarix CME Reference Quality Report  --  date={report['date']}")
    lines.append("=" * 80)
    lines.append(f"Decision               : {report['decision']}")
    if report.get("error"):
        lines.append(f"Error                  : {report['error']}")
    lines.append(
        f"Vendor / dataset / schema : {report['vendor']} / {report['dataset']} / {report['schema']}"
    )
    lines.append(f"Input root             : {report['input_root']}")
    lines.append(f"Output root            : {report['output_root']}")
    lines.append(f"Source files           : {len(report['source_files'])}")
    lines.append(f"Symbols requested      : {', '.join(report['symbols_requested'])}")
    if "symbols_found" in report:
        lines.append(f"Symbols found          : {', '.join(report['symbols_found'])}")
    if "total_rows_read" in report:
        lines.append(f"Total rows read        : {report['total_rows_read']}")
        lines.append(f"Trade rows extracted   : {report['trade_rows_extracted']}")
        lines.append(f"Valid reference trades : {report['valid_reference_trades']}")
        lines.append(f"Rows by action         : {report['rows_by_action']}")
        lines.append(f"Rows by side_raw       : {report['rows_by_side_raw']}")
    lines.append("")
    for symbol, entry in report.get("per_symbol", {}).items():
        lines.append("-" * 80)
        lines.append(f"SYMBOL: {symbol}")
        lines.append(f"  silver_files_exist   : {entry['silver_files_exist']}")
        lines.append(f"  readable             : {entry['readable']} (err={entry['read_error']})")
        lines.append(f"  total_rows           : {entry['total_rows']}")
        lines.append(f"  valid_reference     : {entry['valid_reference_trades']}")
        lines.append(
            f"""  unknown_side         : {entry["unknown_side_count"]} ({("-" if entry["unknown_side_pct"] is None else f"{entry['unknown_side_pct']:.2f}%")})"""
        )
        lines.append(f"  missing_bbo_count    : {entry['missing_bbo_count']}")
        lines.append(f"  invalid_price_count  : {entry['invalid_price_count']}")
        lines.append(f"  invalid_size_count   : {entry['invalid_size_count']}")
        lines.append(f"  ts_event min/max     : {entry['ts_event_min']} / {entry['ts_event_max']}")
        lines.append(f"  ts_recv  min/max     : {entry['ts_recv_min']} / {entry['ts_recv_max']}")
        if entry.get("recv_minus_event_summary"):
            r = entry["recv_minus_event_summary"]
            lines.append(
                f"  recv-event ns        : min={r['min']} mean={r['mean']} p50={r['p50']} p95={r['p95']} p99={r['p99']} max={r['max']}"
            )
        if entry.get("spread_summary"):
            s = entry["spread_summary"]
            lines.append(
                f"  spread               : min={s['min']} mean={s['mean']} p50={s['p50']} p95={s['p95']} p99={s['p99']} max={s['max']}"
            )
    lines.append("-" * 80)
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


def write_reports(config: CmeQualityConfig, report: dict) -> tuple[Path, Path]:
    config.reports_root.mkdir(parents=True, exist_ok=True)
    json_path = config.reports_root / f"cme_reference_quality_{config.date}.json"
    txt_path = config.reports_root / f"cme_reference_quality_{config.date}.txt"
    json_tmp = json_path.with_suffix(".json.tmp")
    json_tmp.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(json_tmp, json_path)
    txt_tmp = txt_path.with_suffix(".txt.tmp")
    txt_tmp.write_text(render_text_report(report), encoding="utf-8")
    os.replace(txt_tmp, txt_path)
    return (json_path, txt_path)
