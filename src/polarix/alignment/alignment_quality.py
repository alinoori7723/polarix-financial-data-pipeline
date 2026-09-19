from __future__ import annotations

import datetime as _dt
import glob
import json
import math
import os
import subprocess
from bisect import bisect_right
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Sequence

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from polarix.ingestion import cme_downloader as cme_dl

REASON_NO_MAPPED_SYMBOL = "no_mapped_symbol"
REASON_NO_MT5_ROWS_FOR_SYMBOL = "no_mt5_rows_for_symbol"
REASON_NO_PRIOR_QUOTE = "no_prior_quote"
REASON_OUTSIDE_TOLERANCE = "outside_tolerance"
REASON_MT5_NOT_JOIN_SAFE = "mt5_not_join_safe"
REASON_NO_OVERLAP_WINDOW = "no_overlap_window"
REASON_MISSING_REQUIRED_COLUMNS = "missing_required_columns"
DEFAULT_SYMBOL_MAP: Mapping[str, str] = {"ES": "SPX500", "NQ": "NDX100"}
DEFAULT_ALIGNMENT_TOLERANCE_MS = 50
DEFAULT_DIAGNOSTIC_TOLERANCES_MS = (50, 100, 250, 500, 1000)
DEFAULT_SAMPLE_UNMATCHED_LIMIT = 1000
DEFAULT_MIN_MATCH_RATE_50MS = 0.5
DEFAULT_MAX_UNMATCHED_VOLUME_RATIO_50MS = 0.5
MS_TO_NS = 1000000
CME_REQUIRED_COLS = (
    "symbol",
    "event_time_utc_ns",
    "price",
    "size",
    "aggressor_side",
    "is_reference_trade_valid",
)
MT5_REQUIRED_COLS = (
    "symbol",
    "time_msc_utc_ms",
    "is_join_safe",
    "mid",
    "spread_price",
    "spread_points",
)


class AlignmentQualityError(RuntimeError):
    pass


@dataclass
class AlignmentQualityThresholds:
    min_match_rate_50ms: float = DEFAULT_MIN_MATCH_RATE_50MS
    max_unmatched_volume_ratio_50ms: float = DEFAULT_MAX_UNMATCHED_VOLUME_RATIO_50MS


@dataclass
class AlignmentQualityConfig:
    date: str
    cme_root: Path
    mt5_root: Path
    reports_root: Path
    symbol_map: Mapping[str, str] = field(default_factory=lambda: dict(DEFAULT_SYMBOL_MAP))
    alignment_tolerance_ms: int = DEFAULT_ALIGNMENT_TOLERANCE_MS
    diagnostic_tolerances_ms: tuple[int, ...] = DEFAULT_DIAGNOSTIC_TOLERANCES_MS
    sample_unmatched_limit: int = DEFAULT_SAMPLE_UNMATCHED_LIMIT
    thresholds: AlignmentQualityThresholds = field(default_factory=AlignmentQualityThresholds)
    write_unmatched_sample: bool = True
    cme_raw_input_root: Optional[Path] = None

    def __post_init__(self) -> None:
        self.cme_root = Path(self.cme_root).resolve()
        self.mt5_root = Path(self.mt5_root).resolve()
        self.reports_root = Path(self.reports_root).resolve()
        if self.cme_raw_input_root is not None:
            self.cme_raw_input_root = Path(self.cme_raw_input_root).resolve()
        if self.alignment_tolerance_ms <= 0:
            raise ValueError("alignment_tolerance_ms must be > 0")
        diag = sorted({int(t) for t in self.diagnostic_tolerances_ms if int(t) > 0})
        if self.alignment_tolerance_ms not in diag:
            diag.append(self.alignment_tolerance_ms)
            diag.sort()
        self.diagnostic_tolerances_ms = tuple(diag)


def _cme_files(cme_root: Path, symbol: str, date: str) -> list[Path]:
    pattern = os.path.join(str(cme_root), f"symbol={symbol}", f"date={date}", "part-*.parquet")
    return sorted((Path(p) for p in glob.glob(pattern)))


def _mt5_files(mt5_root: Path, symbol: str, date: str) -> list[Path]:
    pattern = os.path.join(str(mt5_root), f"symbol={symbol}", f"date={date}", "part-*.parquet")
    return sorted((Path(p) for p in glob.glob(pattern)))


def _read_concat(files: Sequence[Path], required_cols: Sequence[str]) -> Optional[pa.Table]:
    if not files:
        return None
    chunks: list[pa.Table] = []
    for f in files:
        t = pq.ParquetFile(f).read()
        missing = [c for c in required_cols if c not in t.column_names]
        if missing:
            raise AlignmentQualityError(f"{f}: missing required columns {missing}")
        chunks.append(t)
    return pa.concat_tables(chunks)


def _git_hash() -> Optional[str]:
    try:
        repo_root = Path(__file__).resolve().parents[3]
        out = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip() or None
    except Exception:
        return None
    return None


@dataclass
class _SymbolDatasets:
    cme_event_ns: list[int]
    cme_size: list[int]
    cme_price: list[float]
    cme_aggressor: list[str]
    cme_valid: list[bool]
    mt5_time_ms: list[int]
    mt5_join_safe: list[bool]
    mt5_mid: list[Optional[float]]
    mt5_spread_price: list[Optional[float]]
    mt5_spread_points: list[Optional[int]]


def _load_symbol_datasets(cme_table: pa.Table, mt5_table: pa.Table) -> _SymbolDatasets:
    mt5_sorted = mt5_table.sort_by([("time_msc_utc_ms", "ascending")])
    return _SymbolDatasets(
        cme_event_ns=cme_table.column("event_time_utc_ns").to_pylist(),
        cme_size=cme_table.column("size").to_pylist(),
        cme_price=cme_table.column("price").to_pylist(),
        cme_aggressor=cme_table.column("aggressor_side").to_pylist(),
        cme_valid=cme_table.column("is_reference_trade_valid").to_pylist(),
        mt5_time_ms=mt5_sorted.column("time_msc_utc_ms").to_pylist(),
        mt5_join_safe=mt5_sorted.column("is_join_safe").to_pylist(),
        mt5_mid=mt5_sorted.column("mid").to_pylist(),
        mt5_spread_price=mt5_sorted.column("spread_price").to_pylist(),
        mt5_spread_points=mt5_sorted.column("spread_points").to_pylist(),
    )


def _percentile(values: list[float], pct: float) -> Optional[float]:
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


def _summary(values: list[float]) -> dict[str, Optional[float]]:
    arr = [
        float(v) for v in values if v is not None and (not (isinstance(v, float) and math.isnan(v)))
    ]
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


def _align_one_event(
    event_ns: int, mt5_time_ms: Sequence[int], mt5_join_safe: Sequence[bool], tolerance_ms: int
) -> tuple[Optional[int], Optional[float], str]:
    if not mt5_time_ms:
        return (None, None, REASON_NO_MT5_ROWS_FOR_SYMBOL)
    event_ms = event_ns // MS_TO_NS
    upper = bisect_right(mt5_time_ms, event_ms)
    if upper == 0:
        return (None, None, REASON_NO_PRIOR_QUOTE)
    saw_within_tolerance = False
    saw_unsafe_within_tolerance = False
    for i in range(upper - 1, -1, -1):
        delta_ms = event_ms - mt5_time_ms[i]
        if delta_ms > tolerance_ms:
            break
        saw_within_tolerance = True
        if mt5_join_safe[i]:
            return (i, float(delta_ms), "")
        saw_unsafe_within_tolerance = True
    if saw_within_tolerance and saw_unsafe_within_tolerance:
        return (None, None, REASON_MT5_NOT_JOIN_SAFE)
    return (None, None, REASON_OUTSIDE_TOLERANCE)


def _per_symbol_metrics(
    cme_symbol: str,
    mt5_symbol: str,
    datasets: _SymbolDatasets,
    config: AlignmentQualityConfig,
    overlap_present: bool,
) -> tuple[dict, list[dict]]:
    tol = config.alignment_tolerance_ms
    n = len(datasets.cme_event_ns)
    aligned = 0
    aligned_volume = 0
    unmatched_count_by_reason: dict[str, int] = {}
    unmatched_volume_by_reason: dict[str, int] = {}
    alignment_deltas: list[float] = []
    matched_mids: list[float] = []
    matched_spread_prices: list[float] = []
    matched_spread_points: list[int] = []
    unknown_aggressor_count = 0
    unknown_aggressor_volume = 0
    unmatched_sample: list[dict] = []
    by_hour: dict[int, dict[str, int]] = {}
    cme_volume_total = 0
    cme_trades_total = 0
    for i in range(n):
        if not datasets.cme_valid[i]:
            cme_trades_total += 1
            size = int(datasets.cme_size[i] or 0)
            cme_volume_total += size
            reason = REASON_MISSING_REQUIRED_COLUMNS
            unmatched_count_by_reason[reason] = unmatched_count_by_reason.get(reason, 0) + 1
            unmatched_volume_by_reason[reason] = unmatched_volume_by_reason.get(reason, 0) + size
            continue
        cme_trades_total += 1
        size = int(datasets.cme_size[i] or 0)
        cme_volume_total += size
        agg = datasets.cme_aggressor[i]
        if agg == "UNKNOWN":
            unknown_aggressor_count += 1
            unknown_aggressor_volume += size
        event_ns = int(datasets.cme_event_ns[i])
        hour = event_ns // 1000000000 // 3600 % 24
        bucket = by_hour.setdefault(int(hour), {"total": 0, "aligned": 0})
        bucket["total"] += 1
        if not overlap_present:
            reason = REASON_NO_OVERLAP_WINDOW
            unmatched_count_by_reason[reason] = unmatched_count_by_reason.get(reason, 0) + 1
            unmatched_volume_by_reason[reason] = unmatched_volume_by_reason.get(reason, 0) + size
            if len(unmatched_sample) < config.sample_unmatched_limit:
                unmatched_sample.append(
                    {
                        "cme_symbol": cme_symbol,
                        "mt5_symbol": mt5_symbol,
                        "cme_event_time_utc_ns": event_ns,
                        "cme_price": datasets.cme_price[i],
                        "cme_size": size,
                        "cme_aggressor_side": agg,
                        "reason": reason,
                    }
                )
            continue
        idx, delta_ms, reason = _align_one_event(
            event_ns, datasets.mt5_time_ms, datasets.mt5_join_safe, tol
        )
        if idx is None:
            unmatched_count_by_reason[reason] = unmatched_count_by_reason.get(reason, 0) + 1
            unmatched_volume_by_reason[reason] = unmatched_volume_by_reason.get(reason, 0) + size
            if len(unmatched_sample) < config.sample_unmatched_limit:
                unmatched_sample.append(
                    {
                        "cme_symbol": cme_symbol,
                        "mt5_symbol": mt5_symbol,
                        "cme_event_time_utc_ns": event_ns,
                        "cme_price": datasets.cme_price[i],
                        "cme_size": size,
                        "cme_aggressor_side": agg,
                        "reason": reason,
                    }
                )
            continue
        aligned += 1
        aligned_volume += size
        bucket["aligned"] += 1
        if delta_ms is not None:
            alignment_deltas.append(delta_ms)
        mid = datasets.mt5_mid[idx]
        sp = datasets.mt5_spread_price[idx]
        spp = datasets.mt5_spread_points[idx]
        if mid is not None:
            matched_mids.append(float(mid))
        if sp is not None:
            matched_spread_prices.append(float(sp))
        if spp is not None:
            matched_spread_points.append(int(spp))
    unmatched_count = cme_trades_total - aligned
    unmatched_volume = cme_volume_total - aligned_volume
    match_rate = aligned / cme_trades_total if cme_trades_total else None
    unmatched_volume_ratio = unmatched_volume / cme_volume_total if cme_volume_total else None
    diagnostic_rates: dict[int, dict] = {}
    if overlap_present and cme_trades_total:
        for diag_tol in config.diagnostic_tolerances_ms:
            d_aligned = 0
            d_aligned_vol = 0
            for i in range(n):
                if not datasets.cme_valid[i]:
                    continue
                size = int(datasets.cme_size[i] or 0)
                idx, _, _ = _align_one_event(
                    int(datasets.cme_event_ns[i]),
                    datasets.mt5_time_ms,
                    datasets.mt5_join_safe,
                    diag_tol,
                )
                if idx is not None:
                    d_aligned += 1
                    d_aligned_vol += size
            diagnostic_rates[diag_tol] = {
                "aligned_count": d_aligned,
                "aligned_volume": d_aligned_vol,
                "match_rate": d_aligned / cme_trades_total if cme_trades_total else None,
                "unmatched_volume_ratio": 1.0 - d_aligned_vol / cme_volume_total
                if cme_volume_total
                else None,
            }
    match_rate_by_hour = {
        h: {
            "total": v["total"],
            "aligned": v["aligned"],
            "rate": v["aligned"] / v["total"] if v["total"] else None,
        }
        for h, v in sorted(by_hour.items())
    }
    return (
        {
            "cme_symbol": cme_symbol,
            "mt5_symbol": mt5_symbol,
            "cme_trades_total": cme_trades_total,
            "cme_volume_total": cme_volume_total,
            "aligned_count": aligned,
            "unmatched_count": unmatched_count,
            "alignment_match_rate": match_rate,
            "aligned_volume": aligned_volume,
            "unmatched_volume": unmatched_volume,
            "unmatched_volume_ratio": unmatched_volume_ratio,
            "unmatched_count_by_reason": unmatched_count_by_reason,
            "unmatched_volume_by_reason": unmatched_volume_by_reason,
            "alignment_delta_ms_summary": _summary(alignment_deltas),
            "matched_mt5_mid_summary": _summary(matched_mids),
            "matched_mt5_spread_price_summary": _summary(matched_spread_prices),
            "matched_mt5_spread_points_summary": _summary(
                [float(v) for v in matched_spread_points]
            ),
            "cme_trade_size_summary": _summary(
                [float(v) for v in datasets.cme_size if v is not None]
            ),
            "cme_trade_price_summary": _summary(
                [float(v) for v in datasets.cme_price if v is not None]
            ),
            "unknown_aggressor_count": unknown_aggressor_count,
            "unknown_aggressor_volume": unknown_aggressor_volume,
            "match_rate_by_hour": match_rate_by_hour,
            "match_rate_by_session_hour_utc": match_rate_by_hour,
            "diagnostic_rates_by_tolerance_ms": diagnostic_rates,
        },
        unmatched_sample,
    )


def _window_ns_from_event_ns(values: Sequence[int]) -> Optional[tuple[int, int]]:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return (min(vals), max(vals))


def _window_ns_from_ms(values: Sequence[int]) -> Optional[tuple[int, int]]:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return (min(vals) * MS_TO_NS, max(vals) * MS_TO_NS)


def _overlap(
    a: Optional[tuple[int, int]], b: Optional[tuple[int, int]]
) -> Optional[tuple[int, int]]:
    if a is None or b is None:
        return None
    lo = max(a[0], b[0])
    hi = min(a[1], b[1])
    if lo > hi:
        return None
    return (lo, hi)


def _has_alignment_overlap(
    cme_window: Optional[tuple[int, int]],
    mt5_window: Optional[tuple[int, int]],
    alignment_tolerance_ms: int,
) -> Optional[tuple[int, int]]:
    if cme_window is None or mt5_window is None:
        return None
    tolerance_ns = alignment_tolerance_ms * MS_TO_NS
    effective_cme = (cme_window[0] - tolerance_ns, cme_window[1])
    return _overlap(effective_cme, mt5_window)


def _ns_to_iso(ns: Optional[int]) -> Optional[str]:
    if ns is None:
        return None
    return _dt.datetime.fromtimestamp(ns / 1000000000.0, tz=_dt.timezone.utc).isoformat()


def build_alignment_quality_report(config: AlignmentQualityConfig) -> tuple[dict, list[dict]]:
    warnings: list[str] = []
    errors: list[str] = []
    cme_present_any = False
    mt5_present_any = False
    per_symbol: dict[str, dict] = {}
    time_window_cme: dict[str, dict] = {}
    time_window_mt5: dict[str, dict] = {}
    overlap_window: dict[str, dict] = {}
    sample_rows: list[dict] = []
    schema_failure: Optional[str] = None
    for cme_symbol, mt5_symbol in config.symbol_map.items():
        cme_files = _cme_files(config.cme_root, cme_symbol, config.date)
        mt5_files = _mt5_files(config.mt5_root, mt5_symbol, config.date)
        cme_present = bool(cme_files)
        mt5_present = bool(mt5_files)
        cme_present_any = cme_present_any or cme_present
        mt5_present_any = mt5_present_any or mt5_present
        sym_entry: dict = {
            "cme_symbol": cme_symbol,
            "mt5_symbol": mt5_symbol,
            "cme_present": cme_present,
            "mt5_present": mt5_present,
        }
        if not cme_present or not mt5_present:
            sym_entry["status"] = (
                "MISSING_CME_REFERENCE" if not cme_present else "MISSING_MT5_SILVER"
            )
            per_symbol[cme_symbol] = sym_entry
            continue
        try:
            cme_table = _read_concat(cme_files, CME_REQUIRED_COLS)
            mt5_table = _read_concat(mt5_files, MT5_REQUIRED_COLS)
        except AlignmentQualityError as exc:
            schema_failure = str(exc)
            errors.append(str(exc))
            sym_entry["status"] = "MISSING_REQUIRED_COLUMNS"
            sym_entry["error"] = str(exc)
            per_symbol[cme_symbol] = sym_entry
            continue
        if "symbol" in mt5_table.column_names:
            mask = pc.equal(mt5_table.column("symbol"), pa.scalar(mt5_symbol, pa.string()))
            mt5_table = mt5_table.filter(mask)
        cme_window = _window_ns_from_event_ns(cme_table.column("event_time_utc_ns").to_pylist())
        mt5_window = _window_ns_from_ms(mt5_table.column("time_msc_utc_ms").to_pylist())
        ov = _has_alignment_overlap(cme_window, mt5_window, config.alignment_tolerance_ms)
        time_window_cme[cme_symbol] = {
            "min_ns": cme_window[0] if cme_window else None,
            "max_ns": cme_window[1] if cme_window else None,
            "min_utc": _ns_to_iso(cme_window[0]) if cme_window else None,
            "max_utc": _ns_to_iso(cme_window[1]) if cme_window else None,
        }
        time_window_mt5[mt5_symbol] = {
            "min_ns": mt5_window[0] if mt5_window else None,
            "max_ns": mt5_window[1] if mt5_window else None,
            "min_utc": _ns_to_iso(mt5_window[0]) if mt5_window else None,
            "max_utc": _ns_to_iso(mt5_window[1]) if mt5_window else None,
        }
        overlap_window[cme_symbol] = {
            "min_ns": ov[0] if ov else None,
            "max_ns": ov[1] if ov else None,
            "min_utc": _ns_to_iso(ov[0]) if ov else None,
            "max_utc": _ns_to_iso(ov[1]) if ov else None,
            "has_overlap": ov is not None,
        }
        datasets = _load_symbol_datasets(cme_table, mt5_table)
        sym_metrics, sym_sample = _per_symbol_metrics(
            cme_symbol=cme_symbol,
            mt5_symbol=mt5_symbol,
            datasets=datasets,
            config=config,
            overlap_present=ov is not None,
        )
        sym_entry.update(sym_metrics)
        sym_entry["status"] = "OK" if ov is not None else "NO_OVERLAP_WINDOW"
        per_symbol[cme_symbol] = sym_entry
        sample_rows.extend(sym_sample)
    cme_raw_completeness_status: Optional[str] = None
    if config.cme_raw_input_root is not None:
        try:
            raw_docs = cme_dl.find_metadata_for_raw_dir(config.cme_raw_input_root)
            if raw_docs:
                cme_raw_completeness_status = cme_dl.directory_completeness_status(raw_docs)
        except Exception as exc:
            warnings.append(f"failed to read CME raw metadata sidecar(s): {exc}")
    if cme_raw_completeness_status == cme_dl.DATA_COMPLETENESS_TRUNCATED_BY_LIMIT:
        warnings.append(
            "CME raw sample is TRUNCATED_BY_LIMIT (capped by --max-download-records); alignment metrics describe a partial window only"
        )
    elif cme_raw_completeness_status in (
        cme_dl.DATA_COMPLETENESS_UNKNOWN_NO_RECORD_LIMIT,
        cme_dl.DATA_COMPLETENESS_UNKNOWN_ROW_COUNT,
    ):
        warnings.append(
            f"CME raw sample completeness is {cme_raw_completeness_status}; truncation cannot be ruled out"
        )
    decision: str
    decision_reason: Optional[str] = None
    if not cme_present_any:
        decision = "FAIL"
        decision_reason = "MISSING_CME_REFERENCE"
    elif not mt5_present_any:
        decision = "FAIL"
        decision_reason = "MISSING_MT5_SILVER"
    elif schema_failure is not None:
        decision = "FAIL"
        decision_reason = "MISSING_REQUIRED_COLUMNS"
    else:
        any_overlap = any(
            ((per_symbol.get(s) or {}).get("status") == "OK" for s in config.symbol_map)
        )
        if not any_overlap:
            decision = "FAIL"
            decision_reason = "MISSING_OVERLAP_DATA"
        else:
            ok_symbols = [
                s for s in config.symbol_map if (per_symbol.get(s) or {}).get("status") == "OK"
            ]
            partial = len(ok_symbols) < len(config.symbol_map)
            for s in ok_symbols:
                m = per_symbol[s]
                mr = m.get("alignment_match_rate")
                uvr = m.get("unmatched_volume_ratio")
                if mr is not None and mr < config.thresholds.min_match_rate_50ms:
                    partial = True
                    warnings.append(
                        f"{s}: match rate {mr:.4f} below threshold {config.thresholds.min_match_rate_50ms}"
                    )
                if uvr is not None and uvr > config.thresholds.max_unmatched_volume_ratio_50ms:
                    partial = True
                    warnings.append(
                        f"{s}: unmatched_volume_ratio {uvr:.4f} above threshold {config.thresholds.max_unmatched_volume_ratio_50ms}"
                    )
            if cme_raw_completeness_status == cme_dl.DATA_COMPLETENESS_TRUNCATED_BY_LIMIT:
                partial = True
            decision = "PARTIAL" if partial else "PASS"
    overall = {
        "cme_trades_total": 0,
        "cme_volume_total": 0,
        "aligned_count": 0,
        "aligned_volume": 0,
        "unmatched_count": 0,
        "unmatched_volume": 0,
        "alignment_match_rate": None,
        "unmatched_volume_ratio": None,
    }
    for s in config.symbol_map:
        m = per_symbol.get(s) or {}
        if m.get("status") != "OK":
            continue
        for k in (
            "cme_trades_total",
            "cme_volume_total",
            "aligned_count",
            "aligned_volume",
            "unmatched_count",
            "unmatched_volume",
        ):
            overall[k] += int(m.get(k) or 0)
    if overall["cme_trades_total"]:
        overall["alignment_match_rate"] = overall["aligned_count"] / overall["cme_trades_total"]
    if overall["cme_volume_total"]:
        overall["unmatched_volume_ratio"] = (
            overall["unmatched_volume"] / overall["cme_volume_total"]
        )
    report = {
        "date": config.date,
        "cme_root": str(config.cme_root),
        "mt5_root": str(config.mt5_root),
        "reports_root": str(config.reports_root),
        "symbol_map": dict(config.symbol_map),
        "alignment_tolerance_ms": config.alignment_tolerance_ms,
        "diagnostic_tolerances_ms": list(config.diagnostic_tolerances_ms),
        "sample_unmatched_limit": config.sample_unmatched_limit,
        "thresholds": {
            "min_match_rate_50ms": config.thresholds.min_match_rate_50ms,
            "max_unmatched_volume_ratio_50ms": config.thresholds.max_unmatched_volume_ratio_50ms,
        },
        "quality_decision": decision,
        "decision_reason": decision_reason,
        "cme_raw_completeness_status": cme_raw_completeness_status,
        "errors": errors,
        "warnings": warnings,
        "per_symbol": per_symbol,
        "overall": overall,
        "time_window_cme_by_symbol": time_window_cme,
        "time_window_mt5_by_symbol": time_window_mt5,
        "overlap_window_by_symbol": overlap_window,
        "real_overlap_present": any(
            ((overlap_window.get(s) or {}).get("has_overlap") for s in config.symbol_map)
        ),
        "generated_at_utc": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
        "code_version": _git_hash(),
    }
    return (report, sample_rows)


def render_text_report(report: dict) -> str:
    lines: list[str] = []
    lines.append("=" * 80)
    lines.append(f"Polarix Alignment Quality Report  --  date={report['date']}")
    lines.append("=" * 80)
    lines.append(f"Decision               : {report['quality_decision']}")
    if report.get("decision_reason"):
        lines.append(f"Decision reason        : {report['decision_reason']}")
    lines.append(f"Real overlap present   : {report['real_overlap_present']}")
    lines.append(f"Alignment tolerance ms : {report['alignment_tolerance_ms']}")
    lines.append(f"Diagnostic tolerances  : {report['diagnostic_tolerances_ms']}")
    lines.append(f"Symbol map             : {report['symbol_map']}")
    lines.append(f"CME root               : {report['cme_root']}")
    lines.append(f"MT5 root               : {report['mt5_root']}")
    lines.append("")
    for sym, m in report["per_symbol"].items():
        lines.append("-" * 80)
        lines.append(f"SYMBOL: {sym} -> {m.get('mt5_symbol')}")
        lines.append(f"  status              : {m.get('status')}")
        if m.get("status") not in ("OK", "NO_OVERLAP_WINDOW"):
            lines.append(f"  error               : {m.get('error', '-')}")
            continue
        lines.append(f"  cme_present         : {m.get('cme_present')}")
        lines.append(f"  mt5_present         : {m.get('mt5_present')}")
        lines.append(f"  cme_trades_total    : {m.get('cme_trades_total')}")
        lines.append(f"  cme_volume_total    : {m.get('cme_volume_total')}")
        lines.append(f"  aligned_count       : {m.get('aligned_count')}")
        mr = m.get("alignment_match_rate")
        uvr = m.get("unmatched_volume_ratio")
        lines.append(f"  match_rate          : {('-' if mr is None else f'{mr:.4f}')}")
        lines.append(f"  unmatched_volume_ratio: {('-' if uvr is None else f'{uvr:.4f}')}")
        lines.append(f"  unmatched_by_reason : {m.get('unmatched_count_by_reason')}")
        lines.append(f"  unmatched_vol_by_reason: {m.get('unmatched_volume_by_reason')}")
        lines.append(
            f"  unknown_aggressor   : {m.get('unknown_aggressor_count')} count, {m.get('unknown_aggressor_volume')} vol"
        )
        ad = m.get("alignment_delta_ms_summary") or {}
        if any((v is not None for v in ad.values())):
            lines.append(
                f"  alignment_delta_ms  : min={ad.get('min')} mean={ad.get('mean')} p50={ad.get('p50')} p95={ad.get('p95')} p99={ad.get('p99')} max={ad.get('max')}"
            )
        sp = m.get("matched_mt5_spread_price_summary") or {}
        if any((v is not None for v in sp.values())):
            lines.append(
                f"  mt5_spread_price    : min={sp.get('min')} mean={sp.get('mean')} p50={sp.get('p50')} p95={sp.get('p95')} p99={sp.get('p99')} max={sp.get('max')}"
            )
        diag = m.get("diagnostic_rates_by_tolerance_ms") or {}
        if diag:
            lines.append("  diagnostic rates:")
            for tol in sorted(diag, key=int):
                row = diag[tol]
                mr2 = row.get("match_rate")
                uvr2 = row.get("unmatched_volume_ratio")
                lines.append(
                    f"    tolerance={tol}ms  rate={('-' if mr2 is None else f'{mr2:.4f}')}  unmatched_vol_ratio={('-' if uvr2 is None else f'{uvr2:.4f}')}"
                )
        tw_cme = report["time_window_cme_by_symbol"].get(sym, {})
        tw_mt5 = report["time_window_mt5_by_symbol"].get(m.get("mt5_symbol"), {})
        ow = report["overlap_window_by_symbol"].get(sym, {})
        lines.append(f"  cme_window          : {tw_cme.get('min_utc')} -> {tw_cme.get('max_utc')}")
        lines.append(f"  mt5_window          : {tw_mt5.get('min_utc')} -> {tw_mt5.get('max_utc')}")
        lines.append(
            f"  overlap_window      : {ow.get('min_utc')} -> {ow.get('max_utc')} (has_overlap={ow.get('has_overlap')})"
        )
    lines.append("-" * 80)
    overall = report["overall"]
    lines.append("OVERALL")
    lines.append(f"  cme_trades_total    : {overall['cme_trades_total']}")
    mr = overall.get("alignment_match_rate")
    uvr = overall.get("unmatched_volume_ratio")
    lines.append(f"  alignment_match_rate: {('-' if mr is None else f'{mr:.4f}')}")
    lines.append(f"  unmatched_volume_ratio: {('-' if uvr is None else f'{uvr:.4f}')}")
    if report["warnings"]:
        lines.append("Warnings:")
        for w in report["warnings"]:
            lines.append(f"  - {w}")
    if report["errors"]:
        lines.append("Errors:")
        for w in report["errors"]:
            lines.append(f"  - {w}")
    lines.append("-" * 80)
    if (
        report["quality_decision"] == "FAIL"
        and report.get("decision_reason") == "MISSING_OVERLAP_DATA"
    ):
        lines.append(
            "NEXT: fetch a Databento GLBX.MDP3 mbp-1 sample whose UTC window overlaps the MT5 Silver window for this date, then re-run alignment_quality_report.py."
        )
    elif report["quality_decision"] == "FAIL" and report.get("decision_reason") in (
        "MISSING_CME_REFERENCE",
        "MISSING_MT5_SILVER",
    ):
        lines.append(
            f"NEXT: {report['decision_reason']} -- restore the missing dataset or rerun the upstream ingestion for this date."
        )
    elif report["quality_decision"] == "PARTIAL":
        lines.append(
            "NEXT: review per-symbol diagnostic rates above; consider whether the MT5 Silver join-safe threshold or the CME window choice should be revisited."
        )
    else:
        lines.append(
            "NEXT: alignment quality acceptable at the configured tolerance; downstream phases may rely on this as a join-safe baseline."
        )
    lines.append("=" * 80)
    return "\n".join(lines) + "\n"


def write_reports(
    config: AlignmentQualityConfig, report: dict, unmatched_sample: list[dict]
) -> tuple[Path, Path, Optional[Path]]:
    config.reports_root.mkdir(parents=True, exist_ok=True)
    json_path = config.reports_root / f"alignment_quality_{config.date}.json"
    txt_path = config.reports_root / f"alignment_quality_{config.date}.txt"
    json_tmp = json_path.with_suffix(".json.tmp")
    json_tmp.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
    os.replace(json_tmp, json_path)
    txt_tmp = txt_path.with_suffix(".txt.tmp")
    txt_tmp.write_text(render_text_report(report), encoding="utf-8")
    os.replace(txt_tmp, txt_path)
    sample_path: Optional[Path] = None
    if config.write_unmatched_sample and unmatched_sample:
        sample_path = config.reports_root / f"alignment_unmatched_sample_{config.date}.parquet"
        cols = {
            "cme_symbol": [r["cme_symbol"] for r in unmatched_sample],
            "mt5_symbol": [r["mt5_symbol"] for r in unmatched_sample],
            "cme_event_time_utc_ns": [r["cme_event_time_utc_ns"] for r in unmatched_sample],
            "cme_price": [r["cme_price"] for r in unmatched_sample],
            "cme_size": [r["cme_size"] for r in unmatched_sample],
            "cme_aggressor_side": [r["cme_aggressor_side"] for r in unmatched_sample],
            "reason": [r["reason"] for r in unmatched_sample],
        }
        schema = pa.schema(
            [
                ("cme_symbol", pa.string()),
                ("mt5_symbol", pa.string()),
                ("cme_event_time_utc_ns", pa.int64()),
                ("cme_price", pa.float64()),
                ("cme_size", pa.int64()),
                ("cme_aggressor_side", pa.string()),
                ("reason", pa.string()),
            ]
        )
        table = pa.Table.from_pydict(cols, schema=schema)
        tmp = sample_path.with_suffix(".parquet.tmp")
        pq.write_table(table, tmp, compression="zstd")
        os.replace(tmp, sample_path)
    return (json_path, txt_path, sample_path)


def parse_symbol_map(spec: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in spec.split(","):
        item = raw.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"symbol-map entry {item!r} must contain '='")
        k, v = item.split("=", 1)
        k, v = (k.strip(), v.strip())
        if not k or not v:
            raise ValueError(f"symbol-map entry {item!r} has empty key or value")
        out[k] = v
    if not out:
        raise ValueError("symbol-map must contain at least one entry")
    return out
