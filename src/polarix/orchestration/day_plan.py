from __future__ import annotations

import datetime as _dt
import glob
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

DEFAULT_SYMBOL_MAP: Mapping[str, str] = {"ES": "SPX500", "NQ": "NDX100"}
DEFAULT_PRE_ROLL_MINUTES = 5
DEFAULT_POST_ROLL_MINUTES = 5


def _mt5_files_for(mt5_root: Path, symbol: str, date: str) -> list[Path]:
    pattern = os.path.join(str(mt5_root), f"symbol={symbol}", f"date={date}", "part-*.parquet")
    return sorted((Path(p) for p in glob.glob(pattern)))


def mt5_window_ms(mt5_root: Path, mt5_symbols: list[str], date: str) -> Optional[tuple[int, int]]:
    import polars as pl

    mn: Optional[int] = None
    mx: Optional[int] = None
    for sym in mt5_symbols:
        for f in _mt5_files_for(mt5_root, sym, date):
            try:
                col = pl.read_parquet(f, columns=["time_msc_utc_ms"])["time_msc_utc_ms"]
            except Exception:
                continue
            cmin = col.min()
            cmax = col.max()
            if cmin is None or cmax is None:
                continue
            cmin = int(cmin)
            cmax = int(cmax)
            mn = cmin if mn is None or cmin < mn else mn
            mx = cmax if mx is None or cmax > mx else mx
    if mn is None or mx is None:
        return None
    return (mn, mx)


def _ms_to_iso(ms: Optional[int]) -> Optional[str]:
    if ms is None:
        return None
    return _dt.datetime.fromtimestamp(ms / 1000.0, tz=_dt.timezone.utc).isoformat()


@dataclass
class DayPlan:
    date: str
    mt5_symbols_present: list[str]
    mt5_window_ms: Optional[tuple[int, int]]
    mt5_min_utc: Optional[str]
    mt5_max_utc: Optional[str]
    recommended_cme_start_utc: Optional[str]
    recommended_cme_end_utc: Optional[str]
    cme_raw_files: list[str]
    cme_normalized_files: list[str]
    gold_feature_files: list[str]
    eda_report_path: Optional[str]
    bar_alignment_report_path: Optional[str]
    estimated_local_disk_bytes: int

    def to_dict(self) -> dict:
        return {
            "date": self.date,
            "mt5_symbols_present": list(self.mt5_symbols_present),
            "mt5_window_ms": list(self.mt5_window_ms) if self.mt5_window_ms else None,
            "mt5_min_utc": self.mt5_min_utc,
            "mt5_max_utc": self.mt5_max_utc,
            "recommended_cme_start_utc": self.recommended_cme_start_utc,
            "recommended_cme_end_utc": self.recommended_cme_end_utc,
            "cme_raw_files": list(self.cme_raw_files),
            "cme_normalized_files": list(self.cme_normalized_files),
            "gold_feature_files": list(self.gold_feature_files),
            "eda_report_path": self.eda_report_path,
            "bar_alignment_report_path": self.bar_alignment_report_path,
            "estimated_local_disk_bytes": self.estimated_local_disk_bytes,
        }


def list_available_dates(mt5_root: Path, mt5_symbols: list[str]) -> list[str]:
    dates: set[str] = set()
    for sym in mt5_symbols:
        sym_dir = Path(mt5_root) / f"symbol={sym}"
        if not sym_dir.exists():
            continue
        for d in sym_dir.iterdir():
            if d.is_dir() and d.name.startswith("date="):
                dates.add(d.name.split("=", 1)[1])
    return sorted(dates)


def _file_size_sum(paths: list[str]) -> int:
    total = 0
    for p in paths:
        try:
            total += Path(p).stat().st_size
        except OSError:
            continue
    return total


def _shift_iso(iso: str, *, minutes: int) -> str:
    t = _dt.datetime.fromisoformat(iso).astimezone(_dt.timezone.utc)
    return (t + _dt.timedelta(minutes=minutes)).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_day_plan(
    date: str,
    *,
    data_root: Path,
    reports_root: Path,
    symbol_map: Mapping[str, str] = DEFAULT_SYMBOL_MAP,
    pre_roll_minutes: int = DEFAULT_PRE_ROLL_MINUTES,
    post_roll_minutes: int = DEFAULT_POST_ROLL_MINUTES,
) -> DayPlan:
    data_root = Path(data_root)
    reports_root = Path(reports_root)
    mt5_root = data_root / "normalized" / "mt5_ticks"
    mt5_symbols = list(symbol_map.values())
    cme_symbols = list(symbol_map.keys())
    present_syms: list[str] = []
    for sym in mt5_symbols:
        if _mt5_files_for(mt5_root, sym, date):
            present_syms.append(sym)
    win = mt5_window_ms(mt5_root, mt5_symbols, date)
    if win is not None:
        mn_ms, mx_ms = win
        start_iso = _shift_iso(_ms_to_iso(mn_ms), minutes=-pre_roll_minutes)
        end_iso = _shift_iso(_ms_to_iso(mx_ms), minutes=post_roll_minutes)
    else:
        mn_ms = mx_ms = None
        start_iso = end_iso = None
    cme_raw_dir = data_root / "raw" / "cme_sample" / f"date={date}"
    cme_raw_files = (
        sorted((str(p) for p in cme_raw_dir.glob("**/*.parquet"))) if cme_raw_dir.exists() else []
    )
    cme_norm_root = data_root / "normalized" / "cme_reference" / "reference_trades"
    cme_norm_files: list[str] = []
    for sym in cme_symbols:
        cme_norm_files.extend(
            (
                str(p)
                for p in (cme_norm_root / f"symbol={sym}" / f"date={date}").glob("part-*.parquet")
            )
        )
    cme_norm_files.sort()
    gold_root = data_root / "features" / "bar_features"
    gold_files: list[str] = []
    for cme_sym, mt5_sym in symbol_map.items():
        pair = f"{cme_sym}_{mt5_sym}"
        pair_date_dir = gold_root / f"symbol_pair={pair}" / f"date={date}"
        if pair_date_dir.exists():
            for bucket_dir in pair_date_dir.iterdir():
                if bucket_dir.is_dir() and bucket_dir.name.startswith("bucket="):
                    gold_files.extend((str(p) for p in bucket_dir.glob("part-*.parquet")))
    gold_files.sort()
    eda_path = reports_root / f"feature_eda_{date}.json"
    bar_align_path = reports_root / f"bar_alignment_quality_{date}.json"
    return DayPlan(
        date=date,
        mt5_symbols_present=present_syms,
        mt5_window_ms=(mn_ms, mx_ms) if win else None,
        mt5_min_utc=_ms_to_iso(mn_ms) if win else None,
        mt5_max_utc=_ms_to_iso(mx_ms) if win else None,
        recommended_cme_start_utc=start_iso,
        recommended_cme_end_utc=end_iso,
        cme_raw_files=cme_raw_files,
        cme_normalized_files=cme_norm_files,
        gold_feature_files=gold_files,
        eda_report_path=str(eda_path) if eda_path.exists() else None,
        bar_alignment_report_path=str(bar_align_path) if bar_align_path.exists() else None,
        estimated_local_disk_bytes=_file_size_sum(cme_raw_files)
        + _file_size_sum(cme_norm_files)
        + _file_size_sum(gold_files),
    )


def build_multiday_plan(
    *,
    data_root: Path,
    reports_root: Path,
    dates: Optional[list[str]] = None,
    symbol_map: Mapping[str, str] = DEFAULT_SYMBOL_MAP,
    pre_roll_minutes: int = DEFAULT_PRE_ROLL_MINUTES,
    post_roll_minutes: int = DEFAULT_POST_ROLL_MINUTES,
) -> dict:
    data_root = Path(data_root)
    reports_root = Path(reports_root)
    mt5_root = data_root / "normalized" / "mt5_ticks"
    mt5_symbols = list(symbol_map.values())
    if dates is None:
        dates = list_available_dates(mt5_root, mt5_symbols)
    plans = [
        build_day_plan(
            d,
            data_root=data_root,
            reports_root=reports_root,
            symbol_map=symbol_map,
            pre_roll_minutes=pre_roll_minutes,
            post_roll_minutes=post_roll_minutes,
        )
        for d in dates
    ]
    overall = {
        "dates_requested": list(dates),
        "data_root": str(data_root.resolve()),
        "reports_root": str(reports_root.resolve()),
        "symbol_map": dict(symbol_map),
        "pre_roll_minutes": pre_roll_minutes,
        "post_roll_minutes": post_roll_minutes,
        "per_day": [p.to_dict() for p in plans],
        "summary": {
            "total_dates": len(plans),
            "dates_with_mt5_silver": sum((1 for p in plans if p.mt5_window_ms)),
            "dates_with_cme_raw": sum((1 for p in plans if p.cme_raw_files)),
            "dates_with_cme_normalized": sum((1 for p in plans if p.cme_normalized_files)),
            "dates_with_gold_features": sum((1 for p in plans if p.gold_feature_files)),
            "dates_with_eda_report": sum((1 for p in plans if p.eda_report_path)),
            "estimated_local_disk_bytes_total": sum((p.estimated_local_disk_bytes for p in plans)),
        },
        "generated_at_utc": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
        "this_tool": "Read-only planner; does NOT call Databento. No API key is required. It only inspects locally-available Parquet/JSON.",
    }
    return overall


def render_multiday_plan_text(plan: dict) -> str:
    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("Polarix Multi-Day Plan")
    lines.append("=" * 80)
    lines.append(f"data_root        : {plan['data_root']}")
    lines.append(f"reports_root     : {plan['reports_root']}")
    lines.append(f"symbol_map       : {plan['symbol_map']}")
    lines.append(f"pre/post roll min: {plan['pre_roll_minutes']} / {plan['post_roll_minutes']}")
    summary = plan["summary"]
    lines.append(
        f"summary          : total={summary['total_dates']}  with_mt5={summary['dates_with_mt5_silver']}  with_cme_raw={summary['dates_with_cme_raw']}  with_cme_norm={summary['dates_with_cme_normalized']}  with_gold={summary['dates_with_gold_features']}  with_eda={summary['dates_with_eda_report']}"
    )
    lines.append(f"local disk bytes : {summary['estimated_local_disk_bytes_total']}")
    lines.append("-" * 80)
    for day in plan["per_day"]:
        lines.append(f"DATE {day['date']}")
        lines.append(f"  mt5_symbols_present     : {day['mt5_symbols_present']}")
        lines.append(f"  mt5_window_utc          : {day['mt5_min_utc']}  ->  {day['mt5_max_utc']}")
        lines.append(
            f"  recommended_cme_window  : {day['recommended_cme_start_utc']}  ->  {day['recommended_cme_end_utc']}"
        )
        lines.append(f"  cme_raw_files           : {len(day['cme_raw_files'])}")
        lines.append(f"  cme_normalized_files    : {len(day['cme_normalized_files'])}")
        lines.append(f"  gold_feature_files      : {len(day['gold_feature_files'])}")
        lines.append(f"  eda_report_path         : {day['eda_report_path']}")
        lines.append(f"  bar_alignment_report    : {day['bar_alignment_report_path']}")
        lines.append(f"  estimated_local_bytes   : {day['estimated_local_disk_bytes']}")
    lines.append("=" * 80)
    return "\n".join(lines) + "\n"
