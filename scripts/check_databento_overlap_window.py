from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path


def _add_src_to_path() -> None:
    here = Path(__file__).resolve()
    src = here.parents[1] / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


_add_src_to_path()
from polarix.alignment.alignment_quality import DEFAULT_SYMBOL_MAP, parse_symbol_map
from polarix.normalization import silver_paths

DEFAULT_MT5_ROOT = Path(".polarix/data/normalized/mt5_ticks")


def _mt5_window_ms(files: list[Path]) -> tuple[int | None, int | None]:
    import pyarrow.parquet as pq

    if not files:
        return (None, None)
    mn: int | None = None
    mx: int | None = None
    for f in files:
        t = pq.ParquetFile(f).read(columns=["time_msc_utc_ms"])
        vals = t.column("time_msc_utc_ms").to_pylist()
        for v in vals:
            if v is None:
                continue
            mn = v if mn is None or v < mn else mn
            mx = v if mx is None or v > mx else mx
    return (mn, mx)


def _ms_to_iso(ms: int | None) -> str | None:
    if ms is None:
        return None
    return _dt.datetime.fromtimestamp(ms / 1000, tz=_dt.timezone.utc).isoformat()


def _recommend_window(
    mn_ms: int | None, mx_ms: int | None, pre_min: int, post_min: int
) -> tuple[str | None, str | None]:
    if mn_ms is None or mx_ms is None:
        return (None, None)
    start_ms = mn_ms - pre_min * 60000
    end_ms = mx_ms + post_min * 60000
    return (_ms_to_iso(start_ms), _ms_to_iso(end_ms))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inspect MT5 Silver windows for Databento planning")
    p.add_argument("--date", required=True, help="YYYY-MM-DD")
    p.add_argument("--mt5-root", default=str(DEFAULT_MT5_ROOT))
    p.add_argument(
        "--run-id",
        default=None,
        help="read ONLY the run-scoped MT5 Silver partition symbol=<S>/date=<DATE>/run_id=<RUN_ID>/. Without --run-id the planner fails closed when multiple run partitions exist for the date (it must not recommend a mixed-run window).",
    )
    p.add_argument(
        "--symbol-map",
        default=",".join((f"{k}={v}" for k, v in DEFAULT_SYMBOL_MAP.items())),
        help="Comma-separated CME=MT5 pairs (default ES=SPX500,NQ=NDX100)",
    )
    p.add_argument(
        "--pre-roll-minutes",
        type=int,
        default=5,
        help="Minutes to pad before MT5 start when recommending a CME window.",
    )
    p.add_argument(
        "--post-roll-minutes",
        type=int,
        default=5,
        help="Minutes to pad after MT5 end when recommending a CME window.",
    )
    p.add_argument(
        "--previous-error",
        default=None,
        help="Optional verbatim Databento error returned on the last attempt; this script will echo it but not act on it.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        symbol_map = parse_symbol_map(args.symbol_map)
    except ValueError as exc:
        print(f"[check_databento_overlap_window] bad --symbol-map: {exc}", file=sys.stderr)
        return 2
    mt5_root = Path(args.mt5_root)
    mt5_windows: dict[str, dict] = {}
    cme_recommendations: dict[str, dict] = {}
    mt5_symbols = list(symbol_map.values())
    layout_notes: list[str] = []
    if args.run_id is None:
        date_run_ids = silver_paths.list_date_run_ids(mt5_root, mt5_symbols, args.date)
        if len(date_run_ids) > 1:
            print(
                json.dumps(
                    {
                        "date": args.date,
                        "mt5_root": str(mt5_root.resolve()),
                        "error": "AMBIGUOUS_MULTIPLE_RUN_PARTITIONS",
                        "available_run_ids": date_run_ids,
                        "message": f"multiple run-scoped MT5 Silver partitions exist for date={args.date}: {date_run_ids}. Refusing to recommend a Databento window from mixed-run Silver. Re-run with --run-id <RUN_ID> to inspect a single run.",
                        "this_tool": "Read-only planner; does NOT call Databento. Fails closed on cross-run ambiguity.",
                    },
                    indent=2,
                )
            )
            return 2
    selected_run_ids: set[str] = set()
    silver_layout_versions: set[str] = set()
    for cme_symbol, mt5_symbol in symbol_map.items():
        selection = silver_paths.select_silver(mt5_root, mt5_symbol, args.date, args.run_id)
        files = selection.files
        silver_layout_versions.add(selection.layout_version)
        if selection.selected_run_id:
            selected_run_ids.add(selection.selected_run_id)
        for note in selection.warnings:
            layout_notes.append(note)
        if not files:
            mt5_windows[mt5_symbol] = {"files": 0, "min_utc": None, "max_utc": None}
            cme_recommendations[cme_symbol] = {
                "mt5_symbol": mt5_symbol,
                "recommended_start_utc": None,
                "recommended_end_utc": None,
                "note": f"no MT5 Silver Parquet found for symbol={mt5_symbol} date={args.date}; cannot recommend a CME window.",
            }
            continue
        mn_ms, mx_ms = _mt5_window_ms(files)
        mt5_windows[mt5_symbol] = {
            "files": len(files),
            "min_ms": mn_ms,
            "max_ms": mx_ms,
            "min_utc": _ms_to_iso(mn_ms),
            "max_utc": _ms_to_iso(mx_ms),
        }
        rec_start, rec_end = _recommend_window(
            mn_ms, mx_ms, args.pre_roll_minutes, args.post_roll_minutes
        )
        cme_recommendations[cme_symbol] = {
            "mt5_symbol": mt5_symbol,
            "recommended_start_utc": rec_start,
            "recommended_end_utc": rec_end,
            "pre_roll_minutes": args.pre_roll_minutes,
            "post_roll_minutes": args.post_roll_minutes,
            "note": f"Use this window for the Databento GLBX.MDP3 mbp-1 historical request so the resulting CME sample overlaps the MT5 Silver capture for {mt5_symbol}.",
        }
    report_selected_run_id = args.run_id or (
        next(iter(selected_run_ids)) if len(selected_run_ids) == 1 else None
    )
    layout_version = (
        silver_paths.SILVER_LAYOUT_VERSION_RUN_SCOPED
        if silver_paths.SILVER_LAYOUT_VERSION_RUN_SCOPED in silver_layout_versions
        else silver_paths.SILVER_LAYOUT_VERSION_LEGACY
    )
    out: dict = {
        "date": args.date,
        "mt5_root": str(mt5_root.resolve()),
        "symbol_map": dict(symbol_map),
        "requested_run_id": args.run_id,
        "selected_run_id": report_selected_run_id,
        "silver_layout_version": layout_version,
        "silver_layout_notes": layout_notes,
        "mt5_silver_windows": mt5_windows,
        "cme_databento_recommendations": cme_recommendations,
        "this_tool": "Read-only planner; does NOT call Databento. No API key is required. It only inspects locally-available MT5 Silver Parquet.",
    }
    if args.previous_error:
        out["previous_databento_error_echoed"] = args.previous_error
        out["operator_note"] = (
            "Previous error was echoed verbatim. If your Databento subscription does not yet cover the recommended window, retry later or provide an explicitly licensed sample."
        )
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
