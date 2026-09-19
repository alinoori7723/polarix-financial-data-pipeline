from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
from pathlib import Path


def _add_src_to_path() -> None:
    here = Path(__file__).resolve()
    src = here.parents[1] / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


_add_src_to_path()
from polarix.orchestration.day_plan import (
    DEFAULT_POST_ROLL_MINUTES,
    DEFAULT_PRE_ROLL_MINUTES,
    DEFAULT_SYMBOL_MAP,
    build_multiday_plan,
    render_multiday_plan_text,
)

DEFAULT_DATA_ROOT = Path(".polarix/data")
DEFAULT_REPORTS_ROOT = Path(".polarix/reports")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Polarix multi-day plan (read-only)")
    p.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    p.add_argument("--reports-root", default=str(DEFAULT_REPORTS_ROOT))
    p.add_argument(
        "--dates",
        default=None,
        help="Comma-separated YYYY-MM-DD; default: all dates found under MT5 Silver",
    )
    p.add_argument(
        "--symbol-map", default=",".join((f"{k}={v}" for k, v in DEFAULT_SYMBOL_MAP.items()))
    )
    p.add_argument("--pre-roll-minutes", type=int, default=DEFAULT_PRE_ROLL_MINUTES)
    p.add_argument("--post-roll-minutes", type=int, default=DEFAULT_POST_ROLL_MINUTES)
    return p.parse_args(argv)


def _parse_symbol_map(spec: str) -> dict[str, str]:
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


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        symbol_map = _parse_symbol_map(args.symbol_map)
    except ValueError as exc:
        print(f"[build_multiday_plan] bad --symbol-map: {exc}", file=sys.stderr)
        return 2
    dates = None
    if args.dates:
        dates = [s.strip() for s in args.dates.split(",") if s.strip()]
        if not dates:
            print("[build_multiday_plan] --dates parsed empty", file=sys.stderr)
            return 2
    plan = build_multiday_plan(
        data_root=Path(args.data_root),
        reports_root=Path(args.reports_root),
        dates=dates,
        symbol_map=symbol_map,
        pre_roll_minutes=args.pre_roll_minutes,
        post_roll_minutes=args.post_roll_minutes,
    )
    timestamp = _dt.datetime.now(tz=_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    reports = Path(args.reports_root)
    reports.mkdir(parents=True, exist_ok=True)
    jp = reports / f"multiday_plan_{timestamp}.json"
    tp = reports / f"multiday_plan_{timestamp}.txt"
    jtmp = jp.with_suffix(".json.tmp")
    jtmp.write_text(json.dumps(plan, indent=2, sort_keys=True, default=str), encoding="utf-8")
    os.replace(jtmp, jp)
    ttmp = tp.with_suffix(".txt.tmp")
    ttmp.write_text(render_multiday_plan_text(plan), encoding="utf-8")
    os.replace(ttmp, tp)
    print(
        json.dumps(
            {
                "status": "OK",
                "json_plan": str(jp),
                "txt_plan": str(tp),
                "dates_listed": len(plan["per_day"]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
