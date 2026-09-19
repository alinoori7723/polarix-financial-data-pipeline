from __future__ import annotations

import datetime as _dt
import glob
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import polars as pl

from polarix.features.bar_feature_contract import (
    FEATURE_QUALITY_DIAGNOSTIC_ONLY,
    FORBIDDEN_COLUMNS,
    FeatureContract,
    assert_contract_invariants,
)

REQUIRED_SYMBOL_PAIRS = ("ES_SPX500", "NQ_NDX100")
REQUIRED_BUCKET_LABELS = ("15s", "60s")


@dataclass
class QualityConfig:
    date: str
    features_root: Path
    reports_root: Path
    required_pairs: tuple[str, ...] = REQUIRED_SYMBOL_PAIRS
    required_buckets: tuple[str, ...] = REQUIRED_BUCKET_LABELS

    def __post_init__(self) -> None:
        self.features_root = Path(self.features_root).resolve()
        self.reports_root = Path(self.reports_root).resolve()


def _percentile(values: list[float], pct: float) -> Optional[float]:
    arr = sorted(
        (
            float(v)
            for v in values
            if v is not None and (not (isinstance(v, float) and math.isnan(v)))
        )
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


def _summary(values: Iterable[float]) -> dict[str, Optional[float]]:
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


def _read_manifest(features_root: Path, date: str) -> Optional[dict]:
    path = features_root / f"date={date}" / "bar_features_manifest.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _list_part_files(features_root: Path, pair: str, date: str, bucket: str) -> list[Path]:
    pattern = os.path.join(
        str(features_root),
        f"symbol_pair={pair}",
        f"date={date}",
        f"bucket={bucket}",
        "part-*.parquet",
    )
    return sorted((Path(p) for p in glob.glob(pattern)))


def _read_pair_bucket(
    features_root: Path, pair: str, date: str, bucket: str
) -> Optional[pl.DataFrame]:
    files = _list_part_files(features_root, pair, date, bucket)
    if not files:
        return None
    return pl.concat([pl.read_parquet(p) for p in files], how="vertical_relaxed")


def _verify_readable(files: list[Path]) -> tuple[bool, Optional[str]]:
    if not files:
        return (False, "no parquet files")
    try:
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


def build_feature_quality_report(config: QualityConfig) -> dict:
    warnings: list[str] = []
    errors: list[str] = []
    manifest = _read_manifest(config.features_root, config.date)
    if manifest is None:
        return {
            "date": config.date,
            "features_root": str(config.features_root),
            "reports_root": str(config.reports_root),
            "decision": "FAIL",
            "decision_reason": "MISSING_INPUT_DATA",
            "errors": ["bar_features_manifest.json not found"],
            "warnings": [],
            "per_pair": {},
            "generated_at_utc": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
        }
    contract = FeatureContract(builder_version=manifest.get("builder_version", "unknown"))
    try:
        assert_contract_invariants(contract)
    except Exception as exc:
        errors.append(f"contract invariants failed: {exc}")
    per_pair: dict[str, dict] = {}
    decision = "PASS"
    decision_reason: Optional[str] = None
    pair_rows_total = 0
    pair_eligible_total = 0
    for pair in sorted(
        set(list(config.required_pairs) + list(manifest.get("rows_by_pair_bucket") or {}))
    ):
        pair_entry: dict = {"by_bucket": {}}
        pair_rows = 0
        pair_eligible = 0
        for bucket in sorted(
            set(
                list(config.required_buckets)
                + list((manifest.get("rows_by_pair_bucket") or {}).get(pair, {}))
            )
        ):
            files = _list_part_files(config.features_root, pair, config.date, bucket)
            readable, read_err = _verify_readable(files)
            bucket_entry: dict = {
                "files": [str(p) for p in files],
                "readable": readable,
                "read_error": read_err,
                "total_rows": 0,
                "model_eligible_rows": 0,
                "diagnostic_only_rows": 0,
                "reject_reason_counts": {},
                "feature_quality_counts": {},
                "column_null_rate": {},
                "column_finite_rate": {},
            }
            if not files:
                pair_entry["by_bucket"][bucket] = bucket_entry
                continue
            if not readable:
                errors.append(f"{pair}/{bucket}: output not readable ({read_err})")
                pair_entry["by_bucket"][bucket] = bucket_entry
                continue
            df = _read_pair_bucket(config.features_root, pair, config.date, bucket)
            assert df is not None
            bucket_entry["total_rows"] = df.height
            pair_rows += df.height
            present_forbidden = [c for c in FORBIDDEN_COLUMNS if c in df.columns]
            if present_forbidden:
                errors.append(f"{pair}/{bucket}: forbidden columns present: {present_forbidden}")
            for col in (
                "cme_open_price",
                "cme_close_price",
                "cme_high_price",
                "cme_low_price",
                "cme_vwap",
                "mt5_mid_open",
                "mt5_mid_close",
                "mt5_mid_high",
                "mt5_mid_low",
                "mt5_mid_twap",
                "mt5_mid_mean",
            ):
                if col in contract.model_feature_candidate_columns:
                    errors.append(
                        f"contract violation: absolute-price column {col!r} listed as model feature"
                    )
            if "feature_quality_flag" in df.columns:
                grp = df.group_by("feature_quality_flag").agg(pl.len().alias("n"))
                bucket_entry["feature_quality_counts"] = {
                    row["feature_quality_flag"]: int(row["n"]) for row in grp.to_dicts()
                }
            if "bar_reject_reason" in df.columns:
                grp = (
                    df.filter(pl.col("bar_reject_reason").is_not_null())
                    .group_by("bar_reject_reason")
                    .agg(pl.len().alias("n"))
                )
                bucket_entry["reject_reason_counts"] = {
                    row["bar_reject_reason"]: int(row["n"]) for row in grp.to_dicts()
                }
            if "is_model_eligible_candidate" in df.columns:
                bucket_entry["model_eligible_rows"] = int(
                    df["is_model_eligible_candidate"].cast(pl.Int64).sum() or 0
                )
                pair_eligible += bucket_entry["model_eligible_rows"]
            if "feature_quality_flag" in df.columns:
                bucket_entry["diagnostic_only_rows"] = int(
                    (df["feature_quality_flag"] == FEATURE_QUALITY_DIAGNOSTIC_ONLY)
                    .cast(pl.Int64)
                    .sum()
                    or 0
                )
            candidate_cols = [
                c for c in contract.model_feature_candidate_columns if c in df.columns
            ]
            null_rate = {}
            finite_rate = {}
            for col in candidate_cols:
                total = df.height
                if total == 0:
                    null_rate[col] = None
                    finite_rate[col] = None
                    continue
                nulls = int(df[col].null_count())
                null_rate[col] = nulls / total
                if df[col].dtype in (pl.Float64, pl.Float32):
                    non_null = df.filter(pl.col(col).is_not_null())[col]
                    finite_count = (
                        int(non_null.is_finite().cast(pl.Int64).sum() or 0) if non_null.len() else 0
                    )
                    finite_rate[col] = finite_count / total
                else:
                    finite_rate[col] = (total - nulls) / total
            bucket_entry["column_null_rate"] = null_rate
            bucket_entry["column_finite_rate"] = finite_rate
            key_summaries: dict[str, dict] = {}
            for col in (
                "basis_close_bps",
                "basis_change_bps",
                "return_diff_close_to_close",
                "cme_signed_volume_ratio",
                "cme_high_low_range_bps",
                "mt5_spread_price_p95",
                "mt5_spread_price_max",
            ):
                if col in df.columns:
                    key_summaries[col] = _summary(df[col].to_list())
            bucket_entry["key_summary_stats"] = key_summaries
            pair_entry["by_bucket"][bucket] = bucket_entry
        pair_entry["total_rows"] = pair_rows
        pair_entry["model_eligible_rows"] = pair_eligible
        per_pair[pair] = pair_entry
        pair_rows_total += pair_rows
        pair_eligible_total += pair_eligible
    if errors:
        decision = "FAIL"
        decision_reason = "CONTRACT_OR_OUTPUT_FAILURE"
    elif not any((per_pair[p]["total_rows"] for p in config.required_pairs if p in per_pair)):
        decision = "FAIL"
        decision_reason = "MISSING_INPUT_DATA"
    elif not all((per_pair.get(p, {}).get("total_rows") for p in config.required_pairs)):
        decision = "PARTIAL"
        warnings.append(
            f"one or more required pairs missing rows: {[p for p in config.required_pairs if not per_pair.get(p, {}).get('total_rows')]}"
        )
    else:
        all_pairs_eligible = True
        for p in config.required_pairs:
            pair_info = per_pair.get(p, {})
            buckets = pair_info.get("by_bucket", {})
            has_eligible_in_req_bucket = any(
                (
                    buckets.get(b, {}).get("model_eligible_rows", 0) > 0
                    for b in config.required_buckets
                )
            )
            if not has_eligible_in_req_bucket:
                all_pairs_eligible = False
                warnings.append(
                    f"pair {p} has no model-eligible rows at required buckets {config.required_buckets}"
                )
        decision = "PASS" if all_pairs_eligible else "PARTIAL"
    return {
        "date": config.date,
        "features_root": str(config.features_root),
        "reports_root": str(config.reports_root),
        "manifest": manifest,
        "feature_contract": contract.to_dict(),
        "per_pair": per_pair,
        "total_rows": pair_rows_total,
        "total_model_eligible_rows": pair_eligible_total,
        "decision": decision,
        "decision_reason": decision_reason,
        "errors": errors,
        "warnings": warnings
        + [
            "z-scores are sample-local; do NOT claim production stationarity from a single date's sample."
        ],
        "generated_at_utc": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
    }


def render_text_report(report: dict) -> str:
    lines: list[str] = []
    lines.append("=" * 80)
    lines.append(f"Polarix Bar-Feature Quality Report  --  date={report['date']}")
    lines.append("=" * 80)
    lines.append(f"Decision                : {report['decision']}")
    if report.get("decision_reason"):
        lines.append(f"Decision reason         : {report['decision_reason']}")
    lines.append(f"Features root           : {report['features_root']}")
    lines.append(f"Total rows              : {report.get('total_rows', 0)}")
    lines.append(f"Model eligible rows     : {report.get('total_model_eligible_rows', 0)}")
    lines.append("")
    for pair, pe in report["per_pair"].items():
        lines.append("-" * 80)
        lines.append(f"PAIR: {pair}")
        lines.append(f"  total_rows           : {pe.get('total_rows', 0)}")
        lines.append(f"  model_eligible_rows  : {pe.get('model_eligible_rows', 0)}")
        for bucket, be in (pe.get("by_bucket") or {}).items():
            lines.append(f"  bucket {bucket}:")
            lines.append(f"    files               : {len(be['files'])}")
            lines.append(f"    readable            : {be['readable']} ({be['read_error']})")
            lines.append(f"    total_rows          : {be['total_rows']}")
            lines.append(f"    model_eligible_rows : {be['model_eligible_rows']}")
            lines.append(f"    diagnostic_only_rows: {be['diagnostic_only_rows']}")
            qc = be.get("feature_quality_counts") or {}
            lines.append(f"    feature_quality_flag: {qc}")
            rc = be.get("reject_reason_counts") or {}
            if rc:
                lines.append(f"    reject_reasons      : {rc}")
            null_rates = be.get("column_null_rate") or {}
            top_null = sorted(null_rates.items(), key=lambda kv: -(kv[1] or 0))[:5]
            lines.append(f"    top null-rate cols  : {top_null}")
            for col, s in (be.get("key_summary_stats") or {}).items():
                lines.append(
                    f"    {col:32s}: min={s['min']} mean={s['mean']} p50={s['p50']} p95={s['p95']} p99={s['p99']} max={s['max']}"
                )
    lines.append("-" * 80)
    if report["warnings"]:
        lines.append("Warnings:")
        for w in report["warnings"]:
            lines.append(f"  - {w}")
    if report["errors"]:
        lines.append("Errors:")
        for w in report["errors"]:
            lines.append(f"  - {w}")
    lines.append("-" * 80)
    if report["decision"] == "PASS":
        lines.append(
            "NEXT: Phase 2D feature contract ACCEPTED. The Gold candidate feature table is ready for future feature design / EDA. This report does NOT authorize trading, model training, or CVD aggregation."
        )
    elif report["decision"] == "PARTIAL":
        lines.append(
            "NEXT: feature output exists but model-eligible coverage is insufficient at the required bucket sizes. Inspect per-pair metrics and consider whether more data is needed before shipping the Gold table as a baseline."
        )
    else:
        lines.append(
            "NEXT: decision FAIL -- review the errors and rerun build_bar_features.py once inputs / contract are restored."
        )
    lines.append("=" * 80)
    return "\n".join(lines) + "\n"


def write_reports(config: QualityConfig, report: dict) -> tuple[Path, Path]:
    config.reports_root.mkdir(parents=True, exist_ok=True)
    json_path = config.reports_root / f"bar_feature_quality_{config.date}.json"
    txt_path = config.reports_root / f"bar_feature_quality_{config.date}.txt"
    json_tmp = json_path.with_suffix(".json.tmp")
    json_tmp.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
    os.replace(json_tmp, json_path)
    txt_tmp = txt_path.with_suffix(".txt.tmp")
    txt_tmp.write_text(render_text_report(report), encoding="utf-8")
    os.replace(txt_tmp, txt_path)
    return (json_path, txt_path)
