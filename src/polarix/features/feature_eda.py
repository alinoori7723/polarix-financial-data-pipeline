from __future__ import annotations

import datetime as _dt
import glob
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import polars as pl

from polarix.features.feature_correlation import (
    DEFAULT_PVALUE_ALPHA,
    DEFAULT_SMALL_SAMPLE_MIN_ROWS,
    compute_correlation_summary,
    scipy_available,
)
from polarix.features.feature_statistics import (
    DEFAULT_HEAVY_TAIL_THRESHOLD,
    compute_distribution_summary,
    compute_intra_sample_stability,
    compute_missingness_summary,
)

EDA_VERSION = "0.1.0"
DEFAULT_SYMBOL_PAIRS = ("ES_SPX500", "NQ_NDX100")
DEFAULT_BUCKET_SIZES = ("15s", "60s")
DEFAULT_TOP_N = 15
DECISION_REASON_MISSING_FEATURE_DATA = "MISSING_FEATURE_DATA"
DECISION_REASON_MISSING_FEATURE_MANIFEST = "MISSING_FEATURE_MANIFEST"
DECISION_REASON_CONTRACT_OR_OUTPUT_FAILURE = "CONTRACT_OR_OUTPUT_FAILURE"
DECISION_REASON_INSUFFICIENT_ELIGIBLE_ROWS = "INSUFFICIENT_ELIGIBLE_ROWS"
TINY_SAMPLE_NON_PREDICTIVE_WARNING = "EDA is a pipeline sanity check only; not predictive evidence."
ABSOLUTE_PRICE_COLUMNS = (
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
)
FORBIDDEN_COLUMN_NAMES = (
    "label",
    "target",
    "y",
    "y_true",
    "future_return",
    "forward_return",
    "return_t_plus_1",
    "outcome",
    "cvd",
    "cvd_cumulative",
    "cvd_running",
    "cvd_total",
    "cme_max_1s_volume_share",
    "cme_max_1s_signed_volume_share",
    "decision_lag_ms",
)


class FeatureEDAError(RuntimeError):
    pass


@dataclass
class FeatureEDAConfig:
    date: str
    features_root: Path
    reports_root: Path
    symbol_pairs: tuple[str, ...] = DEFAULT_SYMBOL_PAIRS
    bucket_sizes: tuple[str, ...] = DEFAULT_BUCKET_SIZES
    include_diagnostic_5s: bool = False
    max_correlation_features: Optional[int] = None
    correlation_method: str = "pearson"
    small_sample_min_rows: int = DEFAULT_SMALL_SAMPLE_MIN_ROWS
    pvalue_alpha: float = DEFAULT_PVALUE_ALPHA
    heavy_tail_threshold: float = DEFAULT_HEAVY_TAIL_THRESHOLD
    top_n: int = DEFAULT_TOP_N
    force: bool = False
    dry_run: bool = False

    def __post_init__(self) -> None:
        self.features_root = Path(self.features_root).resolve()
        self.reports_root = Path(self.reports_root).resolve()
        if self.correlation_method != "pearson":
            raise FeatureEDAError(
                f"correlation_method={self.correlation_method!r} unsupported in Phase 2E; only 'pearson' is allowed."
            )


def _read_manifest(features_root: Path, date: str) -> Optional[dict]:
    path = features_root / f"date={date}" / "bar_features_manifest.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _list_parts(features_root: Path, pair: str, date: str, bucket: str) -> list[Path]:
    pattern = os.path.join(
        str(features_root),
        f"symbol_pair={pair}",
        f"date={date}",
        f"bucket={bucket}",
        "part-*.parquet",
    )
    return sorted((Path(p) for p in glob.glob(pattern)))


def _load_pair_bucket(
    features_root: Path, pair: str, date: str, bucket: str
) -> Optional[pl.DataFrame]:
    files = _list_parts(features_root, pair, date, bucket)
    if not files:
        return None
    return pl.concat([pl.read_parquet(p) for p in files], how="vertical_relaxed")


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


def _validate_manifest_contract(manifest: dict) -> list[str]:
    errors: list[str] = []
    contract = manifest.get("feature_contract") or {}
    candidates = set(contract.get("model_feature_candidate_columns") or [])
    for col in ABSOLUTE_PRICE_COLUMNS:
        if col in candidates:
            errors.append(
                f"contract violation: absolute-price column {col!r} listed as model_feature_candidate"
            )
    for col in FORBIDDEN_COLUMN_NAMES:
        if col in candidates:
            errors.append(
                f"contract violation: forbidden column {col!r} listed as model_feature_candidate"
            )
    if not candidates:
        errors.append("contract has no model_feature_candidate_columns")
    return errors


def _build_overview(
    config: FeatureEDAConfig,
    manifest: dict,
    rows_by_pair_bucket_eligible: dict[str, dict[str, int]],
    rows_by_pair_bucket_diag: dict[str, dict[str, int]],
    rows_by_pair_bucket_excluded: dict[str, dict[str, int]],
    feature_roles: dict[str, list[str]],
) -> dict:
    total_eligible = sum((sum(b.values()) for b in rows_by_pair_bucket_eligible.values()))
    total_diag = sum((sum(b.values()) for b in rows_by_pair_bucket_diag.values()))
    total_excluded = sum((sum(b.values()) for b in rows_by_pair_bucket_excluded.values()))
    small_sample_flag = total_eligible < config.small_sample_min_rows
    return {
        "date": config.date,
        "features_root": str(config.features_root),
        "reports_root": str(config.reports_root),
        "symbol_pairs": list(config.symbol_pairs),
        "bucket_sizes": list(config.bucket_sizes),
        "include_diagnostic_5s": config.include_diagnostic_5s,
        "small_sample_min_rows": config.small_sample_min_rows,
        "small_sample_flag": small_sample_flag,
        "total_eligible_rows": total_eligible,
        "total_diagnostic_only_rows": total_diag,
        "total_excluded_rows": total_excluded,
        "rows_by_symbol_pair_bucket_eligible": rows_by_pair_bucket_eligible,
        "rows_by_symbol_pair_bucket_diagnostic": rows_by_pair_bucket_diag,
        "rows_by_symbol_pair_bucket_excluded": rows_by_pair_bucket_excluded,
        "feature_manifest_path": str(
            config.features_root / f"date={config.date}" / "bar_features_manifest.json"
        ),
        "model_feature_candidate_count": len(feature_roles["model_feature_candidate_columns"]),
        "diagnostic_feature_count": len(feature_roles["diagnostic_feature_columns"]),
        "non_feature_column_count": len(feature_roles["non_feature_columns"]),
        "quality_column_count": len(feature_roles["quality_columns"]),
        "identity_column_count": len(feature_roles["identity_columns"]),
        "scipy_available": scipy_available(),
    }


@dataclass
class _LoadedData:
    eligible: Optional[pl.DataFrame]
    diagnostic: Optional[pl.DataFrame]
    rows_by_pair_bucket_eligible: dict[str, dict[str, int]]
    rows_by_pair_bucket_diag: dict[str, dict[str, int]]
    rows_by_pair_bucket_excluded: dict[str, dict[str, int]]
    feature_files_present: bool = False
    feature_file_count: int = 0


def _load_filtered(config: FeatureEDAConfig) -> _LoadedData:
    eligible_frames: list[pl.DataFrame] = []
    diagnostic_frames: list[pl.DataFrame] = []
    rows_eligible: dict[str, dict[str, int]] = {}
    rows_diag: dict[str, dict[str, int]] = {}
    rows_excluded: dict[str, dict[str, int]] = {}
    feature_file_count = 0
    buckets_to_read: list[str] = list(config.bucket_sizes)
    if config.include_diagnostic_5s and "5s" not in buckets_to_read:
        buckets_to_read = ["5s"] + buckets_to_read
    for pair in config.symbol_pairs:
        rows_eligible[pair] = {}
        rows_diag[pair] = {}
        rows_excluded[pair] = {}
        for bucket in buckets_to_read:
            files = _list_parts(config.features_root, pair, config.date, bucket)
            feature_file_count += len(files)
            df = _load_pair_bucket(config.features_root, pair, config.date, bucket)
            if df is None:
                continue
            if "is_model_eligible_candidate" not in df.columns:
                raise FeatureEDAError(
                    f"feature parquet for {pair}/{bucket} missing 'is_model_eligible_candidate' -- not a Phase 2D output"
                )
            ok = df.filter(pl.col("is_model_eligible_candidate"))
            diag = (
                df.filter(pl.col("feature_quality_flag") == "DIAGNOSTIC_ONLY")
                if "feature_quality_flag" in df.columns
                else pl.DataFrame()
            )
            excluded = df.height - ok.height - diag.height
            rows_eligible[pair][bucket] = ok.height
            rows_diag[pair][bucket] = diag.height
            rows_excluded[pair][bucket] = excluded
            if ok.height:
                eligible_frames.append(ok)
            if diag.height:
                diagnostic_frames.append(diag)
    eligible = pl.concat(eligible_frames, how="vertical_relaxed") if eligible_frames else None
    diagnostic = pl.concat(diagnostic_frames, how="vertical_relaxed") if diagnostic_frames else None
    return _LoadedData(
        eligible=eligible,
        diagnostic=diagnostic,
        rows_by_pair_bucket_eligible=rows_eligible,
        rows_by_pair_bucket_diag=rows_diag,
        rows_by_pair_bucket_excluded=rows_excluded,
        feature_files_present=feature_file_count > 0,
        feature_file_count=feature_file_count,
    )


@dataclass
class EDAResult:
    config: FeatureEDAConfig
    report: dict
    missingness_summary: Optional[pl.DataFrame]
    distribution_summary: Optional[pl.DataFrame]
    correlation_summary: Optional[pl.DataFrame]
    stability_summary: Optional[pl.DataFrame]
    paths: dict


def run_eda(config: FeatureEDAConfig) -> EDAResult:
    started_at_utc = _dt.datetime.now(tz=_dt.timezone.utc).isoformat()
    errors: list[str] = []
    warnings: list[str] = []
    manifest = _read_manifest(config.features_root, config.date)
    if manifest is None:
        return EDAResult(
            config=config,
            report={
                "date": config.date,
                "features_root": str(config.features_root),
                "reports_root": str(config.reports_root),
                "quality_decision": "FAIL",
                "decision_reason": "MISSING_FEATURE_MANIFEST",
                "errors": [
                    f"feature manifest not found at {config.features_root / f'date={config.date}' / 'bar_features_manifest.json'}"
                ],
                "warnings": [],
                "generated_at_utc": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
            },
            missingness_summary=None,
            distribution_summary=None,
            correlation_summary=None,
            stability_summary=None,
            paths={},
        )
    contract_errors = _validate_manifest_contract(manifest)
    errors.extend(contract_errors)
    contract = manifest.get("feature_contract") or {}
    feature_roles = {
        "identity_columns": list(contract.get("identity_columns") or []),
        "quality_columns": list(contract.get("quality_columns") or []),
        "non_feature_columns": list(contract.get("non_feature_columns") or []),
        "diagnostic_feature_columns": list(contract.get("diagnostic_feature_columns") or []),
        "model_feature_candidate_columns": list(
            contract.get("model_feature_candidate_columns") or []
        ),
    }
    model_features = feature_roles["model_feature_candidate_columns"]
    diagnostic_features = feature_roles["diagnostic_feature_columns"]
    if not model_features:
        errors.append("manifest has no model_feature_candidate_columns")
    try:
        loaded = _load_filtered(config)
    except FeatureEDAError as exc:
        return EDAResult(
            config=config,
            report={
                "date": config.date,
                "features_root": str(config.features_root),
                "reports_root": str(config.reports_root),
                "quality_decision": "FAIL",
                "decision_reason": DECISION_REASON_CONTRACT_OR_OUTPUT_FAILURE,
                "errors": [str(exc)],
                "warnings": [],
                "generated_at_utc": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
            },
            missingness_summary=None,
            distribution_summary=None,
            correlation_summary=None,
            stability_summary=None,
            paths={},
        )
    overview = _build_overview(
        config,
        manifest,
        loaded.rows_by_pair_bucket_eligible,
        loaded.rows_by_pair_bucket_diag,
        loaded.rows_by_pair_bucket_excluded,
        feature_roles,
    )
    overview["feature_files_present"] = loaded.feature_files_present
    overview["feature_file_count"] = loaded.feature_file_count
    if not loaded.feature_files_present:
        return EDAResult(
            config=config,
            report={
                **overview,
                "quality_decision": "FAIL",
                "decision_reason": DECISION_REASON_MISSING_FEATURE_DATA,
                "errors": errors
                + [
                    f"no Gold feature Parquet files found under {config.features_root}/symbol_pair=*/date={config.date}/"
                ],
                "warnings": warnings,
                "feature_roles": feature_roles,
                "generated_at_utc": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
            },
            missingness_summary=None,
            distribution_summary=None,
            correlation_summary=None,
            stability_summary=None,
            paths={},
        )
    main_df = loaded.eligible if loaded.eligible is not None else pl.DataFrame()
    missingness_summary = compute_missingness_summary(main_df, feature_columns=model_features)
    distribution_summary = compute_distribution_summary(
        main_df,
        feature_columns=model_features,
        heavy_tail_threshold=config.heavy_tail_threshold,
        small_sample_min_rows=config.small_sample_min_rows,
    )
    correlation_summary = compute_correlation_summary(
        main_df,
        feature_columns=model_features,
        pvalue_alpha=config.pvalue_alpha,
        small_sample_min_rows=config.small_sample_min_rows,
        max_features=config.max_correlation_features,
    )
    stability_summary = compute_intra_sample_stability(main_df, feature_columns=model_features)
    diagnostic_block: dict = {}
    if (
        config.include_diagnostic_5s
        and loaded.diagnostic is not None
        and (loaded.diagnostic.height > 0)
    ):
        diag_dist = compute_distribution_summary(
            loaded.diagnostic,
            feature_columns=model_features + diagnostic_features,
            heavy_tail_threshold=config.heavy_tail_threshold,
            small_sample_min_rows=config.small_sample_min_rows,
        )
        diagnostic_block = {
            "rows": loaded.diagnostic.height,
            "distribution_summary_rows": diag_dist.height,
            "distribution_summary": diag_dist.to_dicts()[: config.top_n * 4],
        }
    eligible_cols = set(main_df.columns)
    for col in ABSOLUTE_PRICE_COLUMNS:
        if col in model_features:
            errors.append(
                f"runtime violation: absolute-price column {col!r} declared as model_feature_candidate"
            )
    for col in FORBIDDEN_COLUMN_NAMES:
        if col in eligible_cols:
            errors.append(f"runtime violation: forbidden column {col!r} present in feature data")
    forbidden_column_check = {
        "absolute_prices_in_candidates": [c for c in ABSOLUTE_PRICE_COLUMNS if c in model_features],
        "forbidden_columns_present": [c for c in FORBIDDEN_COLUMN_NAMES if c in eligible_cols],
    }
    data_mutation_check = {
        "rows_loaded": int(main_df.height),
        "columns_loaded": list(main_df.columns),
        "transformations_applied": [],
    }
    top_missing = (
        missingness_summary.sort("null_rate", descending=True, nulls_last=True)
        .head(config.top_n)
        .to_dicts()
    )
    top_heavy = (
        distribution_summary.filter(pl.col("heavy_tail_score").is_not_null())
        .sort("heavy_tail_score", descending=True, nulls_last=True)
        .head(config.top_n)
        .to_dicts()
    )
    top_abs_corr = (
        correlation_summary.filter(pl.col("abs_correlation").is_not_null())
        .sort("abs_correlation", descending=True, nulls_last=True)
        .head(config.top_n)
        .to_dicts()
    )
    stat_sig_count = int(
        correlation_summary["is_statistically_significant"].cast(pl.Int64).sum() or 0
    )
    small_sample_corr_count = int(
        correlation_summary["is_small_sample_warning"].cast(pl.Int64).sum() or 0
    )
    decision: str
    decision_reason: Optional[str] = None
    total_eligible = overview["total_eligible_rows"]
    if errors:
        decision = "FAIL"
        decision_reason = DECISION_REASON_CONTRACT_OR_OUTPUT_FAILURE
    elif total_eligible == 0:
        decision = "PARTIAL"
        decision_reason = DECISION_REASON_INSUFFICIENT_ELIGIBLE_ROWS
        warnings.append(
            f"total_eligible_rows=0 (with {overview['total_excluded_rows']} excluded by the quality filter); feature files are present but no row passed is_model_eligible_candidate. {TINY_SAMPLE_NON_PREDICTIVE_WARNING}"
        )
    elif total_eligible < config.small_sample_min_rows:
        decision = "PARTIAL"
        decision_reason = DECISION_REASON_INSUFFICIENT_ELIGIBLE_ROWS
        warnings.append(
            f"total_eligible_rows={total_eligible} is below small_sample_min_rows={config.small_sample_min_rows}; {TINY_SAMPLE_NON_PREDICTIVE_WARNING}"
        )
    else:
        pairs_with_eligible = [
            p
            for p in config.symbol_pairs
            if sum(loaded.rows_by_pair_bucket_eligible.get(p, {}).values()) > 0
        ]
        if len(pairs_with_eligible) < len(config.symbol_pairs):
            decision = "PARTIAL"
            warnings.append(
                f"missing eligible rows for symbol_pairs: {[p for p in config.symbol_pairs if p not in pairs_with_eligible]}"
            )
        else:
            decision = "PASS"
    if not scipy_available():
        warnings.append(
            "scipy is not installed in this environment; correlation p-values are unavailable (pvalue_available=false). Statistical significance cannot be evaluated. Install scipy to enable p-values."
        )
    paths: dict = {}
    if not config.dry_run:
        config.reports_root.mkdir(parents=True, exist_ok=True)
        miss_path = config.reports_root / f"feature_missingness_summary_{config.date}.parquet"
        dist_path = config.reports_root / f"feature_distribution_summary_{config.date}.parquet"
        corr_path = config.reports_root / f"feature_correlation_summary_{config.date}.parquet"
        if (miss_path.exists() or dist_path.exists() or corr_path.exists()) and (not config.force):
            raise FeatureEDAError(
                "EDA outputs already exist for this date; pass --force to overwrite"
            )
        for path, df in (
            (miss_path, missingness_summary),
            (dist_path, distribution_summary),
            (corr_path, correlation_summary),
        ):
            tmp = path.with_suffix(".parquet.tmp")
            df.write_parquet(tmp, compression="zstd")
            os.replace(tmp, path)
        paths.update(
            {
                "missingness_summary_path": str(miss_path),
                "distribution_summary_path": str(dist_path),
                "correlation_summary_path": str(corr_path),
            }
        )
    report = {
        **overview,
        "feature_roles": feature_roles,
        "quality_decision": decision,
        "decision_reason": decision_reason,
        "errors": errors,
        "warnings": warnings,
        "top_missing_features": top_missing,
        "top_heavy_tail_features": top_heavy,
        "top_abs_correlations": top_abs_corr,
        "statistically_significant_correlations_count": stat_sig_count,
        "small_sample_correlations_count": small_sample_corr_count,
        "forbidden_column_check": forbidden_column_check,
        "data_mutation_check": data_mutation_check,
        "diagnostic_block": diagnostic_block,
        "pvalue_alpha": config.pvalue_alpha,
        "scipy_available": scipy_available(),
        "code_version": _git_hash(),
        "eda_version": EDA_VERSION,
        "started_at_utc": started_at_utc,
        "generated_at_utc": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
        "missingness_summary_path": paths.get("missingness_summary_path"),
        "distribution_summary_path": paths.get("distribution_summary_path"),
        "correlation_summary_path": paths.get("correlation_summary_path"),
    }
    return EDAResult(
        config=config,
        report=report,
        missingness_summary=missingness_summary,
        distribution_summary=distribution_summary,
        correlation_summary=correlation_summary,
        stability_summary=stability_summary,
        paths=paths,
    )


def render_text_report(report: dict) -> str:
    lines: list[str] = []
    lines.append("=" * 80)
    lines.append(f"Polarix Feature EDA Report  --  date={report['date']}")
    lines.append("=" * 80)
    lines.append(f"Decision                  : {report['quality_decision']}")
    if report.get("decision_reason"):
        lines.append(f"Decision reason           : {report['decision_reason']}")
    lines.append(f"Features root             : {report['features_root']}")
    lines.append(f"Symbol pairs              : {report.get('symbol_pairs')}")
    lines.append(f"Bucket sizes              : {report.get('bucket_sizes')}")
    lines.append(f"Include diagnostic 5s     : {report.get('include_diagnostic_5s')}")
    lines.append(f"Eligible rows             : {report.get('total_eligible_rows')}")
    lines.append(f"Diagnostic-only rows      : {report.get('total_diagnostic_only_rows')}")
    lines.append(f"Excluded rows             : {report.get('total_excluded_rows')}")
    lines.append(f"Small-sample threshold    : {report.get('small_sample_min_rows')}")
    lines.append(f"Small-sample flag         : {report.get('small_sample_flag')}")
    lines.append(f"scipy available           : {report.get('scipy_available')}")
    lines.append(f"pvalue_alpha              : {report.get('pvalue_alpha')}")
    lines.append(
        f"sig correlations count    : {report.get('statistically_significant_correlations_count')}"
    )
    lines.append(f"small-sample corr count   : {report.get('small_sample_correlations_count')}")
    lines.append("")
    lines.append("-" * 80)
    lines.append("ROWS BY (symbol_pair, bucket)  -- eligible / diagnostic / excluded")
    e = report.get("rows_by_symbol_pair_bucket_eligible") or {}
    d = report.get("rows_by_symbol_pair_bucket_diagnostic") or {}
    x = report.get("rows_by_symbol_pair_bucket_excluded") or {}
    pairs = sorted(set(list(e) + list(d) + list(x)))
    for p in pairs:
        buckets = sorted(set(list(e.get(p, {})) + list(d.get(p, {})) + list(x.get(p, {}))))
        for b in buckets:
            lines.append(
                f"  {p} / {b}:  eligible={e.get(p, {}).get(b, 0)}  diagnostic={d.get(p, {}).get(b, 0)}  excluded={x.get(p, {}).get(b, 0)}"
            )
    lines.append("-" * 80)
    lines.append("FEATURE ROLES")
    roles = report.get("feature_roles") or {}
    lines.append(f"  identity_columns          : {len(roles.get('identity_columns') or [])}")
    lines.append(f"  quality_columns           : {len(roles.get('quality_columns') or [])}")
    lines.append(f"  non_feature_columns       : {len(roles.get('non_feature_columns') or [])}")
    lines.append(
        f"  diagnostic_feature_cols   : {len(roles.get('diagnostic_feature_columns') or [])}"
    )
    lines.append(
        f"  model_feature_candidates  : {len(roles.get('model_feature_candidate_columns') or [])}"
    )
    lines.append("-" * 80)
    lines.append("TOP NULL-RATE FEATURES (eligible rows)")
    for row in (report.get("top_missing_features") or [])[:10]:
        lines.append(
            f"  [{row.get('symbol_pair')} / {row.get('bucket_size')}] {row.get('feature_column'):42s}  null_rate={row.get('null_rate')}  finite_rate={row.get('finite_rate')}"
        )
    lines.append("-" * 80)
    lines.append("TOP HEAVY-TAIL FEATURES (read-only diagnostic)")
    for row in (report.get("top_heavy_tail_features") or [])[:10]:
        lines.append(
            f"  [{row.get('symbol_pair')} / {row.get('bucket_size')}] {row.get('feature_column'):42s}  heavy_tail_score={row.get('heavy_tail_score')}  p99={row.get('p99')}  outlier_rate_iqr={row.get('outlier_rate_iqr')}"
        )
    lines.append("-" * 80)
    lines.append("TOP ABSOLUTE CORRELATIONS (NOT predictive evidence)")
    for row in (report.get("top_abs_correlations") or [])[:10]:
        lines.append(
            f"  [{row.get('symbol_pair')} / {row.get('bucket_size')}] {row.get('feature_a')} <-> {row.get('feature_b')}  r={row.get('correlation')}  p={row.get('p_value')}  n={row.get('n_obs')}  small_sample={row.get('is_small_sample_warning')}"
        )
    lines.append("-" * 80)
    if report.get("warnings"):
        lines.append("Warnings:")
        for w in report["warnings"]:
            lines.append(f"  - {w}")
    if report.get("errors"):
        lines.append("Errors:")
        for w in report["errors"]:
            lines.append(f"  - {w}")
    lines.append("")
    lines.append(
        "NOTE: This EDA is READ-ONLY. No winsorization, clipping, imputation, or normalization was applied to the Gold dataset. Heavy-tail flags are diagnostic only. Correlation values from a small sample are a pipeline sanity check, NOT predictive evidence."
    )
    if report["quality_decision"] == "FAIL":
        lines.append(f"NEXT: {report.get('decision_reason')} -- inspect errors above.")
    elif report["quality_decision"] == "PARTIAL":
        lines.append(
            "NEXT: pipeline works; collect more dates / sessions before treating this output as a stationary baseline."
        )
    else:
        lines.append(
            "NEXT: dataset is large enough for EDA, but predictive use still requires out-of-sample validation in a later phase."
        )
    lines.append("=" * 80)
    return "\n".join(lines) + "\n"


def write_reports(config: FeatureEDAConfig, result: EDAResult) -> tuple[Path, Path]:
    config.reports_root.mkdir(parents=True, exist_ok=True)
    json_path = config.reports_root / f"feature_eda_{config.date}.json"
    txt_path = config.reports_root / f"feature_eda_{config.date}.txt"
    if (json_path.exists() or txt_path.exists()) and (not config.force):
        raise FeatureEDAError("EDA reports already exist for this date; pass --force to overwrite")
    json_tmp = json_path.with_suffix(".json.tmp")
    json_tmp.write_text(
        json.dumps(result.report, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    os.replace(json_tmp, json_path)
    txt_tmp = txt_path.with_suffix(".txt.tmp")
    txt_tmp.write_text(render_text_report(result.report), encoding="utf-8")
    os.replace(txt_tmp, txt_path)
    return (json_path, txt_path)


def parse_symbol_pairs(spec: str) -> tuple[str, ...]:
    items = tuple((s.strip() for s in spec.split(",") if s.strip()))
    if not items:
        raise ValueError("symbol-pairs must contain at least one entry")
    return items


def parse_bucket_size_labels(spec: str) -> tuple[str, ...]:
    items = tuple((s.strip() for s in spec.split(",") if s.strip()))
    if not items:
        raise ValueError("bucket-sizes must contain at least one entry")
    return items
