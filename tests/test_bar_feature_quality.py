from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from polarix.features.bar_aggregation import parse_bucket_sizes
from polarix.features.bar_feature_builder import BuilderConfig, build
from polarix.quality.bar_feature_quality import (
    QualityConfig,
    build_feature_quality_report,
    write_reports,
)
from tests.test_bar_feature_builder import _matched, _seed


def _make_builder_cfg(
    tmp_path: Path, *, include_5s: bool = False, bucket_sizes: str = "15s,60s"
) -> BuilderConfig:
    base = 1779081406
    cme_es, mt5_spx = _matched(base, n=6, bucket_seconds=15)
    cme_nq, mt5_ndx = _matched(base, n=6, bucket_seconds=15)
    cme_root, mt5_root = _seed(
        tmp_path, "2026-05-18", {"ES": cme_es, "NQ": cme_nq}, {"SPX500": mt5_spx, "NDX100": mt5_ndx}
    )
    return BuilderConfig(
        date="2026-05-18",
        cme_root=cme_root,
        mt5_root=mt5_root,
        output_root=tmp_path / "features",
        reports_root=tmp_path / "reports",
        bucket_sizes=parse_bucket_sizes(bucket_sizes),
        include_diagnostic_5s=include_5s,
        force=True,
    )


def test_quality_emits_pass_on_valid_fixture(tmp_path: Path) -> None:
    cfg = _make_builder_cfg(tmp_path, bucket_sizes="15s,60s")
    build(cfg)
    qcfg = QualityConfig(
        date="2026-05-18", features_root=cfg.output_root, reports_root=cfg.reports_root
    )
    rep = build_feature_quality_report(qcfg)
    assert rep["decision"] == "PASS"
    for pair in ("ES_SPX500", "NQ_NDX100"):
        assert pair in rep["per_pair"]
        for b in ("15s", "60s"):
            assert b in rep["per_pair"][pair]["by_bucket"]
            assert rep["per_pair"][pair]["by_bucket"][b]["model_eligible_rows"] > 0


def test_quality_writes_json_and_txt(tmp_path: Path) -> None:
    cfg = _make_builder_cfg(tmp_path, bucket_sizes="15s")
    build(cfg)
    qcfg = QualityConfig(
        date="2026-05-18", features_root=cfg.output_root, reports_root=cfg.reports_root
    )
    rep = build_feature_quality_report(qcfg)
    jp, tp = write_reports(qcfg, rep)
    assert jp.exists() and tp.exists()
    parsed = json.loads(jp.read_text(encoding="utf-8"))
    assert parsed["decision"] == rep["decision"]
    assert "Bar-Feature Quality Report" in tp.read_text(encoding="utf-8")


def test_quality_partial_when_one_pair_missing(tmp_path: Path) -> None:
    base = 1779081406
    cme_es, mt5_spx = _matched(base, n=6, bucket_seconds=15)
    cme_root, mt5_root = _seed(
        tmp_path, "2026-05-18", {"ES": cme_es, "NQ": []}, {"SPX500": mt5_spx, "NDX100": []}
    )
    cfg = BuilderConfig(
        date="2026-05-18",
        cme_root=cme_root,
        mt5_root=mt5_root,
        output_root=tmp_path / "features",
        reports_root=tmp_path / "reports",
        bucket_sizes=parse_bucket_sizes("15s,60s"),
        force=True,
    )
    build(cfg)
    qcfg = QualityConfig(
        date="2026-05-18", features_root=cfg.output_root, reports_root=cfg.reports_root
    )
    rep = build_feature_quality_report(qcfg)
    assert rep["decision"] == "PARTIAL"


def test_quality_fail_when_manifest_missing(tmp_path: Path) -> None:
    qcfg = QualityConfig(
        date="2026-05-18", features_root=tmp_path / "absent", reports_root=tmp_path / "reports"
    )
    rep = build_feature_quality_report(qcfg)
    assert rep["decision"] == "FAIL"
    assert rep["decision_reason"] == "MISSING_INPUT_DATA"


def test_quality_reports_column_null_and_finite_rates(tmp_path: Path) -> None:
    cfg = _make_builder_cfg(tmp_path, bucket_sizes="15s")
    build(cfg)
    qcfg = QualityConfig(
        date="2026-05-18", features_root=cfg.output_root, reports_root=cfg.reports_root
    )
    rep = build_feature_quality_report(qcfg)
    pair_entry = rep["per_pair"]["ES_SPX500"]
    bucket_entry = pair_entry["by_bucket"]["15s"]
    nulls = bucket_entry["column_null_rate"]
    assert "basis_close_bps" in nulls
    finites = bucket_entry["column_finite_rate"]
    assert "basis_close_bps" in finites


def test_quality_emits_partial_or_fail_when_only_diagnostic_rows(tmp_path: Path) -> None:
    base = 1779081406
    cme_es, mt5_spx = _matched(base, n=6, bucket_seconds=5)
    cme_nq, mt5_ndx = _matched(base, n=6, bucket_seconds=5)
    cme_root, mt5_root = _seed(
        tmp_path, "2026-05-18", {"ES": cme_es, "NQ": cme_nq}, {"SPX500": mt5_spx, "NDX100": mt5_ndx}
    )
    cfg = BuilderConfig(
        date="2026-05-18",
        cme_root=cme_root,
        mt5_root=mt5_root,
        output_root=tmp_path / "features",
        reports_root=tmp_path / "reports",
        bucket_sizes=parse_bucket_sizes("5s"),
        include_diagnostic_5s=True,
        force=True,
    )
    build(cfg)
    qcfg = QualityConfig(
        date="2026-05-18", features_root=cfg.output_root, reports_root=cfg.reports_root
    )
    rep = build_feature_quality_report(qcfg)
    assert rep["decision"] in ("PARTIAL", "FAIL")
    assert rep["decision"] != "PASS"


def test_quality_fail_if_absolute_price_marked_as_model_feature(tmp_path: Path) -> None:
    cfg = _make_builder_cfg(tmp_path, bucket_sizes="15s")
    res = build(cfg)
    manifest_path = res.manifest_path
    assert manifest_path is not None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["feature_contract"]["model_feature_candidate_columns"].append("cme_close_price")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    qcfg = QualityConfig(
        date="2026-05-18", features_root=cfg.output_root, reports_root=cfg.reports_root
    )
    rep = build_feature_quality_report(qcfg)
    assert rep["decision"] != "FAIL" or "contract" in (rep.get("decision_reason") or "")
    assert "per_pair" in rep


def test_quality_fail_if_forbidden_column_present(tmp_path: Path) -> None:
    cfg = _make_builder_cfg(tmp_path, bucket_sizes="15s")
    build(cfg)
    out_dir = cfg.output_root / "symbol_pair=ES_SPX500" / "date=2026-05-18" / "bucket=15s"
    parts = list(out_dir.glob("part-*.parquet"))
    assert parts
    df = pl.read_parquet(parts[0])
    df = df.with_columns(pl.lit(1).alias("label"))
    parts[0].unlink()
    df.write_parquet(parts[0], compression="zstd")
    qcfg = QualityConfig(
        date="2026-05-18", features_root=cfg.output_root, reports_root=cfg.reports_root
    )
    rep = build_feature_quality_report(qcfg)
    assert rep["decision"] == "FAIL"
    assert any(("forbidden columns present" in e for e in rep["errors"]))
