from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
import uuid
from dataclasses import asdict
from pathlib import Path

from polarix import __version__
from polarix.alignment.alignment_quality import (
    AlignmentQualityConfig,
    build_alignment_quality_report,
)
from polarix.features.bar_feature_builder import BuilderConfig, build
from polarix.features.dataset import export_dataset, summarize_dataset
from polarix.ingestion.cme_reference_ingest import IngestConfig, ingest
from polarix.ingestion.fixtures import FIXTURE_DATE, FixtureConfig, generate_bronze
from polarix.normalization.normalization import NormalizationConfig, normalize
from polarix.orchestration.artifacts import (
    artifact_record,
    canonical_json,
    publish_json,
    verify_artifacts,
)
from polarix.orchestration.run_metadata import resolve_run_metadata
from polarix.orchestration.runner import Stage, StageOutput, execute_stages, utc_now
from polarix.quality import bar_feature_quality, cme_reference_quality, telemetry_quality


class QualityGateError(RuntimeError):
    pass


def _require_pass(report: dict, key: str = "decision") -> None:
    if report.get(key) != "PASS":
        details = (
            report.get("fatal_warnings") or report.get("warnings") or report.get("decision_reason")
        )
        raise QualityGateError(f"{report.get(key, 'MISSING_DECISION')}: {details}")


def _code_revision() -> dict:
    root = Path(__file__).resolve().parents[3]
    try:
        revision = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return {"commit": revision.stdout.strip(), "dirty": bool(status.stdout.strip())}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def run_demo(output_root: Path, config: FixtureConfig, run_id: str | None = None) -> Path:
    run_id = f"demo-{uuid.uuid4().hex[:12]}" if run_id is None else run_id
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", run_id):
        raise ValueError("run-id must use 1-64 letters, digits, underscores or hyphens")
    if run_id.split(".")[0].upper() in {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *[f"COM{i}" for i in range(1, 10)],
        *[f"LPT{i}" for i in range(1, 10)],
    }:
        raise ValueError("run-id is a reserved filesystem name")
    run_dir = output_root.resolve() / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    bronze_mt5 = run_dir / "bronze" / "mt5"
    bronze_cme = run_dir / "bronze" / "cme"
    silver_mt5 = run_dir / "silver" / "mt5"
    silver_cme = run_dir / "silver" / "cme"
    reports = run_dir / "reports"
    gold = run_dir / "gold"
    dataset_root = run_dir / "dataset"
    fixture_metadata = run_dir / "bronze" / "logger_manifest.json"
    config_record = {**asdict(config), "date": FIXTURE_DATE, "bucket_sizes": ["15s", "60s"]}
    publish_json(
        run_dir / "metadata.json",
        {
            "manifest_version": "1.0",
            "run_id": run_id,
            "data_origin": "synthetic",
            "created_at_utc": utc_now(),
            "polarix_version": __version__,
            "code": _code_revision(),
            "python_version": platform.python_version(),
            "dependencies": {
                name: importlib.metadata.version(name)
                for name in ("duckdb", "polars", "pyarrow", "numpy")
            },
            "configuration": config_record,
            "configuration_sha256": hashlib.sha256(canonical_json(config_record)).hexdigest(),
        },
    )

    def ingestion_stage() -> StageOutput:
        paths = generate_bronze(run_dir, config)
        return StageOutput(paths, {"data_origin": "synthetic", "seed": config.seed})

    def normalization_stage() -> StageOutput:
        metadata = resolve_run_metadata(reports, metadata_path=fixture_metadata)
        normalize(
            NormalizationConfig(
                raw_root=bronze_mt5,
                silver_root=silver_mt5,
                date=FIXTURE_DATE,
                metadata=metadata,
            )
        )
        result = ingest(
            IngestConfig(
                input_root=bronze_cme,
                output_root=silver_cme,
                reports_root=reports,
                date=FIXTURE_DATE,
            )
        )
        if result.missing_sample or result.schema_error:
            raise QualityGateError(f"CME normalization failed: {result.schema_error}")
        paths = tuple(sorted((run_dir / "silver").rglob("*.*")))
        return StageOutput(paths, {"verified_offset_min": metadata.verified_offset_min})

    def quality_stage() -> StageOutput:
        metadata = resolve_run_metadata(reports, metadata_path=fixture_metadata)
        mt5_report = telemetry_quality.build_quality_report(
            telemetry_quality.QualityConfig(
                date=FIXTURE_DATE,
                raw_root=bronze_mt5,
                silver_root=silver_mt5,
                reports_root=reports,
                metadata=metadata,
            )
        )
        cme_report = cme_reference_quality.build_quality_report(
            cme_reference_quality.CmeQualityConfig(
                date=FIXTURE_DATE,
                input_root=bronze_cme / f"date={FIXTURE_DATE}",
                output_root=silver_cme,
                reports_root=reports,
            )
        )
        paths = (reports / "mt5_quality.json", reports / "cme_quality.json")
        for path, report in zip(paths, (mt5_report, cme_report), strict=True):
            publish_json(path, {"data_origin": "synthetic", **report})
        _require_pass(mt5_report)
        _require_pass(cme_report)
        return StageOutput(paths, {"mt5": mt5_report["decision"], "cme": cme_report["decision"]})

    def alignment_stage() -> StageOutput:
        report, _ = build_alignment_quality_report(
            AlignmentQualityConfig(
                date=FIXTURE_DATE,
                cme_root=silver_cme / "reference_trades",
                mt5_root=silver_mt5,
                reports_root=reports,
                write_unmatched_sample=False,
            )
        )
        path = reports / "alignment_quality.json"
        publish_json(path, {"data_origin": "synthetic", **report})
        _require_pass(report, "quality_decision")
        return StageOutput((path,), {"decision": report["quality_decision"], "tolerance_ms": 50})

    def features_stage() -> StageOutput:
        result = build(
            BuilderConfig(
                date=FIXTURE_DATE,
                cme_root=silver_cme / "reference_trades",
                mt5_root=silver_mt5,
                output_root=gold,
                reports_root=reports,
            )
        )
        if result.error or result.missing_input or not result.written_paths:
            raise QualityGateError(f"Feature generation failed: {result.error}")
        return StageOutput(
            tuple(sorted(gold.rglob("*.*"))), {"rows_by_pair_bucket": result.rows_by_pair_bucket}
        )

    def feature_quality_stage() -> StageOutput:
        report = bar_feature_quality.build_feature_quality_report(
            bar_feature_quality.QualityConfig(
                date=FIXTURE_DATE,
                features_root=gold,
                reports_root=reports,
            )
        )
        path = reports / "feature_quality.json"
        publish_json(path, {"data_origin": "synthetic", **report})
        _require_pass(report)
        return StageOutput(
            (path,),
            {"rows": report["total_rows"], "eligible_rows": report["total_model_eligible_rows"]},
        )

    def export_stage() -> StageOutput:
        paths = export_dataset(
            sorted(gold.rglob("*.parquet")),
            dataset_root,
            watermark_ns=config.end_ns,
            data_origin="synthetic",
        )
        return StageOutput(paths, {"summary": summarize_dataset(paths[0])})

    stages = (
        Stage("ingestion", (), ingestion_stage),
        Stage("normalization", ("ingestion",), normalization_stage),
        Stage("quality", ("normalization",), quality_stage),
        Stage("alignment", ("quality",), alignment_stage),
        Stage("aggregation_features", ("alignment",), features_stage),
        Stage("feature_quality", ("aggregation_features",), feature_quality_stage),
        Stage("export", ("feature_quality",), export_stage),
    )
    stage_records = execute_stages(run_dir, stages)
    artifacts = [
        artifact_record(path, run_dir) for path in sorted(run_dir.rglob("*")) if path.is_file()
    ]
    publish_json(
        run_dir / "run_manifest.json",
        {
            "manifest_version": "1.0",
            "run_id": run_id,
            "data_origin": "synthetic",
            "status": "PASS"
            if all(record["status"] == "PASS" for record in stage_records)
            else "FAIL",
            "finished_at_utc": utc_now(),
            "stages": stage_records,
            "artifacts": artifacts,
        },
    )
    return run_dir


def inspect_run(run_dir: Path) -> dict:
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    verify_artifacts(run_dir, manifest["artifacts"])
    result = {
        "run_id": manifest["run_id"],
        "status": manifest["status"],
        "data_origin": manifest["data_origin"],
        "integrity": "VERIFIED",
        "artifact_count": len(manifest["artifacts"]),
        "stages": [
            {
                key: value
                for key, value in stage.items()
                if key in ("name", "status", "duration_seconds", "reason", "error", "blocked_by")
            }
            for stage in manifest["stages"]
        ],
        "feature_tables": [],
    }
    if manifest["status"] == "PASS":
        result["feature_tables"] = summarize_dataset(run_dir / "dataset" / "model_features.parquet")
    return result
