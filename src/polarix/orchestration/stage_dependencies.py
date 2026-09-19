from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional

from polarix.ingestion import cme_downloader as cme_dl
from polarix.orchestration import pipeline_status as ps

ARTIFACT_CME_NORMALIZED = "CME_NORMALIZED"
ARTIFACT_CME_QUALITY_REPORT = "CME_QUALITY_REPORT"
ARTIFACT_ALIGNMENT_REPORT = "ALIGNMENT_REPORT"
ARTIFACT_BAR_ALIGNMENT_REPORT = "BAR_ALIGNMENT_REPORT"
ARTIFACT_GOLD_FEATURES = "GOLD_FEATURES"
ARTIFACT_BAR_FEATURE_QUALITY_REPORT = "BAR_FEATURE_QUALITY_REPORT"
ARTIFACT_EDA_REPORT = "EDA_REPORT"


@dataclass(frozen=True)
class StageSpec:
    name: str
    script: str
    base_args: tuple[str, ...]
    depends_on: tuple[str, ...]
    is_required: bool
    supports_force: bool
    artifact_kind: str

    def date_argv_tail(self, date: str, *, with_force: bool) -> list[str]:
        argv: list[str] = ["--date", date, *self.base_args]
        if with_force and self.supports_force:
            argv.append("--force")
        return argv


SYNTHETIC_CME_DOWNLOAD = "CME_DOWNLOAD"
DEFAULT_STAGE_GRAPH: tuple[StageSpec, ...] = (
    StageSpec(
        name="CME_INGEST",
        script="ingest_cme_reference_sample.py",
        base_args=(),
        depends_on=(SYNTHETIC_CME_DOWNLOAD,),
        is_required=True,
        supports_force=True,
        artifact_kind=ARTIFACT_CME_NORMALIZED,
    ),
    StageSpec(
        name="CME_QUALITY",
        script="cme_reference_quality_report.py",
        base_args=(),
        depends_on=("CME_INGEST",),
        is_required=True,
        supports_force=False,
        artifact_kind=ARTIFACT_CME_QUALITY_REPORT,
    ),
    StageSpec(
        name="ALIGNMENT_QUALITY",
        script="alignment_quality_report.py",
        base_args=(),
        depends_on=("CME_QUALITY",),
        is_required=True,
        supports_force=True,
        artifact_kind=ARTIFACT_ALIGNMENT_REPORT,
    ),
    StageSpec(
        name="BAR_ALIGNMENT",
        script="bar_alignment_quality_report.py",
        base_args=("--write-buckets",),
        depends_on=("ALIGNMENT_QUALITY",),
        is_required=True,
        supports_force=True,
        artifact_kind=ARTIFACT_BAR_ALIGNMENT_REPORT,
    ),
    StageSpec(
        name="BUILD_BAR_FEATURES",
        script="build_bar_features.py",
        base_args=("--include-diagnostic-5s",),
        depends_on=("BAR_ALIGNMENT",),
        is_required=True,
        supports_force=True,
        artifact_kind=ARTIFACT_GOLD_FEATURES,
    ),
    StageSpec(
        name="BAR_FEATURE_QUALITY",
        script="bar_feature_quality_report.py",
        base_args=(),
        depends_on=("BUILD_BAR_FEATURES",),
        is_required=False,
        supports_force=False,
        artifact_kind=ARTIFACT_BAR_FEATURE_QUALITY_REPORT,
    ),
    StageSpec(
        name="FEATURE_EDA",
        script="feature_eda_report.py",
        base_args=(),
        depends_on=("BUILD_BAR_FEATURES", "BAR_FEATURE_QUALITY"),
        is_required=False,
        supports_force=True,
        artifact_kind=ARTIFACT_EDA_REPORT,
    ),
)
SATISFIED_STATUSES = frozenset({ps.STAGE_OK, ps.STAGE_SKIPPED_ALREADY_EXISTS_TRUSTED})
UNSATISFIED_STATUSES = frozenset(
    {
        ps.STAGE_FAIL,
        ps.STAGE_SKIPPED_UPSTREAM_FAILED,
        ps.STAGE_SKIPPED_ALREADY_EXISTS_UNTRUSTED,
        ps.STAGE_SKIPPED_MISSING_INPUT,
    }
)


def _has_glob(pattern: str) -> bool:
    return bool(glob.glob(pattern))


def _cme_normalized_present(data_root: Path, date: str, cme_symbols: Iterable[str]) -> bool:
    root = data_root / "normalized" / "cme_reference" / "reference_trades"
    for sym in cme_symbols:
        pat = os.path.join(str(root), f"symbol={sym}", f"date={date}", "part-*.parquet")
        if not _has_glob(pat):
            return False
    return True


def _gold_features_present(data_root: Path, date: str, symbol_pairs: Iterable[str]) -> bool:
    root = data_root / "features" / "bar_features"
    for pair in symbol_pairs:
        pair_dir = root / f"symbol_pair={pair}" / f"date={date}"
        if not pair_dir.exists():
            return False
        if not any(
            (
                p.glob("part-*.parquet")
                for p in pair_dir.iterdir()
                if p.is_dir() and p.name.startswith("bucket=")
            )
        ):
            return False
    return True


def _report_present(reports_root: Path, name: str, date: str) -> bool:
    return (reports_root / f"{name}_{date}.json").exists()


def detect_artifact_present(
    spec: StageSpec,
    *,
    date: str,
    data_root: Path,
    reports_root: Path,
    cme_symbols: Iterable[str],
    symbol_pairs: Iterable[str],
) -> bool:
    if spec.artifact_kind == ARTIFACT_CME_NORMALIZED:
        return _cme_normalized_present(data_root, date, cme_symbols)
    if spec.artifact_kind == ARTIFACT_GOLD_FEATURES:
        return _gold_features_present(data_root, date, symbol_pairs)
    name_map = {
        ARTIFACT_CME_QUALITY_REPORT: "cme_reference_quality",
        ARTIFACT_ALIGNMENT_REPORT: "alignment_quality",
        ARTIFACT_BAR_ALIGNMENT_REPORT: "bar_alignment_quality",
        ARTIFACT_BAR_FEATURE_QUALITY_REPORT: "bar_feature_quality",
        ARTIFACT_EDA_REPORT: "feature_eda",
    }
    if spec.artifact_kind in name_map:
        return _report_present(reports_root, name_map[spec.artifact_kind], date)
    raise ValueError(f"unknown artifact_kind: {spec.artifact_kind!r}")


@dataclass(frozen=True)
class CmeRawTrust:
    raw_present: bool
    completeness_status: Optional[str]
    trust_status: str
    reason: str


def evaluate_cme_raw_trust(
    *, raw_files: list[str], allow_truncated_cme_sample: bool
) -> CmeRawTrust:
    if not raw_files:
        return CmeRawTrust(
            raw_present=False,
            completeness_status=None,
            trust_status=ps.ARTIFACT_ABSENT,
            reason="no CME raw Parquet found for this date",
        )
    raw_dir = Path(raw_files[0]).parent
    docs = cme_dl.find_metadata_for_raw_dir(raw_dir)
    if not docs:
        return CmeRawTrust(
            raw_present=True,
            completeness_status=None,
            trust_status=ps.ARTIFACT_TRUSTED_LEGACY_NO_SIDECAR,
            reason="no Phase 2G.2 metadata sidecar; treating as legacy trusted-with-warning",
        )
    status = cme_dl.directory_completeness_status(docs)
    if status == cme_dl.DATA_COMPLETENESS_COMPLETE_WITHIN_LIMIT:
        return CmeRawTrust(
            raw_present=True,
            completeness_status=status,
            trust_status=ps.ARTIFACT_TRUSTED,
            reason="COMPLETE_WITHIN_LIMIT",
        )
    if status == cme_dl.DATA_COMPLETENESS_TRUNCATED_BY_LIMIT:
        if allow_truncated_cme_sample:
            return CmeRawTrust(
                raw_present=True,
                completeness_status=status,
                trust_status=ps.ARTIFACT_TRUSTED_EXPLORATORY,
                reason="TRUNCATED_BY_LIMIT but --allow-truncated-cme-sample",
            )
        return CmeRawTrust(
            raw_present=True,
            completeness_status=status,
            trust_status=ps.ARTIFACT_EXISTS_BUT_UNTRUSTED,
            reason="TRUNCATED_BY_LIMIT and --allow-truncated-cme-sample not set",
        )
    return CmeRawTrust(
        raw_present=True,
        completeness_status=status,
        trust_status=ps.ARTIFACT_EXISTS_BUT_UNTRUSTED,
        reason=f"CME raw sidecar reports {status}",
    )


@dataclass
class StageTrustVerdict:
    artifact_present: bool
    artifact_trust_status: str
    trust_chain_intact: bool
    reason: str


def evaluate_stage_trust(
    spec: StageSpec,
    *,
    artifact_present: bool,
    cme_raw_trust: CmeRawTrust,
    upstream_trust: Mapping[str, "StageTrustVerdict"],
) -> StageTrustVerdict:
    if not artifact_present:
        return StageTrustVerdict(
            artifact_present=False,
            artifact_trust_status=ps.ARTIFACT_ABSENT,
            trust_chain_intact=False,
            reason=f"{spec.name}: no on-disk artifact",
        )
    raw_trust = cme_raw_trust.trust_status
    raw_trusted = raw_trust in (
        ps.ARTIFACT_TRUSTED,
        ps.ARTIFACT_TRUSTED_EXPLORATORY,
        ps.ARTIFACT_TRUSTED_LEGACY_NO_SIDECAR,
    )
    if not raw_trusted:
        return StageTrustVerdict(
            artifact_present=True,
            artifact_trust_status=ps.ARTIFACT_EXISTS_BUT_UNTRUSTED,
            trust_chain_intact=False,
            reason=f"{spec.name}: artifact present but CME raw lineage is {raw_trust} ({cme_raw_trust.reason})",
        )
    for dep in spec.depends_on:
        if dep == SYNTHETIC_CME_DOWNLOAD:
            continue
        up = upstream_trust.get(dep)
        if up is None:
            return StageTrustVerdict(
                artifact_present=True,
                artifact_trust_status=ps.ARTIFACT_EXISTS_BUT_UNTRUSTED,
                trust_chain_intact=False,
                reason=f"{spec.name}: upstream {dep} trust verdict missing",
            )
        if up.artifact_trust_status not in (
            ps.ARTIFACT_TRUSTED,
            ps.ARTIFACT_TRUSTED_EXPLORATORY,
            ps.ARTIFACT_TRUSTED_LEGACY_NO_SIDECAR,
        ):
            return StageTrustVerdict(
                artifact_present=True,
                artifact_trust_status=ps.ARTIFACT_EXISTS_BUT_UNTRUSTED,
                trust_chain_intact=False,
                reason=f"{spec.name}: upstream {dep} is {up.artifact_trust_status} -- lineage broken",
            )
    return StageTrustVerdict(
        artifact_present=True,
        artifact_trust_status=raw_trust,
        trust_chain_intact=True,
        reason=f"{spec.name}: lineage intact ({raw_trust})",
    )


def classify_day_status(
    *, stage_graph: tuple[StageSpec, ...], stage_status: Mapping[str, str]
) -> str:
    bad_required = False
    bad_optional = False
    for spec in stage_graph:
        st = stage_status.get(spec.name, ps.STAGE_SKIPPED_MISSING_INPUT)
        if st in UNSATISFIED_STATUSES:
            if spec.is_required:
                bad_required = True
            else:
                bad_optional = True
    if bad_required:
        return ps.STATUS_DAY_FAILED
    if bad_optional:
        return ps.STATUS_DAY_PARTIAL
    return ps.STATUS_DAY_COMPLETE


def first_upstream_failed(spec: StageSpec, stage_status: Mapping[str, str]) -> Optional[str]:
    for dep in spec.depends_on:
        if dep == SYNTHETIC_CME_DOWNLOAD:
            continue
        if stage_status.get(dep) in UNSATISFIED_STATUSES:
            return dep
    return None


def zombie_artifact_risk(
    *, stage_graph: tuple[StageSpec, ...], stage_records: Mapping[str, dict]
) -> bool:
    name_to_spec = {s.name: s for s in stage_graph}
    for spec in stage_graph:
        rec = stage_records.get(spec.name)
        if rec is None:
            continue
        if rec.get("status") not in UNSATISFIED_STATUSES:
            continue
        for other in stage_graph:
            if other.name == spec.name:
                continue
            if not _has_upstream(other, spec.name, name_to_spec):
                continue
            other_rec = stage_records.get(other.name) or {}
            if other_rec.get("artifact_present"):
                return True
    return False


def _has_upstream(spec: StageSpec, name: str, graph: Mapping[str, StageSpec]) -> bool:
    visited: set[str] = set()
    frontier = list(spec.depends_on)
    while frontier:
        nxt = frontier.pop()
        if nxt in visited:
            continue
        visited.add(nxt)
        if nxt == name:
            return True
        up = graph.get(nxt)
        if up is None:
            continue
        frontier.extend(up.depends_on)
    return False
