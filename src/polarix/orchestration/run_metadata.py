from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

VALID_TIMESTAMP_SEMANTICS_STATUSES = ("OFFSET_VERIFIED_FOR_SESSION", "UTC_EPOCH_VERIFIED")
VALID_CLOCK_STATUSES = ("CALIBRATION_COARSE_OK", "CALIBRATION_OK", "CALIBRATION_FINE_OK")
_LIVE_RUN_SUMMARY_RE = re.compile("^live_run_(?P<run_id>[^_]+)_summary\\.json$")
RUN_SCOPED_DIRNAME = "logger_runs"
RUN_SCOPED_MANIFEST = "logger_manifest.json"
RUN_SCOPED_HEALTH = "logger_health.json"
RUN_SCOPED_SUMMARY = "summary.json"
ERROR_AMBIGUOUS_OR_MISSING_RUN_METADATA = "AMBIGUOUS_OR_MISSING_RUN_METADATA"
ERROR_UNVERIFIED_RUN_METADATA = "UNVERIFIED_RUN_METADATA"


class RunMetadataError(RuntimeError):
    def __init__(self, message: str, error_tag: Optional[str] = None) -> None:
        super().__init__(message)
        self.error_tag = error_tag


@dataclass(frozen=True)
class RunMetadata:
    source_path: Path
    source_kind: str
    run_id: Optional[str]
    verified_offset_min: int
    timestamp_semantics_status: str
    clock_status: dict
    broker_metadata: dict
    run_started_at_utc: Optional[str]
    run_ended_at_utc: Optional[str]
    final_decision: Optional[str]
    symbols: tuple[str, ...]
    source_type: str = field(default="legacy_latest", kw_only=True)
    latest_data_file_mtime_utc: Optional[str] = field(default=None, kw_only=True)

    @property
    def verified_offset_ms(self) -> int:
        return int(self.verified_offset_min) * 60 * 1000

    def run_window_utc(self) -> tuple[Optional[str], Optional[str]]:
        start = self.run_started_at_utc
        end_candidates = [t for t in (self.run_ended_at_utc, self.latest_data_file_mtime_utc) if t]
        end = max(end_candidates) if end_candidates else None
        return (start, end)

    @property
    def broker_server(self) -> str:
        return str(
            self.broker_metadata.get("account_server") or self.broker_metadata.get("server") or ""
        )

    @property
    def broker_company(self) -> str:
        return str(
            self.broker_metadata.get("account_company") or self.broker_metadata.get("company") or ""
        )

    @property
    def account_login_hash(self) -> str:
        return str(
            self.broker_metadata.get("account_login_hash")
            or self.broker_metadata.get("login_hash")
            or ""
        )

    def to_dict(self) -> dict:
        return {
            "metadata_source_path": str(self.source_path),
            "metadata_source_kind": self.source_kind,
            "selected_metadata_source_type": self.source_type,
            "selected_metadata_path": str(self.source_path),
            "selected_run_id": self.run_id,
            "run_id": self.run_id,
            "verified_offset_min": self.verified_offset_min,
            "verified_offset_ms": self.verified_offset_ms,
            "timestamp_semantics_status": self.timestamp_semantics_status,
            "metadata_verified_for_normalization": True,
            "clock_status": self.clock_status,
            "broker_metadata": self.broker_metadata,
            "broker_server": self.broker_server,
            "broker_company": self.broker_company,
            "account_login_hash": self.account_login_hash,
            "run_started_at_utc": self.run_started_at_utc,
            "run_ended_at_utc": self.run_ended_at_utc,
            "latest_data_file_mtime_utc": self.latest_data_file_mtime_utc,
            "final_decision": self.final_decision,
            "symbols": list(self.symbols),
        }


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RunMetadataError(f"metadata file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RunMetadataError(f"metadata file is not valid JSON: {path}: {exc}") from exc


def _validate_offset_and_status(
    verified_offset_min: object, status: object, source_path: Path
) -> tuple[int, str]:
    if verified_offset_min is None:
        raise RunMetadataError(
            f"verified_offset_min missing in {source_path}; refusing to infer offset"
        )
    if isinstance(verified_offset_min, bool) or not isinstance(verified_offset_min, int):
        raise RunMetadataError(
            f"verified_offset_min must be an integer (got {type(verified_offset_min).__name__}) in {source_path}"
        )
    if not isinstance(status, str) or status not in VALID_TIMESTAMP_SEMANTICS_STATUSES:
        raise RunMetadataError(
            f"timestamp_semantics.status='{status}' is not in {VALID_TIMESTAMP_SEMANTICS_STATUSES} in {source_path}; fail closed"
        )
    return (int(verified_offset_min), status)


def _list_live_run_summaries(reports_root: Path) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    if not reports_root.exists():
        return out
    for child in reports_root.iterdir():
        if not child.is_file():
            continue
        m = _LIVE_RUN_SUMMARY_RE.match(child.name)
        if m:
            out.append((m.group("run_id"), child))
    out.sort(key=lambda t: t[0])
    return out


def _from_live_run_summary(
    path: Path, doc: dict, run_id: Optional[str], *, source_type: str = "legacy_latest"
) -> RunMetadata:
    ts = doc.get("timestamp_semantics") or {}
    verified_offset_min = ts.get("verified_offset_min")
    status = ts.get("status")
    if verified_offset_min is None or status is None:
        raise RunMetadataError(
            f"live_run summary {path} does not embed timestamp_semantics.verified_offset_min and timestamp_semantics.status; supply --metadata-path pointing at logger_manifest.json or logger_health.json instead"
        )
    voff, st = _validate_offset_and_status(verified_offset_min, status, path)
    broker = doc.get("broker_metadata") or {}
    return RunMetadata(
        source_path=path,
        source_kind="live_run_summary",
        source_type=source_type,
        run_id=run_id,
        verified_offset_min=voff,
        timestamp_semantics_status=st,
        clock_status=dict(doc.get("clock_status") or {}),
        broker_metadata=dict(broker),
        run_started_at_utc=doc.get("started_at_utc"),
        run_ended_at_utc=doc.get("ended_at_utc"),
        final_decision=doc.get("final_decision"),
        symbols=tuple(doc.get("symbols") or ()),
        latest_data_file_mtime_utc=doc.get("latest_data_file_mtime_utc"),
    )


def _from_logger_manifest(
    path: Path, doc: dict, *, source_type: str = "legacy_latest", run_id: Optional[str] = None
) -> RunMetadata:
    ts = doc.get("timestamp_semantics") or {}
    voff, st = _validate_offset_and_status(ts.get("verified_offset_min"), ts.get("status"), path)
    return RunMetadata(
        source_path=path,
        source_kind="logger_manifest",
        source_type=source_type,
        run_id=run_id,
        verified_offset_min=voff,
        timestamp_semantics_status=st,
        clock_status={},
        broker_metadata={},
        run_started_at_utc=doc.get("generated_at_utc"),
        run_ended_at_utc=doc.get("generated_at_utc"),
        final_decision=None,
        symbols=tuple(doc.get("symbols") or ()),
    )


def _from_logger_health(
    path: Path, doc: dict, *, source_type: str = "legacy_latest", run_id: Optional[str] = None
) -> RunMetadata:
    ts = doc.get("timestamp_semantics") or {}
    voff, st = _validate_offset_and_status(ts.get("verified_offset_min"), ts.get("status"), path)
    account = dict(doc.get("account") or {})
    broker = dict(doc.get("broker") or {})
    merged_broker = {
        "account_server": account.get("server"),
        "account_company": account.get("company"),
        "account_login_hash": account.get("login_hash"),
        "terminal_company": broker.get("terminal_company"),
        "terminal_name": broker.get("terminal_name"),
        "terminal_build": broker.get("terminal_build"),
    }
    return RunMetadata(
        source_path=path,
        source_kind="logger_health",
        source_type=source_type,
        run_id=run_id,
        verified_offset_min=voff,
        timestamp_semantics_status=st,
        clock_status=dict(doc.get("clock_health") or {}),
        broker_metadata=merged_broker,
        run_started_at_utc=doc.get("generated_at_utc"),
        run_ended_at_utc=doc.get("generated_at_utc"),
        final_decision=None,
        symbols=tuple(doc.get("symbols") or ()),
    )


def _enrich_with_companion_files(md: RunMetadata, reports_root: Path) -> RunMetadata:
    broker = dict(md.broker_metadata)
    clock = dict(md.clock_status)
    symbols = tuple(md.symbols)
    manifest_path = reports_root / "logger_manifest.json"
    health_path = reports_root / "logger_health.json"

    def _missing(d: dict, key: str) -> bool:
        v = d.get(key)
        return v is None or v == ""

    if md.source_kind != "logger_health" and health_path.exists():
        try:
            doc = _load_json(health_path)
            account = dict(doc.get("account") or {})
            term = dict(doc.get("broker") or {})
            if _missing(broker, "account_server"):
                broker["account_server"] = account.get("server")
            if _missing(broker, "account_company"):
                broker["account_company"] = account.get("company")
            if _missing(broker, "account_login_hash"):
                broker["account_login_hash"] = account.get("login_hash")
            if _missing(broker, "terminal_company"):
                broker["terminal_company"] = term.get("terminal_company")
            if _missing(broker, "terminal_name"):
                broker["terminal_name"] = term.get("terminal_name")
            if _missing(broker, "terminal_build"):
                broker["terminal_build"] = term.get("terminal_build")
            if not clock:
                clock = dict(doc.get("clock_health") or {})
            if not symbols:
                symbols = tuple(doc.get("symbols") or ())
        except RunMetadataError:
            pass
    if not symbols and md.source_kind != "logger_manifest" and manifest_path.exists():
        try:
            doc = _load_json(manifest_path)
            symbols = tuple(doc.get("symbols") or ())
        except RunMetadataError:
            pass
    return RunMetadata(
        source_path=md.source_path,
        source_kind=md.source_kind,
        source_type=md.source_type,
        run_id=md.run_id,
        verified_offset_min=md.verified_offset_min,
        timestamp_semantics_status=md.timestamp_semantics_status,
        clock_status=clock,
        broker_metadata=broker,
        run_started_at_utc=md.run_started_at_utc,
        run_ended_at_utc=md.run_ended_at_utc,
        final_decision=md.final_decision,
        symbols=symbols,
    )


def _select_live_run_summary(
    reports_root: Path, run_id: Optional[str], candidates: Iterable[tuple[str, Path]]
) -> Optional[tuple[str, Path]]:
    items = list(candidates)
    if run_id is not None:
        for rid, path in items:
            if rid == run_id:
                return (rid, path)
        raise RunMetadataError(
            f"live_run summary for --run-id {run_id} not found under {reports_root}"
        )
    pass_items: list[tuple[str, Path]] = []
    for rid, path in items:
        try:
            doc = _load_json(path)
        except RunMetadataError:
            continue
        if doc.get("final_decision") == "PASS":
            pass_items.append((rid, path))
    if not pass_items:
        return None
    pass_items.sort(key=lambda t: t[0])
    return pass_items[-1]


def _run_scoped_root(reports_root: Path) -> Path:
    return Path(reports_root) / RUN_SCOPED_DIRNAME


def list_run_scoped_runs(reports_root: Path) -> list[tuple[str, Path]]:
    root = _run_scoped_root(reports_root)
    if not root.exists():
        return []
    out: list[tuple[str, Path]] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        out.append((child.name, child))
    out.sort(key=lambda t: t[0])
    return out


def _resolve_from_run_scoped_dir(run_dir: Path, run_id: str) -> RunMetadata:
    manifest_path = run_dir / RUN_SCOPED_MANIFEST
    health_path = run_dir / RUN_SCOPED_HEALTH
    summary_path = run_dir / RUN_SCOPED_SUMMARY
    if manifest_path.exists():
        try:
            doc = _load_json(manifest_path)
            md = _from_logger_manifest(manifest_path, doc, source_type="run_scoped", run_id=run_id)
            return _enrich_with_run_scoped_companions(md, run_dir, run_id)
        except RunMetadataError:
            pass
    if summary_path.exists():
        try:
            doc = _load_json(summary_path)
            md = _from_live_run_summary(summary_path, doc, run_id, source_type="run_scoped")
            return _enrich_with_run_scoped_companions(md, run_dir, run_id)
        except RunMetadataError:
            pass
    if health_path.exists():
        try:
            doc = _load_json(health_path)
            md = _from_logger_health(health_path, doc, source_type="run_scoped", run_id=run_id)
            return _enrich_with_run_scoped_companions(md, run_dir, run_id)
        except RunMetadataError:
            pass
    raise RunMetadataError(
        f"run-scoped metadata under {run_dir} does not carry a verified offset or a valid timestamp_semantics status",
        error_tag=ERROR_UNVERIFIED_RUN_METADATA,
    )


def _enrich_with_run_scoped_companions(md: RunMetadata, run_dir: Path, run_id: str) -> RunMetadata:
    broker = dict(md.broker_metadata)
    clock = dict(md.clock_status)
    symbols = tuple(md.symbols)
    run_started = md.run_started_at_utc
    run_ended = md.run_ended_at_utc
    latest_mtime = md.latest_data_file_mtime_utc
    final_decision = md.final_decision

    def _missing(d: dict, key: str) -> bool:
        v = d.get(key)
        return v is None or v == ""

    health_path = run_dir / RUN_SCOPED_HEALTH
    if md.source_kind != "logger_health" and health_path.exists():
        try:
            doc = _load_json(health_path)
            account = dict(doc.get("account") or {})
            term = dict(doc.get("broker") or {})
            if _missing(broker, "account_server"):
                broker["account_server"] = account.get("server")
            if _missing(broker, "account_company"):
                broker["account_company"] = account.get("company")
            if _missing(broker, "account_login_hash"):
                broker["account_login_hash"] = account.get("login_hash")
            if _missing(broker, "terminal_company"):
                broker["terminal_company"] = term.get("terminal_company")
            if _missing(broker, "terminal_name"):
                broker["terminal_name"] = term.get("terminal_name")
            if _missing(broker, "terminal_build"):
                broker["terminal_build"] = term.get("terminal_build")
            if not clock:
                clock = dict(doc.get("clock_health") or {})
            if not symbols:
                symbols = tuple(doc.get("symbols") or ())
        except RunMetadataError:
            pass
    summary_path = run_dir / RUN_SCOPED_SUMMARY
    if summary_path.exists():
        try:
            doc = _load_json(summary_path)
            sbroker = dict(doc.get("broker_metadata") or {})
            for k, v in sbroker.items():
                if _missing(broker, k):
                    broker[k] = v
            if not symbols:
                symbols = tuple(doc.get("symbols") or symbols)
            if doc.get("started_at_utc"):
                run_started = doc["started_at_utc"]
            if doc.get("ended_at_utc"):
                run_ended = doc["ended_at_utc"]
            if doc.get("latest_data_file_mtime_utc"):
                latest_mtime = doc["latest_data_file_mtime_utc"]
            if doc.get("final_decision") and final_decision is None:
                final_decision = doc["final_decision"]
        except RunMetadataError:
            pass
    return RunMetadata(
        source_path=md.source_path,
        source_kind=md.source_kind,
        source_type="run_scoped",
        run_id=run_id,
        verified_offset_min=md.verified_offset_min,
        timestamp_semantics_status=md.timestamp_semantics_status,
        clock_status=clock,
        broker_metadata=broker,
        run_started_at_utc=run_started,
        run_ended_at_utc=run_ended,
        final_decision=final_decision,
        symbols=symbols,
        latest_data_file_mtime_utc=latest_mtime,
    )


def _verified_run_scoped_for_date(
    reports_root: Path, date: Optional[str]
) -> list[tuple[str, RunMetadata]]:
    out: list[tuple[str, RunMetadata]] = []
    date_prefix = date.replace("-", "") if date else None
    for run_id, run_dir in list_run_scoped_runs(reports_root):
        if date_prefix:
            m = re.search("\\d{8}", run_id)
            if not m or m.group(0) != date_prefix:
                continue
        try:
            md = _resolve_from_run_scoped_dir(run_dir, run_id)
        except RunMetadataError:
            continue
        out.append((run_id, md))
    return out


def resolve_run_metadata(
    reports_root: Path,
    metadata_path: Optional[Path] = None,
    run_id: Optional[str] = None,
    *,
    date: Optional[str] = None,
    allow_latest_metadata: bool = True,
    strict_run_metadata: bool = False,
) -> RunMetadata:
    reports_root = Path(reports_root)
    if metadata_path is not None:
        return _resolve_explicit_path(reports_root, Path(metadata_path))
    if run_id is not None:
        run_scoped_dir = _run_scoped_root(reports_root) / run_id
        if run_scoped_dir.exists():
            return _resolve_from_run_scoped_dir(run_scoped_dir, run_id)
        summaries = _list_live_run_summaries(reports_root)
        chosen = _select_live_run_summary(reports_root, run_id, summaries)
        if chosen is None:
            raise RunMetadataError(
                f"no run-scoped directory at {run_scoped_dir} and no live_run_{run_id}_summary.json under {reports_root}",
                error_tag=ERROR_AMBIGUOUS_OR_MISSING_RUN_METADATA,
            )
        rid, path = chosen
        doc = _load_json(path)
        md = _from_live_run_summary(path, doc, rid, source_type="legacy_latest")
        return _enrich_with_companion_files(md, reports_root)
    verified_run_scoped = _verified_run_scoped_for_date(reports_root, date)
    if len(verified_run_scoped) == 1:
        return verified_run_scoped[0][1]
    if len(verified_run_scoped) > 1:
        if not allow_latest_metadata:
            ids = ", ".join((rid for rid, _ in verified_run_scoped))
            raise RunMetadataError(
                f"multiple verified run-scoped runs for date={date}: [{ids}]; pass --run-id <id> to choose one explicitly, or --allow-latest-metadata to fall back to legacy resolution",
                error_tag=ERROR_AMBIGUOUS_OR_MISSING_RUN_METADATA,
            )
        return verified_run_scoped[-1][1]
    if not allow_latest_metadata:
        run_scoped_root = _run_scoped_root(reports_root)
        raise RunMetadataError(
            f"no verified run-scoped metadata under {run_scoped_root} for date={date!r}; pass --run-id, --metadata-path, or --allow-latest-metadata explicitly",
            error_tag=ERROR_AMBIGUOUS_OR_MISSING_RUN_METADATA,
        )
    return _resolve_legacy(reports_root, run_id=None)


def _resolve_explicit_path(reports_root: Path, path: Path) -> RunMetadata:
    if not path.exists():
        raise RunMetadataError(f"explicit --metadata-path not found: {path}")
    doc = _load_json(path)
    name = path.name.lower()
    if _LIVE_RUN_SUMMARY_RE.match(path.name):
        rid_match = _LIVE_RUN_SUMMARY_RE.match(path.name)
        assert rid_match is not None
        md = _from_live_run_summary(
            path, doc, rid_match.group("run_id"), source_type="explicit_path"
        )
    elif name == RUN_SCOPED_SUMMARY or ("summary" in name and "live_run" in name):
        md = _from_live_run_summary(path, doc, None, source_type="explicit_path")
    elif name == "logger_manifest.json" or "manifest" in name:
        md = _from_logger_manifest(path, doc, source_type="explicit_path")
    elif name == "logger_health.json" or "health" in name:
        md = _from_logger_health(path, doc, source_type="explicit_path")
    else:
        try:
            md = _from_live_run_summary(path, doc, None, source_type="explicit_path")
        except RunMetadataError:
            try:
                md = _from_logger_manifest(path, doc, source_type="explicit_path")
            except RunMetadataError:
                md = _from_logger_health(path, doc, source_type="explicit_path")
    parent = path.parent
    if parent.parent.name == RUN_SCOPED_DIRNAME and parent.name:
        return _enrich_with_run_scoped_companions(md, parent, parent.name)
    return _enrich_with_companion_files(md, reports_root)


def _resolve_legacy(reports_root: Path, run_id: Optional[str]) -> RunMetadata:
    summaries = _list_live_run_summaries(reports_root)
    if summaries:
        chosen = _select_live_run_summary(reports_root, run_id, summaries)
        if chosen is not None:
            rid, path = chosen
            doc = _load_json(path)
            try:
                md = _from_live_run_summary(path, doc, rid, source_type="legacy_latest")
            except RunMetadataError:
                md = None
            else:
                return _enrich_with_companion_files(md, reports_root)
    manifest_path = reports_root / "logger_manifest.json"
    if manifest_path.exists():
        doc = _load_json(manifest_path)
        md = _from_logger_manifest(manifest_path, doc, source_type="legacy_latest")
        return _enrich_with_companion_files(md, reports_root)
    health_path = reports_root / "logger_health.json"
    if health_path.exists():
        doc = _load_json(health_path)
        md = _from_logger_health(health_path, doc, source_type="legacy_latest")
        return _enrich_with_companion_files(md, reports_root)
    raise RunMetadataError(
        f"no run-metadata source available under {reports_root}; expected a live_run_*_summary.json, logger_manifest.json, or logger_health.json",
        error_tag=ERROR_AMBIGUOUS_OR_MISSING_RUN_METADATA,
    )


def write_run_scoped_metadata(
    reports_root: Path,
    run_id: str,
    *,
    manifest_src: Optional[Path],
    health_src: Optional[Path],
    summary_doc: Optional[dict],
    console_log_src: Optional[Path] = None,
    supervisor_health_src: Optional[Path] = None,
) -> Path:
    import json as _json
    import os as _os
    import shutil as _shutil

    run_dir = _run_scoped_root(reports_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    def _copy(src: Optional[Path], dst_name: str) -> None:
        if src is None or not Path(src).exists():
            return
        dst = run_dir / dst_name
        tmp = dst.with_suffix(dst.suffix + ".tmp")
        _shutil.copy2(str(src), str(tmp))
        _os.replace(str(tmp), str(dst))

    _copy(manifest_src, RUN_SCOPED_MANIFEST)
    _copy(health_src, RUN_SCOPED_HEALTH)
    _copy(console_log_src, "console.log")
    _copy(supervisor_health_src, "supervisor_health.jsonl")
    if summary_doc is not None:
        summary_path = run_dir / RUN_SCOPED_SUMMARY
        tmp = summary_path.with_suffix(".json.tmp")
        tmp.write_text(
            _json.dumps(summary_doc, indent=2, sort_keys=True, default=str), encoding="utf-8"
        )
        _os.replace(str(tmp), str(summary_path))
    return run_dir
