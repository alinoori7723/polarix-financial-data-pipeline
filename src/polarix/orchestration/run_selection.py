from __future__ import annotations

import datetime as _dt
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional

POLICY_LONGEST_VERIFIED = "longest_verified"
POLICY_LATEST_VERIFIED = "latest_verified"
POLICY_EXPLICIT = "explicit"
POLICY_FAIL_ON_MULTIPLE = "fail_on_multiple"
VALID_POLICIES = (
    POLICY_LONGEST_VERIFIED,
    POLICY_LATEST_VERIFIED,
    POLICY_EXPLICIT,
    POLICY_FAIL_ON_MULTIPLE,
)
REASON_NO_RUN_SCOPED_DIR = "NO_RUN_SCOPED_DIR_FOR_DATE"
REASON_UNVERIFIED = "UNVERIFIED_RUN_METADATA"
REASON_NOT_ASSIGNED_TO_DATE = "RUN_NOT_ASSIGNED_TO_DATE"
REASON_AMBIGUOUS = "AMBIGUOUS_RUN_METADATA"
REASON_NO_VERIFIED_RUN_METADATA = "NO_VERIFIED_RUN_METADATA"
REASON_EXPLICIT_MAP_MISSING_ENTRY = "EXPLICIT_MAP_MISSING_ENTRY"
REASON_EXPLICIT_MAP_RUN_NOT_FOUND = "EXPLICIT_MAP_RUN_NOT_FOUND"
REASON_EXPLICIT_MAP_RUN_NOT_VERIFIED = "EXPLICIT_MAP_RUN_NOT_VERIFIED"
REASON_EXPLICIT_MAP_RUN_WRONG_DATE = "EXPLICIT_MAP_RUN_WRONG_DATE"
WARNING_CROSS_MIDNIGHT = "CROSS_MIDNIGHT_RUN_ASSIGNED_TO_START_DATE"
_DATE8_RE = re.compile("(\\d{8})T(\\d{6})Z")


@dataclass(frozen=True)
class RunCandidate:
    run_id: str
    run_dir: Path
    summary: Optional[dict]
    started_at_utc: Optional[str]
    ended_at_utc: Optional[str]
    duration_seconds: Optional[float]
    run_assignment_date: Optional[str]
    ended_date: Optional[str]
    crosses_midnight: bool
    metadata_verified_for_normalization: bool
    verified_offset_min: Optional[int]
    timestamp_semantics_status: Optional[str]
    final_decision: Optional[str]
    rejected_reason: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "started_at_utc": self.started_at_utc,
            "ended_at_utc": self.ended_at_utc,
            "duration_seconds": self.duration_seconds,
            "run_assignment_date": self.run_assignment_date,
            "ended_date": self.ended_date,
            "crosses_midnight": self.crosses_midnight,
            "metadata_verified_for_normalization": self.metadata_verified_for_normalization,
            "verified_offset_min": self.verified_offset_min,
            "timestamp_semantics_status": self.timestamp_semantics_status,
            "final_decision": self.final_decision,
            "rejected_reason": self.rejected_reason,
        }


@dataclass(frozen=True)
class RunSelection:
    date: str
    policy: str
    selected_run_id: Optional[str]
    selected_candidate: Optional[RunCandidate]
    candidates: tuple[RunCandidate, ...]
    rejected: tuple[RunCandidate, ...]
    error_reason: Optional[str]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "date": self.date,
            "policy": self.policy,
            "selected_run_id": self.selected_run_id,
            "selected_run": self.selected_candidate.to_dict() if self.selected_candidate else None,
            "selected_run_started_at_utc": self.selected_candidate.started_at_utc
            if self.selected_candidate
            else None,
            "selected_run_ended_at_utc": self.selected_candidate.ended_at_utc
            if self.selected_candidate
            else None,
            "selected_run_duration_seconds": self.selected_candidate.duration_seconds
            if self.selected_candidate
            else None,
            "selected_run_crosses_midnight": self.selected_candidate.crosses_midnight
            if self.selected_candidate
            else None,
            "candidates": [c.to_dict() for c in self.candidates],
            "rejected": [c.to_dict() for c in self.rejected],
            "error_reason": self.error_reason,
            "warnings": list(self.warnings),
        }


def _parse_utc(value: object) -> Optional[_dt.datetime]:
    if value is None or not isinstance(value, str) or (not value):
        return None
    try:
        s = value.replace("Z", "+00:00")
        return _dt.datetime.fromisoformat(s).astimezone(_dt.timezone.utc)
    except Exception:
        return None


def _utc_date(value: object) -> Optional[str]:
    dt = _parse_utc(value)
    return dt.date().isoformat() if dt else None


def _read_summary(run_dir: Path) -> Optional[dict]:
    p = run_dir / "summary.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_manifest_status(run_dir: Path) -> tuple[Optional[str], Optional[int]]:
    p = run_dir / "logger_manifest.json"
    if not p.exists():
        return (None, None)
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return (None, None)
    ts = doc.get("timestamp_semantics") or {}
    status = ts.get("status")
    voff = ts.get("verified_offset_min")
    return (
        status if isinstance(status, str) else None,
        voff if isinstance(voff, int) and (not isinstance(voff, bool)) else None,
    )


def _is_verified(
    summary: Optional[dict], manifest_status: Optional[str], manifest_offset: Optional[int]
) -> tuple[bool, Optional[str], Optional[int]]:
    if summary is not None and "metadata_verified_for_normalization" in summary:
        verified = bool(summary["metadata_verified_for_normalization"])
        status = summary.get("timestamp_semantics_status") or manifest_status
        offset = summary.get("verified_offset_min")
        if not isinstance(offset, int) or isinstance(offset, bool):
            offset = manifest_offset
        return (verified, status, offset)
    if manifest_status in ("OFFSET_VERIFIED_FOR_SESSION", "UTC_EPOCH_VERIFIED") and isinstance(
        manifest_offset, int
    ):
        return (True, manifest_status, manifest_offset)
    return (False, manifest_status, manifest_offset)


def discover_run_candidates(reports_root: Path) -> list[RunCandidate]:
    root = Path(reports_root) / "logger_runs"
    if not root.exists():
        return []
    out: list[RunCandidate] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        summary = _read_summary(child)
        manifest_status, manifest_offset = _read_manifest_status(child)
        started = summary.get("started_at_utc") if summary else None
        ended = summary.get("ended_at_utc") if summary else None
        run_assignment_date = _utc_date(started)
        ended_date = _utc_date(ended)
        start_dt = _parse_utc(started)
        end_dt = _parse_utc(ended)
        duration: Optional[float] = None
        if start_dt is not None and end_dt is not None:
            duration = (end_dt - start_dt).total_seconds()
        if run_assignment_date is None:
            m = _DATE8_RE.search(child.name)
            if m:
                d = m.group(1)
                run_assignment_date = f"{d[0:4]}-{d[4:6]}-{d[6:8]}"
        crosses = bool(run_assignment_date and ended_date and (run_assignment_date != ended_date))
        verified, ts_status, voff = _is_verified(summary, manifest_status, manifest_offset)
        out.append(
            RunCandidate(
                run_id=child.name,
                run_dir=child,
                summary=summary,
                started_at_utc=started,
                ended_at_utc=ended,
                duration_seconds=duration,
                run_assignment_date=run_assignment_date,
                ended_date=ended_date,
                crosses_midnight=crosses,
                metadata_verified_for_normalization=verified,
                verified_offset_min=voff,
                timestamp_semantics_status=ts_status,
                final_decision=summary.get("final_decision") if summary else None,
            )
        )
    out.sort(key=lambda c: (c.started_at_utc or "", c.run_id))
    return out


def parse_run_id_map(spec: Optional[str]) -> dict[str, str]:
    if not spec:
        return {}
    out: dict[str, str] = {}
    for raw in spec.split(","):
        item = raw.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"run-id-map entry {item!r} must contain '='")
        k, v = item.split("=", 1)
        k, v = (k.strip(), v.strip())
        if not k or not v:
            raise ValueError(f"run-id-map entry {item!r} has empty key or value")
        out[k] = v
    return out


def _annotate_rejected(c: RunCandidate, reason: str) -> RunCandidate:
    return RunCandidate(
        run_id=c.run_id,
        run_dir=c.run_dir,
        summary=c.summary,
        started_at_utc=c.started_at_utc,
        ended_at_utc=c.ended_at_utc,
        duration_seconds=c.duration_seconds,
        run_assignment_date=c.run_assignment_date,
        ended_date=c.ended_date,
        crosses_midnight=c.crosses_midnight,
        metadata_verified_for_normalization=c.metadata_verified_for_normalization,
        verified_offset_min=c.verified_offset_min,
        timestamp_semantics_status=c.timestamp_semantics_status,
        final_decision=c.final_decision,
        rejected_reason=reason,
    )


def select_run_for_date(
    date: str,
    candidates: Iterable[RunCandidate],
    *,
    policy: str = POLICY_LONGEST_VERIFIED,
    run_id_map: Optional[Mapping[str, str]] = None,
) -> RunSelection:
    if policy not in VALID_POLICIES:
        raise ValueError(f"unknown policy {policy!r}; expected one of {VALID_POLICIES}")
    cand_list = list(candidates)
    run_id_map = dict(run_id_map or {})
    warnings: list[str] = []
    assigned: list[RunCandidate] = []
    rejected: list[RunCandidate] = []
    for c in cand_list:
        if c.run_assignment_date == date:
            assigned.append(c)
        else:
            rejected.append(_annotate_rejected(c, REASON_NOT_ASSIGNED_TO_DATE))
    if any((c.crosses_midnight for c in assigned)):
        warnings.append(WARNING_CROSS_MIDNIGHT)
    if not assigned:
        return RunSelection(
            date=date,
            policy=policy,
            selected_run_id=None,
            selected_candidate=None,
            candidates=tuple(cand_list),
            rejected=tuple(rejected),
            error_reason=REASON_NO_RUN_SCOPED_DIR,
            warnings=tuple(warnings),
        )
    if policy == POLICY_EXPLICIT:
        target = run_id_map.get(date)
        if target is None:
            for c in assigned:
                if not c.metadata_verified_for_normalization:
                    rejected.append(_annotate_rejected(c, REASON_UNVERIFIED))
            return RunSelection(
                date=date,
                policy=policy,
                selected_run_id=None,
                selected_candidate=None,
                candidates=tuple(cand_list),
                rejected=tuple(rejected),
                error_reason=REASON_EXPLICIT_MAP_MISSING_ENTRY,
                warnings=tuple(warnings),
            )
        match = next((c for c in assigned if c.run_id == target), None)
        if match is None:
            for c in assigned:
                if not c.metadata_verified_for_normalization:
                    rejected.append(_annotate_rejected(c, REASON_UNVERIFIED))
            return RunSelection(
                date=date,
                policy=policy,
                selected_run_id=None,
                selected_candidate=None,
                candidates=tuple(cand_list),
                rejected=tuple(rejected),
                error_reason=REASON_EXPLICIT_MAP_RUN_NOT_FOUND,
                warnings=tuple(warnings),
            )
        if not match.metadata_verified_for_normalization:
            for c in assigned:
                if c is not match and (not c.metadata_verified_for_normalization):
                    rejected.append(_annotate_rejected(c, REASON_UNVERIFIED))
            rejected.append(_annotate_rejected(match, REASON_UNVERIFIED))
            return RunSelection(
                date=date,
                policy=policy,
                selected_run_id=None,
                selected_candidate=None,
                candidates=tuple(cand_list),
                rejected=tuple(rejected),
                error_reason=REASON_EXPLICIT_MAP_RUN_NOT_VERIFIED,
                warnings=tuple(warnings),
            )
        for c in assigned:
            if c.run_id == match.run_id:
                continue
            rejected.append(
                _annotate_rejected(
                    c,
                    REASON_UNVERIFIED
                    if not c.metadata_verified_for_normalization
                    else "EXPLICIT_MAP_NOT_SELECTED",
                )
            )
        return RunSelection(
            date=date,
            policy=policy,
            selected_run_id=match.run_id,
            selected_candidate=match,
            candidates=tuple(cand_list),
            rejected=tuple(rejected),
            error_reason=None,
            warnings=tuple(warnings),
        )
    verified: list[RunCandidate] = []
    for c in assigned:
        if c.metadata_verified_for_normalization:
            verified.append(c)
        else:
            rejected.append(_annotate_rejected(c, REASON_UNVERIFIED))
    if not verified:
        return RunSelection(
            date=date,
            policy=policy,
            selected_run_id=None,
            selected_candidate=None,
            candidates=tuple(cand_list),
            rejected=tuple(rejected),
            error_reason=REASON_NO_VERIFIED_RUN_METADATA,
            warnings=tuple(warnings),
        )
    if policy == POLICY_FAIL_ON_MULTIPLE:
        if len(verified) > 1:
            return RunSelection(
                date=date,
                policy=policy,
                selected_run_id=None,
                selected_candidate=None,
                candidates=tuple(cand_list),
                rejected=tuple(rejected),
                error_reason=REASON_AMBIGUOUS,
                warnings=tuple(warnings),
            )
        chosen = verified[0]
        return RunSelection(
            date=date,
            policy=policy,
            selected_run_id=chosen.run_id,
            selected_candidate=chosen,
            candidates=tuple(cand_list),
            rejected=tuple(rejected),
            error_reason=None,
            warnings=tuple(warnings),
        )
    if policy == POLICY_LONGEST_VERIFIED:
        verified.sort(
            key=lambda c: (-(c.duration_seconds or 0.0), c.started_at_utc or "", c.run_id)
        )
    elif policy == POLICY_LATEST_VERIFIED:
        verified.sort(key=lambda c: (c.started_at_utc or "", c.run_id), reverse=True)
    chosen = verified[0]
    for c in verified[1:]:
        rejected.append(_annotate_rejected(c, f"POLICY_{policy.upper()}_NOT_SELECTED"))
    return RunSelection(
        date=date,
        policy=policy,
        selected_run_id=chosen.run_id,
        selected_candidate=chosen,
        candidates=tuple(cand_list),
        rejected=tuple(rejected),
        error_reason=None,
        warnings=tuple(warnings),
    )
