from __future__ import annotations

import datetime as _dt
import json
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    return str(obj)


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding="utf-8"
    )
    os.replace(tmp, path)


def write_health_snapshot(
    reports_root: Path, payload: dict, name: str = "logger_health.json"
) -> Path:
    out = reports_root / name
    payload = dict(payload)
    payload["generated_at_utc"] = _dt.datetime.now(tz=_dt.timezone.utc).isoformat()
    _atomic_write_json(out, payload)
    return out


def write_manifest(reports_root: Path, payload: dict, name: str = "logger_manifest.json") -> Path:
    out = reports_root / name
    payload = dict(payload)
    payload["generated_at_utc"] = _dt.datetime.now(tz=_dt.timezone.utc).isoformat()
    _atomic_write_json(out, payload)
    return out


def write_critical_health(
    reports_root: Path, payload: dict, name_prefix: str = "logger_critical"
) -> Path:
    now = _dt.datetime.now(tz=_dt.timezone.utc)
    stamp = now.strftime("%Y%m%dT%H%M%S%f") + "Z"
    out = reports_root / f"{name_prefix}-{stamp}.json"
    payload = dict(payload)
    payload.setdefault("severity", "CRITICAL")
    payload["generated_at_utc"] = now.isoformat()
    _atomic_write_json(out, payload)
    return out
