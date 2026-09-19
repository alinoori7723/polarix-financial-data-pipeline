from __future__ import annotations

import datetime as dt
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from polarix.orchestration.artifacts import artifact_record, publish_json, verify_artifacts


@dataclass(frozen=True)
class StageOutput:
    paths: tuple[Path, ...] = ()
    metrics: dict | None = None


@dataclass(frozen=True)
class Stage:
    name: str
    depends_on: tuple[str, ...]
    action: Callable[[], StageOutput]


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def execute_stages(run_dir: Path, stages: tuple[Stage, ...]) -> list[dict]:
    seen = set()
    for stage in stages:
        if (
            not re.fullmatch(r"[a-z][a-z0-9_]*", stage.name)
            or stage.name in seen
            or not set(stage.depends_on).issubset(seen)
        ):
            raise ValueError(f"Invalid stage dependency order: {stage.name}")
        seen.add(stage.name)
    records: dict[str, dict] = {}
    metadata_path = run_dir / "metadata.json"
    all_artifacts = [artifact_record(metadata_path, run_dir)] if metadata_path.is_file() else []
    for stage in stages:
        started = time.perf_counter()
        record = {
            "name": stage.name,
            "depends_on": list(stage.depends_on),
            "started_at_utc": utc_now(),
            "status": "PENDING",
            "artifacts": [],
            "metrics": {},
        }
        blocked = [name for name in stage.depends_on if records[name]["status"] != "PASS"]
        if blocked:
            record.update(status="SKIPPED", reason="UPSTREAM_FAILED", blocked_by=blocked)
        else:
            try:
                verify_artifacts(run_dir, all_artifacts)
                output = stage.action()
                verify_artifacts(run_dir, all_artifacts)
                artifacts = [artifact_record(path, run_dir) for path in output.paths]
                all_artifacts.extend(artifacts)
                record.update(status="PASS", artifacts=artifacts, metrics=output.metrics or {})
            except Exception as exc:
                record.update(status="FAIL", reason=type(exc).__name__, error=str(exc))
        record["finished_at_utc"] = utc_now()
        record["duration_seconds"] = round(time.perf_counter() - started, 6)
        publish_json(run_dir / "stages" / f"{stage.name}.json", record)
        records[stage.name] = record
    return list(records.values())
