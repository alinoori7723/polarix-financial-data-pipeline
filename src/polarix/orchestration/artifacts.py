from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


class ArtifactIntegrityError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def publish_json(path: Path, value: Any) -> None:
    payload = canonical_json(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def artifact_record(path: Path, run_dir: Path) -> dict:
    relative = path.resolve().relative_to(run_dir.resolve()).as_posix()
    record = {"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
    if path.suffix == ".parquet":
        parquet = pq.ParquetFile(path)
        record["rows"] = parquet.metadata.num_rows
        record["schema"] = {field.name: str(field.type) for field in parquet.schema_arrow}
    return record


def verify_artifacts(run_dir: Path, records: list[dict]) -> None:
    root = run_dir.resolve()
    for record in records:
        path = (root / record["path"]).resolve()
        if not path.is_relative_to(root):
            raise ArtifactIntegrityError(f"Artifact escapes run directory: {record['path']}")
        if not path.is_file():
            raise ArtifactIntegrityError(f"Missing artifact: {record['path']}")
        if path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
            raise ArtifactIntegrityError(f"Artifact changed: {record['path']}")
