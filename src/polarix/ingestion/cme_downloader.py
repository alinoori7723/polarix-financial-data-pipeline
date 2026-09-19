from __future__ import annotations

import datetime as _dt
import inspect
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

DATA_COMPLETENESS_COMPLETE_WITHIN_LIMIT = "COMPLETE_WITHIN_LIMIT"
DATA_COMPLETENESS_TRUNCATED_BY_LIMIT = "TRUNCATED_BY_LIMIT"
DATA_COMPLETENESS_UNKNOWN_NO_RECORD_LIMIT = "UNKNOWN_NO_RECORD_LIMIT"
DATA_COMPLETENESS_UNKNOWN_ROW_COUNT = "UNKNOWN_ROW_COUNT"
DOWNLOAD_BLOCKED_NO_RECORD_LIMIT = "DOWNLOAD_BLOCKED_NO_RECORD_LIMIT"
DOWNLOAD_BLOCKED_PHYSICAL_LIMIT_UNSUPPORTED = "DOWNLOAD_BLOCKED_PHYSICAL_LIMIT_UNSUPPORTED"
DOWNLOAD_BLOCKED_MISSING_API_KEY = "DOWNLOAD_BLOCKED_MISSING_API_KEY"
DOWNLOAD_OK = "DOWNLOAD_OK"
DOWNLOAD_OK_TRUNCATED = "DOWNLOAD_OK_TRUNCATED"
DOWNLOAD_FAILED = "DOWNLOAD_FAILED"
DOWNLOAD_METADATA_ONLY = "DOWNLOAD_METADATA_ONLY"
CRITICAL_DOWNLOAD_WITHOUT_PHYSICAL_LIMIT = "DOWNLOAD_WITHOUT_PHYSICAL_LIMIT"


class CmeDownloadError(RuntimeError):
    pass


@dataclass(frozen=True)
class DownloadRequest:
    dataset: str
    schema: str
    symbols: tuple[str, ...]
    stype_in: str
    start_utc: str
    end_utc: str
    output_path: Path
    max_download_records: Optional[int]
    allow_download_without_physical_limit: bool = False
    metadata_only: bool = False
    dry_run: bool = False
    api_key_present: Optional[bool] = None

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "schema": self.schema,
            "symbols": list(self.symbols),
            "stype_in": self.stype_in,
            "start_utc": self.start_utc,
            "end_utc": self.end_utc,
            "output_path": str(self.output_path),
            "max_download_records": self.max_download_records,
            "allow_download_without_physical_limit": self.allow_download_without_physical_limit,
            "metadata_only": self.metadata_only,
            "dry_run": self.dry_run,
        }


@dataclass(frozen=True)
class DownloadResult:
    status: str
    data_completeness_status: Optional[str]
    truncation_warning: bool
    records_downloaded: Optional[int]
    output_path: Optional[Path]
    metadata_path: Optional[Path]
    file_size_bytes: Optional[int]
    physical_limit_applied: bool
    physical_limit_type: Optional[str]
    physical_limit_value: Optional[int]
    api_call_parameters: dict
    databento_sdk_version: Optional[str]
    critical_warnings: tuple[str, ...]
    warnings: tuple[str, ...]
    error: Optional[str]

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "data_completeness_status": self.data_completeness_status,
            "truncation_warning": self.truncation_warning,
            "records_downloaded": self.records_downloaded,
            "output_path": str(self.output_path) if self.output_path else None,
            "metadata_path": str(self.metadata_path) if self.metadata_path else None,
            "file_size_bytes": self.file_size_bytes,
            "physical_limit_applied": self.physical_limit_applied,
            "physical_limit_type": self.physical_limit_type,
            "physical_limit_value": self.physical_limit_value,
            "api_call_parameters": dict(self.api_call_parameters),
            "databento_sdk_version": self.databento_sdk_version,
            "critical_warnings": list(self.critical_warnings),
            "warnings": list(self.warnings),
            "error": self.error,
        }


def _sdk_supports_limit(client: object) -> bool:
    try:
        ts = getattr(client, "timeseries", None)
        if ts is None:
            return False
        target = getattr(ts, "get_range", None)
        if not callable(target):
            return False
        sig = inspect.signature(target)
        return "limit" in sig.parameters
    except Exception:
        return False


def _count_parquet_rows(path: Path) -> Optional[int]:
    try:
        import pyarrow.parquet as pq

        return pq.ParquetFile(str(path)).metadata.num_rows
    except Exception:
        return None


def _safe_get(d: dict, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, default)
    return cur


def _utc_iso() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).isoformat()


def _detect_completeness(records: Optional[int], limit: Optional[int]) -> str:
    if records is None:
        return DATA_COMPLETENESS_UNKNOWN_ROW_COUNT
    if limit is None:
        return DATA_COMPLETENESS_UNKNOWN_NO_RECORD_LIMIT
    if records >= limit:
        return DATA_COMPLETENESS_TRUNCATED_BY_LIMIT
    return DATA_COMPLETENESS_COMPLETE_WITHIN_LIMIT


def _databento_version() -> Optional[str]:
    try:
        import databento

        return getattr(databento, "__version__", None)
    except Exception:
        return None


METADATA_FIELDS = (
    "requested_start_utc",
    "requested_end_utc",
    "dataset",
    "schema",
    "symbols",
    "stype_in",
    "output_path",
    "file_size_bytes",
    "max_download_records",
    "records_downloaded",
    "data_completeness_status",
    "truncation_warning",
    "created_at_utc",
    "databento_sdk_version",
    "api_call_parameters",
    "physical_limit_applied",
    "physical_limit_type",
    "physical_limit_value",
)


def metadata_path_for(output_path: Path) -> Path:
    return output_path.with_suffix(output_path.suffix + ".metadata.json")


def write_metadata_file(
    request: DownloadRequest,
    *,
    output_path: Path,
    records_downloaded: Optional[int],
    file_size_bytes: Optional[int],
    api_call_parameters: dict,
    physical_limit_applied: bool,
    physical_limit_type: Optional[str],
    physical_limit_value: Optional[int],
    databento_sdk_version: Optional[str],
) -> Path:
    completeness = _detect_completeness(records_downloaded, request.max_download_records)
    truncation = completeness == DATA_COMPLETENESS_TRUNCATED_BY_LIMIT
    payload = {
        "requested_start_utc": request.start_utc,
        "requested_end_utc": request.end_utc,
        "dataset": request.dataset,
        "schema": request.schema,
        "symbols": list(request.symbols),
        "stype_in": request.stype_in,
        "output_path": str(output_path),
        "file_size_bytes": file_size_bytes,
        "max_download_records": request.max_download_records,
        "records_downloaded": records_downloaded,
        "data_completeness_status": completeness,
        "truncation_warning": truncation,
        "created_at_utc": _utc_iso(),
        "databento_sdk_version": databento_sdk_version,
        "api_call_parameters": dict(api_call_parameters),
        "physical_limit_applied": physical_limit_applied,
        "physical_limit_type": physical_limit_type,
        "physical_limit_value": physical_limit_value,
    }
    metadata_path = metadata_path_for(output_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = metadata_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    os.replace(tmp, metadata_path)
    return metadata_path


def read_metadata_file(metadata_path: Path) -> Optional[dict]:
    if not metadata_path.exists():
        return None
    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def find_metadata_for_raw_dir(raw_dir: Path) -> list[dict]:
    out: list[dict] = []
    if not raw_dir.exists():
        return out
    for meta in raw_dir.glob("*.metadata.json"):
        doc = read_metadata_file(meta)
        if doc is not None:
            out.append(doc)
    return out


def directory_completeness_status(metadata_docs: Iterable[dict]) -> str:
    docs = list(metadata_docs)
    if not docs:
        return DATA_COMPLETENESS_UNKNOWN_NO_RECORD_LIMIT
    statuses = {d.get("data_completeness_status") for d in docs}
    if DATA_COMPLETENESS_TRUNCATED_BY_LIMIT in statuses:
        return DATA_COMPLETENESS_TRUNCATED_BY_LIMIT
    if statuses == {DATA_COMPLETENESS_COMPLETE_WITHIN_LIMIT}:
        return DATA_COMPLETENESS_COMPLETE_WITHIN_LIMIT
    return DATA_COMPLETENESS_UNKNOWN_ROW_COUNT


def run_download(
    request: DownloadRequest, *, client_factory: Optional[callable] = None
) -> DownloadResult:
    if request.dry_run or request.metadata_only:
        api_params = _api_call_parameters(
            request, with_limit=request.max_download_records is not None
        )
        return DownloadResult(
            status=DOWNLOAD_METADATA_ONLY if request.metadata_only else DOWNLOAD_OK,
            data_completeness_status=None,
            truncation_warning=False,
            records_downloaded=None,
            output_path=None,
            metadata_path=None,
            file_size_bytes=None,
            physical_limit_applied=False,
            physical_limit_type=None,
            physical_limit_value=None,
            api_call_parameters=api_params,
            databento_sdk_version=_databento_version(),
            critical_warnings=(),
            warnings=("dry-run/metadata-only: no network call",),
            error=None,
        )
    if request.max_download_records is None and (not request.allow_download_without_physical_limit):
        return DownloadResult(
            status=DOWNLOAD_BLOCKED_NO_RECORD_LIMIT,
            data_completeness_status=DATA_COMPLETENESS_UNKNOWN_NO_RECORD_LIMIT,
            truncation_warning=False,
            records_downloaded=None,
            output_path=None,
            metadata_path=None,
            file_size_bytes=None,
            physical_limit_applied=False,
            physical_limit_type=None,
            physical_limit_value=None,
            api_call_parameters={},
            databento_sdk_version=_databento_version(),
            critical_warnings=(),
            warnings=(),
            error="--max-download-records is required for a real Databento download; pass it explicitly or --allow-download-without-physical-limit.",
        )
    api_key_present = request.api_key_present
    if api_key_present is None:
        api_key_present = bool(os.environ.get("DATABENTO_API_KEY"))
    if not api_key_present and client_factory is None:
        return DownloadResult(
            status=DOWNLOAD_BLOCKED_MISSING_API_KEY,
            data_completeness_status=None,
            truncation_warning=False,
            records_downloaded=None,
            output_path=None,
            metadata_path=None,
            file_size_bytes=None,
            physical_limit_applied=False,
            physical_limit_type=None,
            physical_limit_value=None,
            api_call_parameters={},
            databento_sdk_version=_databento_version(),
            critical_warnings=(),
            warnings=(),
            error="DATABENTO_API_KEY not set; refusing to construct client",
        )
    try:
        if client_factory is not None:
            client = client_factory()
        else:
            import databento

            client = databento.Historical(key=os.environ["DATABENTO_API_KEY"])
    except Exception as exc:
        return DownloadResult(
            status=DOWNLOAD_FAILED,
            data_completeness_status=None,
            truncation_warning=False,
            records_downloaded=None,
            output_path=None,
            metadata_path=None,
            file_size_bytes=None,
            physical_limit_applied=False,
            physical_limit_type=None,
            physical_limit_value=None,
            api_call_parameters={},
            databento_sdk_version=_databento_version(),
            critical_warnings=(),
            warnings=(),
            error=f"client construction failed: {type(exc).__name__}: {exc}",
        )
    supports_limit = _sdk_supports_limit(client)
    critical_warnings: list[str] = []
    physical_limit_applied = False
    physical_limit_type: Optional[str] = None
    physical_limit_value: Optional[int] = None
    if request.max_download_records is not None:
        if not supports_limit:
            if not request.allow_download_without_physical_limit:
                return DownloadResult(
                    status=DOWNLOAD_BLOCKED_PHYSICAL_LIMIT_UNSUPPORTED,
                    data_completeness_status=None,
                    truncation_warning=False,
                    records_downloaded=None,
                    output_path=None,
                    metadata_path=None,
                    file_size_bytes=None,
                    physical_limit_applied=False,
                    physical_limit_type=None,
                    physical_limit_value=None,
                    api_call_parameters={},
                    databento_sdk_version=_databento_version(),
                    critical_warnings=(),
                    warnings=(),
                    error="installed databento SDK does not expose limit on timeseries.get_range; real download blocked. Pass --allow-download-without-physical-limit to override.",
                )
            critical_warnings.append(CRITICAL_DOWNLOAD_WITHOUT_PHYSICAL_LIMIT)
        else:
            physical_limit_applied = True
            physical_limit_type = "record_limit"
            physical_limit_value = int(request.max_download_records)
    else:
        critical_warnings.append(CRITICAL_DOWNLOAD_WITHOUT_PHYSICAL_LIMIT)
    request.output_path.parent.mkdir(parents=True, exist_ok=True)
    api_params = _api_call_parameters(request, with_limit=physical_limit_applied)
    try:
        kwargs = dict(api_params)
        kwargs.pop("output_path", None)
        store = client.timeseries.get_range(**kwargs)
        store.to_parquet(str(request.output_path))
    except Exception as exc:
        return DownloadResult(
            status=DOWNLOAD_FAILED,
            data_completeness_status=None,
            truncation_warning=False,
            records_downloaded=None,
            output_path=None,
            metadata_path=None,
            file_size_bytes=None,
            physical_limit_applied=physical_limit_applied,
            physical_limit_type=physical_limit_type,
            physical_limit_value=physical_limit_value,
            api_call_parameters=api_params,
            databento_sdk_version=_databento_version(),
            critical_warnings=tuple(critical_warnings),
            warnings=(),
            error=f"databento get_range / to_parquet raised: {type(exc).__name__}: {exc}",
        )
    records = _count_parquet_rows(request.output_path)
    try:
        file_size = request.output_path.stat().st_size
    except OSError:
        file_size = None
    metadata_path = write_metadata_file(
        request,
        output_path=request.output_path,
        records_downloaded=records,
        file_size_bytes=file_size,
        api_call_parameters=api_params,
        physical_limit_applied=physical_limit_applied,
        physical_limit_type=physical_limit_type,
        physical_limit_value=physical_limit_value,
        databento_sdk_version=_databento_version(),
    )
    completeness = _detect_completeness(records, request.max_download_records)
    truncated = completeness == DATA_COMPLETENESS_TRUNCATED_BY_LIMIT
    warnings_out: list[str] = []
    if truncated:
        warnings_out.append("records_downloaded == max_download_records -> TRUNCATED_BY_LIMIT")
    return DownloadResult(
        status=DOWNLOAD_OK_TRUNCATED if truncated else DOWNLOAD_OK,
        data_completeness_status=completeness,
        truncation_warning=truncated,
        records_downloaded=records,
        output_path=request.output_path,
        metadata_path=metadata_path,
        file_size_bytes=file_size,
        physical_limit_applied=physical_limit_applied,
        physical_limit_type=physical_limit_type,
        physical_limit_value=physical_limit_value,
        api_call_parameters=api_params,
        databento_sdk_version=_databento_version(),
        critical_warnings=tuple(critical_warnings),
        warnings=tuple(warnings_out),
        error=None,
    )


def _api_call_parameters(request: DownloadRequest, *, with_limit: bool) -> dict:
    params: dict = {
        "dataset": request.dataset,
        "schema": request.schema,
        "symbols": list(request.symbols),
        "stype_in": request.stype_in,
        "start": request.start_utc,
        "end": request.end_utc,
    }
    if with_limit and request.max_download_records is not None:
        params["limit"] = int(request.max_download_records)
    return params
