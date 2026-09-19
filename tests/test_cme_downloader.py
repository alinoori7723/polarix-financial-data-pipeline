from __future__ import annotations

import ast
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from polarix.ingestion import cme_downloader as cme_dl


class _FakeStore:
    def __init__(self, table: pa.Table) -> None:
        self._table = table

    def to_parquet(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(self._table, path)


class _FakeTimeseriesWithLimit:
    def __init__(self, rows_to_emit: int) -> None:
        self._rows_to_emit = rows_to_emit
        self.calls: list[dict] = []

    def get_range(
        self,
        *,
        dataset: str,
        schema: str,
        symbols,
        stype_in: str,
        start: str,
        end: str,
        limit: int | None = None,
    ) -> _FakeStore:
        self.calls.append(
            {
                "dataset": dataset,
                "schema": schema,
                "symbols": list(symbols),
                "stype_in": stype_in,
                "start": start,
                "end": end,
                "limit": limit,
            }
        )
        n = self._rows_to_emit
        if limit is not None:
            n = min(n, limit)
        table = pa.table({"price": [1.0] * n, "size": [1] * n})
        return _FakeStore(table)


class _FakeTimeseriesNoLimit:
    def __init__(self, rows_to_emit: int) -> None:
        self._rows_to_emit = rows_to_emit
        self.calls: list[dict] = []

    def get_range(
        self, *, dataset: str, schema: str, symbols, stype_in: str, start: str, end: str
    ) -> _FakeStore:
        self.calls.append(
            {
                "dataset": dataset,
                "schema": schema,
                "symbols": list(symbols),
                "stype_in": stype_in,
                "start": start,
                "end": end,
            }
        )
        table = pa.table({"price": [1.0] * self._rows_to_emit, "size": [1] * self._rows_to_emit})
        return _FakeStore(table)


class _FakeClient:
    def __init__(self, timeseries) -> None:
        self.timeseries = timeseries


def _request(
    tmp_path: Path, *, max_records, allow_override=False, dry_run=False, metadata_only=False
) -> cme_dl.DownloadRequest:
    return cme_dl.DownloadRequest(
        dataset="GLBX.MDP3",
        schema="mbp-1",
        symbols=("ES.c.0",),
        stype_in="continuous",
        start_utc="2026-04-01T12:00:00Z",
        end_utc="2026-04-01T13:00:00Z",
        output_path=tmp_path / "raw" / "es" / "out.parquet",
        max_download_records=max_records,
        allow_download_without_physical_limit=allow_override,
        metadata_only=metadata_only,
        dry_run=dry_run,
        api_key_present=True,
    )


def test_1_passes_limit_into_get_range(tmp_path):
    ts = _FakeTimeseriesWithLimit(rows_to_emit=5)

    def factory():
        return _FakeClient(ts)

    req = _request(tmp_path, max_records=1000)
    result = cme_dl.run_download(req, client_factory=factory)
    assert result.status == cme_dl.DOWNLOAD_OK
    assert ts.calls and ts.calls[0]["limit"] == 1000
    assert result.physical_limit_applied is True
    assert result.physical_limit_type == "record_limit"
    assert result.physical_limit_value == 1000


def test_2_fails_closed_without_limit_and_without_override(tmp_path):
    req = _request(tmp_path, max_records=None, allow_override=False)
    result = cme_dl.run_download(
        req, client_factory=lambda: _FakeClient(_FakeTimeseriesWithLimit(0))
    )
    assert result.status == cme_dl.DOWNLOAD_BLOCKED_NO_RECORD_LIMIT
    assert result.data_completeness_status == cme_dl.DATA_COMPLETENESS_UNKNOWN_NO_RECORD_LIMIT
    assert "max-download-records" in (result.error or "")


def test_3_override_records_critical_warning(tmp_path):
    ts = _FakeTimeseriesWithLimit(rows_to_emit=2)
    req = _request(tmp_path, max_records=None, allow_override=True)
    result = cme_dl.run_download(req, client_factory=lambda: _FakeClient(ts))
    assert result.status == cme_dl.DOWNLOAD_OK
    assert cme_dl.CRITICAL_DOWNLOAD_WITHOUT_PHYSICAL_LIMIT in result.critical_warnings


def test_4_counts_parquet_rows_and_writes_metadata(tmp_path):
    ts = _FakeTimeseriesWithLimit(rows_to_emit=3)
    req = _request(tmp_path, max_records=10)
    result = cme_dl.run_download(req, client_factory=lambda: _FakeClient(ts))
    assert result.status == cme_dl.DOWNLOAD_OK
    assert result.records_downloaded == 3
    md = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert md["records_downloaded"] == 3
    assert md["max_download_records"] == 10
    assert md["data_completeness_status"] == cme_dl.DATA_COMPLETENESS_COMPLETE_WITHIN_LIMIT


def test_5_truncation_detected_when_records_match_limit(tmp_path):
    ts = _FakeTimeseriesWithLimit(rows_to_emit=999999)
    req = _request(tmp_path, max_records=100)
    result = cme_dl.run_download(req, client_factory=lambda: _FakeClient(ts))
    assert result.status == cme_dl.DOWNLOAD_OK_TRUNCATED
    assert result.records_downloaded == 100
    assert result.data_completeness_status == cme_dl.DATA_COMPLETENESS_TRUNCATED_BY_LIMIT
    assert result.truncation_warning is True
    md = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert md["truncation_warning"] is True


def test_6_complete_below_limit(tmp_path):
    ts = _FakeTimeseriesWithLimit(rows_to_emit=5)
    req = _request(tmp_path, max_records=100)
    result = cme_dl.run_download(req, client_factory=lambda: _FakeClient(ts))
    assert result.data_completeness_status == cme_dl.DATA_COMPLETENESS_COMPLETE_WITHIN_LIMIT
    assert result.truncation_warning is False


def test_7_api_params_carry_real_limit(tmp_path):
    ts = _FakeTimeseriesWithLimit(rows_to_emit=4)
    req = _request(tmp_path, max_records=50)
    result = cme_dl.run_download(req, client_factory=lambda: _FakeClient(ts))
    assert result.api_call_parameters["limit"] == 50
    assert result.physical_limit_applied is True


def test_8_blocks_when_sdk_lacks_limit_and_no_override(tmp_path):
    ts = _FakeTimeseriesNoLimit(rows_to_emit=3)
    req = _request(tmp_path, max_records=100, allow_override=False)
    result = cme_dl.run_download(req, client_factory=lambda: _FakeClient(ts))
    assert result.status == cme_dl.DOWNLOAD_BLOCKED_PHYSICAL_LIMIT_UNSUPPORTED
    assert result.physical_limit_applied is False
    assert ts.calls == []


def test_9_override_when_sdk_lacks_limit_records_critical_warning(tmp_path):
    ts = _FakeTimeseriesNoLimit(rows_to_emit=3)
    req = _request(tmp_path, max_records=100, allow_override=True)
    result = cme_dl.run_download(req, client_factory=lambda: _FakeClient(ts))
    assert result.status == cme_dl.DOWNLOAD_OK
    assert cme_dl.CRITICAL_DOWNLOAD_WITHOUT_PHYSICAL_LIMIT in result.critical_warnings
    assert "limit" not in result.api_call_parameters
    assert result.physical_limit_applied is False


def test_10_dry_run_does_not_construct_client(tmp_path):
    calls = []

    def factory():
        calls.append(1)
        return _FakeClient(_FakeTimeseriesWithLimit(0))

    req = _request(tmp_path, max_records=100, dry_run=True)
    result = cme_dl.run_download(req, client_factory=factory)
    assert calls == []
    assert result.status == cme_dl.DOWNLOAD_OK
    assert result.records_downloaded is None
    assert not req.output_path.exists()


def test_10b_metadata_only_does_not_construct_client(tmp_path):
    calls = []

    def factory():
        calls.append(1)
        return _FakeClient(_FakeTimeseriesWithLimit(0))

    req = _request(tmp_path, max_records=100, metadata_only=True)
    result = cme_dl.run_download(req, client_factory=factory)
    assert calls == []
    assert result.status == cme_dl.DOWNLOAD_METADATA_ONLY


def test_11_directory_completeness_aggregation():
    f = cme_dl.directory_completeness_status
    assert f([]) == cme_dl.DATA_COMPLETENESS_UNKNOWN_NO_RECORD_LIMIT
    docs_all_complete = [
        {"data_completeness_status": cme_dl.DATA_COMPLETENESS_COMPLETE_WITHIN_LIMIT},
        {"data_completeness_status": cme_dl.DATA_COMPLETENESS_COMPLETE_WITHIN_LIMIT},
    ]
    assert f(docs_all_complete) == cme_dl.DATA_COMPLETENESS_COMPLETE_WITHIN_LIMIT
    docs_one_trunc = [
        {"data_completeness_status": cme_dl.DATA_COMPLETENESS_COMPLETE_WITHIN_LIMIT},
        {"data_completeness_status": cme_dl.DATA_COMPLETENESS_TRUNCATED_BY_LIMIT},
    ]
    assert f(docs_one_trunc) == cme_dl.DATA_COMPLETENESS_TRUNCATED_BY_LIMIT


def test_12_only_cme_downloader_calls_get_range():
    repo_root = Path(__file__).resolve().parents[1]
    src = repo_root / "src"
    offenders: list[str] = []
    for py in src.rglob("*.py"):
        if py.name == "cme_downloader.py":
            continue
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            if (
                isinstance(fn, ast.Attribute)
                and fn.attr == "get_range"
                and isinstance(fn.value, ast.Attribute)
                and (fn.value.attr == "timeseries")
            ):
                offenders.append(f"{py.relative_to(repo_root)}:{node.lineno}")
    assert offenders == [], f"unexpected get_range callers: {offenders}"


def test_13_download_script_does_not_import_databento_directly():
    repo_root = Path(__file__).resolve().parents[1]
    src = (repo_root / "scripts" / "download_databento_sample.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name != "databento", (
                    "download_databento_sample.py must not import the databento SDK directly; it must go through cme_downloader"
                )
        elif isinstance(node, ast.ImportFrom):
            assert node.module != "databento", (
                "download_databento_sample.py must not import from databento"
            )
