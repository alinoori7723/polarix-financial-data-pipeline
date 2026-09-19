from __future__ import annotations

import datetime as _dt
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

DEFAULT_PRE_ROLL_MINUTES = 5
DEFAULT_POST_ROLL_MINUTES = 5
DEFAULT_MAX_RETRIES = 1
DEFAULT_SYMBOLS = ("ES.c.0", "NQ.c.0")
DEFAULT_STYPE_IN = "continuous"
DEFAULT_DATASET = "GLBX.MDP3"
DEFAULT_SCHEMA = "mbp-1"
DEFAULT_TIMEOUT_SECONDS = 1800
STATUS_DOWNLOAD_OK = "DATABENTO_DOWNLOAD_OK"
STATUS_RANGE_UNAVAILABLE = "DATABENTO_RANGE_UNAVAILABLE"
STATUS_DOWNLOAD_FAILED = "DATABENTO_DOWNLOAD_FAILED"
STATUS_DOWNLOAD_SKIPPED_EXISTS = "DATABENTO_DOWNLOAD_SKIPPED_EXISTS"
STATUS_DOWNLOAD_BLOCKED_BY_FLAGS = "DATABENTO_DOWNLOAD_BLOCKED_BY_FLAGS"
STATUS_DOWNLOAD_MISSING_API_KEY = "DATABENTO_DOWNLOAD_MISSING_API_KEY"
STATUS_DOWNLOAD_MISSING_MT5_WINDOW = "DATABENTO_DOWNLOAD_MISSING_MT5_WINDOW"
STATUS_DOWNLOAD_BLOCKED_BY_COST_GUARD = "DOWNLOAD_BLOCKED_BY_COST_GUARD"
STATUS_DOWNLOAD_BLOCKED_BY_PHYSICAL_LIMIT_GUARD = "DOWNLOAD_BLOCKED_BY_PHYSICAL_LIMIT_GUARD"
STATUS_DOWNLOAD_BLOCKED_MISSING_ACK = "DOWNLOAD_BLOCKED_MISSING_ACK"
STATUS_DOWNLOAD_BLOCKED_DRY_RUN = "DOWNLOAD_BLOCKED_DRY_RUN"
RANGE_UNAVAILABLE_TOKENS = (
    "dataset_unavailable_range",
    "data_not_available_range",
    "subscription",
    "not licensed",
    "outside available range",
    "exceeds the subscription range",
)


class DatabentoDownloadError(RuntimeError):
    pass


@dataclass(frozen=True)
class DatabentoDownloadPlan:
    date: str
    start_utc: str
    end_utc: str
    symbols: tuple[str, ...]
    stype_in: str
    dataset: str
    schema: str
    output_root: str
    output_filename: str
    argv: list[str]
    will_download: bool
    block_reason: Optional[str]
    max_retries: int

    def to_dict(self) -> dict:
        return {
            "date": self.date,
            "start_utc": self.start_utc,
            "end_utc": self.end_utc,
            "symbols": list(self.symbols),
            "stype_in": self.stype_in,
            "dataset": self.dataset,
            "schema": self.schema,
            "output_root": self.output_root,
            "output_filename": self.output_filename,
            "argv": list(self.argv),
            "will_download": self.will_download,
            "block_reason": self.block_reason,
            "max_retries": self.max_retries,
        }


def _ms_to_iso_utc(ms: int) -> str:
    return _dt.datetime.fromtimestamp(ms / 1000.0, tz=_dt.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _shift_iso(iso: str, *, minutes: int) -> str:
    t = _dt.datetime.fromisoformat(iso.replace("Z", "+00:00")).astimezone(_dt.timezone.utc)
    return (t + _dt.timedelta(minutes=minutes)).strftime("%Y-%m-%dT%H:%M:%SZ")


def assert_window_is_single_day(date: str, start_utc: str, end_utc: str) -> None:
    target = _dt.date.fromisoformat(date)
    t0 = _dt.datetime.fromisoformat(start_utc.replace("Z", "+00:00")).astimezone(_dt.timezone.utc)
    t1 = _dt.datetime.fromisoformat(end_utc.replace("Z", "+00:00")).astimezone(_dt.timezone.utc)
    duration_h = (t1 - t0).total_seconds() / 3600.0
    if duration_h <= 0:
        raise DatabentoDownloadError(f"databento plan has non-positive duration {duration_h:.3f}h")
    if duration_h > 26.0:
        raise DatabentoDownloadError(
            f"databento plan window is multi-day ({duration_h:.3f}h between {start_utc} and {end_utc}); Phase 2F forbids monolithic multi-day requests"
        )
    if not t0.date() <= target <= t1.date():
        raise DatabentoDownloadError(
            f"databento plan window {start_utc}/{end_utc} does not cover target date {date}"
        )


def _output_filename(date: str, symbols: Sequence[str]) -> str:
    sym_part = "_".join((s.replace(".", "") for s in symbols))
    return f"databento_GLBX_MDP3_mbp-1_{sym_part}_{date.replace('-', '')}.parquet"


def plan_day_download(
    date: str,
    *,
    mt5_window_ms: Optional[tuple[int, int]],
    output_root: Path | str,
    symbols: Sequence[str] = DEFAULT_SYMBOLS,
    stype_in: str = DEFAULT_STYPE_IN,
    dataset: str = DEFAULT_DATASET,
    schema: str = DEFAULT_SCHEMA,
    pre_roll_minutes: int = DEFAULT_PRE_ROLL_MINUTES,
    post_roll_minutes: int = DEFAULT_POST_ROLL_MINUTES,
    allow_databento_download: bool = False,
    download_cme: bool = False,
    api_key_present: Optional[bool] = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    max_download_records: Optional[int] = None,
    allow_download_without_physical_limit: bool = False,
) -> DatabentoDownloadPlan:
    symbols_t = tuple(symbols)
    output_root = str(Path(output_root).resolve())
    if mt5_window_ms is None:
        return DatabentoDownloadPlan(
            date=date,
            start_utc="",
            end_utc="",
            symbols=symbols_t,
            stype_in=stype_in,
            dataset=dataset,
            schema=schema,
            output_root=output_root,
            output_filename=_output_filename(date, symbols_t),
            argv=[],
            will_download=False,
            block_reason=STATUS_DOWNLOAD_MISSING_MT5_WINDOW,
            max_retries=max_retries,
        )
    mn_ms, mx_ms = mt5_window_ms
    start_utc = _shift_iso(_ms_to_iso_utc(mn_ms), minutes=-pre_roll_minutes)
    end_utc = _shift_iso(_ms_to_iso_utc(mx_ms), minutes=post_roll_minutes)
    assert_window_is_single_day(date, start_utc, end_utc)
    if api_key_present is None:
        api_key_present = bool(os.environ.get("DATABENTO_API_KEY"))
    out_dir = Path(output_root) / f"date={date}"
    output_filename = _output_filename(date, symbols_t)
    argv = [
        sys.executable,
        "-u",
        str(Path("scripts") / "download_databento_sample.py"),
        "--dataset",
        dataset,
        "--schema",
        schema,
        "--symbols",
        ",".join(symbols_t),
        "--stype-in",
        stype_in,
        "--start",
        start_utc,
        "--end",
        end_utc,
        "--output-root",
        str(out_dir),
        "--output-name",
        output_filename,
    ]
    if max_download_records is not None:
        argv.extend(["--max-download-records", str(int(max_download_records))])
    if allow_download_without_physical_limit:
        argv.append("--allow-download-without-physical-limit")
    will_download = allow_databento_download and download_cme and api_key_present
    block_reason: Optional[str] = None
    if not (allow_databento_download and download_cme):
        block_reason = STATUS_DOWNLOAD_BLOCKED_BY_FLAGS
    elif not api_key_present:
        block_reason = STATUS_DOWNLOAD_MISSING_API_KEY
    return DatabentoDownloadPlan(
        date=date,
        start_utc=start_utc,
        end_utc=end_utc,
        symbols=symbols_t,
        stype_in=stype_in,
        dataset=dataset,
        schema=schema,
        output_root=output_root,
        output_filename=output_filename,
        argv=argv,
        will_download=will_download,
        block_reason=block_reason,
        max_retries=max_retries,
    )


def _classify_error(stderr_text: str) -> Optional[str]:
    low = (stderr_text or "").lower()
    for token in RANGE_UNAVAILABLE_TOKENS:
        if token in low:
            return STATUS_RANGE_UNAVAILABLE
    return None


@dataclass
class DownloadOutcome:
    status: str
    plan: DatabentoDownloadPlan
    attempts: int
    returncode: Optional[int]
    output_path: Optional[Path]
    stdout_tail: str
    stderr_tail: str

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "attempts": self.attempts,
            "returncode": self.returncode,
            "output_path": str(self.output_path) if self.output_path else None,
            "stdout_tail": self.stdout_tail,
            "stderr_tail": self.stderr_tail,
            "plan": self.plan.to_dict(),
        }


def run_day_download(
    plan: DatabentoDownloadPlan,
    *,
    repo_root: Path,
    skip_if_exists: bool = True,
    timeout_seconds: int = 1800,
    runner: Optional["callable[[list[str], int], subprocess.CompletedProcess]"] = None,
) -> DownloadOutcome:
    output_path = Path(plan.output_root) / f"date={plan.date}" / plan.output_filename
    if not plan.will_download:
        return DownloadOutcome(
            status=plan.block_reason or STATUS_DOWNLOAD_BLOCKED_BY_FLAGS,
            plan=plan,
            attempts=0,
            returncode=None,
            output_path=None,
            stdout_tail="",
            stderr_tail="",
        )
    if skip_if_exists and output_path.exists() and (output_path.stat().st_size > 0):
        return DownloadOutcome(
            status=STATUS_DOWNLOAD_SKIPPED_EXISTS,
            plan=plan,
            attempts=0,
            returncode=None,
            output_path=output_path,
            stdout_tail="",
            stderr_tail="",
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if runner is None:

        def runner(argv, timeout):
            return subprocess.run(
                argv, cwd=str(repo_root), capture_output=True, text=True, timeout=timeout
            )

    attempts = 0
    last_rc: Optional[int] = None
    last_stdout = ""
    last_stderr = ""
    final_status = STATUS_DOWNLOAD_FAILED
    while attempts <= plan.max_retries:
        attempts += 1
        try:
            result = runner(plan.argv, timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            last_stderr = f"TIMEOUT after {timeout_seconds}s: {exc}"
            final_status = STATUS_DOWNLOAD_FAILED
            continue
        last_rc = result.returncode
        last_stdout = (result.stdout or "")[-2000:]
        last_stderr = (result.stderr or "")[-2000:]
        if last_rc == 0 and output_path.exists() and (output_path.stat().st_size > 0):
            return DownloadOutcome(
                status=STATUS_DOWNLOAD_OK,
                plan=plan,
                attempts=attempts,
                returncode=last_rc,
                output_path=output_path,
                stdout_tail=last_stdout,
                stderr_tail=last_stderr,
            )
        classified = _classify_error(last_stderr) or _classify_error(last_stdout)
        if classified == STATUS_RANGE_UNAVAILABLE:
            return DownloadOutcome(
                status=STATUS_RANGE_UNAVAILABLE,
                plan=plan,
                attempts=attempts,
                returncode=last_rc,
                output_path=None,
                stdout_tail=last_stdout,
                stderr_tail=last_stderr,
            )
        final_status = STATUS_DOWNLOAD_FAILED
    return DownloadOutcome(
        status=final_status,
        plan=plan,
        attempts=attempts,
        returncode=last_rc,
        output_path=None,
        stdout_tail=last_stdout,
        stderr_tail=last_stderr,
    )
