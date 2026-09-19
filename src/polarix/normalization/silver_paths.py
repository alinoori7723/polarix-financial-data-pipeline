from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

SILVER_LAYOUT_VERSION_LEGACY = "1-date-level"
SILVER_LAYOUT_VERSION_RUN_SCOPED = "2-run-scoped"
LAYOUT_RUN_SCOPED = "run_scoped"
LAYOUT_LEGACY_DATE_LEVEL = "legacy_date_level"
LAYOUT_MISSING = "missing"


def symbol_date_dir(silver_root: Path, symbol: str, date: str) -> Path:
    return Path(silver_root) / f"symbol={symbol}" / f"date={date}"


def run_scoped_silver_dir(silver_root: Path, symbol: str, date: str, run_id: str) -> Path:
    return symbol_date_dir(silver_root, symbol, date) / f"run_id={run_id}"


def legacy_silver_dir(silver_root: Path, symbol: str, date: str) -> Path:
    return symbol_date_dir(silver_root, symbol, date)


def _part_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted((Path(p) for p in glob.glob(os.path.join(str(directory), "part-*.parquet"))))


def list_run_ids(silver_root: Path, symbol: str, date: str) -> list[str]:
    date_dir = symbol_date_dir(silver_root, symbol, date)
    if not date_dir.exists():
        return []
    out: list[str] = []
    for child in date_dir.iterdir():
        if child.is_dir() and child.name.startswith("run_id="):
            run_id = child.name.split("=", 1)[1]
            if _part_files(child):
                out.append(run_id)
    out.sort()
    return out


def legacy_part_files(silver_root: Path, symbol: str, date: str) -> list[Path]:
    return _part_files(symbol_date_dir(silver_root, symbol, date))


def list_date_run_ids(
    silver_root: Path, symbol_prefix_filter: Optional[list[str]], date: str
) -> list[str]:
    root = Path(silver_root)
    if not root.exists():
        return []
    found: set[str] = set()
    for child in root.iterdir():
        if not (child.is_dir() and child.name.startswith("symbol=")):
            continue
        symbol = child.name.split("=", 1)[1]
        if symbol_prefix_filter is not None and symbol not in symbol_prefix_filter:
            continue
        found.update(list_run_ids(root, symbol, date))
    return sorted(found)


@dataclass
class SilverSelection:
    symbol: str
    date: str
    requested_run_id: Optional[str]
    layout: str
    layout_version: str
    selected_run_id: Optional[str]
    files: list[Path] = field(default_factory=list)
    available_run_ids: list[str] = field(default_factory=list)
    legacy_files_present: bool = False
    ambiguous: bool = False
    ok: bool = True
    error: Optional[str] = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "date": self.date,
            "requested_run_id": self.requested_run_id,
            "layout": self.layout,
            "silver_layout_version": self.layout_version,
            "selected_run_id": self.selected_run_id,
            "file_count": len(self.files),
            "available_run_ids": list(self.available_run_ids),
            "legacy_files_present": self.legacy_files_present,
            "ambiguous": self.ambiguous,
            "ok": self.ok,
            "error": self.error,
            "warnings": list(self.warnings),
        }


def select_silver(
    silver_root: Path, symbol: str, date: str, run_id: Optional[str] = None
) -> SilverSelection:
    available = list_run_ids(silver_root, symbol, date)
    legacy_files = legacy_part_files(silver_root, symbol, date)
    legacy_present = bool(legacy_files)
    if run_id is not None:
        run_dir = run_scoped_silver_dir(silver_root, symbol, date, run_id)
        files = _part_files(run_dir)
        if files:
            warnings: list[str] = []
            if legacy_present:
                warnings.append(
                    f"{symbol}/{date}: legacy date-level Silver part files also exist alongside run-scoped partitions; the legacy files were ignored because --run-id was supplied."
                )
            return SilverSelection(
                symbol=symbol,
                date=date,
                requested_run_id=run_id,
                layout=LAYOUT_RUN_SCOPED,
                layout_version=SILVER_LAYOUT_VERSION_RUN_SCOPED,
                selected_run_id=run_id,
                files=files,
                available_run_ids=available,
                legacy_files_present=legacy_present,
                ambiguous=False,
                ok=True,
                error=None,
                warnings=warnings,
            )
        return SilverSelection(
            symbol=symbol,
            date=date,
            requested_run_id=run_id,
            layout=LAYOUT_MISSING,
            layout_version=SILVER_LAYOUT_VERSION_RUN_SCOPED,
            selected_run_id=run_id,
            files=[],
            available_run_ids=available,
            legacy_files_present=legacy_present,
            ambiguous=False,
            ok=False,
            error=f"no run-scoped Silver partition for symbol={symbol} date={date} run_id={run_id} (available run_ids: {available or 'none'})",
        )
    if len(available) > 1:
        return SilverSelection(
            symbol=symbol,
            date=date,
            requested_run_id=None,
            layout=LAYOUT_RUN_SCOPED,
            layout_version=SILVER_LAYOUT_VERSION_RUN_SCOPED,
            selected_run_id=None,
            files=[],
            available_run_ids=available,
            legacy_files_present=legacy_present,
            ambiguous=True,
            ok=False,
            error=f"multiple run-scoped Silver partitions exist for symbol={symbol} date={date}: {available}; pass --run-id <RUN_ID> to choose one explicitly",
        )
    if len(available) == 1:
        only = available[0]
        return SilverSelection(
            symbol=symbol,
            date=date,
            requested_run_id=None,
            layout=LAYOUT_RUN_SCOPED,
            layout_version=SILVER_LAYOUT_VERSION_RUN_SCOPED,
            selected_run_id=only,
            files=_part_files(run_scoped_silver_dir(silver_root, symbol, date, only)),
            available_run_ids=available,
            legacy_files_present=legacy_present,
            ambiguous=False,
            ok=True,
            error=None,
            warnings=[
                f"{symbol}/{date}: --run-id omitted; auto-selected the only run-scoped partition run_id={only}."
            ],
        )
    if legacy_present:
        return SilverSelection(
            symbol=symbol,
            date=date,
            requested_run_id=None,
            layout=LAYOUT_LEGACY_DATE_LEVEL,
            layout_version=SILVER_LAYOUT_VERSION_LEGACY,
            selected_run_id=None,
            files=legacy_files,
            available_run_ids=[],
            legacy_files_present=True,
            ambiguous=False,
            ok=True,
            error=None,
            warnings=[
                f"{symbol}/{date}: reading LEGACY date-level Silver (non-run-scoped); these part files may contain rows from more than one run. Re-normalize with --run-id for run-isolated output."
            ],
        )
    return SilverSelection(
        symbol=symbol,
        date=date,
        requested_run_id=None,
        layout=LAYOUT_MISSING,
        layout_version=SILVER_LAYOUT_VERSION_RUN_SCOPED,
        selected_run_id=None,
        files=[],
        available_run_ids=[],
        legacy_files_present=False,
        ambiguous=False,
        ok=True,
        error=None,
        warnings=[f"{symbol}/{date}: no Silver data found."],
    )
