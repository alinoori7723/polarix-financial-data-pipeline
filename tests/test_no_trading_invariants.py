from __future__ import annotations

import re
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
FORBIDDEN_SUBSTRINGS = (
    "order_send",
    "order_check",
    "order_calc_margin",
    "order_calc_profit",
    "position_close",
    "positions_close",
    "position_modify",
    "trade_request",
)
_ALLOWLIST_FILES = {"mt5_readonly.py", "test_no_trading_invariants.py"}


def _python_files() -> list[Path]:
    return [p for p in SRC.rglob("*.py") if "__pycache__" not in p.parts]


@pytest.mark.parametrize("forbidden", FORBIDDEN_SUBSTRINGS)
def test_no_forbidden_trading_symbol_in_source(forbidden: str):
    offenders: list[str] = []
    for path in _python_files():
        if path.name in _ALLOWLIST_FILES:
            continue
        text = path.read_text(encoding="utf-8")
        if forbidden in text:
            offenders.append(str(path))
    assert not offenders, f"forbidden symbol {forbidden!r} found in: {offenders}"


def test_logger_does_not_import_trading_attributes():
    import ast

    bad: list[tuple[str, str]] = []
    for path in _python_files():
        if path.name in _ALLOWLIST_FILES:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                name = node.attr
                if name in FORBIDDEN_SUBSTRINGS:
                    bad.append((str(path), name))
    assert not bad, f"trading attribute access detected: {bad}"


def test_no_trade_request_construction():
    pat = re.compile("TradeRequest|MqlTradeRequest", re.IGNORECASE)
    offenders = []
    for path in _python_files():
        if path.name in _ALLOWLIST_FILES:
            continue
        if pat.search(path.read_text(encoding="utf-8")):
            offenders.append(str(path))
    assert not offenders, f"trade request shape found in: {offenders}"


def test_readonly_class_has_no_trading_methods():
    from polarix.ingestion.mt5_readonly import ReadOnlyMT5

    method_names = [m for m in dir(ReadOnlyMT5) if not m.startswith("_")]
    leaked = [m for m in method_names if m in FORBIDDEN_SUBSTRINGS]
    assert not leaked, f"ReadOnlyMT5 leaked trading methods: {leaked}"
