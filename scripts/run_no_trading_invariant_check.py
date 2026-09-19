from __future__ import annotations

import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
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
ALLOWLIST = {"mt5_readonly.py"}
TRADE_REQUEST_PAT = re.compile("TradeRequest|MqlTradeRequest", re.IGNORECASE)


def _files() -> list[Path]:
    return [p for p in SRC.rglob("*.py") if "__pycache__" not in p.parts]


def main() -> int:
    violations: list[dict] = []
    for path in _files():
        if path.name in ALLOWLIST:
            continue
        text = path.read_text(encoding="utf-8")
        for token in FORBIDDEN_SUBSTRINGS:
            if token in text:
                violations.append({"file": str(path), "kind": "substring", "token": token})
        if TRADE_REQUEST_PAT.search(text):
            violations.append({"file": str(path), "kind": "trade_request_struct"})
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_SUBSTRINGS:
                violations.append(
                    {"file": str(path), "kind": "attribute_access", "attr": node.attr}
                )
    report = {
        "scanned_files": len(_files()),
        "src_root": str(SRC),
        "violations": violations,
        "ok": not violations,
    }
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
