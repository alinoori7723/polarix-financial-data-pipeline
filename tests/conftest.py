from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(autouse=True)
def _isolate_databento_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABENTO_API_KEY", raising=False)
