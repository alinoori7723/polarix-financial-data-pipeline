from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SENSITIVE_MARKERS = (
    "PRIVATE_LOGIN",
    "PRIVATE_ACCOUNT_NAME",
    "PRIVATE_BALANCE",
    "PRIVATE_SERVER",
    "PRIVATE_TERMINAL_PATH",
    "PRIVATE_DATA_PATH",
    "PRIVATE_ERROR_DETAILS",
)


class FakeMT5:
    def __init__(self) -> None:
        self.initialized = True
        self.shutdown_called = False
        self.account = SimpleNamespace(
            login=SENSITIVE_MARKERS[0],
            name=SENSITIVE_MARKERS[1],
            balance=SENSITIVE_MARKERS[2],
            server=SENSITIVE_MARKERS[3],
        )
        self.terminal = SimpleNamespace(
            path=SENSITIVE_MARKERS[4],
            data_path=SENSITIVE_MARKERS[5],
        )

    def initialize(self) -> bool:
        return self.initialized

    def shutdown(self) -> None:
        self.shutdown_called = True

    def terminal_info(self):
        return self.terminal

    def account_info(self):
        return self.account

    def last_error(self):
        return (-10003, SENSITIVE_MARKERS[6])

    def symbol_select(self, symbol, enabled):
        return True

    def symbol_info(self, symbol):
        return None

    def symbol_info_tick(self, symbol):
        return None


def load_smoke_test(monkeypatch, client):
    monkeypatch.setitem(sys.modules, "MetaTrader5", client)
    path = Path(__file__).resolve().parents[1] / "scripts" / "mt5_smoke_test.py"
    spec = importlib.util.spec_from_file_location("polarix_mt5_smoke", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reports_presence_without_account_or_terminal_details(monkeypatch, capsys) -> None:
    client = FakeMT5()
    smoke = load_smoke_test(monkeypatch, client)
    assert smoke.main() == 0
    output = capsys.readouterr()
    assert "Terminal: available" in output.out
    assert "Account: available" in output.out
    assert "SYMBOL: SPX500" in output.out
    assert "SYMBOL: NDX100" in output.out
    assert not any(marker in output.out + output.err for marker in SENSITIVE_MARKERS)
    assert client.shutdown_called


@pytest.mark.parametrize("missing", ["terminal", "account"])
def test_missing_connection_metadata_fails_and_closes_client(monkeypatch, capsys, missing) -> None:
    client = FakeMT5()
    setattr(client, missing, None)
    smoke = load_smoke_test(monkeypatch, client)
    assert smoke.main() == 1
    output = capsys.readouterr()
    assert "ERROR:" in output.out
    assert "SYMBOL:" not in output.out
    assert not any(marker in output.out + output.err for marker in SENSITIVE_MARKERS)
    assert client.shutdown_called


def test_initialization_failure_reports_only_numeric_error_code(monkeypatch, capsys) -> None:
    client = FakeMT5()
    client.initialized = False
    smoke = load_smoke_test(monkeypatch, client)
    assert smoke.main() == 1
    output = capsys.readouterr()
    assert "code: -10003" in output.out
    assert not any(marker in output.out + output.err for marker in SENSITIVE_MARKERS)
