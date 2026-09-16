from __future__ import annotations

import json

from ledger.cli import main


def test_add_account_then_balance(tmp_path, capsys):
    ledger_file = tmp_path / "ledger.json"
    main(["--file", str(ledger_file), "add-account", "chk", "Checking", "--opening-balance", "100"])
    main(["--file", str(ledger_file), "balance", "chk"])
    out = capsys.readouterr().out
    assert out.strip() == "100.00"


def test_add_transaction_updates_balance(tmp_path, capsys):
    ledger_file = tmp_path / "ledger.json"
    main(["--file", str(ledger_file), "add-account", "chk", "Checking", "--opening-balance", "100"])
    main(["--file", str(ledger_file), "add", "chk", "-20.5", "--description", "coffee"])
    main(["--file", str(ledger_file), "balance", "chk"])
    out = capsys.readouterr().out
    assert out.strip() == "79.50"


def test_list_prints_each_transaction(tmp_path, capsys):
    ledger_file = tmp_path / "ledger.json"
    main(["--file", str(ledger_file), "add-account", "chk", "Checking"])
    main(["--file", str(ledger_file), "add", "chk", "10", "--description", "gift"])
    main(["--file", str(ledger_file), "add", "chk", "-3", "--description", "snack"])
    capsys.readouterr()
    main(["--file", str(ledger_file), "list", "chk"])
    out = capsys.readouterr().out
    assert "gift" in out
    assert "snack" in out


def test_add_account_persists_to_file(tmp_path):
    ledger_file = tmp_path / "ledger.json"
    main(["--file", str(ledger_file), "add-account", "chk", "Checking", "--opening-balance", "10"])
    data = json.loads(ledger_file.read_text(encoding="utf-8"))
    assert data["accounts"][0]["id"] == "chk"
