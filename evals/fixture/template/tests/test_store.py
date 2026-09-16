from __future__ import annotations

import pytest

from ledger.models import Account, Transaction
from ledger.store import Store


def test_add_account_then_get_account():
    store = Store()
    store.add_account(Account(id="chk", name="Checking", opening_balance=50.0))
    assert store.get_account("chk").name == "Checking"


def test_add_account_duplicate_raises():
    store = Store()
    store.add_account(Account(id="chk", name="Checking"))
    with pytest.raises(ValueError):
        store.add_account(Account(id="chk", name="Checking Again"))


def test_get_account_unknown_raises_key_error():
    store = Store()
    with pytest.raises(KeyError):
        store.get_account("nope")


def test_add_transaction_unknown_account_raises():
    store = Store()
    with pytest.raises(KeyError):
        store.add_transaction(Transaction(account_id="nope", amount=5.0))


def test_transactions_for_filters_by_account():
    store = Store()
    store.add_account(Account(id="chk", name="Checking"))
    store.add_account(Account(id="sav", name="Savings"))
    store.add_transaction(Transaction(account_id="chk", amount=10.0))
    store.add_transaction(Transaction(account_id="sav", amount=5.0))
    assert [t.account_id for t in store.transactions_for("chk")] == ["chk"]


def test_save_and_load_round_trip(tmp_path):
    store = Store()
    store.add_account(Account(id="chk", name="Checking", opening_balance=100.0))
    store.add_transaction(Transaction(account_id="chk", amount=-10.0, description="coffee"))
    path = tmp_path / "ledger.json"
    store.save(path)

    loaded = Store.load(path)
    assert loaded.get_account("chk").name == "Checking"
    assert [t.description for t in loaded.transactions_for("chk")] == ["coffee"]


def test_load_missing_file_returns_empty_store(tmp_path):
    store = Store.load(tmp_path / "does-not-exist.json")
    assert store.accounts == {}
    assert store.transactions == []
