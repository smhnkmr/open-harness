from __future__ import annotations

from ledger.models import Account, Transaction
from ledger.report import (
    account_balance,
    all_balances,
    format_summary,
    overdrawn_accounts,
    transaction_count,
)
from ledger.store import Store


def _store_with_two_accounts():
    store = Store()
    store.add_account(Account(id="chk", name="Checking", opening_balance=100.0))
    store.add_account(Account(id="sav", name="Savings", opening_balance=50.0))
    store.add_transaction(Transaction(account_id="chk", amount=-20.0, description="coffee"))
    store.add_transaction(Transaction(account_id="sav", amount=5.0, description="interest"))
    return store


def test_account_balance_adds_opening_balance_and_transactions():
    store = _store_with_two_accounts()
    assert account_balance(store, "chk") == 80.0
    assert account_balance(store, "sav") == 55.0


def test_account_balance_with_no_transactions_is_opening_balance():
    store = Store()
    store.add_account(Account(id="chk", name="Checking", opening_balance=42.0))
    assert account_balance(store, "chk") == 42.0


def test_all_balances_covers_every_account():
    store = _store_with_two_accounts()
    assert all_balances(store) == {"chk": 80.0, "sav": 55.0}


def test_overdrawn_accounts_flags_negative_balance():
    store = Store()
    store.add_account(Account(id="chk", name="Checking", opening_balance=10.0))
    store.add_transaction(Transaction(account_id="chk", amount=-25.0, description="rent"))
    assert overdrawn_accounts(store) == ["chk"]


def test_overdrawn_accounts_empty_when_all_positive():
    store = _store_with_two_accounts()
    assert overdrawn_accounts(store) == []


def test_transaction_count_counts_only_that_account():
    store = _store_with_two_accounts()
    assert transaction_count(store, "chk") == 1
    assert transaction_count(store, "sav") == 1


def test_format_summary_lists_every_account_sorted_by_id():
    store = _store_with_two_accounts()
    summary = format_summary(store)
    assert summary == "Checking (chk): 80.00\nSavings (sav): 55.00"
