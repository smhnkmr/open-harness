from __future__ import annotations

from ledger.models import Account, Transaction


def test_account_to_dict_round_trip():
    account = Account(id="chk", name="Checking", opening_balance=100.0)
    data = account.to_dict()
    assert Account.from_dict(data) == account


def test_account_defaults_to_zero_opening_balance():
    account = Account(id="chk", name="Checking")
    assert account.opening_balance == 0.0


def test_transaction_to_dict_round_trip():
    txn = Transaction(account_id="chk", amount=-12.5, description="coffee")
    data = txn.to_dict()
    assert Transaction.from_dict(data) == txn


def test_transaction_is_credit_and_is_debit():
    credit = Transaction(account_id="chk", amount=10.0)
    debit = Transaction(account_id="chk", amount=-10.0)
    assert credit.is_credit and not credit.is_debit
    assert debit.is_debit and not debit.is_credit
