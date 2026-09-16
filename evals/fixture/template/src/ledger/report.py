"""Balance and summary reporting over a `Store`.

Everything here is a pure function of a `Store` (plus, for `account_balance`,
one account id): nothing in this module touches the filesystem or mutates
its arguments.
"""

from __future__ import annotations

from ledger.store import Store


def account_balance(store: Store, account_id: str) -> float:
    """Return the current balance for `account_id`.

    The balance is the account's opening balance plus every transaction
    recorded against it, in order.
    """
    account = store.get_account(account_id)
    total = account.opening_balance
    for txn in store.transactions_for(account_id):
        total += txn.amount
    return round(total, 2)


def all_balances(store: Store) -> dict[str, float]:
    """Return a mapping of account id to current balance, for every account."""
    return {account_id: account_balance(store, account_id) for account_id in store.accounts}


def overdrawn_accounts(store: Store) -> list[str]:
    """Return the ids of accounts whose current balance is negative.

    The result is sorted so callers get a stable order regardless of how
    the accounts were inserted into the store.
    """
    return sorted(account_id for account_id, balance in all_balances(store).items() if balance < 0)


def transaction_count(store: Store, account_id: str) -> int:
    """Return how many transactions have been recorded against `account_id`.

    Raises if the account doesn't exist, same as `account_balance`, so
    callers can't silently get a count of zero for a typo'd account id.
    """
    store.get_account(account_id)
    return len(store.transactions_for(account_id))


def format_summary(store: Store) -> str:
    """Return a one-line-per-account balance report, sorted by account id."""
    lines = []
    for account_id in sorted(store.accounts):
        account = store.accounts[account_id]
        balance = account_balance(store, account_id)
        lines.append(f"{account.name} ({account_id}): {balance:.2f}")
    return "\n".join(lines)
