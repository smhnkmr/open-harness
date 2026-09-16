"""JSON file persistence for accounts and transactions.

`Store` is an in-memory container; `Store.load` / `Store.save` are the only
places that touch the filesystem, and they always go through plain JSON so
the ledger file stays easy to read by hand.
"""

from __future__ import annotations

import json
from pathlib import Path

from ledger.models import Account, Transaction


class Store:
    """Holds every account and transaction for one ledger file."""

    def __init__(
        self,
        accounts: list[Account] | None = None,
        transactions: list[Transaction] | None = None,
    ) -> None:
        self.accounts: dict[str, Account] = {a.id: a for a in (accounts or [])}
        self.transactions: list[Transaction] = list(transactions or [])

    def add_account(self, account: Account) -> None:
        if account.id in self.accounts:
            raise ValueError(f"account already exists: {account.id!r}")
        self.accounts[account.id] = account

    def get_account(self, account_id: str) -> Account:
        try:
            return self.accounts[account_id]
        except KeyError:
            raise KeyError(f"unknown account: {account_id}") from None

    def add_transaction(self, transaction: Transaction) -> None:
        if transaction.account_id not in self.accounts:
            raise KeyError(f"unknown account: {transaction.account_id}")
        self.transactions.append(transaction)

    def transactions_for(self, account_id: str) -> list[Transaction]:
        """Return transactions for `account_id`, in insertion order."""
        return [t for t in self.transactions if t.account_id == account_id]

    def to_dict(self) -> dict:
        return {
            "accounts": [a.to_dict() for a in self.accounts.values()],
            "transactions": [t.to_dict() for t in self.transactions],
        }

    @classmethod
    def from_dict(cls, data: dict) -> Store:
        accounts = [Account.from_dict(a) for a in data.get("accounts", [])]
        transactions = [Transaction.from_dict(t) for t in data.get("transactions", [])]
        return cls(accounts=accounts, transactions=transactions)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> Store:
        """Load a store from `path`, or return an empty one if it's missing.

        A missing file is treated as "no ledger yet" rather than an error,
        so the CLI can be pointed at a fresh file without a separate init
        step.
        """
        path = Path(path)
        if not path.exists():
            return cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(data)
