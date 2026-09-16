"""Data models for the ledger package.

Both types are plain dataclasses with a `to_dict` / `from_dict` pair so the
`store` module can round-trip them through JSON without any extra
dependency (no dataclasses-json, no pydantic).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Account:
    """A named account with an opening balance.

    `id` is the short handle used everywhere else in the package (CLI
    arguments, transaction references); `name` is the human-readable label.
    """

    id: str
    name: str
    opening_balance: float = 0.0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "opening_balance": self.opening_balance,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Account:
        return cls(
            id=data["id"],
            name=data["name"],
            opening_balance=float(data.get("opening_balance", 0.0)),
        )


@dataclass
class Transaction:
    """A single credit (positive amount) or debit (negative amount).

    A transaction always belongs to exactly one account, referenced by
    `account_id`. There is no timestamp field: transactions are ordered by
    insertion order in the store, which is enough for this project's needs.
    """

    account_id: str
    amount: float
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "account_id": self.account_id,
            "amount": self.amount,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Transaction:
        return cls(
            account_id=data["account_id"],
            amount=float(data["amount"]),
            description=data.get("description", ""),
        )

    @property
    def is_credit(self) -> bool:
        return self.amount >= 0

    @property
    def is_debit(self) -> bool:
        return self.amount < 0
