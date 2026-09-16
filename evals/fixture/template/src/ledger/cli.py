"""Command-line interface for the ledger package.

Every subcommand takes a `--file` pointing at the JSON ledger to operate on;
subcommands that change the ledger (`add-account`, `add`) save it back
before exiting, and read-only subcommands (`list`, `balance`) just print.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ledger.models import Account, Transaction
from ledger.report import account_balance
from ledger.store import Store


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ledger", description="A tiny personal ledger.")
    parser.add_argument("--file", required=True, help="path to the JSON ledger file")
    sub = parser.add_subparsers(dest="command", required=True)

    add_account = sub.add_parser("add-account", help="create a new account")
    add_account.add_argument("account_id")
    add_account.add_argument("name")
    add_account.add_argument("--opening-balance", type=float, default=0.0)

    add = sub.add_parser("add", help="record a transaction")
    add.add_argument("account_id")
    add.add_argument("amount", type=float)
    add.add_argument("--description", default="")

    list_cmd = sub.add_parser("list", help="list transactions for an account")
    list_cmd.add_argument("account_id")

    balance = sub.add_parser("balance", help="print an account's current balance")
    balance.add_argument("account_id")

    return parser


def _cmd_add_account(store: Store, args: argparse.Namespace) -> None:
    store.add_account(
        Account(id=args.account_id, name=args.name, opening_balance=args.opening_balance)
    )


def _cmd_add(store: Store, args: argparse.Namespace) -> None:
    store.add_transaction(
        Transaction(account_id=args.account_id, amount=args.amount, description=args.description)
    )


def _cmd_list(store: Store, args: argparse.Namespace) -> None:
    for txn in store.transactions_for(args.account_id):
        print(f"{txn.amount:+.2f}\t{txn.description}")


def _cmd_balance(store: Store, args: argparse.Namespace) -> None:
    print(f"{account_balance(store, args.account_id):.2f}")


_WRITE_COMMANDS = {"add-account", "add"}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    store = Store.load(args.file)

    handlers = {
        "add-account": _cmd_add_account,
        "add": _cmd_add,
        "list": _cmd_list,
        "balance": _cmd_balance,
    }
    handlers[args.command](store, args)

    if args.command in _WRITE_COMMANDS:
        store.save(Path(args.file))

    return 0


if __name__ == "__main__":
    sys.exit(main())
