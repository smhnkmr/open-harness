# ledger

A small personal-finance ledger, used as a synthetic project for evaluating
coding agents. It tracks accounts and transactions in a single JSON file and
offers a report module plus a command-line interface.

## Layout

- `src/ledger/models.py` — the `Account` and `Transaction` dataclasses.
- `src/ledger/store.py` — loads and saves the ledger as JSON.
- `src/ledger/report.py` — balance and summary calculations over a store.
- `src/ledger/cli.py` — an argparse-based CLI (`add-account`, `add`, `list`, `balance`).
- `tests/` — pytest coverage for the modules above.

## Usage

```console
$ python -m ledger.cli --file mine.json add-account chk Checking --opening-balance 100
$ python -m ledger.cli --file mine.json add chk -20.5 --description coffee
$ python -m ledger.cli --file mine.json balance chk
79.50
```

## Development

```console
$ pip install -e .[dev]
$ pytest
$ ruff check .
```
