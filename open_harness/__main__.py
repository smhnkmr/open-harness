"""`python -m open_harness` delegates to the stdio CLI entry point."""

from __future__ import annotations

import sys

from open_harness.clients.stdio import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
