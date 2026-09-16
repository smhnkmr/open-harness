"""Append-only JSONL event log.

Spec: open-harness-spec.md sections 3.1, 4.3, 12.1, 13.

One JSON object per line: `{"seq": n, "ts": iso8601, "kind": ..., **payload}`.
Never rewritten or deleted; compaction adds a `compact_boundary` record, it
does not remove earlier ones. Writes are buffered and flushed per record with
no `fsync` (crash-durability is explicitly not a goal here).

Message replay record shapes (payload keys beyond seq/ts/kind), all carrying
the *whole* `Message` under `"message"` so replay never has to guess how to
regroup blocks:

- `user_message`      {"message": <serialize_message(Message(role="user", ...))>}
- `assistant_message` {"message": <serialize_message(Message(role="assistant", ...))>, ...extra fields}
- `tool_result`       {"message": <serialize_message(Message(role="user", blocks=[tool_result block], ...))>}

A `tool_result` record is exactly a `user_message` whose first block happens
to be a `tool_result` block (one record per tool result message, matching how
the loop pushes each tool's result as its own message); `replay_messages`
treats the two kinds identically.

`replay_messages` rebuilds the message list after the last `compact_boundary`
(or from the start if there is none), in log order, so the loop can hand the
result straight to `gateway.call`.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self, TextIO

from open_harness.kernel.serde import deserialize_message
from open_harness.model.types import Message


class EventLog:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._seq = self._last_seq()
        # Append mode preserves any existing content (resume). newline="" so
        # we control line endings ourselves and never write "\r\n" on Windows.
        self._fh: TextIO = self.path.open("a", encoding="utf-8", newline="")
        # Listeners receive every appended record (used to mirror the log to a client).
        self.listeners: list[Callable[[dict[str, Any]], None]] = []

    @property
    def session_id(self) -> str:
        return self.path.stem

    def _last_seq(self) -> int:
        last = 0
        for record in self.read():
            seq = record.get("seq")
            if isinstance(seq, int) and seq > last:
                last = seq
        return last

    def append(self, kind: str, **payload: Any) -> dict[str, Any]:
        self._seq += 1
        record: dict[str, Any] = {
            "seq": self._seq,
            "ts": datetime.now(UTC).isoformat(),
            "kind": kind,
            **payload,
        }
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fh.flush()
        for listener in self.listeners:
            listener(record)
        return record

    def read(self) -> Iterator[dict[str, Any]]:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)

    def last_boundary(self) -> int | None:
        seq: int | None = None
        for record in self.read():
            if record.get("kind") == "compact_boundary":
                seq = record.get("seq")
        return seq

    def replay_messages(self) -> list[Message]:
        boundary = self.last_boundary()
        messages: list[Message] = []
        for record in self.read():
            if boundary is not None and record.get("seq", 0) <= boundary:
                continue
            if record.get("kind") in ("user_message", "assistant_message", "tool_result"):
                messages.append(deserialize_message(record["message"]))
        return messages

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def new_session(root: Path) -> EventLog:
    """Create `root/<uuid>.jsonl` and return an `EventLog` over it.

    Does not write a `session_start` record itself; the loop owns turn/session
    lifecycle records and writes them with real payload (cwd, config, resolved
    roles, ...).
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    session_id = str(uuid.uuid4())
    return EventLog(root / f"{session_id}.jsonl")
