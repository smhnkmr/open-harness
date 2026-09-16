"""Stream-json stdio client and CLI entry point.

Spec: open-harness-spec.md sections 12.1, 12.2, 12.3.

Protocol
--------
stdin lines are JSON ops:
    {"op": "turn_input", "text": "..."}
    {"op": "approve", "request_id": "...", "behavior": "allow" | "deny" | "allow_always"}
    {"op": "interrupt"}
    {"op": "shutdown"}
    ... plus set_mode, set_role, compact, new_context (spec 12.1), forwarded to
    the loop unchanged by `next_op`.

stdout lines are JSON events mirroring event-log records, plus two
client-only kinds not persisted verbatim in the log:
    {"kind": "approval_request", "request_id", "tool", "content", "reason", "suggested_rule"}
    {"kind": "user_question", "request_id", "question", "options"}
    {"kind": "result", "reason": <TerminalReason>, "turns": n, "usage": {...}}
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import uuid
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any, TextIO

from open_harness.config import load_config
from open_harness.kernel.log import EventLog, new_session
from open_harness.policy.types import Decision, ToolCallRequest


class StdioClient:
    """Reads ops from stdin on a background thread, writes events to stdout.

    A dedicated reader thread drains stdin continuously so that `interrupted()`
    reflects an `{"op": "interrupt"}` line the instant it arrives, even while
    the main thread is blocked inside a tool call or waiting on the model --
    not just when it happens to call `next_op()`. Every other op is queued in
    arrival order; `next_op()` and the `ask_*` waiters both pull from that same
    queue, so nothing sent on stdin is ever dropped.

    `ask_human` and `ask_user` both block on a matching `approve` op keyed by
    `request_id` (the vocabulary in `kernel.events.OpKind` has no separate
    "answer" op, so `ask_user` reuses `approve`'s shape: its `behavior` field
    carries the chosen answer text -- see the report for this contract note).
    """

    def __init__(
        self,
        stdin: TextIO | None = None,
        stdout: TextIO | None = None,
        *,
        quiet: bool = False,
        non_interactive: bool = False,
        background: bool = True,
    ) -> None:
        self._stdin: TextIO = stdin if stdin is not None else sys.stdin
        self._stdout: TextIO = stdout if stdout is not None else sys.stdout
        self.quiet = quiet
        self.non_interactive = non_interactive

        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._queue: deque[dict[str, Any]] = deque()
        self._interrupted = False
        self._eof = False

        self._thread: threading.Thread | None = None
        if background:
            self._thread = threading.Thread(target=self._reader_loop, name="stdio-reader", daemon=True)
            self._thread.start()

    # ------------------------------------------------------------------ reader thread

    def _reader_loop(self) -> None:
        while True:
            line = self._stdin.readline()
            with self._cv:
                if line == "":
                    self._eof = True
                    self._cv.notify_all()
                    return
                text = line.strip()
                if not text:
                    continue
                try:
                    op = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if op.get("op") == "interrupt":
                    # `interrupted()` is the sole signal for this op; it is
                    # not also surfaced through next_op()/ask_*'s queue.
                    self._interrupted = True
                    self._cv.notify_all()
                    continue
                self._queue.append(op)
                self._cv.notify_all()

    def interrupted(self) -> bool:
        with self._lock:
            return self._interrupted

    # ------------------------------------------------------------------ ops in / events out

    def emit(self, record: dict[str, Any]) -> None:
        if self.quiet:
            return
        self._stdout.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._stdout.flush()

    def next_op(self) -> dict[str, Any] | None:
        with self._cv:
            while not self._queue and not self._eof:
                self._cv.wait()
            if self._queue:
                return self._queue.popleft()
            return None

    def _wait_for(self, predicate: Callable[[dict[str, Any]], bool]) -> dict[str, Any] | None:
        with self._cv:
            while True:
                for i, op in enumerate(self._queue):
                    if predicate(op):
                        del self._queue[i]
                        return op
                if self._eof:
                    return None
                self._cv.wait()

    # ------------------------------------------------------------------ human-in-the-loop

    def ask_human(self, req: ToolCallRequest, decision: Decision, *, request_id: str | None = None) -> str:
        request_id = request_id or str(uuid.uuid4())
        self.emit(
            {
                "kind": "approval_request",
                "request_id": request_id,
                "tool": req.tool_name,
                "content": req.permission_content,
                "reason": decision.reason,
                "suggested_rule": decision.suggested_rule,
            }
        )
        if self.non_interactive:
            # One-shot mode: nobody can answer. Fail closed and say so. (spec P2)
            self.emit({"kind": "info", "msg": f"denied (non-interactive): {req.tool_name} {req.permission_content}"})
            return "deny"
        op = self._wait_for(lambda o: o.get("op") == "approve" and o.get("request_id") in (request_id, "*"))
        if op is None:
            return "deny"
        return op.get("behavior", "deny")

    def ask_user(self, question: str, options: list[str], *, request_id: str | None = None) -> str:
        request_id = request_id or str(uuid.uuid4())
        self.emit(
            {
                "kind": "user_question",
                "request_id": request_id,
                "question": question,
                "options": options,
            }
        )
        op = self._wait_for(lambda o: o.get("op") == "approve" and o.get("request_id") in (request_id, "*"))
        if op is None:
            return ""
        return op.get("behavior", "")


# --------------------------------------------------------------------------- CLI entry point


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="open-harness")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--cwd", type=Path, default=None)
    parser.add_argument("--resume", type=str, default=None, metavar="SESSION_ID")
    parser.add_argument("-p", "--prompt", type=str, default=None)
    parser.add_argument("--output-format", choices=("text", "stream-json"), default=None)
    parser.add_argument("--show-session", type=str, default=None, metavar="ID|last",
                        help="pretty-print a session log and exit")
    parser.add_argument("--list-sessions", action="store_true", help="list session logs and exit")
    parser.add_argument("--client", choices=("auto", "stdio", "terminal"), default="auto",
                        help="which client surface to run: rich terminal REPL, raw stream-json stdio, "
                             "or auto-detect from stdin (default)")
    return parser


def _last_turn_count(log: EventLog) -> int:
    turns = 0
    for record in log.read():
        if record.get("kind") == "turn_end":
            turns = record.get("turns", turns)
    return turns


def _summed_usage(log: EventLog) -> dict[str, int]:
    usage = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}
    for record in log.read():
        if record.get("kind") != "usage":
            continue
        for key in usage:
            usage[key] += int(record.get(key, 0) or 0)
    return usage


def _drive_interactive(session: Any, client: StdioClient) -> None:
    """Fallback multi-turn dispatcher for a `Session` that only exposes
    `run_turn`, driving it from `turn_input` ops until `shutdown` or EOF."""
    while True:
        op = client.next_op()
        if op is None or op.get("op") == "shutdown":
            return
        if op.get("op") != "turn_input":
            continue
        terminal = session.run_turn(op.get("text", ""))
        client.emit(
            {
                "kind": "result",
                "reason": getattr(terminal, "reason", "completed"),
                "turns": _last_turn_count(session.log),
                "usage": _summed_usage(session.log),
            }
        )


def _session_files(root: Path) -> list[Path]:
    return sorted(root.glob("*.jsonl"), key=lambda f: f.stat().st_mtime, reverse=True)


def _list_sessions(root: Path) -> int:
    files = _session_files(root)
    if not files:
        print(f"no sessions under {root}")
        return 1
    print(f"sessions under {root} (newest first):")
    for f in files[:30]:
        first = last = ""
        turns = 0
        for rec in EventLog(f).read():
            if rec.get("kind") == "user_message" and not first:
                blocks = rec.get("message", {}).get("blocks", [])
                first = next((b.get("text", "") for b in blocks if b.get("type") == "text"), "")
            if rec.get("kind") == "turn_end":
                turns = rec.get("turns", turns)
                last = rec.get("reason", "")
        print(f"  {f.stem}  turns={turns:<3} end={last:<12} {first[:70]!r}")
    return 0


def _show_session(root: Path, which: str) -> int:
    files = _session_files(root)
    if which == "last":
        if not files:
            print(f"no sessions under {root}")
            return 1
        path = files[0]
    else:
        path = root / f"{which}.jsonl"
        if not path.exists():
            matches = [f for f in files if f.stem.startswith(which)]
            if len(matches) != 1:
                print(f"no unique session matching {which!r} under {root}")
                return 1
            path = matches[0]
    usage = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}
    print(f"session {path.stem}")
    print(f"{path}\n")
    for rec in EventLog(path).read():
        kind = rec.get("kind")
        seq = rec.get("seq")
        if kind in ("user_message", "assistant_message", "tool_result"):
            msg = rec.get("message", {})
            for b in msg.get("blocks", []):
                t = b.get("type")
                if t == "text":
                    print(f"{seq:>4}  {msg.get('role'):<9} {b.get('text', '').strip()[:300]}")
                elif t == "thinking":
                    print(f"{seq:>4}  thinking  {b.get('text', '').strip()[:160]}")
                elif t == "tool_call":
                    print(f"{seq:>4}  call      {b.get('name')} {json.dumps(b.get('args'))[:200]}")
                elif t == "tool_result":
                    flag = "ERROR " if b.get("is_error") else ""
                    print(f"{seq:>4}  result    {flag}{(b.get('text') or '').strip()[:200]}")
        elif kind == "usage":
            for k in usage:
                usage[k] += rec.get(k, 0)
            print(f"{seq:>4}  usage     in={rec.get('input')} out={rec.get('output')} "
                  f"cache_read={rec.get('cache_read')} cache_write={rec.get('cache_write')}")
        elif kind == "permission_decision":
            print(f"{seq:>4}  policy    {rec.get('tool')} -> {rec.get('behavior')} [{rec.get('step')}] {rec.get('reason')}")
        elif kind == "verifier_result":
            print(f"{seq:>4}  verify    {rec.get('stage')} ok={rec.get('ok')} {str(rec.get('detail', ''))[:120]}")
        elif kind in ("turn_start", "transition", "turn_end", "compact_boundary", "provider_error", "model_resolved", "approval_request", "approval_response", "session_start"):
            rest = {k: v for k, v in rec.items() if k not in ("seq", "ts", "kind")}
            print(f"{seq:>4}  {kind:<9} {json.dumps(rest)[:200]}")
    print(f"\ntotal usage: {usage}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. See this module's docstring for the wire protocol.

    Expected interface of `open_harness.kernel.loop.Session`, imported lazily
    so this module (and its tests) work even when the loop is unavailable:

        class Session:
            def __init__(self, config: Config, cwd: Path, client: StdioClient,
                         log: EventLog) -> None: ...

            def run_turn(self, text: str) -> Terminal:
                '''Run one user turn to completion, writing every event to
                `log` and to `client.emit()` along the way. Returns a Terminal
                (an object with at least a `.reason` attribute -- one of
                `kernel.events.TerminalReason`).'''

            def final_text(self) -> str:
                '''The most recent assistant message's text, for -p/--prompt
                output-format text.'''

            def run(self) -> None:
                '''Optional. Drive the full interactive loop by pulling ops
                from `client.next_op()` until `shutdown`/EOF, calling
                `run_turn` per `turn_input`. If absent, `main()` drives the
                same protocol itself via `run_turn` in a loop.'''
    """
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    # Model output is UTF-8; never let a legacy console code page crash the final print.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    try:
        from open_harness.kernel.loop import Session
    except ImportError as exc:
        print(
            f"open-harness: kernel loop is not available ({exc}); nothing to run.",
            file=sys.stderr,
        )
        return 1

    cwd = (args.cwd or Path.cwd()).resolve()
    output_format = args.output_format or ("text" if args.prompt is not None else "stream-json")

    config = load_config(args.config)
    if args.list_sessions:
        return _list_sessions(config.session_root)
    if args.show_session:
        return _show_session(config.session_root, args.show_session)

    if args.resume:
        log = EventLog(config.session_root / f"{args.resume}.jsonl")
    else:
        log = new_session(config.session_root)

    use_terminal = args.prompt is None and (
        args.client == "terminal" or (args.client == "auto" and sys.stdin.isatty())
    )
    if use_terminal:
        from open_harness.clients.terminal import run_terminal
        try:
            return run_terminal(config, cwd, log)
        finally:
            log.close()

    client = StdioClient(quiet=(args.prompt is not None and output_format == "text"),
                         non_interactive=args.prompt is not None)
    if output_format == "stream-json":
        # Mirror every log record to stdout so the client sees the whole turn,
        # not only approval requests and the final result. (spec 12.1)
        log.listeners.append(client.emit)

    try:
        session = Session(config, cwd, client, log)

        if args.prompt is not None:
            terminal = session.run_turn(args.prompt)
            reason = getattr(terminal, "reason", "completed")
            if output_format == "text":
                print(session.final_text())
            else:
                client.emit(
                    {
                        "kind": "result",
                        "reason": reason,
                        "turns": _last_turn_count(log),
                        "usage": _summed_usage(log),
                    }
                )
            return 0

        if hasattr(session, "run"):
            session.run()
        else:
            _drive_interactive(session, client)
        return 0
    finally:
        log.close()
