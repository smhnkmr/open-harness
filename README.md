# open-harness

A small agent harness prototype. Stateless kernel, append-only event log, vendor-neutral model layer, and a turn that cannot end until the verifier gate passes.

The normative design is in `SPEC.md`. This README covers the prototype only.

## What is in the prototype

| Area | Included |
|---|---|
| Kernel | while-true loop with named transitions, parallel read-only tool batches, JSONL event log with resume |
| Model layer | neutral request and events, kernel-owned tool-call reducer, `anthropic` and `openai-compatible` adapters, roles |
| Tools | read, edit, write, shell, grep, glob, ask_user; large results persisted to disk with a preview |
| Policy | pooled rules, fixed decision order with bypass-immune checks, ask reducer (deterministic, then human) |
| Verification | lint after every edit, test command before a turn may end |
| Context | static prompt with cache boundary, AGENTS.md as first user message, summarise-tier compaction |
| Client | stream-json over stdio, and a `-p` one-shot mode |

Not in the prototype: OS sandbox, classifier stage, sub-agents, memory, MCP, hooks, skills, IDE. The threat model for this prototype is one line: **it runs with your user's permissions and no sandbox; the policy engine is the only guard.**

## Run

```
uv venv && uv pip install -e ".[dev]"
cp open-harness.example.toml open-harness.toml     # edit roles and keys
export ANTHROPIC_API_KEY=...

# one shot
.venv/Scripts/python -m open_harness -p "list the python files here and summarise them"

# interactive over stdio (JSON lines in, JSON lines out)
.venv/Scripts/python -m open_harness --output-format stream-json
{"op": "turn_input", "text": "add a docstring to main.py"}
```

## Live run

Verified against Anthropic Sonnet 4.5 on a scratch project with a deliberate bug. Eight turns, bug fixed, the verifier gate ran the real test command before the turn could end, about 22k tokens served from prompt cache against 56 uncached. Approval requests in one-shot mode are denied and logged rather than blocking.

## Test

```
.venv/Scripts/python -m pytest -q
```

`tests/test_e2e_fake.py` drives the real loop with a scripted adapter and is the test that proves the kernel.

## Layout

```
open_harness/
  kernel/    loop.py gateway.py roles.py log.py serde.py events.py
  model/     types.py adapter.py reducer.py schema.py registry.py adapters/{anthropic,openai_compatible}.py
  tools/     base.py results.py read.py edit.py write.py shell.py grep.py glob.py ask_user.py
  backend/   base.py local.py
  policy/    types.py rules.py safety.py engine.py reducer.py
  verify/    gate.py
  context/   prompt.py compact.py
  clients/   stdio.py
```
