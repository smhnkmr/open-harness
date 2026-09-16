# Eval driver

Runs the same task matrix against open-harness (`oh`) and Claude Code (`cc`)
from one driver, so the two harnesses can be compared on pass rate, cost,
and speed under identical conditions. This is roadmap step 1 in
`SPEC.md` section 18.2: "everything below is judged by it."

Not part of the installed `open_harness` package; run it from the repo
root as a module.

## Prerequisites

- The project venv, with dev extras installed: `uv pip install -e ".[dev]"`
  (or `pip install -e ".[dev]"`).
- A `.env` file next to `open-harness.toml` (repo root) with
  `ANTHROPIC_API_KEY=...`. The driver never reads or prints this file
  itself -- open-harness's own config loading picks it up, and Claude
  Code picks up its own credentials the same way it does interactively.
- The `claude` CLI on `PATH`, already logged in (`claude login` once,
  interactively, before running evals -- the driver runs it non-interactively
  with stdin detached and cannot complete a login flow).

## Run

```
.venv/Scripts/python.exe -m evals.driver \
    --harness oh,cc \
    --model claude-sonnet-5 \
    --tasks all \
    --runs 5 \
    --out evals/results/2026-09-16
```

Full CLI:

```
python -m evals.driver
    --harness oh,cc            comma list, any of oh,cc (default: oh,cc)
    --model claude-sonnet-5    model id, passed to both harnesses (required)
    --tasks all|t01,t04        'all' or a comma list of task ids (default: all)
    --runs 5                   runs per (task, harness) (default: 5)
    --out evals/results/<ts>   output directory (default: evals/results/<timestamp>
                                under the repo root, auto-generated)
    --harness-python PATH      python to launch open-harness with (default: the
                                interpreter running the driver)
    --timeout S                per-turn timeout in seconds, forwarded to both
                                runners (default: each task's own timeout_s)
    --dry-run                  build each task's workspace and run its checker,
                                unmodified, with no harness invoked -- cheap
                                validation of fixtures/checkers
    --resume                   read an existing runs.jsonl under --out and skip
                                any (task, harness, run_index) already recorded
```

A quick sanity check before spending API budget:

```
.venv/Scripts/python.exe -m evals.driver --model claude-sonnet-5 --dry-run \
    --out evals/results/dry
```

This builds every selected task's workspace and runs its checker against
the *untouched* fixture, printing one `<task> dry-run PASS|FAIL|ERROR` line
per task. It never invokes open-harness or Claude Code. A checker that
passes against the untouched workspace usually means it isn't actually
checking the thing the task claims to test.

## Output layout

```
<out>/
  runs.jsonl              one RunRecord.to_json() per line, appended as each
                           run finishes (append-only, safe to tail live)
  report.md                regenerated from runs.jsonl after every run
  <task>/<harness>/run<N>/
    workspace/              the task's fixture, built fresh by
                             evals.fixture.build_workspace(..., mutate=task.mutate,
                             python=<harness python>) and git-committed, then
                             handed to the harness
    turnN.stdout.jsonl      raw harness output per turn (this dir is `run_dir`)
    turnN.stderr.txt
    oh_config.toml          oh only: the generated config for this run
    oh_sessions/<id>.jsonl  oh only: the session event log
```

The task cases themselves are also exported as Google ADK `EvalSet` JSON
at `evals/cases/eval_set.json` (SPEC.md 14.3), regenerated with
`python -m evals.evalset`; a test fails if the file is stale.

Run order is interleaved: for each task, for each run index, for each
harness in `--harness` order -- so within a given run index, `oh` and `cc`
see calls made back-to-back rather than minutes or hours apart, keeping
"API weather" (latency, any provider-side drift) roughly comparable between
the two harnesses for that run.

`report.md` is rewritten after every single run, so a matrix that is
killed partway through (Ctrl-C, a crashed shell, a laptop closing) still
leaves a readable report for everything that finished. Re-run with
`--resume` and the same `--out` to fill in the rest; already-recorded
`(task, harness, run_index)` triples are skipped, not re-run.

## Metrics

All metrics live on `RunRecord` (`evals/types.py`) and are computed the
same way regardless of harness, so the numbers are comparable:

- **api_calls** -- distinct model requests made during the task, across all
  turns (and, for open-harness, across any future sub-agents). Retries
  after an invalid tool call count as separate requests.
- **tool_calls** -- distinct tool invocations across all turns.
- **tokens** (`TokenUsage`) -- input, output, cache_read and cache_write
  token counts, summed across all requests in the task; `tokens.total` is
  their sum. Cache read/write tokens are included because both harnesses
  use prompt caching by default and excluding them would understate real
  cost and overstate apparent token efficiency.
- **cost_usd** -- computed identically for both harnesses from the same
  list price table (`evals.pricing.PRICES`), per model: `usage_by_model`
  splits the tokens by the model that consumed them (Claude Code's
  `modelUsage`, open-harness's per-call `spec`), so Haiku side calls are
  priced at Haiku rates on both sides. This is what makes `oh` and `cc`
  costs comparable at all: neither harness's own accounting is trusted for
  the comparison.
- **reported_cost_usd** -- the harness's own notion of what the task cost,
  when it has one. Claude Code reports a cost per turn; open-harness does
  not currently report cost, so this is `None` for `oh` runs. It is kept
  alongside the computed cost so the pricing-table assumption (see
  Caveats) can be checked against Claude Code's own number.
- **denials** -- permission denials the user (or the non-interactive
  policy) issued during the run.
- **verifier_failures** -- for `oh`, the count of `verifier_result` events
  with `ok=false` (lint/test gate failures before a turn could end); always
  `0` for `cc`, which has no equivalent gate.
- **wall_s** -- wall-clock seconds for the task, as measured by the
  respective runner (or by the driver itself, if the runner crashed before
  returning a record).
- **passed** / **check_detail** -- set by the driver, not the runner: after
  a run returns without `error`, the driver calls the task's own
  `check(workspace, final_texts)` against the resulting workspace and the
  final assistant text of each turn. A runner-level failure (crash,
  timeout) sets `error` instead, and the checker is skipped for that run
  (there is nothing meaningful to check).

## Fairness rules

The whole point of the driver is that a difference in the numbers reflects
a difference between the harnesses, not a difference in how they were
run. Concretely:

- Every run gets its own workspace, built fresh from the same task fixture
  by `evals.fixture.build_workspace` and git-committed before the harness
  ever sees it (so both harnesses start from an identical, diff-able base,
  and any task mutation, e.g. seeding a bug, is applied identically).
- stdin is detached for both harnesses -- neither can fall back to an
  interactive prompt mid-run; a run that would need one should fail loudly
  instead of hanging.
- Both harnesses run in their most permissive-but-still-gated mode
  (accept-edits) rather than requiring per-edit approval, so neither pays
  an interactive-approval tax the other doesn't. The same shell allow list
  is given to both (`_OH_CONFIG_TEMPLATE` rules and `CC_ALLOWED_TOOLS` in
  `evals/runners.py`): python, pytest, ruff, and read-only git, ls, cat,
  find, wc. Anything else is denied on both, and counted.
- Both harnesses get identical project instructions: `build_workspace`
  commits an `AGENTS.md` (read by open-harness) and a byte-identical
  `CLAUDE.md` (read by Claude Code) naming the interpreter that has pytest
  and ruff, since the fixture has no venv of its own.
- MCP is disabled for Claude Code, since open-harness has no MCP client
  yet (SPEC.md 18.1) -- giving `cc` extra tools would not be a fair
  comparison of the harnesses themselves.
- Runs are interleaved (see "Output layout" above) rather than run as two
  separate blocks, so neither harness is systematically favored or
  disadvantaged by when its share of the matrix happened to run.
- Both harnesses are always given the same `--model`.

## Caveats

- **Pricing.** `evals.pricing.PRICES` assumes Claude Sonnet 5 shares
  Sonnet 4.5's list price, because Sonnet 5's price was not yet confirmed
  at the time this table was written. `reported_cost_usd` on `cc` runs is
  recorded specifically so this assumption can be checked against Claude
  Code's own billed cost; if they diverge, trust `reported_cost_usd` over
  `cost_usd` for `cc` and treat cross-harness cost deltas with a grain of
  salt until `evals/pricing.py` is updated.
  Calibration from the first smoke runs (Sonnet 5, six runs): Claude
  Code's reported cost was 15 to 22 percent below the computed one, so
  Sonnet 5's real price is somewhat lower than the table; the ranking
  between harnesses is unaffected because both are priced from the same
  table.
- **n=5 is coarse.** Five runs per (task, harness) is enough to smooth out
  the worst API-latency and sampling noise, but not enough for tight
  confidence intervals on pass rate or cost -- see the spread already
  visible in SPEC.md 18.3's comparison record (e.g. "81 to 90 calls, 458 to
  561 s, $2.27 to $3.03" for a single repeated task). Treat a report as
  "probably true, worth a second look if it matters," not as a settled
  benchmark; `--runs 20` or more is worth considering before relying on a
  single comparison for a real decision.
- **Secrets.** The driver never reads `.env` and never prints API keys or
  other secrets; it only shells out to processes that read their own
  credentials the way they normally would.

## Defects the driver has found so far

- open-harness sent a read-only `find . -path ./.git -prune ... | head` to
  a human because the safety check flagged any shell command naming
  `.git`. Fixed: read-only shell commands skip the protected-directory
  check (secrets stay protected).
- open-harness never routed bare dotfile tokens such as `cat .env` through
  the protected-file check (only tokens containing a slash counted as
  paths). Fixed.
