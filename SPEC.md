# open-harness: design and specification

Status: living document. Version 0.5, 16 September 2026. Section 18 records what is built and what comes next; §18.4 records the decision not to build on Google ADK.
Origin: synthesised from source reading of Claude Code, opencode, Codex CLI, browser-use, Gemini CLI, OpenHands, Deep Agents and LangChain, plus Anthropic's published harness guidance. Companion explainer pages exist for each; this file is the normative record.

Conventions: MUST, SHOULD and MAY are used in the RFC sense. Anything marked `[open]` is undecided. Anything marked `[expires]` is a guard that must carry a reason and a model version and be re-tested at each model release.

---

## 1. Purpose

open-harness is an agent harness for a Fable-class model: the runtime that wraps a language model in a tool loop with context management, isolation, verification and policy so it can act on a codebase or a task over many steps.

Design goal in one sentence: spend engineering on what the model structurally lacks, remove engineering that second-guesses what the model can do.

Two facts drive every decision:

1. The harness moves benchmark scores as much as the model does. Same model, different harness, ten to fifteen points on coding benchmarks.
2. Every component encodes an assumption about what the model cannot do alone, and those assumptions go stale with each model release.

---

## 2. Principles

| # | Principle | Consequence |
|---|---|---|
| P1 | The prompt cache is physics | Static prefix, boundary marker, latched headers, append-only context, changes as tail attachments, byte-identical fork prefixes, pinned tool order |
| P2 | Fail toward asking | Parse failure means ask. Deny before allow. Some checks survive every mode |
| P3 | Truncation is never lossy | Large results go to disk with a preview and a path. Compaction is a boundary marker, not deletion |
| P4 | Everything that changes is an attachment | State changes travel as typed, harness-authored attachments, never as edits to history or as fake user messages |
| P5 | One channel for anything asynchronous | Background shells, agents, cron, remote work and stalls all re-enter as one notification type |
| P6 | The UI is one client among several | The kernel is headless. Every surface pushes operations and reads events |
| P7 | Guards carry expiry dates | Every model-compensating component lives in a profile with a reason and a model version, and is pruned by evidence |
| P8 | Branch on declared capability, never on vendor name | Adapters declare flags. The kernel never inspects a provider string |
| P9 | Verify with ground truth | Compilers, tests, linters and an independent evaluator decide when work is done. The model never grades itself |
| P10 | Ship a threat model | Trust boundaries and known weaknesses are generated and published, not hidden |

---

## 3. Architecture

Three rings.

```
┌──────────────────────────────────────────────────────────────────┐
│ CLIENTS   terminal · headless/SDK · IDE · web · sub-agent · evals │
│           push Op{id, kind, payload}  ─────►  read Event{id, msg} │
├──────────────────────────────────────────────────────────────────┤
│ KERNEL    loop (state machine) · append-only event log · gateway │
│           stateless between events; the log survives crashes     │
├──────────────────────────────────────────────────────────────────┤
│ SUBSTRATE sandboxed tools · verifiers · policy engine · memory   │
│           deterministic; never expires                            │
└──────────────────────────────────────────────────────────────────┘
```

Alongside the rings, two registries and one dataset:

- Adapter registry: vendor adapters, discovered by entry point.
- Profile registry: capability profiles (construction time) and harness profiles (runtime), keyed `provider:model` with provider fallback.
- Capability dataset: generated from models.dev plus per-vendor overrides.

### 3.1 Kernel components

| Component | Responsibility | Never does |
|---|---|---|
| Loop | Runs one turn as a state machine; batches tool calls; applies transitions | Inspect provider names; rewrite history |
| Event log | Append-only record of every op, event, tool call, result, decision and model resolution | Delete; it marks |
| Gateway | Builds the neutral request, calls the adapter, reduces the stream, retries, falls back | Vendor-specific formatting |
| Role resolver | Binds slots (main, explore, classifier, evaluator, compactor, memory, fallback) to `provider:model` once per turn | Switch mid-turn |

### 3.2 Substrate components

| Component | Responsibility |
|---|---|
| Backend | Filesystem and execution behind three primitives: execute, upload, download |
| Sandbox | OS-level isolation around execute; on by default |
| Verifiers | Diagnostics and lint after edits; tests and evaluator before done |
| Policy engine | Fixed decision order over pooled rules and modes |
| Memory store | Typed files with an index and content-hashed citations |

---

## 4. The loop

### 4.1 State

One struct survives an iteration:

```
State {
  messages: list[Message]
  turn: int
  compaction: CompactionTracking
  recovery: { max_output_retries: int, reactive_compact_attempted: bool }
  budget: { tokens_remaining: int | None, usd_remaining: float | None }
  transition: Transition
}
```

### 4.2 Iteration

```
while true:
  msgs   = prepare(state.messages)      # §9.4 compaction tiers, §9.3 attachments
  guard  = fits_window(msgs)            # else Terminal(prompt_too_long)
  reply  = gateway.stream(role.main, msgs, tools)
  calls  = tool_uses(reply)             # by content, never by stop_reason
  if calls is empty:
      gate = verifiers.gate()           # §7
      if gate.ok: return Terminal(completed)
      msgs += as_tool_result(gate.failures); continue
  results = run_batches(calls)          # §4.4
  state = next(state, reply, results, attachments())
```

### 4.3 Transitions

Terminal reasons: `completed`, `max_turns`, `blocking_limit`, `prompt_too_long`, `aborted_streaming`, `aborted_tools`, `stop_hook_prevented`, `hook_stopped`, `model_error`, `image_error`, `budget_exhausted`.

Continue reasons: `next_turn`, `reactive_compact_retry`, `max_output_tokens_escalate` (once, to the model's upper limit), `max_output_tokens_recovery` (at most three, injects a nudge) `[expires]`, `stop_hook_blocking`, `token_budget_continuation`, `verifier_failed`.

Every transition MUST be written to the event log with its reason.

### 4.4 Tool batching

- Consecutive tool calls that are all `concurrency_safe` form one batch and run in parallel, capped at 10 (configurable).
- A tool that is not concurrency-safe runs alone and serially.
- Context changes such as cwd apply after a batch drains, in original order.
- Tools MAY start while the stream is still open. Results are re-emitted in receipt order.
- A sibling abort controller cancels related subprocesses when one shell call in a batch fails.

### 4.5 Loop detection

Mechanical only. Three identical tool calls in a row raise a permission ask. No model-judged loop check `[left out]`.

### 4.6 Termination requires two signatures

The model must stop calling tools, and the verifier gate must pass. There is no next-speaker classifier and no automatic "please continue" `[left out]`.

---

## 5. Model layer

### 5.1 Neutral request and events

```
ModelRequest {
  system: list[Block]           # static prefix, BOUNDARY, dynamic tail
  messages: list[Message]       # user | assistant | tool_result
  tools: list[ToolSpec]         # name, description, json_schema, defer: bool
  max_output_tokens: int
  thinking: ThinkingConfig | None
  cache_hints: CacheHints       # where breakpoints may go; adapters MAY ignore
  role: RoleName
}

Event =
  | TextDelta(text)
  | ThinkingDelta(text, native: NativeBlock | None)
  | ToolCallFragment(index, id | None, name | None, args_fragment: str)
  | Usage(input, output, cache_read, cache_write)
  | Stop(reason)
  | ProviderError(kind, retryable: bool, retry_after: float | None)
```

Message content is a list of blocks in a standard vocabulary: `text`, `thinking`, `tool_call`, `tool_result`, `image`, `file`, `citation`, `non_standard(value)`. Native provider blocks are kept alongside the standard view and MUST NOT be dropped. Each native block carries `needs_server_state: bool`; the gateway omits blocks that need server state when the session cannot supply it.

### 5.2 Adapter contract

An adapter MUST implement:

```
class Adapter:
    name: str
    flags: CapabilityFlags
    def generate(self, model: str, req: ModelRequest) -> list[Event]
    def stream(self, model: str, req: ModelRequest) -> Iterator[Event]
```

Everything else has a default. Required capability flags:

| Flag | Meaning |
|---|---|
| `requires_role_alternation` | Adapter merges consecutive same-role messages (Anthropic) |
| `streams_complete_tool_calls` | Adapter emits one fragment per call (Ollama) |
| `native_structured_output` | Adapter can enforce a JSON schema without a tool trick |
| `supports_cache_control` | Adapter honours cache breakpoints |
| `supports_thinking` | Adapter can request and round-trip reasoning blocks |
| `server_state_blocks` | Some native blocks need server state to be resent (OpenAI Responses) |
| `tool_schema_dialect` | `openai` (default) or a named dialect the adapter converts |

The kernel MUST branch on flags and MUST NOT branch on `adapter.name`.

### 5.3 Kernel-owned reducers

- Tool-call reduction: fragments are concatenated by index, parsed as partial JSON, and any malformed or non-object result becomes `InvalidToolCall{raw, error}` returned to the model as an error result. Adapters never parse tool arguments.
- Tool schema normalisation: any callable, Pydantic model, TypedDict or dict becomes one canonical OpenAI-shaped function schema. Adapters convert from that shape to their dialect.

### 5.4 Shipped adapters

- `anthropic`: cache control at the boundary and on the last message; thinking with signature round-trip; message merging for alternation.
- `openai`: Responses API first, Chat Completions fallback; reasoning items resent only when server state allows.
- `gemini`: GenAI SDK.
- `openai-compatible`: base class for Ollama, Groq, DeepSeek, OpenRouter and any host. A subclass overrides auth, payload quirks, extra usage fields and identity only. Nothing about message conversion or streaming moves.
- `langchain`: bridge that wraps any LangChain chat model so the LangChain ecosystem works on day one. Native adapters replace it where they exist.

### 5.5 Gateway behaviour

- Manual retries: up to 10, 500 ms doubling to 32 s with 25 percent jitter, honouring retry-after. The vendor SDK is configured with zero retries.
- Overload: three attempts, then the `fallback` role if bound.
- Context-limit 400: adjust max output tokens and retry once, then reactive compaction.
- Streaming failure: retry non-streaming once, capped at the model's output limit; partial messages are tombstoned in the log.
- Cost: computed per usage event from the capability dataset's pricing, attributed to the role that produced it.
- Headers and betas latch on for the session once sent `[P1]`.

### 5.6 Profiles

Two registries, same key shape `provider` or `provider:model`, exact key first then provider fallback, additive merge on re-registration, entry-point discovery with per-plugin failure isolation.

Capability profile (construction time), generated, not hand-edited:

```
CapabilityProfile {
  max_input_tokens, max_output_tokens,
  tool_calling, tool_choice, tool_call_streaming,
  structured_output, reasoning_output, reasoning_effort_levels,
  text_inputs, image_inputs, pdf_inputs, audio_inputs, video_inputs,
  image_tool_message, pdf_tool_message,
  pricing: { input, output, cache_read, cache_write }
}
```

Source: models.dev, plus `profile_overrides.toml` per vendor, compiled by a `profiles refresh` command. Unknown keys warn, never fail.

Harness profile (runtime), hand-written, this is the guard registry `[P7]`:

```
HarnessProfile {
  prompt_suffix: str | None
  tool_description_overrides: dict[str, str]
  excluded_tools: set[str]
  extra_middleware: list[Middleware]
  shims: list[Shim]                # e.g. TextToolCallParser
  general_purpose_subagent: { enabled: bool, prompt: str | None }
  added_for: str                   # model and date, mandatory
  reason: str                      # mandatory
}
```

### 5.7 Roles

Slots the kernel calls by name:

| Role | Used when | Default |
|---|---|---|
| `main` | Every turn of the primary agent | required |
| `explore` | Read-only sub-agents | inherits main |
| `classifier` | Policy returned ask | inherits main |
| `evaluator` | Model stopped calling tools | inherits main |
| `compactor` | Summarisation tier | inherits main |
| `memory` | Extraction and consolidation | inherits main |
| `fallback` | Overload or repeated error | unbound |

Rules:
- A role is resolved once per turn and written to the event log.
- Agent definitions and skills MAY declare `model:` directly or name a role, which creates user-defined roles.
- Changing `main` at runtime takes effect at the next compaction boundary.
- Verifiers, sandbox, policy and log have no role; they never involve a model.

### 5.8 Conformance

An adapter is complete when it passes the conformance suite: about forty tests gated by its declared flags, covering invoke, streaming, tool calls with and without arguments, tool choice, error-status tool results, structured output, usage detail including cache metadata, image and PDF inputs, and an end-to-end agent loop. A vendor declares what it supports and must pass what it declares.

### 5.9 User configuration

```toml
[providers.anthropic]  adapter = "anthropic"          api_key_env = "ANTHROPIC_API_KEY"
[providers.openai]     adapter = "openai"             api_key_env = "OPENAI_API_KEY"
[providers.gemini]     adapter = "gemini"             api_key_env = "GEMINI_API_KEY"
[providers.local]      adapter = "openai-compatible"  base_url = "http://localhost:11434/v1"

[roles]
main       = "anthropic:claude-fable-5-1"
explore    = "local:qwen3-coder"
classifier = "openai:gpt-5-mini"
evaluator  = "anthropic:claude-sonnet-5"
```

Adapters and profiles load from entry points and a plugins directory. A managed policy file MAY pin `models.allowed` and is checked before any credential is read.

---

## 6. Tools

### 6.1 Interface

```
Tool {
  name: str
  description(input) -> str
  prompt() -> str                      # long-form doc, injected only when loaded
  input_schema: JsonSchema
  search_hint: str                     # for deferred discovery
  defer: bool
  max_result_chars: int | Infinity
  is_read_only(input) -> bool
  is_concurrency_safe(input) -> bool
  is_destructive(input) -> bool
  validate(input, ctx) -> Result
  permission_matcher(input) -> (pattern) -> bool
  call(input, ctx, on_progress) -> ToolResult
}
```

### 6.2 Core set, always loaded

`read`, `edit`, `write`, `shell`, `grep`, `glob`, `fetch`, `agent`, `ask_user`. Everything else is deferred and discovered through `tool_search`, which accepts `select:Name1,Name2` or free text scored against search hints and prompt text. MCP tools are deferred unless flagged always-load. Tool order is pinned `[P1]`.

### 6.3 Semantics

- `read`: caps by file size before reading and by tokens after. Same file, same range, unchanged mtime returns an unchanged stub. Never persisted to disk.
- `edit`: refuses files not read this session or changed on disk since, unless content still matches byte for byte. Exact match only; multiple matches fail unless replace-all. No fuzzy matchers, no model repair `[left out]`.
- `shell`: parsed with a real parser; per-CLI read-only tables; compound commands split and each segment checked; default timeout 2 min, max 10; output capped at 30,000 chars; only the main thread may change cwd; auto-background when a command blocks too long.
- `grep`, `glob`: ripgrep-backed; capped results; paths relative to cwd.
- Empty results are returned as `(tool completed with no output)`.

### 6.4 Results

- Over `min(max_result_chars, 50,000)`: written to `tool-results/<id>`, replaced by a 2,000-byte preview and the path.
- Per-message aggregate cap 200,000 chars: largest never-seen results persist first; earlier decisions are frozen `[P1]`.
- Results containing images are never persisted.

### 6.5 Backends

```
Backend {
  execute(cmd, timeout, env) -> { stdout, stderr, exit_code }
  upload(files) -> list[Result]
  download(paths) -> list[Result]
}
```

All file tools derive from these three. A composite backend routes by path prefix, mounts durable memory at `/memories`, and pins `execute` to the default backend; execution is never path-routed. A new sandbox provider is one adapter, roughly a hundred lines. Credential provenance is checked: workspace-resolved credentials are compared with process-resolved ones and mixed or half-set pairs are rejected.

Commands run with the project's interpreter first on PATH. The interpreter is taken from the configured verify commands, else a venv found in cwd. Commands never inherit the harness's stdin. (Both learned live: the model reached for a system `python` without pytest, and child processes hung on the client pipe.)

---

## 7. Verification

Two stages. Nothing between them asks the model to grade itself `[P9]`.

After every edit, fast and deterministic:
1. Touch the file in the language server; collect diagnostics.
2. Run the configured linter on the file.
3. Append only errors to the tool result. Passing output is swallowed.

Before `done` is accepted:
1. Run the declared test command; must be green.
2. Run the `evaluator` role on the sprint contract in a fresh context that never shares the generator's. Rubric rules: the criteria list is frozen after the first grading pass; a `satisfied` verdict that did not check every criterion is downgraded to `needs_revision`; internally inconsistent grader output is rejected.
3. Pass returns `completed`. Failure re-enters the loop as a tool result with the failing criteria.

Sprint contract: a list of testable criteria agreed before work starts, stored in the event log, editable only by the user or by an `update_contract` tool that requires approval.

---

## 8. Safety

### 8.1 Sandbox

On by default. Modes `read-only`, `workspace-write`, `full-access`. Network off in `workspace-write` unless an allowlist is configured. Git internals and the harness's own config stay read-only inside writable roots. Implementations: Seatbelt on macOS, bubblewrap plus seccomp on Linux, restricted tokens on Windows. `excluded_commands` is a convenience, not a security boundary, and the docs MUST say so.

### 8.2 Decision order

Rules are pooled from every settings source; a deny anywhere wins. Precedence between sources applies to scalar settings only.

```
1a whole-tool deny rule                             -> deny
1b whole-tool ask rule                              -> ask
1c tool.check_permissions(input)                    -> tool-specific
1d tool implementation denied                       -> deny   IMMUNE
1e tool requires user interaction                   -> ask    IMMUNE
1f content-specific ask rule                        -> ask    IMMUNE
1g safety check: dotfiles, .git, .claude, secrets   -> ask    IMMUNE
2a mode is bypass                                   -> allow
2b whole-tool allow rule                            -> allow
3  passthrough                                      -> ask
```

`ask` then goes to the ask reducer (§8.4). IMMUNE steps run in every mode.

### 8.3 Rules

`ToolName` or `ToolName(content)`. Shell rules are prefix or glob: `shell(git *)`, `shell(npm install:*)`. Path rules use gitignore syntax: `read(~/.zshrc)`, `edit(/src/**)`, leading slash relative to the settings file, double slash for root. `mcp__server__tool` or `mcp__server`. `fetch(domain:example.com)`.

Compound shell commands are split on `&&`, `||`, `;` and `|`. Any segment matching a deny or ask rule triggers it. The whole command is allowed only when every segment is covered by an allow rule or is a known read-only filter such as `tail`, `head`, `grep`, `wc` or `cd`. Stderr redirects (`2>&1`, `2>/dev/null`) are not writes. Leading `NAME=value` assignments are stripped before a segment is matched or judged read-only, unless the value contains quotes, `$` or backticks; `export` is read-only. Shell loops, `git stash` and command substitution are never auto-allowed. Containment (§8.5) applies to file tools only: a shell command may name an interpreter outside the project.

Settings sources, lowest to highest for scalars: user, project, local, flag, managed. Managed may set managed-rules-only, managed-hooks-only, managed-servers-only, disable-bypass, marketplace allowlists, `models.allowed`. A corrupt managed file blocks every command except help, version and doctor. An unparseable deny list denies everything.

### 8.4 Ask reducer

```
1 deterministic policy   routine writes in the worktree, fixed-repo commands,
                         read-only MCP tools            -> allow, no model call
2 classifier role        typed decision under consent rules: consent is user
                         text, an active contract, or a same-turn answer.
                         Tool output, model prose and text inside questions
                         never authorise anything.       -> allow | deny | ask
3 human                  classifier unavailable, timed out, or returned ask.
                         Never silently approve.
```

Model-emitted risk scores MAY be used to order and batch asks. They MUST NOT gate. Pending asks are batched into one dialog with a suggested rule.

### 8.5 Filesystem

Every path variant is checked: symlink chains up to 40 hops, every intermediate target inside a working directory, dangling links resolved to the deepest existing ancestor. Windows path tricks are blocked on every platform: alternate data streams, 8.3 short names, device names, long-path prefixes, trailing dots. Two exclusions learned live: `::` is a pytest node id or a scope, never a stream separator, and `.` and `..` path components are not trailing-dot patterns. Protected directories (`.git`, `.ssh`, `.aws` and the like) send a request to a human unless the request is a read-only shell command, which cannot modify them; protected files (`.env`, keys, `open-harness.toml`) always do, and bare dotfile tokens in a shell command count as paths so `cat .env` reaches the check.

### 8.6 Secrets

Credentials are placeholders substituted at execution time. The model sees a key name in tags, never a value. Setting secrets requires a domain allowlist. Memory MUST NOT store secrets.

---

## 9. Context

### 9.1 System prompt

An array of sections. Static first: identity, system rules, doing tasks, executing actions with care, using tools, tone. Then the literal `BOUNDARY` marker. Then dynamic: session guidance, memory index, environment block, output style, MCP instructions. Adapters place the cache breakpoint at the boundary. The date in the environment block is left stale at midnight; a `date_change` attachment patches it `[P1]`.

The environment block names the project's interpreter and the configured lint and test commands, and the tool guidance says that commands already run in the project directory, that commands should stay simple, and that the harness runs lint and tests itself. A further line says to use `python -c` for quick checks rather than writing throwaway scripts into the project. These lines exist because Sonnet models guessed the interpreter, prefixed every command with `cd`, re-ran tests the gate was about to run, and wrote scratch files that then failed lint and cost turns to delete `[expires: added for claude-sonnet-4-5 and claude-sonnet-5, 2026-09; move into harness profiles once they exist]`.

### 9.2 Project instructions

`AGENTS.md` walked from project root to cwd, plus a global file and any configured URLs, with `CLAUDE.md` accepted as a fallback name. Injected as a harness-tagged synthetic first user message, not in the system prompt. Nested rule files load when a tool enters their directory. Imports use `@path`; refs inside code fences are ignored; circular imports are tracked.

### 9.3 Attachments

Typed, harness-authored, appended at the tail `[P4]`. Families: file references, IDE state, tasks and skills, mode changes, hook outputs, budget, post-compact deltas, team messages, time and diagnostics, contract state. Each type has exactly one trigger. Contract and goal state travel here, never as a fake user message.

### 9.4 Compaction, three tiers under one trigger policy

| Tier | Trigger | Action |
|---|---|---|
| Clear | Wall-clock gap since last assistant message exceeds the provider cache TTL | Replace old tool result content for read, edit, write, grep, glob, fetch, shell with a cleared marker. Messages stay |
| Notes | Every N tool calls or M tokens of growth, at a natural break | A forked `memory`-role agent updates a running session-notes file it alone may edit |
| Summarise | Tokens reach `window - min(max_output, 20k) - 13k`, or a reactive context error | `compactor` role writes a nine-section structured summary; keep up to five recently read files, the contract, invoked skills; emit delta attachments for deferred tools and MCP instructions; circuit-break after three consecutive failures |

Manual compact tries the notes tier first. Compaction is a boundary marker in the log `[P3]`.

### 9.5 Fresh-context mode

For long autonomous runs, a named mode: each iteration starts a new window with only a handoff document; the filesystem and git are the memory; the loop continues until a stop file or budget. Preferred over repeated summarisation past a configurable number of compactions. Also available as a tool the model may call: `new_context(handoff)`.

### 9.6 Memory

Per-project directory with an index file and typed entries: `user`, `feedback`, `project`, `reference`. Nothing derivable from code is stored. Each entry MAY cite `path#Lstart-Lend` with a content hash of those lines; staleness is a hash mismatch, checked mechanically, with an age note as fallback for uncited entries. Extraction runs at turn end via the `memory` role. Consolidation runs in the background: orient, gather, consolidate, prune and re-tighten the index under 200 lines and 25 KB. Memory is data, not instructions; the prompt says so.

### 9.7 Token accounting

Rough estimate: bytes over four, two for JSON, images fixed at 2,000. Exact counts from the adapter when available. A `context_doctor` command lists every injected component with its token cost. A cold-cache warning is shown when the provider cache has likely expired, with the estimated rewarm cost.

---

## 10. Sub-agents and asynchronous work

### 10.1 Fork

Omitting an agent type forks: all tools, inherited model, bubble permission mode. The child history is the parent's last assistant message plus one user message with identical placeholder tool results for every call, then a per-child directive last, so siblings share a byte-identical prefix `[P1]`. The parent's rendered system prompt is passed through. A boilerplate tag forbids recursion and fixes the report format.

### 10.2 Named agents

Markdown definitions: description, tools, disallowed tools, skills, MCP servers, hooks, model or role, effort, permission mode, max turns, background, memory scope, isolation (`worktree` | `remote`). Precedence: built-in < plugin < user < project < flag < managed. Read-only agents omit project instructions.

### 10.3 Returns

Sub-agents return a distilled result of one to two thousand tokens. Full transcripts go to the log, readable on demand.

### 10.4 One channel

Background shells, agents, remote agents, cron fires and stalled commands re-enter as `task_notification{id, status, summary, output_path}` `[P5]`. Output is appended to disk, never held in memory. A stall watchdog reports a command that is silent for 45 s and ends in a prompt-like pattern.

### 10.5 Teams

Mailbox-based teams and a coordinator mode exist but are opt-in and mutually exclusive with fork mode. Plain text output is invisible to other agents; only `send_message` crosses.

---

## 11. Extensibility

### 11.1 Hooks

Events: `PreToolUse`, `PostToolUse`, `PostToolUseFailure`, `UserPromptSubmit`, `SessionStart`, `SessionEnd`, `Stop`, `SubagentStart`, `SubagentStop`, `PreCompact`, `PostCompact`, `PermissionRequest`, `PermissionDenied`, `ContractUpdated`, `Notification`.

Types: `command` (bash or powershell, sync or async), `prompt` (small model), `agent` (verifier agent), `http` (POST with env allowlist). Optional `if` in rule syntax filters firing.

Contract: JSON on stdin; JSON or text on stdout; exit 0 success, exit 2 blocking with stderr to the model, other codes non-blocking. All hooks for an event run in parallel with individual timeouts, ten minutes default, 1.5 s for `SessionEnd`. Managed settings MAY restrict to managed hooks only.

In-process hook API: where an event coincides with a Google ADK plugin callback (`before_model`, `after_model`, `before_tool`, `after_tool`, `before_agent`, `after_agent`, `on_event`, `on_tool_error`, `on_model_error`) the Python name SHOULD match it, so a hook written against ADK's shape ports with a rename. A `before_tool` hook MAY return a substitute result, which short-circuits the call and is logged as such. This is a naming convention, not a dependency on ADK.

### 11.2 Skills

Directory with `SKILL.md` and frontmatter: name, description, user-invocable, model or role, allowed-tools, arguments, when-to-use, disable-model-invocation, hooks, fork. Only name, description and when-to-use enter the prompt until invoked. Bundled skills MAY extract reference files to disk on first use.

### 11.3 Plugins

Manifest declaring commands, agents, skills, hooks, MCP servers, LSP servers, output styles, adapters, profiles, and typed user config whose sensitive values go to the keychain. Marketplaces from URL, git, npm or path. Reserved-name guard for official-looking names. Path traversal checks. No transitive trust across marketplaces. Startup loads from cache so clones never block the first prompt.

### 11.4 MCP

Three scopes: user, project, managed. Transports: stdio, streamable HTTP, SSE, in-process. OAuth with dynamic registration, PKCE, cross-process refresh lock. Tool names `mcp__server__tool`. Descriptions capped at 2,048 chars. Outputs over 25,000 tokens go to a file. Project servers start pending and require approval. Elicitation supported; sampling not.

---

## 12. Interfaces

### 12.1 Protocol

```
Op    = turn_input | interrupt | approve | set_mode | set_role | compact | new_context | shutdown
Event = turn_start | text_delta | thinking_delta | tool_call_start | tool_call_end
      | approval_request | task_notification | usage | compact_boundary | turn_end | error
```

Clients push ops and read events. Any socket transport MUST be authenticated; there is no unauthenticated local server `[left out]`.

### 12.2 Surfaces

Terminal UI, headless with `text | json | stream-json` output and a stream-json control channel over stdio, IDE via lock-file discovery and an MCP connection, web and mobile via an authenticated bridge that spawns child sessions, an SDK, and an A2A server that exposes a session as an Agent2Agent task endpoint so orchestrators built on other frameworks can call open-harness as one agent. All are thin. The A2A surface is planned after the terminal and headless surfaces are stable (§18.2).

The terminal renders every log record through a listener, so the screen and the transcript are the same data. Its approval prompt offers allow once, allow always, deny, and deny with a message that reaches the model. A line typed at the prompt that is clearly a next instruction is queued for the next turn rather than dropped. In one-shot mode approvals fail closed and are logged.

### 12.3 Exit codes

Zero on success, one on any error, with the error subtype in the final event.

---

## 13. Observability and honesty

- The event log is the audit trail; every model resolution, permission decision and transition is recorded.
- Cost per role per turn.
- `context_doctor`, cold-cache warnings, and a compaction boundary that names what was removed.
- A generated `THREAT_MODEL.md` listing trust boundaries and known weaknesses, regenerated on release.

---

## 14. Self-maintenance

### 14.1 Guard registry

Every model-compensating component lives in a harness profile with `reason` and `added_for`. The registry itself, the event log, the sandbox, the verifiers, the immune checks and secrets substitution never expire.

### 14.2 Pruning loop

At each model release, an outer agent edits the surfaces for that model's profile (prompt text, tool descriptions, shims, middleware) against a train and holdout split with matching strata, and a change is accepted only if `train.passed + holdout.passed` strictly increases. Every decision is persisted. No human in the loop during the run; humans write the config and review the report.

### 14.3 Evals

Two suites: diagnostic unit evals that each assert one behaviour, and a holistic battery across autonomous terminal tasks, conversation with a simulated user, context retrieval and research. The harness variant (`bare` versus `product`) is an explicit input alongside the model. Scores are `pass@k` and `avg@k` over at least three rollouts. Intermittent tasks are kept because they discriminate.

Case format: eval cases are stored as Google ADK `EvalSet` JSON (eval set, eval cases, conversation turns with expected tool trajectory and final response) so the same cases run under `adk eval` unchanged. Tool-trajectory scoring follows ADK's metric definitions where they exist (`TOOL_TRAJECTORY_AVG_SCORE`, `FINAL_RESPONSE_MATCH_V2`); harness-specific measures (API calls, cost per role, denials, files touched outside scope, verifier retries) are extra fields the driver adds. The format is borrowed; the driver and runners are ours.

---

## 15. Left out, deliberately

| Feature | Seen in | Why not |
|---|---|---|
| Next-speaker classifier and auto-continue | Gemini CLI | Compensates for early stopping; the verifier gate replaces it |
| LLM repair inside the edit tool | Gemini CLI, opencode | Hides mistakes; exact failure teaches better |
| LLM loop judge | Gemini CLI | Mechanical detection suffices |
| Pixel-based browser perception | Operator | Accessibility tree with element refs is deterministic and cheaper |
| Seven MCP config scopes | Claude Code | Three suffice |
| Unauthenticated local server | dcode | Listed threat |
| Three independent eviction policies | Deep Agents | One trigger policy over three tiers |
| Goal state as a user message | dcode | Injection surface; use attachments |
| Hard-coded provider table | LangChain | Entry-point discovery |
| Model-emitted risk as the gate | OpenHands | The actor must not grade its own risk |

---

## 16. Open questions

- `[open]` Exact contract for `needs_server_state` when a session moves between adapters.
- `[open]` Whether the `explore` role should default to a cheaper model automatically when one is configured.
- `[open]` Contract format for the sprint contract: free text with criteria, or a schema.
- `[open]` Where the pruning loop runs: local, CI, or a hosted job.
- `[open]` Whether teams should share a single event log or one per teammate.

---

## 17. Sources

Source readings, September 2026: Claude Code (local snapshot), opencode, OpenAI Codex CLI, browser-use, Google Gemini CLI, OpenHands and its SDK, LangChain Deep Agents and dcode, LangChain core and partner packages. Published guidance: Anthropic on building effective agents, context engineering, long-running harnesses and managed agents; SWE-agent on agent-computer interfaces; Cursor and LangChain on harness engineering.

Google ADK (Python v2.9.1, September 2026): studied as a candidate foundation rather than as a source of design, see §18.4. Companion page: "Should open-harness Build on ADK?" Evidence read: adk-python source (flows, plugins, models, compaction, resumability, code executors), adk.dev 2.0 notes, adk-python issues #265, #994, #3289, #3828, #4482, #4801, #7004, litellm issues #18950, #25561, #29491, allenporter/adk-coder, adk-samples software-bug-assistant, gemini-cli issue #8256, Google's May 2026 post on pause-and-resume agents, Simon Willison on how coding agents work, Addy Osmani on agent harness engineering.

## 18. Implementation status and roadmap

Status as of 16 September 2026, against the prototype at https://github.com/smhnkmr/open-harness.

### 18.1 Built and live-verified

| Area | State |
|---|---|
| Loop (§4) | State machine with named transitions, parallel read-only batches, invalid-call retry, max-output recovery, two-signature turn end. No streaming tool start, no token budget, no mechanical loop detection yet. |
| Event log (§3) | JSONL, resume, listeners, `--show-session` and `--list-sessions` viewers. No fork, no file-history snapshots. |
| Model layer (§5) | Neutral types, two-method adapter, kernel reducer, `anthropic` and `openai-compatible` adapters, roles with inheritance, retry and fallback, adaptive-thinking switch. No capability dataset, no harness profile registry, no conformance suite, no Gemini or LangChain bridge adapters. The openai-compatible adapter has run only against fakes. |
| Tools (§6) | read, edit, write, shell, grep, glob, ask_user; truncate-to-disk; local backend with PATH prepend. No fetch, agent, tool search, MCP. |
| Verification (§7) | Lint after edit, test gate before done. No evaluator role, no sprint contract. |
| Safety (§8) | Rules, fixed decision order, immune checks, compound splitting with read-only leniency, ask reducer stages 1 and 3. No OS sandbox, no classifier stage, no secrets substitution, no persisted rules. |
| Context (§9) | Static prompt with boundary, environment block, project instructions, summarise tier. No clear or notes tiers, no attachments beyond instructions and verifier, no memory, no fresh-context mode. |
| Interfaces (§12) | stdio stream-json client, one-shot mode, terminal client with approval prompts and six slash commands. No IDE, bridge or SDK. |
| Evals (§14.3) | `evals/` driver: ten tasks on a synthetic `ledger` fixture, per-run committed workspaces, both harnesses from one driver with stdin detached, interleaved order, identical allow lists and project instructions, per-model list-price costing, resumable `runs.jsonl` and a markdown report. Cases exported as ADK `EvalSet` JSON. No simulated-user conversations, no `pass@k` over three rollouts yet (one rollout per run, `--runs` sets k). |

Three live comparisons against Claude Code (a bug fix, a five-turn conversation, a nine-minute feature task) found nine harness defects, all fixed with regression tests, and left outcome, durability and cost within noise of each other. See §18.3.

### 18.2 Roadmap, in order

1. **Eval driver.** Built (0.5). Ten task types, five runs each, same model, both harnesses from one driver with stdin detached. Scores pass rate, API calls, cost, wall time, denials, and files touched outside scope. Everything below is judged by it. First full matrix recorded in §18.3. Still to add: files-touched-outside-scope as a first-class metric on every edit task (today only t10 records it); lint once per tool batch on the set of edited files instead of after each edit, since 40 of the 47 lint failures in the matrix were transient states inside a multi-edit batch that the model's next queued edit fixed; a simulated-user conversation task type.
2. **Friction-free permissions.** Shell `permission_content` defaults to the first two tokens so "allow always" yields a reusable rule; rules persist to `.open-harness/rules.toml`.
3. **Input during a turn.** Reader thread on stdin, Escape aborts at the next check point, cooperative cancel in the reducer for mid-stream abort.
4. **Diff preview at approval** for edits outside the worktree and in accept-edits mode.
5. **Harness profiles.** The registry from §5.6. First entry: the Sonnet guidance lines from §9.1, with reason and date.
6. **Multi-vendor live.** `explore` on a local Ollama model, `classifier` on a small OpenAI model, eval suite across three vendors.
7. **Classifier stage** of the ask reducer (§8.4 stage 2).
8. **A2A server surface** (§12.2): one session per A2A task, events mapped to task status updates, approvals surfaced as input-required states.

Step 1 stores its cases in the ADK `EvalSet` format (§14.3) from the first commit, so no migration is needed later. Step 5's hook names follow §11.1.

Deferred until those are done: sub-agents and forks, memory with hashed citations, compaction tiers one and two, OS sandbox, MCP, hooks, remaining stream-json ops.

Re-check triggers. The ADK decision in §18.4 is revisited if either becomes true: ADK exposes a public injection point for the LLM flow (today `LlmAgent._llm_flow` is private), or the Anthropic-path issues for thinking with tool use and streamed tool arguments close on both adk-python and litellm. Each is checkable in an afternoon.

### 18.3 Comparison record

| Test | Claude Code | open-harness | Outcome |
|---|---|---|---|
| Bug fix, same model | 4 calls, $0.39, 107k cache read per call | 6 calls, about $0.08, 2.5k cache read per call | both fixed it; ours also ran the gate |
| Five-turn conversation, Sonnet 4.5 | 17 calls, 222 s, $0.66 | 24 calls, 196 s, about $0.33 | equal answers; ours 7 extra calls in one turn from self-verification |
| Nine-minute feature, Sonnet 5 | 66 calls, 540 s, $2.23, 3 unrelated files edited | 81 to 90 calls, 458 to 561 s, $2.27 to $3.03, in scope | both complete and green; cost within noise |

Defects found by these tests, all fixed: child stdin inheritance hang; TOML key ordering; one-shot approvals blocking; console encoding; shell containment on interpreter paths; `::` and `./` false positives; `2>&1` treated as a write; lint path unquoted in bash; provider-error log key collision; adaptive thinking; interpreter not on PATH; prompts swallowed at approval; bare `[stderr]` rendering.

Defects found by the eval driver's first smoke runs, both fixed: a read-only `find` naming `.git` was sent to a human (§8.5 now exempts read-only shell commands from the protected-directory check, secrets excepted); bare dotfile tokens such as `cat .env` never reached the protected-file check (§8.5 now treats them as paths).

**First full matrix, 16 September 2026.** Ten tasks, five runs each, Sonnet 5 on both, from `evals/driver.py`; report and run records in `evals/reports/20260916-sonnet5-full/`.

| | open-harness | Claude Code |
|---|---|---|
| Pass rate | 50/50 | 50/50 |
| API calls, total | 349 | 425 |
| Tool calls, total | 413 | 460 |
| Cache read tokens per call, mean | 2.4k to 10.2k by task | 28k to 39k by task |
| Cost, list price for both | $2.99 | $10.08 (Claude Code reported $8.21) |
| Wall time, total | 1589 s | 1722 s |
| Denials | 25 | 35 |
| Verifier failures | 47, all lint after an edit, none at the test gate | not applicable |
| Files changed on the vague task (t10) | 1 in every run | 1 in every run |

Reading. Outcome is tied at 100 percent, so this matrix cannot rank the harnesses on correctness; it ranks them on what correctness costs. open-harness spends 18 percent fewer API calls, 8 percent less wall time, and 30 to 37 percent of the money, almost entirely because its cached prefix is a tenth the size. Per-task spread is wide (t05 on Claude Code: 6 to 17 calls; t10 on open-harness: 15 to 31 calls), so single-task deltas under about 30 percent are noise at n=5. On t10 open-harness averaged 21 calls to Claude Code's 16: both harnesses' models wrote scratch preview scripts, but open-harness denied `rm` on them, costing about two calls per affected run; the test gate never bounced a t10 run. Denials on both sides were dominated by compound `cd ... && ...` chains, `xargs` pipes, env-assignment prefixes and quoted interpreter paths; `rm`, `del` and `sed -i` denials are the policy working as designed. None changed an outcome.

Changes made from the matrix: leading `NAME=value` assignments are stripped before rule matching (§8.3), `export` is read-only, the eval config gives open-harness the quoted interpreter rule it already gave Claude Code, and the prompt tells the model not to write throwaway scripts into the project (§9.1). The eval driver itself had one defect: a relative `--out` put a relative `src` on the checkers' `PYTHONPATH`; paths are now resolved.

### 18.4 Decision: not built on Google ADK

Assessed 16 September 2026 against ADK Python v2.9.1. Twenty spec requirements were mapped to ADK primitives: three native (event log, evals, MCP and A2A), nine partial, three in conflict, five absent. The conflicts are the kernel: ADK's tool loop is selected by a private property and turn end is decided by `is_final_response()`, its canonical message type is Gemini's `Content`/`Part` with every other vendor converted, and its model classes branch on provider name, which P8 forbids. The absences are the verifier gate, the independent evaluator, coding tools, the guard registry and secrets substitution. Field evidence: no mature coding harness has shipped on ADK, Gemini CLI does not use it, the one attempt (adk-coder) wrote its own permission engine, and Claude through ADK has open issues for thinking with tool use, streamed tool arguments and cache control.

Decision: keep the kernel; adopt the `EvalSet` format (§14.3), the plugin callback names (§11.1) and an A2A surface (§12.2). No runtime dependency on ADK. An ADK-backed model adapter (wrapping `BaseLlm` classes) is possible in a few hundred lines but is not planned, because it would re-import the LiteLLM streaming bugs the native adapters avoid.

## 19. Changelog

- 0.5, 2026-09-16: eval driver built (`evals/`, roadmap step 1); cases exported in ADK `EvalSet` format; two safety-check defects found by it fixed; stream-json result record carries the session id. First full matrix recorded in §18.3: 100/100 on both harnesses, open-harness at a third of the cost; env-assignment prefixes stripped in rule matching; scratch-script guard line.
- 0.4, 2026-09-16: Google ADK assessed and declined as a foundation (§18.4); eval cases adopt the ADK `EvalSet` format (§14.3); in-process hook names follow ADK's plugin callbacks (§11.1); A2A server added as a planned surface (§12.2) and roadmap step 8; re-check triggers recorded.
- 0.3, 2026-09-16: implementation status and roadmap added; rules for compound commands, stderr redirects and shell containment; environment block names interpreter and verify commands; terminal pending-input queue. Nine live-found defects fixed.
- 0.2, 2026-09-16: terminal client with approval prompts; session-scoped rules and mode changes recorded in the log and replayed on resume; deny-with-message reaches the model; gateway streams text deltas to clients.
- 0.1, 2026-09-16: first consolidated spec from the five explainer pages and the model-layer study.
