# Development documentation (SA)

## Project overview

This folder documents the StudyAdministrator (SA) backend in `../`:

- `crates/sa-core`: core logic (config, skill loading, OpenAI-compatible client, agent loop)
- `crates/sa`: backend daemon that runs the agent loop and exposes WebSocket

The terminal frontend is a separate project located at:

- `../sa-cli`

## Running

1. Create `sa.toml` from `sa.example.toml`.
2. (Optional) Create a local `Agents.md` (gitignored). If it references files in backticks
   (e.g. `SOUL.md`, `USER.md`), the daemon will preload them into the prompt.
3. Start the daemon:

```bash
cargo run -p sa --release
```

4. Start the interactive CLI (bottom input box) in the separate CLI project:

```bash
cd ../sa-cli
cargo run --release
```

## Build (size + speed)

This repo tunes the standard `--release` profile in `Cargo.toml` for **both**
size and runtime performance.

Example:

```bash
cargo build --release
```

Interactive commands (CLI):

- Type text and press Enter → interrupts current task (if any) and submits a new task
- If a structured question is pending, Enter answers that question instead of creating a new task
- `/stop` → interrupt current task
- `/exit` → quit CLI

One-shot mode (for scripting):

```bash
cd ../sa-cli
cargo run --release -- run "用一句话解释这个项目的结构"
```

## WebSocket API (v0)

All messages are JSON text frames.

Client → Server:

- `{"type":"submit","task_id":"<optional uuid>","task":"..."}`
- `{"type":"get_history","from_event_id":123}`
- `{"type":"interrupt","task_id":"<uuid>"}`
- `{"type":"answer_question","answer":{...}}`

Server → Client:

- `{"type":"accepted","task_id":"..."}`
- `{"type":"history","events":[...]}`
- `{"type":"event","event":{...}}`
- `{"type":"question","question":{...}}`
- `{"type":"pending_questions","questions":[...]}` 
- `{"type":"question_resolved","question_id":"..."}`

Event fields:

- `event_id` (u64): monotonically increasing identifier
- `ts` (RFC3339 string): server timestamp (UTC)
- `task_id` (string): which task produced the event
- `kind` (string): `log` | `tool` | `message` | `final` | `error`
- `message` (string): human-readable text

Structured question fields:

- `question_id` (UUID): used to correlate the answer
- `task_id` (UUID): top-level task waiting on the answer
- `prompt` (string): user-facing prompt
- `mode` (string): `single_choice` | `multi_choice` | `text`
- `options` (array): selectable options for choice-based prompts
- `allow_free_text` (bool): whether extra text is allowed

## Built-in tools

- `Read`: read a UTF-8 text file under the workspace root
- `Write`: create a file only if it does not already exist
- `Edit`: modify an existing file, but only after `Read` has been used on it in the same agent session
- `Bash`: run commands through Git Bash (`bash -lc`)
- `Send`: push a user-facing message into the CLI event stream
- `Ask`: emit a structured question and block until an answer arrives
- `Skill`: load a named skill's `SKILL.md`
- `SubAgent`: run a nested child agent with parent-supplied context; child output is traced back into the parent task stream

## Known issues

- This is intentionally minimal: no authentication on the WebSocket server.
- `SubAgent` recursion is intentionally bounded by a hard depth limit.
- Streaming token output is not implemented; events are per-step.
- Long-term memory is persisted to `.sa/memory.jsonl` (gitignored).

## Changelog

- 0.1.0: initial minimal autonomous agent + WS + CLI.
- 0.2.0: persistent long-term memory, Agents.md reload + persona preloading, interrupt, TUI CLI, infinite retry/backoff.
- 0.2.1: CLI always requests WS history on connect (prevents missing `final` after disconnect).
- 0.2.2: TUI ignores key release events (fixes double-typed input on some terminals).
- 0.2.3: TUI cursor uses Unicode display width (fixes cursor drift for CJK/emoji input).
- 0.2.4: add optimized build profiles (`--release` / `--profile release-small`).
- 0.2.5: keep one tuned `--release` profile (size + speed).
- 0.3.0: split the backend daemon and CLI into separate projects.
- 0.4.0: rename the backend to StudyAdministrator (SA), rename binaries/config to `sa`.
- 0.5.0: replace the built-in toolset with `Read` / `Write` / `Edit` / `Bash` / `Send` / `Ask` / `Skill` / `SubAgent`, add structured question WS messages, and support nested sub-agents.

## Traceability (extracted from `../zeroclaw`)

This minimal implementation is intentionally small, but it is conceptually extracted
from these ZeroClaw modules:

- OpenAI-compatible HTTP calling: `../zeroclaw/src/providers/compatible.rs`
- Agent tool-calling loop patterns: `../zeroclaw/src/agent/loop_.rs`
- Skills config / skill concepts: `../zeroclaw/src/config/schema.rs` (skills section)
