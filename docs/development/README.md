# Development documentation (Claw)

## Project overview

This folder documents the minimal Rust agent in `../`:

- `crates/claw-core`: core logic (config, skill loading, OpenAI-compatible client, agent loop)
- `crates/claw`: backend daemon that runs the agent loop and exposes WebSocket

The terminal frontend is a separate project located at:

- `../claw-cli`

## Running

1. Create `claw.toml` from `claw.example.toml`.
2. (Optional) Create a local `Agents.md` (gitignored). If it references files in backticks
   (e.g. `SOUL.md`, `USER.md`), the daemon will preload them into the prompt.
3. Start the daemon:

```bash
cargo run -p claw --release
```

4. Start the interactive CLI (bottom input box) in the separate CLI project:

```bash
cd ../claw-cli
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
- `/stop` → interrupt current task
- `/exit` → quit CLI

One-shot mode (for scripting):

```bash
cd ../claw-cli
cargo run --release -- run "用一句话解释这个项目的结构"
```

## WebSocket API (v0)

All messages are JSON text frames.

Client → Server:

- `{"type":"submit","task_id":"<optional uuid>","task":"..."}`
- `{"type":"get_history","from_event_id":123}`
- `{"type":"interrupt","task_id":"<uuid>"}`

Server → Client:

- `{"type":"accepted","task_id":"..."}`
- `{"type":"history","events":[...]}`
- `{"type":"event","event":{...}}`

Event fields:

- `event_id` (u64): monotonically increasing identifier
- `ts` (RFC3339 string): server timestamp (UTC)
- `task_id` (string): which task produced the event
- `kind` (string): `log` | `tool` | `final` | `error`
- `message` (string): human-readable text

## Known issues

- This is intentionally minimal: no authentication on the WebSocket server.
- The agent toolset is minimal (shell + basic file ops); expand as needed.
- Streaming token output is not implemented; events are per-step.
- Long-term memory is persisted to `.claw/memory.jsonl` (gitignored).

## Changelog

- 0.1.0: initial minimal autonomous agent + WS + CLI.
- 0.2.0: persistent long-term memory, Agents.md reload + persona preloading, interrupt, TUI CLI, infinite retry/backoff.
- 0.2.1: CLI always requests WS history on connect (prevents missing `final` after disconnect).
- 0.2.2: TUI ignores key release events (fixes double-typed input on some terminals).
- 0.2.3: TUI cursor uses Unicode display width (fixes cursor drift for CJK/emoji input).
- 0.2.4: add optimized build profiles (`--release` / `--profile release-small`).
- 0.2.5: keep one tuned `--release` profile (size + speed).
- 0.3.0: rename backend daemon to `claw`, split `claw-cli` into `../claw-cli`.

## Traceability (extracted from `../zeroclaw`)

This minimal implementation is intentionally small, but it is conceptually extracted
from these ZeroClaw modules:

- OpenAI-compatible HTTP calling: `../zeroclaw/src/providers/compatible.rs`
- Agent tool-calling loop patterns: `../zeroclaw/src/agent/loop_.rs`
- Skills config / skill concepts: `../zeroclaw/src/config/schema.rs` (skills section)
