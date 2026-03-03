# Claw (minimal agent)

`claw` is a **minimal, runnable, autonomous Rust agent** extracted/simplified from the ideas in
`../zeroclaw`:

- Calls an **OpenAI-compatible** `POST /v1/chat/completions` API.
- Reads `Agents.md` from the workspace as the base instruction prompt.
- Discovers and loads `SKILL.md`-based skills (Codex/Agents skills format).
- Runs a **tool-calling loop** (shell/file ops) to act autonomously.
- Exposes a **WebSocket** server so a CLI can connect/disconnect without stopping the agent.
- Persists **long-term memory** to `.claw/memory.jsonl` (gitignored).

## Quick start

1. Create a local config:

   - Copy `claw.example.toml` to `claw.toml`
   - Fill in `llm.api_key`

2. (Optional but recommended) Create a local `Agents.md` in this folder.

   - It is **gitignored** by default.
   - You can reference additional persona/memory files in backticks (e.g. `SOUL.md`, `USER.md`);
     the daemon will preload them and inject into the prompt.

3. Start the agent daemon (WebSocket server):

```bash
cargo run -p claw-agentd
```

4. In another terminal, start the interactive CLI (bottom input box):

```bash
cargo run -p claw-cli
```

## Build (size vs speed)

This repo defines two optimized build profiles:

- Fast runtime (default release): `cargo build --release`
- Smallest binaries: `cargo build --profile release-small`

## Docs

Development docs live in `./docs/development/`.
