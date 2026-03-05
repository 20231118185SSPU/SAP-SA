# StudyAdministrator (SA)

`SA` is a **minimal, runnable, autonomous learning committee agent** extracted
and simplified from the ideas in `../zeroclaw`:

- Calls an **OpenAI-compatible** `POST /v1/chat/completions` API.
- Uses a **Chinese built-in framework prompt** adapted from `zeroclaw`, then injects local `Agents.md`.
- Discovers and loads `SKILL.md`-based skills (Codex/Agents skills format).
- Runs a **tool-calling loop** with built-in `Read` / `Write` / `Edit` / `Bash` /
  `Fetch` / `Search` / `MemorySearch` / `MemoryGet` / `Send` / `Show` / `Ask` / `Skill` / `SubAgent`.
- Exposes a **WebSocket** server so a CLI can connect/disconnect without stopping the agent.
- Loads long-term memory from workspace Markdown files (`MEMORY.md`, `memory.md`, `memory/*.md`) in an OpenClaw-style workflow.

## Built-in tools

- `Read`: read a UTF-8 text file inside the workspace
- `Write`: create a new file only if it does not already exist
- `Edit`: edit an existing file, but only after that file has been `Read`
- `Bash`: run commands through Git Bash (`bash -lc`)
- `Fetch`: send an HTTP request to a known URL
- `Search`: search the web and return candidate titles/snippets/URLs
- `MemorySearch`: search Markdown memory files on demand
- `MemoryGet`: read a bounded slice from one memory Markdown file
- `Send`: send a user-facing message to the connected CLI
- `Show`: display a workspace file in the CLI's dedicated show pane
- `Ask`: ask a structured question and block until the user answers
- `Skill`: read `SKILL.md` or another skill-relative file without exposing the real skill install path
- `SubAgent`: launch a nested child agent with explicit parent context

Best-practice behavior baked into the prompt:

- Use `Search` before `Fetch` when the exact URL is unknown.
- Use `MemorySearch` before answering history/preferences/todos, then `MemoryGet` only for the needed lines.
- Keep `Send` short.
- Use `Show` for dense output (reports, generated files, long explanations, code) instead of flooding the user with plain text.

## Quick start

1. Create a local config:

   - Copy `sa.example.toml` to `sa.toml`
   - Fill in `llm.api_key`

2. (Optional but recommended) Create a local `Agents.md` in this folder.

   - It is **gitignored** by default.
   - You can reference additional persona/context files in backticks (e.g. `SOUL.md`, `USER.md`);
     the daemon will preload them and inject into the prompt.
   - Memory files follow a separate OpenClaw-style flow: keep curated memory in `MEMORY.md` / `memory.md`,
     daily notes in `memory/*.md`, and let the agent use `MemorySearch` / `MemoryGet` on demand.

3. Start the agent daemon (WebSocket server):

```bash
cargo run -p sa --release
```

4. In another terminal, start the interactive CLI (bottom input box).

   The CLI is a **separate project** located at `../sa-cli`:

```bash
cd ../sa-cli
cargo run --release
```

## Build (size vs speed)

This repo tunes the standard `--release` profile for both size and speed:

- Build: `cargo build --release`

## Docs

Development docs live in `./docs/development/`.
