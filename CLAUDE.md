# SAP-6.2

AI agent system: Rust backend (sa-core) + React 19 frontend (WebUI).

## Build Commands

```bash
# Rust backend
export PATH="$PATH:/c/Users/杨先生/.cargo/bin"
cargo build -p sa --release

# WebUI frontend
cd WebUI && npm install && npm run build

# Tests
cd WebUI && npx vitest run
cargo test -p sa-core
```

## Architecture

- `sa/` — Rust Cargo workspace (sa-core library + sa binary)
- `WebUI/` — React 19 + Vite + TypeScript + Tailwind CSS 4
- `workspace/` — Runtime workspace (skills, memory, config)
- `docs/` — Investigation reports
- `bin/` — Compiled binaries

See `AGENTS.md` files in each directory for detailed context.

## Key Patterns

- Memory system: unified SQLite store (`memory_store.rs`) with FTS5
- Frontend state: Zustand stores
- Protocol: WebSocket JSON between frontend and backend
- LLM: OpenAI-compatible API (DeepSeek, etc.)
- Skills: SKILL.md format, discovered at runtime

## Gotchas

- Cargo not in default PATH — use `export PATH="$PATH:/c/Users/杨先生/.cargo/bin"`
- `sa.toml` is gitignored (local config with API keys)
- `workspace/skills/` are third-party installed skills, don't modify
- `openspec/` and `.omc/` are tool state, gitignored
