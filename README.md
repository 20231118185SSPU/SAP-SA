# Claw (minimal agent)

`claw` is a **minimal, runnable, autonomous Rust agent** extracted/simplified from the ideas in
`../zeroclaw`:

- Calls an **OpenAI-compatible** `POST /v1/chat/completions` API.
- Reads `Agents.md` from the workspace as the base instruction prompt.
- Discovers and loads `SKILL.md`-based skills (Codex/Agents skills format).
- Runs a **tool-calling loop** (shell/file ops) to act autonomously.
- Exposes a **WebSocket** server so a CLI can connect/disconnect without stopping the agent.

## Quick start

1. Create a local config:

   - Copy `claw.example.toml` to `claw.toml`
   - Fill in `llm.api_key`

2. Start the agent daemon (WebSocket server):

```bash
cargo run -p claw-agentd
```

3. In another terminal, send a task via the CLI:

```bash
cargo run -p claw-cli -- run "请在当前工作区创建一个README并解释如何运行"
```

## Docs

Development docs live in `./docs/development/`.

