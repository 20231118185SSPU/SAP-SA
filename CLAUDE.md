# sa/

Rust 后端 Agent 引擎。Cargo workspace: `sa-core`（核心逻辑库）+ `sa`（守护进程二进制）。

## Build Commands

```bash
export PATH="$PATH:/c/Users/杨先生/.cargo/bin"
cargo build -p sa --release       # 构建守护进程
cargo build --workspace --release # 构建全部
cargo test -p sa-core             # 核心库测试
cargo test --workspace            # 全部测试
```

## Architecture

- `crates/sa-core/` — 核心逻辑（记忆、工具、协议、技能、LLM）
- `crates/sa/` — 守护进程（WS 服务、事件分发、Agent 调度）
- `bindings/` — TypeScript 协议类型（由 generate-types.js 生成）
- `workflows/` — YAML 工作流定义
- `scripts/generate-types.js` — Rust → TypeScript 类型同步

## Key Patterns

- Memory system: unified SQLite store (`memory_store.rs`) with FTS5
- Protocol: WebSocket JSON between frontend and backend
- LLM: OpenAI-compatible API (DeepSeek, etc.)
- Skills: SKILL.md format, discovered at runtime

## Gotchas

- Cargo 不在默认 PATH — 用 `export PATH="$PATH:/c/Users/杨先生/.cargo/bin"`
- `sa.toml` 包含 API 密钥，被 gitignore
- `bindings/` 由脚本生成，不要手动修改
- 新功能先在 `sa-core` 实现，再在 `sa` 中接入
