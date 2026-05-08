<!-- Generated: 2026-05-06 | Updated: 2026-05-08 -->

# SAP-6.2

## Purpose

StudyAdministrator (SA) — 本地运行的"学习委员 Agent"系统。包含 Rust 后端引擎、React WebUI 前端、以及 workspace 运行时环境。通过 WebSocket 通信，支持自治工具调用循环、分层记忆系统、技能系统和 MCP 工具集成。

## Key Files

| File | Description |
|------|-------------|
| `.gitignore` | 排除 target/、node_modules/、sa.toml、MEMORY.md 等 |
| `AGENTS.md` | 本文件 — 项目级 AI Agent 导航 |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `sa/` | Rust 后端 Agent 引擎（见 `sa/AGENTS.md`） |
| `WebUI/` | React/Vite Web 前端（见 `WebUI/AGENTS.md`） |
| `workspace/` | Agent 运行时环境：配置、技能、记忆、会话（见 `workspace/AGENTS.md`） |
| `docs/` | 调查报告与分析文档（见 `docs/AGENTS.md`） |
| `bin/` | 编译产物（见 `bin/AGENTS.md`） |

## Architecture

```text
用户 ←→ WebUI (React/Vite) ←WebSocket→ sa (Rust 后端) ←→ LLM API
                                    ↕
                              workspace/
                              ├── skills/    (SKILL.md 技能包)
                              ├── memory/    (分层记忆系统)
                              ├── sessions/  (会话持久化)
                              └── scripts/   (MCP 辅助脚本)
```

## Build Commands

| Command | What It Builds |
|---------|----------------|
| `cargo build -p sa --release` | Rust 后端二进制（输出 `target/release/sa.exe`） |
| `cd WebUI && npm run build` | WebUI 前端（输出 `WebUI/dist/`） |
| `cargo test --workspace` | Rust 全工作区测试 |
| `cd WebUI && npm test` | WebUI 前端测试（vitest） |

## For AI Agents

### Working In This Directory
- 修改 `sa/` 下的 Rust 代码后用 `cargo build -p sa --release` 验证
- 修改 `WebUI/` 下的前端代码后用 `cd WebUI && npm run build` 验证
- `workspace/` 下的配置文件（sa.toml、skill 等）运行时热重载，无需重启
- `docs/` 下的调查报告为只读参考，修改前确认不影响进行中的工作

### Testing Requirements
- Rust: `cargo test --workspace`
- WebUI: `cd WebUI && npm test`

### Common Patterns
- 后端与前端通过 WebSocket JSON 协议通信
- 工具调用走自治循环：模型返回 tool_call → 执行 → 结果回传 → 下一轮
- 记忆分层：MEMORY.md（长期）→ memory/*.md（日常）→ memory/dreams/*.md（审计）
- 记忆存储统一使用 `memory_store.rs`（已合并旧的多模块记忆实现）

## Dependencies

### Internal
- `sa-core` crate: 核心逻辑（记忆、工具、协议、技能、成本控制）
- `sa` crate: 二进制入口（main.rs、config.rs、bash 集成）

### External
- Rust (Cargo workspace, edition 2024)
- Node.js + Vite + React 19 + TypeScript
- OpenAI / Anthropic 兼容 LLM API
- SQLite (rusqlite with bundled feature)
- tokio async runtime

<!-- MANUAL: -->
