<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-06 | Updated: 2026-05-06 -->

# crates/

## Purpose

SA 后端的 Rust crate 工作区。`sa-core` 提供核心逻辑（配置、提示词、技能、记忆、工具、LLM 客户端），`sa` 是薄层守护进程（WebSocket、任务队列、事件广播）。

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `sa-core/` | 核心逻辑库 — 所有业务逻辑在此（见 `sa-core/AGENTS.md`） |
| `sa/` | 守护进程 — WS 服务、事件分发、Agent 调度（见 `sa/AGENTS.md`） |

## For AI Agents

### Working In This Directory
- 优先修改 `sa-core/`，守护进程 `sa/` 只处理 WS 和调度
- 新功能先在 `sa-core` 实现，再在 `sa` 中接入
- 编译验证：`cargo build --workspace --release`

### Common Patterns
- `sa-core` 暴露 `pub mod`，`sa` 通过 `use sa_core::*` 引用
- 两个 crate 共享同一个 Cargo.lock

<!-- MANUAL: -->
