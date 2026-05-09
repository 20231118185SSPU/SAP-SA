<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-09 | Updated: 2026-05-09 -->

# sa/

## Purpose

Rust 后端 Agent 引擎。Cargo workspace 包含两个 crate：`sa-core`（核心逻辑库）和 `sa`（守护进程二进制）。通过 WebSocket 与 WebUI 通信，支持自治工具调用循环、分层记忆系统、技能代谢、MCP 工具集成。

## Key Files

| File | Description |
|------|-------------|
| `BOOTSTRAP.md` | Agent 启动引导指令 |
| `README.md` | 后端说明文档 |
| `USER.md` | 用户画像配置 |
| `sa.example.toml` | 配置模板（实际 sa.toml 被 gitignore） |
| `.gitignore` | 排除 sa.toml、target/、runtime/ 等 |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `crates/` | Cargo workspace crate 目录（见 `crates/AGENTS.md`） |
| `docs/` | 开发文档与计划（见 `docs/AGENTS.md`） |
| `bindings/` | 语言绑定 |
| `scripts/` | 构建与辅助脚本 |
| `skills/` | 内置技能（如 image-pipeline） |
| `tools/` | 工具定义 |
| `workflows/` | 工作流定义 |
| `interactions/` | 交互记录 |
| `memory/` | 记忆存储 |
| `sessions/` | 会话持久化 |
| `runtime/` | 运行时状态（agents、workflows、works） |

## Architecture

```text
sa (Cargo workspace)
├── crates/sa-core/   ← 核心逻辑库（记忆、工具、协议、技能、LLM）
└── crates/sa/        ← 守护进程二进制（main.rs、WS 服务、事件分发）
```

## For AI Agents

### Working In This Directory
- Cargo 不在默认 PATH — 使用 `export PATH="$PATH:/c/Users/杨先生/.cargo/bin"`
- 构建：`cargo build -p sa --release`
- 测试：`cargo test -p sa-core`
- `sa.toml` 包含 API 密钥，被 gitignore，不要提交
- `runtime/`、`sessions/`、`memory/`、`interactions/` 是运行时数据，被 gitignore

### Common Patterns
- 所有核心逻辑在 `sa-core` crate，`sa` crate 仅负责启动和配置
- WebSocket JSON 协议与前端通信
- 工具调用走自治循环：model tool_call → 执行 → 结果回传 → 下一轮
- 技能发现：扫描 workspace/skills/ 下的 SKILL.md 文件

## Dependencies

### Internal
- `sa-core` crate — 核心逻辑（被 sa crate 依赖）
- `workspace/` — 运行时技能、配置、记忆

### External
- tokio — 异步运行时
- rusqlite (bundled) — SQLite 存储
- reqwest — HTTP 客户端
- tokio-tungstenite — WebSocket
- serde / serde_json — 序列化

<!-- MANUAL: -->
