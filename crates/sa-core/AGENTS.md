<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-06 | Updated: 2026-05-08 -->

# sa-core/

## Purpose

SA 的核心逻辑库。包含 Agent 循环、配置加载、提示词构建、技能系统、统一记忆存储（SQLite）、工具实现、MCP 客户端、LLM API 调用、安全检查等所有业务逻辑。守护进程 `sa` 是薄层包装。

## Key Files

| File | Description |
|------|-------------|
| `src/lib.rs` | 模块导出入口，声明所有 pub mod |
| `src/config.rs` | 配置加载与解析（sa.toml → SaConfig） |
| `src/agent.rs` | Agent 自治循环核心（AgentRunner） |
| `src/openai.rs` | LLM API 客户端（OpenAI/Anthropic 兼容） |
| `src/tools.rs` | 工具注册与调度入口 |
| `src/memory_store.rs` | **统一 SQLite 记忆存储**（替代旧多模块） |
| `src/memory.rs` | 记忆搜索/读取工具函数（BM25、metadata 评分） |
| `src/working_memory.rs` | 工作记忆层（热缓冲 + pinned slots） |
| `src/skills.rs` | 技能注册与加载 |
| `src/session.rs` | 会话持久化（JSONL） |
| `src/dream.rs` | Nightly dream 记忆提炼 |
| `src/compact.rs` | 会话压缩恢复 |
| `src/commands.rs` | 命令注册表 |
| `src/ws_protocol.rs` | WebSocket 协议定义 |
| `src/mcp_client.rs` | MCP 工具服务器客户端 |
| `src/path_guard.rs` | 路径安全检查 |
| `src/bash_safety.rs` | Bash 命令安全检查 |
| `src/fetch_safety.rs` | HTTP 请求安全检查 |
| `src/task_audit.rs` | 任务审计 |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `src/tools/` | 工具实现拆分（见 `src/tools/AGENTS.md`） |
| `src/cache/` | 缓存模块（见 `src/cache/AGENTS.md`） |
| `src/runtime/` | 运行时状态管理（见 `src/runtime/AGENTS.md`） |
| `src/memory/` | 空目录（旧记忆模块已合并至 `memory_store.rs`） |

## Module Categories

### Agent Core
- `agent.rs` — 自治循环：发消息 → 收 tool_call → 执行 → 回传 → 下一轮
- `commands.rs` — 命令注册（Skill、内置命令）
- `session.rs` — JSONL 会话持久化与恢复
- `compact.rs` — 会话压缩与摘要

### Memory System
- `memory_store.rs` — **统一 SQLite 存储**：episodic memories、semantic facts、pinned slots（三表合一，替代旧 14 个模块）
- `memory.rs` — BM25 搜索、metadata 评分、实体提取、事实抽取
- `memory_filter.rs` — 记忆过滤
- `memory_scope.rs` — 记忆作用域
- `working_memory.rs` — 工作记忆热缓冲（hot_buffer + pinned_slots + scratchpad）
- `dream.rs` — Nightly dream 记忆提炼
- `cost_budget.rs` — 每日 token 成本预算

### Safety
- `path_guard.rs` — 路径安全（防止目录穿越）
- `bash_safety.rs` — Bash 命令安全检查
- `fetch_safety.rs` — HTTP 请求安全检查
- `task_audit.rs` — 任务执行审计
- `pii_detector.rs` — 个人信息检测
- `field_encryption.rs` — 字段加密
- `adversary.rs` — 对抗性检测

### Tools
- `tools.rs` — 工具注册与调度（见 `src/tools/AGENTS.md`）

### LLM & Protocol
- `openai.rs` — OpenAI/Anthropic 兼容 API 客户端
- `ws_protocol.rs` — WebSocket 消息协议
- `mcp_client.rs` — MCP 客户端
- `mcp_protocol.rs` — MCP 协议实现
- `mcp_transport.rs` — MCP 传输层（stdio/http/sse）

### Search & Index
- `file_search.rs` — 文件搜索
- `file_index.rs` — 文件索引
- `file_analyzer.rs` — 文件分析
- `search_backends.rs` — 搜索后端

### Skills & Workflow
- `skills.rs` — 技能注册
- `skill_search.rs` — 技能搜索
- `skill_metabolism.rs` — 技能生命周期（自动淘汰/合并）
- `workflow.rs` / `workflow_engine.rs` — 工作流引擎
- `plan_engine.rs` — 计划引擎

### Other
- `config.rs` — 配置加载
- `agents_md.rs` — AGENTS.md 解析
- `cache_monitor.rs` — 缓存监控
- `tool_cache.rs` — 工具缓存
- `ws_identity.rs` — WebSocket 身份标识
- `retry.rs` — 重试逻辑
- `cancel.rs` — 取消机制
- `interaction_history.rs` — 交互历史

## For AI Agents

### Working In This Directory
- 新工具：在 `src/tools/` 下新建文件，在 `tools.rs` 注册
- 记忆相关修改：优先扩展 `memory_store.rs`，避免新建独立模块
- 安全相关改动需同步检查 `bash_safety.rs`、`fetch_safety.rs`、`path_guard.rs`
- 编译验证：`cargo build -p sa-core`

### Testing Requirements
- `cargo test -p sa-core`
- 安全模块测试尤其重要，覆盖边界情况

### Common Patterns
- 所有模块通过 `lib.rs` 的 `pub mod` 导出
- 工具函数返回 `Result<T, anyhow::Error>`
- 异步操作用 tokio
- 配置通过 `SaConfig` 结构体传递
- 记忆操作统一通过 `MemoryStore`（SQLite + Mutex）

### Migration Notes
- `memory_store.rs` 已替代：cold_store、crystallizer、vector_store、semantic_memory、memory_indexer、memory_l1_index、memory_metabolism、memory_pointer、procedural_memory、timeline_retrieval、search_feedback、mmr_rerank
- `index/` 目录已移除
- `memory/` 目录为空，保留但不再使用

<!-- MANUAL: -->
