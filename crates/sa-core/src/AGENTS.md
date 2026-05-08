<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-08 | Updated: 2026-05-08 -->

# sa-core/src/

## Purpose

SA 核心库源码目录。包含 Agent 自治循环、统一记忆存储（SQLite）、工具执行、LLM API 客户端、WebSocket 协议、MCP 集成、技能系统、安全检查等所有业务逻辑实现。

## Module Table

All public modules declared in `lib.rs`:

| Module | File(s) | Description |
|--------|---------|-------------|
| `adversary` | `adversary.rs` | 对抗性检测（red-team 测试） |
| `agent` | `agent.rs` | Agent 自治循环核心（AgentRunner） |
| `agents_md` | `agents_md.rs` | AGENTS.md 解析与格式化 |
| `bash_safety` | `bash_safety.rs` | Bash 命令安全检查与拦截 |
| `cache` | `cache/` | 缓存子模块（MemoryCache） |
| `cache_monitor` | `cache_monitor.rs` | 缓存性能监控与 token 统计 |
| `cancel` | `cancel.rs` | 取消机制（CancelToken） |
| `commands` | `commands.rs` | 命令注册表 |
| `compact` | `compact.rs` | 会话上下文压缩与摘要 |
| `config` | `config.rs` | 配置加载与解析（sa.toml → SaConfig） |
| `cost_budget` | `cost_budget.rs` | 每日 token 成本预算跟踪与强制执行 |
| `dream` | `dream.rs` | Nightly dream 记忆提炼系统 |
| `fetch_safety` | `fetch_safety.rs` | HTTP 请求安全检查 |
| `field_encryption` | `field_encryption.rs` | 字段加密 |
| `file_analyzer` | `file_analyzer.rs` | 文件内容分析 |
| `file_index` | `file_index.rs` | 文件索引构建 |
| `file_search` | `file_search.rs` | 文件搜索 |
| `interaction_history` | `interaction_history.rs` | 交互历史记录 |
| `mcp_client` | `mcp_client.rs` | MCP 工具服务器客户端 |
| `mcp_protocol` | `mcp_protocol.rs` | MCP 协议消息定义 |
| `mcp_transport` | `mcp_transport.rs` | MCP 传输层（stdio/HTTP/SSE） |
| `memory` | `memory.rs` | 记忆搜索/读取工具函数（BM25、metadata 评分） |
| `memory_filter` | `memory_filter.rs` | 记忆过滤逻辑 |
| `memory_scope` | `memory_scope.rs` | 记忆作用域管理 |
| `memory_store` | `memory_store.rs` | **统一 SQLite 记忆存储**（替代旧多模块） |
| `noise_assessment` | `noise_assessment.rs` | P0 噪音评估与控制 |
| `openai` | `openai.rs` | OpenAI/Anthropic 兼容 LLM API 客户端 |
| `path_guard` | `path_guard.rs` | 路径安全检查（防目录穿越） |
| `pii_detector` | `pii_detector.rs` | 个人信息检测 |
| `plan_engine` | `plan_engine.rs` | 计划引擎（LLM 生成计划并逐步执行） |
| `retry` | `retry.rs` | 重试逻辑与延迟 |
| `runtime` | `runtime/` | 运行时状态管理（RuntimeState、RuntimeStore） |
| `search_backends` | `search_backends.rs` | 搜索后端实现 |
| `session` | `session.rs` | 会话持久化（JSONL） |
| `skill_metabolism` | `skill_metabolism.rs` | 技能生命周期管理（自动淘汰/合并） |
| `skill_search` | `skill_search.rs` | 技能搜索 |
| `skills` | `skills.rs` | 技能注册与加载（SkillRegistry） |
| `task_audit` | `task_audit.rs` | 任务执行审计 |
| `tool_cache` | `tool_cache.rs` | 工具调用缓存 |
| `tools` | `tools.rs` + `tools/` | 工具注册、调度与实现 |
| `working_memory` | `working_memory.rs` | 工作记忆层（热缓冲 + pinned slots） |
| `workflow` | `workflow.rs` | 工作流定义 |
| `workflow_engine` | `workflow_engine.rs` | 工作流执行引擎 |
| `ws_identity` | `ws_identity.rs` | WebSocket 身份标识 |
| `ws_protocol` | `ws_protocol.rs` | WebSocket JSON 协议消息类型 |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `tools/` | 工具实现拆分（agent_ops、file_ops、shell_ops、memory_ops、misc_ops、mcp_ops） |
| `cache/` | 缓存模块（MemoryCache — moka 驱动的 search/file 缓存） |
| `runtime/` | 运行时状态（RuntimeState 类型定义、RuntimeStore 文件系统持久化） |
| `memory/` | 空目录（旧记忆模块已合并至 `memory_store.rs`） |

## Module Categories

### Agent Core
- `agent.rs` — 自治循环：发消息 → 收 tool_call → 执行 → 回传 → 下一轮
- `commands.rs` — 命令注册（Skill、内置命令）
- `session.rs` — JSONL 会话持久化与恢复
- `compact.rs` — 会话上下文压缩（保留最新 suffix，摘要旧 prefix）

### Memory System
- `memory_store.rs` — **统一 SQLite 存储**：episodic memories、semantic facts、pinned slots（三表合一，替代旧 14 个模块）
- `memory.rs` — BM25 搜索、metadata 评分、实体提取、事实抽取
- `memory_filter.rs` — 记忆过滤
- `memory_scope.rs` — 记忆作用域
- `working_memory.rs` — 工作记忆热缓冲（hot_buffer + pinned_slots + scratchpad）
- `dream.rs` — Nightly dream 记忆提炼（session → topic MEMORY.md → 审计）
- `cost_budget.rs` — 每日 token 成本预算

### Tools
- `tools.rs` — 工具注册与调度入口（ToolExecutor）
- `tools/agent_ops.rs` — SubAgent、TransferInput
- `tools/file_ops.rs` — Read/Write/Edit 文件操作
- `tools/shell_ops.rs` — Bash 命令执行
- `tools/memory_ops.rs` — memory_search/memory_get/memory_set/pin/unpin
- `tools/misc_ops.rs` — Send/Show/Ask/Skill 等
- `tools/mcp_ops.rs` — MCP 工具调用

### Safety
- `path_guard.rs` — 路径安全（防目录穿越）
- `bash_safety.rs` — Bash 命令安全检查
- `fetch_safety.rs` — HTTP 请求安全检查
- `task_audit.rs` — 任务执行审计
- `pii_detector.rs` — 个人信息检测
- `field_encryption.rs` — 字段加密
- `adversary.rs` — 对抗性检测

### LLM & Protocol
- `openai.rs` — OpenAI/Anthropic 兼容客户端（chat_completions / responses / anthropic_messages 三种 wire API）
- `ws_protocol.rs` — WebSocket 消息协议（ClientMessage / ServerMessage tagged enums）
- `mcp_client.rs` — MCP 客户端
- `mcp_protocol.rs` — MCP 协议消息
- `mcp_transport.rs` — MCP 传输层（stdio / HTTP / SSE）

### Search & Index
- `file_search.rs` — 文件搜索
- `file_index.rs` — 文件索引
- `file_analyzer.rs` — 文件分析
- `search_backends.rs` — 搜索后端

### Skills & Workflow
- `skills.rs` — 技能注册（SkillRegistry）
- `skill_search.rs` — 技能搜索
- `skill_metabolism.rs` — 技能生命周期（自动淘汰/合并/晋升）
- `workflow.rs` — 工作流定义
- `workflow_engine.rs` — 工作流执行引擎
- `plan_engine.rs` — 计划引擎（LLM 生成 → 逐步执行）

### Other
- `config.rs` — 配置加载（sa.toml → SaConfig）
- `agents_md.rs` — AGENTS.md 解析
- `cache_monitor.rs` — 缓存性能监控
- `tool_cache.rs` — 工具调用缓存
- `ws_identity.rs` — WebSocket 身份标识
- `retry.rs` — 重试逻辑
- `cancel.rs` — 取消机制
- `interaction_history.rs` — 交互历史

## For AI Agents

### Working In This Directory
- 新工具：在 `tools/` 下新建文件，在 `tools.rs` 的 `tool_definitions()` 注册
- 新记忆模块：优先扩展 `memory_store.rs`，避免新建独立模块
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
- WebSocket 消息用 tagged enum（serde tag = "type"）

### Migration Notes
- `memory_store.rs` 已替代旧模块：cold_store、crystallizer、vector_store、semantic_memory、memory_indexer、memory_l1_index、memory_metabolism、memory_pointer、procedural_memory、timeline_retrieval、search_feedback、mmr_rerank
- `memory/` 目录为空，保留但不再使用

<!-- MANUAL: -->
