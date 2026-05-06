<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-06 | Updated: 2026-05-06 -->

# sa-core/

## Purpose

SA 的核心逻辑库。包含 Agent 循环、配置加载、提示词构建、技能系统、记忆系统、工具实现、MCP 客户端、LLM API 调用、安全检查等所有业务逻辑。守护进程 `sa` 是薄层包装。

## Key Files

| File | Description |
|------|-------------|
| `src/lib.rs` | 模块导出入口，声明所有 pub mod |
| `src/config.rs` | 配置加载与解析（sa.toml） |
| `src/agent.rs` | Agent 自治循环核心（AgentRunner） |
| `src/openai.rs` | LLM API 客户端（OpenAI/Anthropic 兼容） |
| `src/tools.rs` | 工具注册与调度入口 |
| `src/memory.rs` | 记忆系统（MemorySearch、MemoryGet） |
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
| `src/tools/` | 工具实现拆分 |
| `src/cache/` | 缓存模块 |
| `src/index/` | 索引模块（SQLite + 向量） |
| `src/runtime/` | 运行时状态管理 |

## Module Categories

### Agent Core
- `agent.rs` — 自治循环：发消息 → 收 tool_call → 执行 → 回传 → 下一轮
- `commands.rs` — 命令注册（Skill、内置命令）
- `session.rs` — JSONL 会话持久化与恢复
- `compact.rs` — 会话压缩与摘要

### Memory System
- `memory.rs` — 记忆搜索/读取入口
- `memory_pointer.rs` — 记忆指针（MEMORY.md 只存指针）
- `memory_metabolism.rs` — 记忆新陈代谢（归档、衰减、晋升）
- `memory_filter.rs` — 记忆过滤
- `memory_scope.rs` — 记忆作用域
- `memory_indexer.rs` — 记忆索引构建
- `memory_l1_index.rs` — L1 索引
- `semantic_memory.rs` — 语义记忆
- `procedural_memory.rs` — 程序性记忆
- `cold_store.rs` — 冷存储
- `timeline_retrieval.rs` — 时间线检索

### Memory Optimization (P0-P4)
- `noise_assessment.rs` — P0: 噪音控制
- `memory_pointer.rs` — P1: 记忆指针
- `memory_metabolism.rs` — P2: 记忆新陈代谢
- `timeline_retrieval.rs` — P3: 时间线检索
- `skill_metabolism.rs` — P4: 技能淘汰

### Safety
- `path_guard.rs` — 路径安全（防止目录穿越）
- `bash_safety.rs` — Bash 命令安全检查
- `fetch_safety.rs` — HTTP 请求安全检查
- `task_audit.rs` — 任务执行审计
- `pii_detector.rs` — 个人信息检测
- `field_encryption.rs` — 字段加密
- `adversary.rs` — 对抗性检测

### Tools
- `tools.rs` — 工具注册与调度
- `tools/file_ops.rs` — Read/Write/Edit 文件操作
- `tools/shell_ops.rs` — Bash 命令执行
- `tools/memory_ops.rs` — MemorySearch/MemoryGet
- `tools/mcp_ops.rs` — MCP 工具调用
- `tools/misc_ops.rs` — Send/Show/Ask/Skill 等
- `tools/agent_ops.rs` — SubAgent/TransferInput

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
- `search_feedback.rs` — 搜索反馈
- `vector_store.rs` — 向量存储
- `index/` — SQLite + sqlite-vec 索引

### Other
- `config.rs` — 配置加载
- `agents_md.rs` — AGENTS.md 解析
- `skills.rs` — 技能注册
- `skill_search.rs` — 技能搜索
- `dream.rs` — Nightly dream 提炼
- `workflow.rs` / `workflow_engine.rs` — 工作流引擎
- `plan_engine.rs` — 计划引擎
- `crystallizer.rs` — 记忆结晶
- `cost_budget.rs` — 成本预算
- `cache_monitor.rs` — 缓存监控
- `tool_cache.rs` — 工具缓存
- `working_memory.rs` — 工作记忆
- `ws_identity.rs` — WebSocket 身份标识
- `retry.rs` — 重试逻辑
- `cancel.rs` — 取消机制
- `interaction_history.rs` — 交互历史
- `mmr_rerank.rs` — MMR 重排

## For AI Agents

### Working In This Directory
- 新工具：在 `src/tools/` 下新建文件，在 `tools.rs` 注册
- 新记忆模块：在 `src/` 下新建，按命名规范（memory_*.rs）
- 安全相关改动需同步检查 `bash_safety.rs`、`fetch_safety.rs`、`path_guard.rs`
- 编译验证：`cargo build -p sa-core`

### Testing Requirements
- `cargo test -p sa-core`
- 安全模块测试尤其重要，覆盖边界情况

### Common Patterns
- 所有模块通过 `lib.rs` 的 `pub mod` 导出
- 工具函数返回 `Result<T, anyhow::Error>`
- 异步操作用 tokio
- 配置通过 `Config` 结构体传递

<!-- MANUAL: -->
