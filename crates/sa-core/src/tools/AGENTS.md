<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-08 | Updated: 2026-05-08 -->

# sa-core/src/tools/

## Purpose

工具实现拆分目录。每个文件实现一类工具操作，通过 `ToolExecutor` 统一调度。主入口 `tools.rs`（父目录）负责工具注册和调度逻辑。

## Key Files

| File | Description |
|------|-------------|
| `agent_ops.rs` | SubAgent 创建、TransferInput 等代理间操作 |
| `file_ops.rs` | Read/Write/Edit/Glob/Grep 文件操作 |
| `shell_ops.rs` | Bash 命令执行（受 bash_safety 约束） |
| `memory_ops.rs` | memory_search/memory_get/memory_set/edit_memory/pin/unpin |
| `misc_ops.rs` | Send/Show/Ask/Skill 等杂项工具 |
| `mcp_ops.rs` | MCP 工具调用代理 |

## Module Details

### agent_ops.rs
代理间操作工具。实现 `AgentOps` trait，包含：
- SubAgent 创建与管理
- TransferInput 在代理间传递输入

### file_ops.rs
文件系统操作工具。实现文件的读取、写入、编辑、glob 搜索和 grep 内容搜索。受 `path_guard.rs` 安全约束。

### shell_ops.rs
Shell 命令执行工具。通过 `bash_safety.rs` 进行命令安全检查后执行。支持后台运行和超时控制。

### memory_ops.rs
记忆操作工具。所有操作通过 `MemoryStore`（SQLite）执行：
- `memory_search` — 语义搜索记忆条目
- `memory_get` — 获取单条记忆
- `memory_set` — 创建/更新记忆
- `edit_memory` — 部分更新记忆
- `forget_memory` — 删除记忆
- `pin_memory` / `unpin_memory` — 管理 pinned slots

### misc_ops.rs
杂项工具实现，包括消息发送、内容展示、交互式问答、技能调用等。

### mcp_ops.rs
MCP 工具调用代理。将工具调用转发至外部 MCP 服务器。

## For AI Agents

### Adding a New Tool
1. 在对应 `*_ops.rs` 文件中添加方法
2. 在 `tools.rs` 的 `tool_definitions()` 中注册工具定义
3. 在 `tools.rs` 的调度逻辑中添加路由
4. 测试：`cargo test -p sa-core`

### Common Patterns
- 所有工具方法签名：`async fn tool_name(&self, args: Value, cancel: &CancelToken) -> Result<String>`
- 参数通过 `serde::Deserialize` 从 JSON 解析
- 异步操作通过 tokio 执行
- 取消通过 `CancelToken` 检查

<!-- MANUAL: -->
