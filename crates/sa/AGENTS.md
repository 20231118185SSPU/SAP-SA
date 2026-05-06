<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-06 | Updated: 2026-05-06 -->

# sa (守护进程)

## Purpose

SA 的后端守护进程。薄层包装 `sa-core`，负责 WebSocket 服务、任务队列、事件广播、Agent 调度、子代理管理。监听 `127.0.0.1:8765/ws`。

## Key Files

| File | Description |
|------|-------------|
| `src/main.rs` | 入口：配置加载、WS 服务、Agent 启动 |
| `src/config.rs` | 运行时配置补充 |
| `src/bash.rs` | Bash 命令执行封装 |
| `src/mirror.rs` | 镜像/同步逻辑 |
| `src/utils.rs` | 工具函数 |
| `src/bin/export_types.rs` | 类型导出工具 |

## For AI Agents

### Working In This Directory
- 这里只放 WS 调度逻辑，业务逻辑放 `sa-core`
- 修改 `main.rs` 前确保理解 WS 事件流
- 编译验证：`cargo build -p sa --release`

### Common Patterns
- axum Router + WebSocketUpgrade 处理 WS 连接
- tokio::spawn 处理并发连接
- Agent 任务通过 channel 与 WS 层通信

### Dependencies

#### Internal
- `sa-core` — 所有业务逻辑

#### External
- axum — HTTP/WS 服务
- tokio — 异步运行时
- futures-util — 流处理
- clap — 命令行参数

<!-- MANUAL: -->
