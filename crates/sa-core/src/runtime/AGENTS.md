<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-08 | Updated: 2026-05-08 -->

# sa-core/src/runtime/

## Purpose

多代理运行时状态管理。负责代理状态、团队状态、任务状态的类型定义和文件系统持久化。守护进程 `sa` 的 Hub 层使用此模块管理多代理生命周期。

## Key Files

| File | Description |
|------|-------------|
| `mod.rs` | 模块导出入口 |
| `state.rs` | 运行时状态类型定义（AgentState、TeamState、RuntimeTaskState 等） |
| `store.rs` | 文件系统持久化层（RuntimeStore） |

## Module Details

### state.rs
定义运行时核心类型：
- `AgentState` — 单个代理的状态（status、config、current_task）
- `AgentStatus` — 代理状态枚举（Idle、Working、Paused 等）
- `TeamState` — 团队状态（成员列表、共享上下文）
- `RuntimeTaskState` — 任务状态
- `RuntimeWorkState` — 工作单元状态
- `MailboxEntry` — 代理间消息
- `PendingQuestionState` — 待回答问题

### store.rs
文件系统持久化层（RuntimeStore）。管理以下目录结构：
```text
runtime/
├── root.json              # 根标记文件
├── team_state.json        # 团队状态
├── agents/<id>/
│   ├── state.json         # 代理状态
│   ├── mailbox.jsonl      # 代理邮箱（append-only）
│   └── pending_question.json
├── tasks/<id>.json        # 任务状态
└── works/<id>.json        # 工作单元状态
```

关键操作：
- 原子写入（通过 tempfile + rename）
- 代理状态读写
- 邮箱消息追加
- 任务状态管理

## For AI Agents

### Working In This Directory
- 状态类型修改影响所有使用 `RuntimeState` 的模块
- 持久化格式变更需考虑向后兼容
- 编译验证：`cargo build -p sa-core`

### Common Patterns
- 所有状态类型通过 serde 序列化为 JSON
- 文件写入使用原子操作（tempfile + fs::rename）
- Mutex 保护并发访问

<!-- MANUAL: -->
