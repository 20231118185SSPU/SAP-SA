# SA Dream Memory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 SA 的记忆系统升级为“原始经历 -> dream 提炼 -> 长期记忆”的分层体系，并在每天本地 0 点自动触发 dream，让 SA 的长期记忆可持续净化与成长。

**Architecture:** 保留现有 `sessions/*.jsonl` 与 compact 机制作为原始经历层；新增 `dream` 调度与状态模块，在后台以隔离 Agent 运行记忆提炼任务；重构系统提示，明确长期记忆、专题记忆、日记与 dream 审计的职责边界，并把 `prompt.md` 作为运行时静态提示词真源。

**Tech Stack:** Rust 2024, tokio, chrono, fs4, anyhow, serde, axum, 现有 `sa-core` AgentRunner / ToolExecutor / SessionStore

---

## File Map

- Modify: `crates/sa-core/src/agent.rs`
  - 将大段静态提示词迁移为基于 `prompt.md` 的静态模板加载，并追加动态 section。
- Create: `crates/sa-core/src/dream.rs`
  - dream 配置、状态、锁、计划时间计算、候选输入文件收集、后台任务 prompt 构造。
- Modify: `crates/sa-core/src/config.rs`
  - 增加 `[dream]` 配置段并补测试。
- Modify: `crates/sa-core/src/lib.rs`
  - 导出 `dream` 模块。
- Modify: `crates/sa-core/src/memory.rs`
  - 调整 memory 搜索边界，支持 `memory/topics/`，排除 `memory/dreams/` 审计层。
- Modify: `crates/sa/src/main.rs`
  - 在 runtime ready 后启动 dream 调度器；dream 使用隔离运行态，不污染主会话。
- Modify: `prompt.md`
  - 作为 SA 静态提示词真源，明确分层记忆与 dream 规则。
- Modify: `sa.example.toml`
  - 暴露 `[dream]` 配置说明。
- Create: `docs/development/dream-memory.md`
  - 记录分层记忆与 dream 设计。
- Modify: `docs/development/README.md`
  - 链接 dream 文档并更新限制说明。
- Modify: `README.md`
  - 更新功能说明。

### Task 1: Dream Core And Config

**Files:**
- Create: `crates/sa-core/src/dream.rs`
- Modify: `crates/sa-core/src/config.rs`
- Modify: `crates/sa-core/src/lib.rs`
- Test: `crates/sa-core/src/dream.rs`
- Test: `crates/sa-core/src/config.rs`

- [ ] **Step 1: 写 dream 配置与时间计算测试**
- [ ] **Step 2: 运行 `cargo test -p sa-core dream config -- --nocapture` 验证测试先失败**
- [ ] **Step 3: 实现 `DreamConfig`、状态文件、文件锁、下一次 0 点触发计算、是否需要补跑判断**
- [ ] **Step 4: 实现 dream 输入文件选择：`MEMORY.md`、`memory/topics/*.md`、最近日记、最近 session 段**
- [ ] **Step 5: 重新运行相关测试直至通过**
- [ ] **Step 6: commit**

### Task 2: Prompt And Memory Policy

**Files:**
- Modify: `prompt.md`
- Modify: `crates/sa-core/src/agent.rs`
- Modify: `crates/sa-core/src/memory.rs`
- Test: `crates/sa-core/src/agent.rs`
- Test: `crates/sa-core/src/memory.rs`

- [ ] **Step 1: 先写测试，约束系统提示必须包含分层记忆与 dream 规则，并且 `memory/dreams/` 不进入普通记忆搜索**
- [ ] **Step 2: 运行 `cargo test -p sa-core agent memory -- --nocapture` 验证测试先失败**
- [ ] **Step 3: 重写 `prompt.md`，参考 Claude Code 的记忆设计但改成 SA 的“学习委员 Agent”语境**
- [ ] **Step 4: 改 `agent.rs`，从 `prompt.md` 加载静态提示词并拼接动态 section**
- [ ] **Step 5: 改 `memory.rs` 搜索边界，支持专题记忆、排除 dream 审计层**
- [ ] **Step 6: 重新运行相关测试直至通过**
- [ ] **Step 7: commit**

### Task 3: Runtime Scheduler Integration

**Files:**
- Modify: `crates/sa/src/main.rs`
- Test: `crates/sa/src/main.rs`

- [ ] **Step 1: 先写测试，约束 runtime ready 后会启动 dream 调度，并且可计算启动补跑与下次午夜触发**
- [ ] **Step 2: 运行 `cargo test -p sa main -- --nocapture` 验证测试先失败**
- [ ] **Step 3: 实现后台 dream 调度器，启动时补跑、之后每天本地 0 点触发**
- [ ] **Step 4: 让 dream 使用隔离 Agent 运行态，不接管主持久化 session，不向同学发送普通交互消息**
- [ ] **Step 5: 重新运行相关测试直至通过**
- [ ] **Step 6: commit**

### Task 4: Documentation And Examples

**Files:**
- Modify: `sa.example.toml`
- Create: `docs/development/dream-memory.md`
- Modify: `docs/development/README.md`
- Modify: `README.md`

- [ ] **Step 1: 写文档，明确原始层 / 提炼层 / 长期层与 dream 职责**
- [ ] **Step 2: 更新示例配置与 README**
- [ ] **Step 3: commit**

### Task 5: Verification And Tag

**Files:**
- Modify: `Cargo.toml` only if verification reveals required dependency changes

- [ ] **Step 1: 运行 `cargo test --all`**
- [ ] **Step 2: 运行 `cargo build --release`**
- [ ] **Step 3: 检查文档与示例配置路径是否一致**
- [ ] **Step 4: 汇总结果并创建关键节点 tag**
