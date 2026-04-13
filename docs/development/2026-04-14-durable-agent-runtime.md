# SA Durable Agent Runtime 进展记录

## 背景

本轮开发的目标是把 SA 从“单任务 worker”逐步演进为“durable 多代理 actor runtime”。

现有仓库在本次提交链路中已经完成的基础能力：

1. 新增 `[team]` 配置段
2. 移除 `llm.max_steps`
3. 新增 durable runtime 状态模型
4. 新增 runtime store（root/team/agent/task/mailbox/pending_question）
5. 启动时自动创建 root/team/agent 基础状态文件
6. 扩展工具接口，加入：
   - `Finish`
   - `Wait`
   - `NotifyParent`
   - `MessageAgent`
   - `BroadcastAgents`
   - `ListAgents`
   - `GetAgent`
   - `GetTask`
   - `TransferInput`
   - 临时工具 `FinishWithoutOutput` 的运行时占位
7. `Bash` 已扩展 `run_in_background`
8. 后端已接入 runtime task registry 和后台 Bash 任务启动入口

## 当前代码落点

### 1. 配置

- 文件：`crates/sa-core/src/config.rs`
- 新增：
  - `TeamConfig`
  - 默认值：
    - `auto_resume = true`
    - `max_active_agents = 4096`
    - `max_concurrent_model_calls = 8`
- 删除：
  - `llm.max_steps`

### 2. runtime 状态模型

- 文件：`crates/sa-core/src/runtime/state.rs`
- 主要类型：
  - `TeamState`
  - `AgentState`
  - `RuntimeTaskState`
  - `MailboxEntry`
  - `PendingQuestionState`
  - `PendingFinishConfirmation`
  - `WaitingDependency`

### 3. runtime store

- 文件：`crates/sa-core/src/runtime/store.rs`
- 覆盖文件：
  - `runtime/root.json`
  - `runtime/team_state.json`
  - `runtime/agents/<agent_id>/state.json`
  - `runtime/agents/<agent_id>/mailbox.jsonl`
  - `runtime/agents/<agent_id>/pending_question.json`
  - `runtime/tasks/<task_id>.json`

### 4. session 泛化

- 文件：`crates/sa-core/src/session.rs`
- 新增：
  - `SessionStore::new_in_relative_dir(...)`
- 用途：
  - 支持 `sessions/agents/<agent_id>/...`

### 5. 工具接口扩展

- 文件：`crates/sa-core/src/tools.rs`
- 当前状态：
  - 类型系统与 schema 已扩展
  - `Finish` / `Wait` / agent communication / background Bash / `GetTask` 已有工具层入口
  - 但完整的 durable supervisor 还未全部接完

### 6. 启动时 durable 根状态初始化

- 文件：`crates/sa/src/main.rs`
- 当前行为：
  - 启动时会初始化 `RuntimeStore`
  - 若不存在 `root.json` / `team_state.json` / root agent state，则自动创建

### 7. 后台 Bash 任务

- 文件：`crates/sa/src/main.rs`
- 当前能力：
  - `Bash(run_in_background=true)` 会创建 runtime task state
  - 输出写入 `runtime/tasks/<task_id>.log`
  - `GetTask` 可以读取任务状态

## 当前未完成的关键部分

下面这些仍然是下一阶段必须完成的工作：

1. 旧 `Hub` 到新 supervisor 的完全切换
2. root agent / child agent 的真实 mailbox 驱动执行
3. `run_quantum()` 在 supervisor 中的正式接入
4. `Wait(agent/work/task)` 的 durable 挂起与恢复
5. `NotifyParent` / `MessageAgent` / `BroadcastAgents` 的真实 mailbox 投递
6. 输入归属转移的真实持久化与 WS 广播
7. `Finish` / `FinishWithoutOutput` 的完整 pending-confirmation 语义
8. `Ask` 的 durable 恢复

## 已建立的 git 节点

- commit: `615ff5e` `refactor: add durable runtime foundations`
- tag: `sa-runtime-foundation-20260414`
- commit: `803fa5b` `feat: add runtime task registry scaffolding`
- tag: `sa-runtime-task-scaffold-20260414`

## 下一步建议顺序

1. 先把 `Hub` 改为真正的 agent supervisor
2. 再把 `run_quantum()` 替换当前 legacy `run_task()` 路径
3. 接着落地 mailbox 驱动的 child runtime
4. 最后补齐 `Wait`、输入归属、pending finish confirmation
