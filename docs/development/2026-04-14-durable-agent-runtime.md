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

## 2026-04-14 第二阶段实装结果

本轮继续推进后，以下关键路径已经真正接通，不再只是类型占位：

1. `main.rs` 已切换为 durable supervisor 驱动
2. `submit` 不再进入 legacy 单 worker 队列，而是：
   - 读取 `team_state.json`
   - 把消息投递到当前输入所有者的 mailbox
   - 必要时创建新的 `active_work_id`
   - 唤醒对应 agent
3. `run_quantum()` 已正式接入 supervisor
   - 每次只跑一个 quantum
   - quantum 结束后根据 `Continue / Ask / Wait / Finish` 更新 durable 状态
4. `Ask` 已改为 durable control flow
   - `Ask` 不再依赖进程内阻塞等待
   - 问题会写入 `pending_question.json`
   - 回答后会把 tool-result 直接写回该 agent 的 session JSONL
   - 重启后可恢复 pending question
5. `Finish` / `FinishWithoutOutput` 已接通
   - root 持有输入且未 `Send/Show` 时，会进入 pending confirmation
   - child 未显式通知父代理时，也会进入 pending confirmation
   - `FinishWithoutOutput()` 只在临时提醒场景中可用
6. `Wait(agent/work/task)` 已接通 durable 挂起/恢复
   - wait 状态持久化到 agent state
   - 背景任务结束、agent idle、work finish 时会唤醒 waiters
   - timeout 也会写入 mailbox 并重新唤醒 agent
7. durable 子代理已经落地
   - `SubAgent` 现在会创建/复用真正的 child agent state
   - child 拥有独立 session 目录 `sessions/agents/<agent_id>/`
   - 父代理通过 mailbox 投递任务和上下文
8. agent 间通信已接通
   - `NotifyParent`
   - `MessageAgent`
   - `BroadcastAgents`
   - `ListAgents`
   - `GetAgent`
   - `TransferInput`
9. 后台 Bash 任务已接入 runtime wait/notify
   - 任务结束后会写入 owner agent mailbox
   - 同时唤醒 `Wait(task)` 的等待者
10. 启动恢复已接通
    - 会重建 pending questions
    - 会恢复 timeout watcher
    - 在 `team.auto_resume = true` 时自动唤醒可恢复的 agent

## 本轮验证

- `cargo test -q -j 1`
- `cargo build --release -q -j 1`

两者均已通过。

## 2026-04-14 审查后修复

在多 agent 并行代码审查后，本轮继续修复了几类高风险问题：

1. `submit` / `Accepted` / `interrupt` 的 work id 对齐
   - `submit` 现在返回稳定的 effective work id
   - 重试同一个 submit id 会返回同一个 work id
   - `Accepted.task_id` 已改为真实 work id，而不是 submit id
2. `interrupt` 不再只是把 agent 设回 idle
   - 现在会把当前 active work 显式终结为 cancelled
   - 清理 active work 元数据
   - 唤醒等待该 work / agent 的等待者
3. `Wait(...)` 不再被任意 mailbox 消息打断
   - 只有真正满足 dependency 的路径才会解除等待
4. `SubAgent(existing_agent_id=...)` 权限边界已收紧
   - 只能复用直属 child
   - 只能复用 idle 的 `Worker`
   - 不能跨树、不能复用 root / sibling / busy agent
   - 复用时要求 capability 配置一致
5. durable 子代理增加稳定的 parent work 绑定
   - `NotifyParent` / `ChildFinished` 不再依赖“父代理此刻的 active_work_id”
   - 改为绑定到创建该 child work 时的 `parent_work_id`
6. mailbox 消息追加做了进程内串行化
   - 避免并发写入生成重复 offset
   - 新增并发测试覆盖
7. `last_mailbox_offset` 的提交时机后移
   - 不再在量子执行前就标记为已消费
   - 避免崩溃后永久跳过尚未入 session 的 mailbox 消息
8. 启动恢复会重整遗留的 `Running` 后台任务
   - 统一标记为失败
   - 向 owner mailbox 投递通知
   - 唤醒 `Wait(task)` 等待者
9. `answer_question` 调整为更可恢复的顺序
   - 先完成 durable 写入，再移除内存 pending
   - 对同一 `tool_call_id` 增加幂等写回检查，避免重复 tool result
10. 增加 control checkpoint
    - `Ask` / `Wait` / `Finish` 的控制决策会先进入 durable checkpoint
    - 重启后可以补写 assistant control message，并继续完成对应 runtime 过渡

## 2026-04-14 work index 补齐

为了解决“older finished work 无法再被 `Wait(kind=work)` 命中”的缺口，本轮补充了 durable work index：

1. 新增 `RuntimeWorkState`
   - 路径：`runtime/works/<work_id>.json`
   - 状态：
     - `running`
     - `finished`
     - `cancelled`
2. work 启动时会建档
   - root 新 work
   - durable child 新 work
   - 以及恢复过程中补建的 active work
3. work 终结时会写入终态
   - `Finish` -> `finished`
   - `interrupt` / 显式取消 -> `cancelled`
4. `Wait(kind=work)` 现在改为查询 durable work index
   - 不再依赖单个 `last_finished_work_id`
   - 因此同一 agent 后续再完成新的 work，也不会让更早完成的 work 丢失可观测性
5. 新增回归测试
   - 证明 older finished work 在 newer finished work 出现后仍然可以被立即识别

## 2026-04-14 交互历史独立持久化

为满足“把所有直接交互单独持久化、便于 grep 与按 id 精准读取”的需求，本轮新增了 interaction history：

1. 新增独立日志
   - 路径：`interactions/history.jsonl`
   - 与 session/runtime state 分离
   - 按追加顺序记录全部直接交互
2. 每条交互日志都包含：
   - 唯一 `uuid`
   - 带本地时区偏移的高精度本地时间戳
   - `work_id`
   - `agent_id`
   - 具体交互 payload
3. 当前已落盘的交互种类：
   - 用户 `submit` 消息
   - `Send`
   - `Ask`
   - `AskAnswer`
   - `Show`
4. `Show` 新增强制参数 `prompt`
   - 必填
   - 用于描述展示内容或展示目的
   - 不作为常规用户文案单独展示，但会进入原始信息流与交互历史
5. `Show` 在交互历史中会保存文件快照
   - 文本内容直接存
   - 二进制内容按编码后字符串存
   - 同时保存 `path`、`file_name`、`title`、`prompt`、`media_type`、`bytes`
6. 新增 `GetInteractionEntry` 工具
   - 通过 `id` 获取交互条目
   - 支持 `summary` / `full` / `slice`
   - 避免大内容一次性塞进上下文
   - 推荐工作流：
     - 先直接 `grep` `interactions/history.jsonl`
     - 找到目标 `id`
     - 再用 `GetInteractionEntry` 做渐进式读取
7. 新增测试覆盖：
   - interaction store 追加/按 id 读取
   - `Show(prompt)` 校验
   - `GetInteractionEntry` 的 summary/full/slice
   - `submit` / `Ask` 的真实主路径写历史

## 2026-04-14 Ask 重连排序补丁

本轮又补了一条容易在前端重连时暴露的细节问题：

1. 后端 `UserQuestion` 协议新增 `created_at`
   - 来源不是前端接收时间
   - 而是后端真正创建 `Ask` 时的时间
   - 对于重启恢复的 pending question，则复用 `pending_question.json` 中持久化的 `created_at`
2. 补这个字段的原因
   - 前端把 `Ask` 视作交互堵塞点
   - 但在重连时，`pending_questions` 与 `history` 是分开发送的
   - 如果没有稳定时间锚点，`Ask` 会被简单追加到 block 末尾
   - 这样会把“更早的历史消息”错误挡在 `Ask` 后面，或者让“本该被阻塞的后续消息”跑到前面
3. 当前前端修正策略
   - 所有交互 block 维护内部 `sortMs`
   - `Ask` 使用后端的 `created_at`
   - `message` 事件使用 event timestamp
   - reconnect / history replay 时统一按 source time 插入，而不是只按接收顺序追加
4. 新增回归覆盖
   - pending question reconnect 时不会再盲目 append
   - 历史消息 replay 到已有未完成 `Ask` 前后时，会按真实时间落位
