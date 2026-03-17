# SA 会话持久化

## 目标

`SA` 的顶层主会话不再只存在于进程内存里，而是会持久化到工作区：

- `sessions/*.jsonl`

设计目标：

- 后端重启后还能继续同一条主会话
- 构建模型请求时可以从 session 文件恢复历史，而不是从空白状态重新开始
- compact 触发后保留旧原始记录，并给新 prompt 一个精确可读的“上一段原始会话文件”路径

## 范围

当前只有顶层主会话会持久化。

子代理不持久化，原因：

- 子代理本身是短生命周期工具
- 持久化子代理会显著扩大状态面和恢复复杂度
- 当前用户需求聚焦在“主 Agent 重启后不要完全重来”

## 存储位置

固定位置：

- `workspace/sessions/`

内部文件：

- `workspace/sessions/.current`
  - 记录当前活跃 session 文件的工作区相对路径
- `workspace/sessions/session-<timestamp>-<uuid>.jsonl`
  - 当前或历史会话段文件

## JSONL 结构

每个 session 段文件都是独立 JSONL。

当前支持三种条目：

1. `session_meta`
2. `compaction_checkpoint`
3. `message`

### `session_meta`

每个文件第一行必须是：

- `type = "session_meta"`
- `schema_version`
- `conversation_id`
- `segment_id`
- `created_at`
- `current_session_path`
- `previous_session_path`

说明：

- `conversation_id` 在整个主会话链上保持稳定
- `segment_id` 每个文件唯一
- `previous_session_path` 指向上一个被 compact 掉的原始 session 段

### `compaction_checkpoint`

只有 compact 之后新切出的 session 文件才会带这个条目。

字段：

- `type = "compaction_checkpoint"`
- `created_at`
- `summary`

作用：

- 恢复 `CompactionState.summary`
- 让当前请求继续沿用 compact 后的摘要上下文

### `message`

每条真实会话消息都会追加一条：

- `type = "message"`
- `created_at`
- `message`

`message` 中保存的是 SA 本地完整消息结构，而不是对外请求时裁剪过的版本，因此会保留：

- `role`
- `content`
- `tool_calls`
- `tool_call_id`
- `request_usage`
- `responses_input_items`

这样做的原因：

- resume 后需要尽可能还原真实本地状态
- token 估算依赖历史 assistant turn 上锚定的 `request_usage`
- `responses` 协议后续轮次可能需要复用 `responses_input_items`

## 运行流程

### 普通运行

顶层任务启动时：

1. 读取 `sessions/.current`
2. 加载当前 session 文件
3. 恢复：
   - `messages`
   - `compaction summary`
4. 把本轮新的 user 消息追加到当前 session 文件
5. 后续 assistant/tool/user follow-up 消息也实时追加

### compact 触发后

compact 完成后不会改写旧文件，而是：

1. 创建新的 `session-*.jsonl`
2. 写入新的 `session_meta`
3. 写入新的 `compaction_checkpoint`
4. 写入 compact 后仍保留的 `kept_messages`
5. 更新 `sessions/.current`

因此：

- 旧文件保留完整原始记录
- 新文件只保留 compact 后继续运行所需的摘要 + retained suffix

## Prompt 注入

恢复顶层会话时，运行时会构造一段“压缩上下文”并注入系统提示。

内容包括：

- 当前压缩后会话文件路径
- 上一段原始会话文件路径（如果存在）
- 当前压缩摘要（如果存在）

这样 Agent 在需要找某条被 compact 掉的原始记录时，可以：

1. 直接用 `Read`
2. 精确读取上一段原始会话文件

而不是让同学重复已经说过的话。

## 崩溃恢复

需要处理的一个特殊问题是：

- assistant 已经发出了 `tool_calls`
- 进程在所有 `tool` 结果落盘前崩溃

如果直接恢复这段尾巴，下一次请求会带着一个“未完成的工具调用轮次”，无法安全重放。

当前策略：

- 加载 session 时检查尾部是否存在未闭合的工具调用轮次
- 如果存在，就截断这一整段不完整 assistant-turn 后缀
- 并把修复后的结果重写回当前 session 文件

这样能保证 resume 后的历史始终是可重放的。

## 当前限制

- 当前没有额外暴露 session 路径配置；目录固定为 `workspace/sessions/`
- 当前只有顶层会话持久化，子代理仍是临时会话
- 当前没有提供“新建会话 / 切换会话 / 分支会话”的独立控制面
