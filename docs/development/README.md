# SA 开发文档

## 项目概览

本目录记录 StudyAdministrator（SA）后端的开发说明。

主要目录：

- `../crates/sa-core`：核心逻辑，包括配置、`Agents.md` 读取、技能扫描、内置工具、MCP、OpenAI 兼容接口调用、Agent 主循环、WS 协议结构
- `../crates/sa`：后端守护进程，负责运行 Agent、缓存事件、处理 WebSocket 连接与重连恢复

专题文档：

- `token-accounting.md`：SA 当前的 token 统计、usage 回填与 compact 估算策略
- `ws-handshake.md`：前后端双向握手与协议身份校验

说明：

- 命令行前端位于独立项目 `../../sa-cli`
- 本轮文档重点覆盖后端与协议，不继续展开前端实现细节

## 运行

1. 基于 `sa.example.toml` 创建 `sa.toml`
2. 配置 `[llm]`
   - `base_url`
   - `api_key`
   - `model`
   - `system_role_name`
   - `reasoning_effort`（可选，例如 `low`、`high`、`xhigh`）
3. 可选配置 `[mcp]`
   - `enabled = true`
   - 配置一个或多个 `[[mcp.servers]]`
   - 当前支持 `stdio`、`http`、`sse`
   - 动态工具注册名格式为 `<server>__<tool>`
4. 在工作区根目录准备 `AGENTS.md` 以及它引用的上下文文件
5. 启动后端：

```bash
cargo run -p sa --release
```

## 构建策略

当前仓库使用一套统一的 `--release` 配置，同时兼顾体积与运行性能。

示例：

```bash
cargo build --release
```

## WebSocket 握手

普通 WS 消息流之前，必须先完成强制握手：

1. 前端先发送 `client_hello`
2. 后端校验 `client_hello`
3. 后端返回 `server_hello`
4. 前端校验 `server_hello`
5. 校验通过后才进入正常协议流

如果握手失败，后端返回 `hello_reject`。

完整协议、字段解释、证明公式、JSON 示例见：

- `ws-handshake.md`

## WebSocket API（正常消息流）

说明：以下消息仅在握手成功后有效。

客户端 -> 后端：

- `{"type":"client_hello","hello":{...}}`：连接首包，必须先发送
- `{"type":"submit","task_id":"<optional uuid>","task":"..."}`
- `{"type":"get_history","from_event_id":123}`
- `{"type":"interrupt","task_id":"<uuid>"}`
- `{"type":"answer_question","answer":{...}}`

后端 -> 客户端：

- `{"type":"server_hello","hello":{...}}`
- `{"type":"hello_reject","reject":{...}}`
- `{"type":"accepted","task_id":"..."}`
- `{"type":"history","events":[...]}`
- `{"type":"event","event":{...}}`
- `{"type":"question","question":{...}}`
- `{"type":"pending_questions","questions":[...]}`
- `{"type":"question_resolved","question_id":"..."}`
- `{"type":"show","file":{...}}`
- `{"type":"recent_shows","files":[...]}`
- `{"type":"error","message":"..."}`

事件字段：

- `event_id`：单调递增事件编号
- `ts`：UTC 时间戳（RFC3339）
- `task_id`：所属顶层任务
- `kind`：`log` | `tool` | `message` | `final` | `error`
- `message`：可读文本

结构化提问字段：

- `question_id`：问题 UUID
- `task_id`：所属顶层任务 UUID
- `prompt`：展示给同学的提问文本
- `mode`：`single_choice` | `multi_choice` | `text`
- `options`：选项数组
- `allow_free_text`：是否允许额外自由输入

## 内置工具

- `Read`：读取工作区内 UTF-8 文本文件
- `Write`：仅在目标文件不存在时创建文件
- `Edit`：编辑已存在文件，且要求当前 Agent 会话中已先读过该文件
- `Bash`：通过 Git Bash 执行命令（`bash -lc`）
- `Fetch`：向指定 URL 发起网络请求
- `Search`：执行网络搜索并返回候选结果
- `MemorySearch`：检索 `MEMORY.md`、`memory.md`、`memory/*.md`
- `MemoryGet`：按路径与可选行范围读取单个 memory Markdown 文件
- `Send`：发送简洁消息给同学
- `Show`：读取一个已存在文件并通过 WS 发送给前端展示
- `Ask`：发起结构化提问并等待同学回答
- `Skill`：按技能名读取 `SKILL.md` 或技能目录下的其他允许文件，不暴露真实宿主路径
- `SubAgent`：启动子代理，传入父代理构造的上下文；子代理完成后把结果回传上级

如果 MCP 已启用，后端启动时还会追加外部动态工具，名称格式为 `<server>__<tool>`。

## MCP

当前 SA 后端已经支持 MCP 工具协议，参考实现来自 `zeroclaw` 的成熟代码路径，目标是减少协议细节错误率。

当前范围：

- 支持连接 MCP server
- 支持列出并注册 MCP tools
- 支持在 Agent 执行中调用 MCP tools

当前不包含：

- MCP resources 的完整接入
- MCP prompts 的完整接入

## 已知限制

- 当前 WebSocket 已增加本机指纹双向握手，但它仍不是面对恶意本地进程的强认证
- `SubAgent` 深度受硬限制保护，避免无限递归
- 目前没有 token 级流式输出，事件粒度仍然是步骤级
- Memory 检索仍是 Markdown 词法搜索，不是 embedding 语义检索
- `Search` 依赖轻量网页搜索解析，若上游页面结构变化，解析逻辑可能需要维护
- MCP 目前主要覆盖 tools，尚未扩展到更完整的协议面

## 更新日志

- `0.1.0`：初始最小可运行后端 Agent + WS
- `0.2.0`：补齐长期记忆、`Agents.md` 重载与上下文预加载、无限重试/退避
- `0.3.0`：将后端与 CLI 拆分为独立项目
- `0.4.0`：统一更名为 StudyAdministrator（SA），后端二进制改为 `sa`
- `0.5.0`：重构内置工具为 `Read` / `Write` / `Edit` / `Bash` / `Send` / `Ask` / `Skill` / `SubAgent`
- `0.6.0`：加入 `Fetch` / `Search` / `Show`，并增强用户可见文件展示协议
- `0.7.0`：记忆加载改为 OpenClaw 风格 Markdown 记忆文件，`Skill` 改为路径隔离读取
- `0.7.1`：加入前端先发起的 WS 双向身份握手，新增 `client_hello` / `server_hello` / `hello_reject`

## 可追溯性

这套最小后端实现虽然做了大量裁剪，但设计来源仍可追溯到已有实现：

- OpenAI 兼容接口调用：`../zeroclaw/src/providers/compatible.rs`
- Agent 工具循环：`../zeroclaw/src/agent/loop_.rs`
- MCP 参考实现：`../zeroclaw` 中 MCP 相关模块
- 记忆装载方式：参考 `../openclaw` 的注入策略并做适配
