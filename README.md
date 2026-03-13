# StudyAdministrator (SA)

`SA` 是一个本地运行的“学习委员 Agent”后端。

它从 `zeroclaw` / `openclaw` / `openai-compatible tool calling` 这类思路中抽取出最核心的部分，只保留最小但可运行的一套能力：

- 调用 OpenAI 兼容的 `POST /v1/chat/completions` 或 `POST /v1/responses`
- 使用内置中文框架提示词
- 注入工作区上下文文件（如 `AGENTS.md`、`SOUL.md`、`USER.md`）
- 加载 `SKILL.md` 形式的技能
- 运行带工具调用的自治 Agent 循环
- 通过 WebSocket 暴露后端服务
- 支持 `Send` / `Ask` / `Show` / `SubAgent`
- 按 OpenClaw 风格处理记忆文件（`MEMORY.md`、`memory/*.md`）

这个仓库只包含后端 Agent。

命令行前端位于独立项目：

- `../sa-cli`

## 项目定位

`SA` 不是通用聊天机器人，而是一个偏执行型、偏协作型的本地学习助手。它的默认角色是“学习委员”：

- 帮同学整理信息
- 读写工作区文件
- 查询资料
- 调用技能
- 通过子代理拆分任务
- 在需要时向用户发消息、提问、展示文件

## 核心能力

### 1. 模型协议调用

后端通过可配置的 OpenAI / Anthropic 兼容接口访问模型，支持配置：

- `base_url`
- `api_key`
- `model`
- `wire_api`
- `auth_style`
- `system_role_name`
- `reasoning_effort`
- `max_steps`

其中 `system_role_name` 用于兼容一些只接受 `developer` 而不是 `system` 的渠道。
`reasoning_effort` 用于给支持的 GPT 推理模型设置思维深度，常见值包括：

- `none`
- `minimal`
- `low`
- `medium`
- `high`
- `xhigh`

如果不填写，就不会向兼容接口发送这个字段。

`wire_api` 用于切换底层协议：

- `chat_completions`
  - 使用 `/v1/chat/completions`
- `responses`
  - 使用 `/v1/responses`
- `anthropic_messages`
  - 使用 `/v1/messages`

如果不填写，默认使用 `chat_completions`。

`auth_style` 用于控制鉴权头格式：

- `bearer`
  - 发送 `Authorization: Bearer ...`
- `x_api_key`
  - 发送 `x-api-key: ...`
- `anthropic_auto`
  - 对 Anthropic-compatible 渠道自动判定：
  - 普通 API key 使用 `x-api-key`
  - setup / OAuth token 使用 `Authorization: Bearer ...`

如果不填写，默认策略如下：

- `chat_completions` / `responses` => `bearer`
- `anthropic_messages` => `anthropic_auto`

另外，主 Agent 运行时的模型调用现在默认优先使用流式请求：

- 目的：降低长请求在某些网关上的 503 / 超时概率
- 行为：SA 在内部聚合流式结果，对上层仍保持原来的完整响应语义
- 回退：如果供应商明确不支持流式，SA 会自动退回一次非流式请求

当前补充说明：

- OpenAI `chat_completions` / `responses` 路径会优先尝试流式请求
- Anthropic `anthropic_messages` 路径当前先走稳定的非流式 `/v1/messages`
- 这样做的目的是直接对接 Claude-compatible 协议，而不是继续依赖某些网关把 OpenAI 请求“转换”为 Claude 请求

### 2. 工作区上下文注入

运行时会把固定内置提示词与工作区上下文拼接为最终提示词。

当前工作区里的这些文件已经纳入版本管理，并会参与上下文构造：

- `AGENTS.md`
- `SOUL.md`
- `USER.md`
- `IDENTITY.md`
- `HEARTBEAT.md`
- `BOOT.md`
- `BOOTSTRAP.md`
- `TOOLS.md`

注意：

- `MEMORY.md` 与 `memory/*.md` 仍按记忆系统单独处理，不走普通上下文预加载逻辑
- `sa.toml` 仍然是本地配置文件，不应提交真实密钥

### 3. 技能系统

`SA` 支持扫描技能目录中的 `SKILL.md`：

- 默认从 `sa.toml` 的 `skills.dirs` 读取技能目录
- 向模型注入“技能名 + 描述”
- 模型需要具体技能内容时，通过 `Skill` 工具按名称读取
- 不向模型暴露宿主机上的真实技能安装路径

### 4. MCP 工具系统

`SA` 支持连接外部 MCP（Model Context Protocol）工具服务器。

当前支持的传输类型：

- `stdio`
- `http`
- `sse`

加载方式：

- 在 `sa.toml` 中配置 `[mcp]`
- 启动时连接所有配置的 MCP server
- 成功连接后，把它们暴露的工具自动注册进 Agent 工具表
- 工具名会带服务器前缀，格式为：`<server>__<tool>`

例如：

- `filesystem__read_file`
- `browser__navigate`

连接失败不会阻止 `SA` 启动；失败的 MCP server 会被记录日志并跳过。

### 5. 记忆系统

记忆系统采用 OpenClaw 风格：

- 根级长期记忆：`MEMORY.md` / `memory.md`
- 日常记忆：`memory/*.md`

使用方式：

- 根级长期记忆可在运行时注入 prompt
- 每日记忆不自动全文注入
- Agent 需要回忆时，优先调用：
  - `MemorySearch`
  - `MemoryGet`

### 6. 自治工具调用循环

后端会重复执行以下循环直到任务完成：

1. 发送当前消息历史到模型
2. 接收模型返回的普通文本或工具调用
3. 执行工具
4. 把工具结果写回消息历史
5. 继续下一轮

同时支持：

- 用户中断当前任务
- 断线后继续运行
- 模型调用失败后的无限重试退避

## 内置工具

当前内置工具如下：

- `Read`
  - 读取工作区内的 UTF-8 文本文件
- `Write`
  - 创建新文件；若目标已存在则拒绝覆盖
- `Edit`
  - 编辑已存在文件；必须先 `Read` 再 `Edit`
- `Bash`
  - 通过 Git Bash 执行命令（`bash -lc`）
- `Fetch`
  - 对已知 URL 发起直接 HTTP 请求
- `Search`
  - 网络搜索，返回候选标题、摘要和 URL
- `MemorySearch`
  - 搜索 `MEMORY.md`、`memory.md`、`memory/*.md`
- `MemoryGet`
  - 读取某个记忆文件的指定片段
- `Send`
  - 向用户发送简短消息
- `Show`
  - 展示一个文件给用户
- `Ask`
  - 向用户发起结构化提问并等待回答
- `Skill`
  - 按技能名读取 `SKILL.md` 或技能目录下的相对文件
- `SubAgent`
  - 启动一个子代理并返回其最终结果

此外，如果配置了 MCP server，工具表中还会出现动态注册的 MCP 工具。

## 目录结构

```text
sa/
├─ crates/
│  ├─ sa-core/   # 核心逻辑：配置、提示词、技能、记忆、工具、OpenAI 客户端
│  └─ sa/        # 后端守护进程：任务队列、WS、事件广播、问题交互、子代理调度
├─ docs/
│  └─ development/
├─ AGENTS.md
├─ SOUL.md
├─ USER.md
├─ IDENTITY.md
├─ HEARTBEAT.md
├─ BOOT.md
├─ BOOTSTRAP.md
├─ TOOLS.md
├─ prompt.md
├─ sa.example.toml
└─ README.md
```

## 快速开始

### 1. 准备配置

复制示例配置：

```bash
copy sa.example.toml sa.toml
```

然后填写：

- `llm.base_url`
- `llm.api_key`
- `llm.model`

如果你的供应商只支持新版 Responses 协议，则继续配置：

```toml
[llm]
wire_api = "responses"
```

如果你的供应商是 Claude / Anthropic-compatible `/v1/messages`，则设置：

```toml
[llm]
wire_api = "anthropic_messages"
```

如果该渠道要求显式 `x-api-key` 或 `Authorization`，也可以继续设置：

```toml
[llm]
auth_style = "anthropic_auto"
```

如果你的渠道不接受 `system` 角色，只接受 `developer`，则设置：

```toml
[llm]
system_role_name = "developer"
```

如果要调节模型的思维深度，可以继续配置：

```toml
[llm]
reasoning_effort = "high"
```

如果要启用 MCP，可以继续配置：

```toml
[mcp]
enabled = true

[[mcp.servers]]
name = "filesystem"
transport = "stdio"
command = "npx"
args = ["-y", "@modelcontextprotocol/server-filesystem", "."]
tool_timeout_secs = 180
```

### 2. 启动后端

```bash
cargo run -p sa --release
```

默认监听：

- `127.0.0.1:8765`
- WS 路径：`/ws`

### 3. 启动测试前端

`sa-cli` 是单独项目，只是当前阶段的测试前端：

```bash
cd ../sa-cli
cargo run --release
```

## 构建

仓库只保留一套调优后的 `release` 配置，兼顾体积和运行性能：

```bash
cargo build --release
```

当前可执行文件：

- 后端：`sa`

## WebSocket 说明

后端通过 WebSocket 与前端通信。

设计原则：

- 前端断开连接不影响 Agent 继续运行
- 事件会缓存在内存中，便于重连后回放
- `Ask` / `Show` / 普通事件都走统一 WS 通道

当前 CLI 的展示策略是：

- 后端仍推送全部消息
- CLI 主要展示 `Send`、`Ask`、`Show`

## 文档

开发文档位于：

- `docs/development/README.md`

如果你要看一份当前提示词快照，可参考：

- `prompt.md`

## 已知限制

- 这是一个最小后端实现，WebSocket 暂未做鉴权
- `Search` 当前使用 DuckDuckGo 轻量 HTML 页面解析，若上游结构变化需要维护
- `MemorySearch` 当前是 Markdown 词法检索，不是向量语义检索
- `SubAgent` 有递归深度上限，避免无限递归
- CLI 只是过渡性的测试前端，不是最终产品形态
- 当前只接入了 MCP 的 tool 能力，还没有接入 MCP resources / prompts

## 版本节点

- `v0.7.0`
  - 切换到 OpenClaw 风格记忆加载
  - 新增 `MemorySearch` / `MemoryGet`
  - `Skill` 改为隔离路径的技能读取
- `v0.7.1`
  - 将动态提示词上下文文件纳入版本管理
