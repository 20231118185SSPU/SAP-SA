# `/v1/messages` 兼容实现说明

## 目标

本文记录 SA 当前对 Anthropic / Claude-compatible `/v1/messages` 协议的兼容实现。

这次支持的原因很直接：

- 有些供应商底层是 Claude 协议
- 它们并不真正支持 `/v1/chat/completions`
- 继续把 OpenAI 请求交给网关“转换”为 Claude 请求，容易出现 `convert_request_failed`

因此 SA 现在直接增加了原生 `anthropic_messages` wire protocol。

## 配置入口

`sa.toml` 中相关配置：

```toml
[llm]
wire_api = "anthropic_messages"
auth_style = "anthropic_auto"
```

支持的 `wire_api` 别名：

- `anthropic_messages`
- `anthropic-messages`
- `anthropic`
- `claude`

支持的 `auth_style`：

- `bearer`
- `x_api_key`
- `anthropic_auto`

默认策略：

- OpenAI 风格协议默认 `bearer`
- `anthropic_messages` 默认 `anthropic_auto`

`anthropic_auto` 的判定规则参考了 `zeroclaw`：

- setup/OAuth token 或明显 JWT-like token => `Authorization: Bearer ...`
- 普通 API key => `x-api-key: ...`

## 请求体映射

SA 内部仍然维护统一的 chat-style 消息模型。

发送到 Anthropic 时，当前映射如下：

- `model` -> `model`
- `max_tokens` -> `max_tokens`
- `system` / `developer` 消息 -> 顶层 `system`
- `user` 消息 -> `messages[].role = "user"` 的 `text` block
- `assistant` 文本 -> `messages[].role = "assistant"` 的 `text` block
- assistant tool call -> `tool_use`
- tool result -> `user` 侧 `tool_result`
- function tool schema -> `tools[].input_schema`

其中系统提示的处理规则是：

- `system`
- `developer`

这两类消息都会被收集起来，并用空行拼接后写入顶层 `system` 字段。

## 工具调用映射

SA 的统一工具调用格式是 OpenAI 风格：

- assistant 侧：`tool_calls[]`
- tool 结果：`role = "tool"` + `tool_call_id`

Anthropic 协议需要把它改写成 block 结构：

- assistant 工具调用：
  - `{"type":"tool_use","id":"...","name":"...","input":{...}}`
- 工具结果：
  - `{"type":"tool_result","tool_use_id":"...","content":"..."}`

当前 SA 的策略：

- assistant 消息如果同时包含文本和工具调用，会先发 `text` block，再发 `tool_use` block
- `tool_call.function.arguments` 会从 JSON 字符串解析成对象
- 解析失败时回退为空对象，保证请求仍可发送

## 响应归一化

Anthropic `/v1/messages` 返回后，SA 会重新归一成自己的统一响应：

- `text` block -> assistant 文本
- `tool_use` block -> `tool_calls[]`
- `usage` -> `ChatUsage`
- `stop_reason = "tool_use"` -> `finish_reason = "tool_calls"`
- `stop_reason = "max_tokens"` -> `finish_reason = "length"`
- `stop_reason = "end_turn"` / `stop_sequence` -> `finish_reason = "stop"`

这样主 Agent 循环、工具执行、compact 逻辑都不需要区分上游到底是 OpenAI 还是 Claude。

## 当前刻意保留的限制

当前实现刻意保持最小可用，不一次性把 Anthropic 全协议搬进来：

- 先走非流式 `/v1/messages`
- 暂未实现 Anthropic SSE streaming 聚合
- 暂未暴露 Anthropic thinking/budget 参数
- 暂未实现图片 block 的正式映射

原因是这轮的核心目标是先修复“Claude-compatible 供应商根本不可用”的问题。

## 参考来源

本次实现主要参考：

- `../../zeroclaw/src/providers/anthropic.rs`
- `../../zeroclaw/src/auth/anthropic_token.rs`

SA 当前实现位于：

- `../../sa/crates/sa-core/src/openai.rs`

## 验证

本轮至少覆盖这些测试：

- `anthropic_auto_auth_uses_x_api_key_for_regular_keys`
- `anthropic_auto_auth_uses_bearer_for_setup_tokens`
- `anthropic_request_extracts_system_tools_and_tool_results`
- `normalize_anthropic_response_extracts_text_and_tool_calls`

并且最终需要通过：

- `cargo test -p sa-core`
- `cargo test --all`
