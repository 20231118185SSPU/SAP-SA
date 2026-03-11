# `/v1/responses` 兼容实现说明

## 目标

本文记录 SA 当前 `/v1/responses` 兼容层的设计，重点说明它如何向 `codex` 的 Rust 官方实现对齐。

这次对齐的目标不是把 `codex` 整个协议层照搬进 SA，而是确保以下关键点与官方 shape 一致，减少“看起来兼容，实际某些供应商会拒绝或行为偏差”的问题：

- 请求体字段形状尽量贴近 `codex-rs`
- `function_call_output.output` 的 wire 编码与官方一致
- SSE 聚合逻辑覆盖 `output_item.added/done` 与 reasoning delta
- assistant 历史回放时保留 reasoning/function-call 等结构化 item，而不是退化成纯文本

## 参考源码

本次实现主要对照以下上游文件：

- `../../codex/codex-rs/codex-api/src/common.rs`
- `../../codex/codex-rs/protocol/src/models.rs`
- `../../codex/codex-rs/codex-api/src/sse/responses.rs`
- `../../codex/codex-rs/core/src/tools/context.rs`

SA 当前实际实现位于：

- `../../sa/crates/sa-core/src/openai.rs`

## 当前请求体形状

当 `llm.wire_api = "responses"` 时，SA 会把内部统一的 chat-style 请求转换成 Responses 风格请求。

当前关键字段如下：

- `model`
- `instructions`
- `input`
- `max_output_tokens`
- `reasoning`
- `tools`
- `tool_choice`
- `parallel_tool_calls`
- `store`
- `stream`
- `include`
- `text`

其中几个重要约定：

- `instructions`：始终是字符串；即使没有 system/developer 文本，也会发送空字符串
- `tools`：始终是数组；内部 `ToolDefinition` 会先转换成原始 JSON value
- `tool_choice`：当前标准化为字符串；有工具时默认 `auto`，无工具时默认 `none`
- `parallel_tool_calls`：当前依据“是否存在工具”自动设置
- `store`：当前固定 `false`
- `include`：当设置了 `reasoning_effort` 时，请求 `reasoning.encrypted_content`
- `text`：当前保留字段，但 SA 还没有把 verbosity/schema 控制暴露到配置层

## `input` 的结构化回放

SA 内部统一保存 `ChatMessage`，但在 Responses 模式下，assistant 历史还会额外保存 `responses_input_items`。

这样做的原因是：

- reasoning item 不能安全地从自然语言重新拼出来
- assistant 的 `function_call` 历史需要原样回放
- 后续 `function_call_output` 需要通过 `call_id` 精确对应前一个 tool call

当前回放的 item 子集包括：

- `message`
- `function_call`
- `function_call_output`
- `reasoning`

其中：

- `message.content` 使用 `input_text` / `output_text` / `input_image`
- `reasoning.summary` 使用 `summary_text[]`
- `reasoning.content` 使用 `reasoning_text[]` / `text[]`
- `function_call_output.output` 使用官方 wire 规则

## `function_call_output.output` 的官方编码

这是这次修复的重点之一。

`codex` 的官方实现里，`function_call_output.output` 不是固定对象，也不是固定字符串，而是：

- 纯文本工具输出时：直接序列化为字符串
- 多模态/结构化工具输出时：直接序列化为数组

SA 现在保持同样规则：

- `Text("ok")` -> `"ok"`
- `ContentItems([...])` -> `[{"type":"input_text",...}, ...]`

这样可以兼容对 wire shape 比较严格的 Responses 供应商。

## SSE 聚合策略

对流式 `/v1/responses`，SA 当前会处理这些关键事件：

- `response.output_text.delta`
- `response.output_text.done`
- `response.output_item.added`
- `response.output_item.done`
- `response.reasoning_text.delta`
- `response.reasoning_summary_part.added`
- `response.reasoning_summary_text.delta`
- `response.completed`
- `response.done`
- `response.failed`
- `error`

聚合规则：

- `output_item.added/done`：按 `id` 或 `call_id` 合并，避免重复保留同一 item
- `output_text.delta`：累积成最终可见文本
- reasoning delta：按 `content_index` / `summary_index` 分桶累积
- 如果服务端没有给出完整的 `response.completed.response`，SA 会用当前累积的数据合成一个 best-effort `ResponsesResponse`

这意味着即使某些供应商的 Responses SSE 比较“半官方”，只发 delta、不发完整 completion body，SA 也仍然能尽量恢复：

- assistant 文本
- function call
- reasoning 历史

## 当前刻意保留的差异

SA 还没有把 `codex` 的所有 Responses 能力完整搬进来，当前保留了这些简化：

- `text.verbosity` 还没有从 `sa.toml` 暴露
- `text.format` / JSON schema 输出控制还没有接入
- `store` 当前固定 `false`
- 目前只实现 SA 自己需要的 item/tool 子集，没有把 `web_search_call`、`image_generation_call` 等所有 Responses item 全量建模

这些差异是有意的，目的是先把“稳定兼容 OpenAI 风格 `/responses` 供应商”这条主路径做扎实。

## 验证方式

本次改动至少应该覆盖以下验证点：

- `cargo test -p sa-core openai`
- 检查 `responses_request_converts_tool_history_and_reasoning_effort`
- 检查 `responses_function_call_output_payload_serializes_like_codex_wire_format`
- 检查 `responses_stream_accumulator_reconstructs_reasoning_from_deltas`

如果后续继续扩展：

- `text.verbosity`
- `output_schema`
- 供应商特定 Responses 变体

应优先继续对照 `codex-rs` 的 request model、protocol model 和 SSE parser，避免再回到“猜一个兼容层”的做法。
