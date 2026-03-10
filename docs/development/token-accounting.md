# SA Token 统计与 Compact 估算

## 目的

本文档说明 StudyAdministrator（SA）当前如何统计 token、如何决定何时触发 compact，以及这套实现和 `OpenClaw` / `ZeroClaw` 的关系。

目标有三点：

- 可追溯：明确代码位置与设计来源
- 可验证：说明每一步到底依据什么数据
- 可解释：解释为什么不是单纯 `chars / 4`

## 结论

当前 SA 已改为 **OpenClaw 风格的混合估算模型**：

1. 优先使用最近一次成功 assistant 响应返回的真实 `usage`
2. 只对该 assistant 之后新增的尾部消息做启发式估算
3. 如果完全没有可用 `usage`，再退回到整包请求的启发式估算
4. compact 改写历史后，会主动清理保留下来的旧 `usage` 快照，避免复用过期窗口大小

这意味着 SA 不再是“全程纯 `chars / 4`”，而是：

`真实 usage 锚点 + 尾部 heuristic`

## 代码位置

- OpenAI 兼容响应与 `usage` 结构：
  - `crates/sa-core/src/openai.rs`
- Agent 主循环中把响应 `usage` 回填到 assistant 消息：
  - `crates/sa-core/src/agent.rs`
- compact / token 估算逻辑：
  - `crates/sa-core/src/compact.rs`

## SA 当前的 token 估算策略

### 1. 真实 usage 从哪里来

SA 调用的是 OpenAI 兼容的 `POST /v1/chat/completions`。

如果服务端返回：

- `usage.prompt_tokens`
- `usage.completion_tokens`
- `usage.total_tokens`

这些字段会先反序列化到 `ChatCompletionsResponse.usage`，随后在 Agent 主循环里回填到本轮 assistant 消息上。

这样做的目的不是把 `usage` 再发回给模型，而是为了给下一轮 compact 估算提供一个“真实锚点”。

### 2. compact 估算时怎么用 usage

`compact.rs` 会在下一轮请求前倒序查找最近一条带 `usage` 的 assistant 消息。

如果找到：

- 优先使用 `usage.total_tokens`
- 如果没有 `total_tokens`，则优先回退到：
  - `input_tokens/prompt_tokens + output_tokens/completion_tokens`
- 只有在这些主字段也缺失时，才退到 cache 细项
- 然后只对这条 assistant 之后新增的消息做 `chars / 4` 估算

最终得到：

`估算请求大小 = 最近 usage 对应的旧请求大小 + usage 之后的尾部消息估算`

### 3. 如果没有 usage

如果当前会话里还没有任何成功 assistant 响应，或者 compact 刚刚重写过历史导致旧 `usage` 被清空，那么 SA 会退回到纯启发式：

- `system/developer` 消息
- synthetic compaction summary
- 当前保留的真实对话消息
- tool definitions
- 少量请求 wrapper 开销

统一按近似字符量估算。

### 4. 为什么 compact 后要清理 usage

这是关键点。

assistant 上的 `usage` 描述的是“旧历史结构下的整包请求大小”。  
一旦 compact 执行：

- 旧前缀消息被删除
- 替换成 synthetic summary
- 保留的是较短的新历史

此时旧 assistant 的 `usage` 已经失真。如果继续使用，极容易出现：

- compact 刚做完
- 下一轮估算仍然以旧大窗口为准
- 结果立刻再次触发 compact

因此 SA 在 compact 完成后，会对保留下来的 assistant 消息清空 `usage`，等下一次真实模型调用成功后，再重新建立新锚点。

## 与 OpenClaw 的关系

SA 当前做法直接参考了 `OpenClaw` / 上游 `pi-coding-agent` 的核心思路：

1. 最近真实 usage 优先
2. 新增尾部消息再估算
3. 估算失败时退回纯 heuristic

可追溯参考：

- `../openclaw/src/agents/compaction.ts`
- `../.tmp/pi-coding-agent-0.55.3/package/dist/core/compaction/compaction.js`

其中上游关键逻辑是：

- 找最后一条有 usage 的 assistant
- 用 `totalTokens`，没有就自己把 usage 各分量相加
- 仅对 trailing messages 调用 `estimateTokens()`

SA 不是逐字复制，而是在更小的 Rust 数据模型里做了等价适配。
其中有一个刻意差异：

- 对 OpenAI 风格的 `prompt_tokens_details.cached_tokens`，SA 不会在已有 `prompt_tokens` 时再次叠加，避免双算

## 与 ZeroClaw 的关系

`ZeroClaw` 更偏向两层模型：

- provider 层：直接相信上游返回的 usage
- gateway / compat / internal utility：很多地方仍然直接 `chars / 4`

SA 这次调整后，位置更接近 `OpenClaw`，不再是简单的 `ZeroClaw compat` 风格估算。

## OpenAI 兼容字段兼容策略

为减少不同兼容网关字段差异带来的问题，SA 的 `ChatUsage` 目前接受这些常见字段：

- `prompt_tokens`
- `completion_tokens`
- `total_tokens`
- `input_tokens`
- `output_tokens`
- `total`
- `prompt_tokens_details.cached_tokens`
- `cache_read_input_tokens`
- `cache_creation_input_tokens`

这让 SA 可以兼容多种“OpenAI 风格但不完全一致”的返回结构。

## 当前限制

当前实现仍有边界，不能误解成“精确 tokenizer 统计”：

- 没有引入真实 tokenizer，因此尾部新增消息仍然是 `chars / 4`
- 如果上游服务端不返回 `usage`，SA 只能退回启发式估算
- `usage.total_tokens` 在不同兼容网关上语义可能不完全一致，SA 当前采用的是与 OpenClaw 同方向的保守用法
- tool definitions 仅在“没有 usage 锚点”时才额外按启发式计入；有锚点时默认认为旧 usage 已覆盖当时的工具 schema
- cache 细项不会在已有 `prompt_tokens/input_tokens` 的情况下再次叠加，避免对 OpenAI 风格响应双算

## 验证方式

可以通过以下方式验证当前实现：

1. 运行 `cargo test`
2. 查看 `compact.rs` 中与 token 估算相关的单元测试
3. 在长会话中观察 compact 日志
4. 对比 compact 前后是否仍然立即重复触发

重点测试点：

- 有 usage 时不再重复把 tool definitions 额外算一遍
- compact 后保留消息的 usage 已被清空
- 没有 usage 时仍能安全退回 heuristic

## 后续可继续优化的方向

如果后续需要进一步向 OpenClaw 靠拢，可继续做：

1. 为不同消息类型增加更细的字符权重
2. 对 tool result 采用更保守的专门估算
3. 引入可选 tokenizer，用于高精度统计
4. 将 compact 前后估算详情暴露到调试日志或状态接口
