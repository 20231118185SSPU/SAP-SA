# SA 上下文压缩（Compaction）提示词设计

本文档记录 SA 的上下文压缩机制所使用的提示词模板，设计参考来自 OpenClaw 上游实现。

上游来源：
- 包名：`@mariozechner/pi-coding-agent@0.55.3`
- npm tarball：`https://registry.npmjs.org/@mariozechner/pi-coding-agent/-/pi-coding-agent-0.55.3.tgz`
- 仓库：`https://github.com/badlogic/pi-mono`

上游本地解压路径（仅供追溯）：
- `G:\AgentProjects\Claw\.tmp\pi-coding-agent-0.55.3\package\dist\core\compaction\compaction.js`
- `G:\AgentProjects\Claw\.tmp\pi-coding-agent-0.55.3\package\dist\core\compaction\utils.js`

## `SUMMARIZATION_SYSTEM_PROMPT`（摘要系统提示）

```text
你是一个上下文摘要助手。你的任务是阅读用户与 AI 助手之间的对话，然后按照指定的格式输出结构化摘要。

不要继续对话。不要回答对话中的任何问题。只输出结构化摘要。
```

## `SUMMARIZATION_PROMPT`（首次摘要提示）

```text
上方的消息是一段需要摘要的对话。请创建一份结构化的上下文检查点摘要，供另一个 LLM 用来接续工作。

严格使用以下格式：

## 目标
[用户想要完成什么？如果会话涉及多个任务，可以列出多项。]

## 约束与偏好
- [用户提到的任何约束、偏好或要求]
- [如果没有提到，写"（无）"]

## 进度
### 已完成
- [x] [已完成的任务/变更]

### 进行中
- [ ] [当前正在做的工作]

### 阻塞
- [阻碍进展的问题（如有）]

## 关键决策
- **[决策内容]**：[简要理由]

## 下一步
1. [按优先级排列的待办事项]

## 关键上下文
- [继续工作所需的数据、示例或参考资料]
- [如果没有，写"（无）"]

保持每个章节简洁。必须保留完整的文件路径、函数名和错误信息原文。
```

## `UPDATE_SUMMARIZATION_PROMPT`（增量更新摘要提示）

```text
上方的消息是需要合入现有摘要的新对话内容。现有摘要在 <previous-summary> 标签中提供。

根据新信息更新现有的结构化摘要。规则：
- 保留现有摘要中的所有信息
- 从新消息中补充新的进度、决策和上下文
- 更新"进度"章节：将已完成的条目从"进行中"移至"已完成"
- 根据实际进展更新"下一步"
- 必须保留完整的文件路径、函数名和错误信息原文
- 如果某些内容已经过时或不再相关，可以移除

严格使用以下格式：

## 目标
[保留已有目标，如果任务范围扩大则补充新目标]

## 约束与偏好
- [保留已有项，补充新发现的约束或偏好]

## 进度
### 已完成
- [x] [包含之前已完成的和新完成的条目]

### 进行中
- [ ] [当前工作 - 根据进展更新]

### 阻塞
- [当前阻塞项 - 已解决的移除]

## 关键决策
- **[决策内容]**：[简要理由]（保留所有已有决策，补充新决策）

## 下一步
1. [根据当前状态更新]

## 关键上下文
- [保留重要上下文，按需补充新内容]

保持每个章节简洁。必须保留完整的文件路径、函数名和错误信息原文。
```

## `TURN_PREFIX_SUMMARIZATION_PROMPT`（轮次前缀摘要提示）

当一个轮次过长需要截断时，保留后半段（近期工作），对前半段生成摘要。

```text
这是一个因过长而被截断的轮次的前半段。后半段（近期工作）已被保留。

请对前半段生成摘要，为保留的后半段提供上下文：

## 原始请求
[用户在这个轮次中要求做什么？]

## 前期进展
- [前半段中的关键决策和已完成的工作]

## 后半段所需上下文
- [理解保留的后半段内容所需的信息]

保持简洁。只聚焦于理解后半段所必需的信息。
```

## `generateSummary()` 上游参考实现

```js
// 上游参考实现，SA 的 Rust 实现以此为蓝本
export async function generateSummary(currentMessages, model, reserveTokens, apiKey, signal, customInstructions, previousSummary) {
    const maxTokens = Math.floor(0.8 * reserveTokens);
    // 如果有上一次摘要，使用增量更新提示；否则使用首次摘要提示
    let basePrompt = previousSummary ? UPDATE_SUMMARIZATION_PROMPT : SUMMARIZATION_PROMPT;
    if (customInstructions) {
        basePrompt = `${basePrompt}\n\nAdditional focus: ${customInstructions}`;
    }
    // 将对话序列化为纯文本，防止模型试图继续对话
    // 先转换为 LLM 消息格式（处理 bashExecution、custom 等自定义类型）
    const llmMessages = convertToLlm(currentMessages);
    const conversationText = serializeConversation(llmMessages);
    // 用标签包裹对话文本，组装最终提示
    let promptText = `<conversation>\n${conversationText}\n</conversation>\n\n`;
    if (previousSummary) {
        promptText += `<previous-summary>\n${previousSummary}\n</previous-summary>\n\n`;
    }
    promptText += basePrompt;
    const summarizationMessages = [
        {
            role: "user",
            content: [{ type: "text", text: promptText }],
            timestamp: Date.now(),
        },
    ];
    const response = await completeSimple(model, { systemPrompt: SUMMARIZATION_SYSTEM_PROMPT, messages: summarizationMessages }, { maxTokens, signal, apiKey, reasoning: "high" });
    if (response.stopReason === "error") {
        throw new Error(`Summarization failed: ${response.errorMessage || "Unknown error"}`);
    }
    const textContent = response.content
        .filter((c) => c.type === "text")
        .map((c) => c.text)
        .join("\n");
    return textContent;
}
```

## 对话序列化参考实现

```js
// 将消息数组序列化为可读的纯文本格式
export function serializeConversation(messages) {
    const parts = [];
    for (const msg of messages) {
        if (msg.role === "user") {
            const content = typeof msg.content === "string"
                ? msg.content
                : msg.content
                    .filter((c) => c.type === "text")
                    .map((c) => c.text)
                    .join("");
            if (content)
                parts.push(`[User]: ${content}`);
        }
        else if (msg.role === "assistant") {
            const textParts = [];
            const thinkingParts = [];
            const toolCalls = [];
            for (const block of msg.content) {
                if (block.type === "text") {
                    textParts.push(block.text);
                }
                else if (block.type === "thinking") {
                    thinkingParts.push(block.thinking);
                }
                else if (block.type === "toolCall") {
                    const args = block.arguments;
                    const argsStr = Object.entries(args)
                        .map(([k, v]) => `${k}=${JSON.stringify(v)}`)
                        .join(", ");
                    toolCalls.push(`${block.name}(${argsStr})`);
                }
            }
            if (thinkingParts.length > 0) {
                parts.push(`[Assistant thinking]: ${thinkingParts.join("\n")}`);
            }
            if (textParts.length > 0) {
                parts.push(`[Assistant]: ${textParts.join("\n")}`);
            }
            if (toolCalls.length > 0) {
                parts.push(`[Assistant tool calls]: ${toolCalls.join("; ")}`);
            }
        }
        else if (msg.role === "toolResult") {
            const content = msg.content
                .filter((c) => c.type === "text")
                .map((c) => c.text)
                .join("");
            if (content) {
                parts.push(`[Tool result]: ${content}`);
            }
        }
    }
    return parts.join("\n\n");
}
```

## 备注

- 四段提示词模板已翻译为中文，供 SA 的 Rust 实现直接使用。
- JS 代码块保留原文，仅作为参考实现蓝本，SA 的实际实现在 Rust 代码中。
- SA 可在 `basePrompt` 之后追加 `Additional focus:` 自定义补充指令。
