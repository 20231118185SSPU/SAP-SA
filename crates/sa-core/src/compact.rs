//! Context compaction for long-running SA conversations.
//!
//! The current SA core uses a very small `ChatMessage` model:
//! - one stable system/developer prompt
//! - user / assistant / tool messages
//! - OpenAI-compatible function calling
//!
//! That means we cannot copy OpenClaw's session-file compaction implementation
//! verbatim. Instead, this module adapts the same core ideas to SA's simpler
//! in-memory conversation state:
//! - reuse the latest assistant usage snapshot when available
//! - estimate only the trailing suffix heuristically with a chars/4 rule
//! - keep the newest suffix intact
//! - summarize the older prefix with the translated prompts from `./compact.md`
//! - re-inject the generated checkpoint summary as a synthetic user message
//!   (this mirrors the upstream OpenClaw behavior)
//! - support incremental summary updates when compaction happens more than once
//!
//! The goal is not perfect token accounting. The goal is safe, predictable,
//! provider-agnostic history reduction that preserves enough context for the
//! next model call to continue work.

use crate::cancel::CancelToken;
use crate::openai::{
    ChatCompletionsError, ChatCompletionsRequest, ChatMessage, ChatUsage, OpenAiClient, ToolCall,
    ToolDefinition,
};
use crate::retry::retry_delay;
use anyhow::Context as _;
use serde::Deserialize;
use std::fmt::Write as _;

/// Synthetic message prefix used to inject the compaction checkpoint back into
/// the active conversation.
///
/// OpenClaw injects compaction summaries as user-role messages. We do the same
/// here because it keeps the summary visible to the model without changing the
/// original system/developer instruction block.
pub const COMPACTION_SUMMARY_PREFIX: &str = "此前的对话历史已被压缩为以下摘要：\n\n<summary>\n";

/// Synthetic message suffix paired with [`COMPACTION_SUMMARY_PREFIX`].
pub const COMPACTION_SUMMARY_SUFFIX: &str = "\n</summary>";

/// System prompt used by the summarization model call.
pub const SUMMARIZATION_SYSTEM_PROMPT: &str = r#"你是一个上下文摘要助手。你的任务是阅读用户与 AI 助手之间的对话，然后按照指定的格式输出结构化摘要。

不要继续对话。不要回答对话中的任何问题。只输出结构化摘要。"#;

/// Prompt used when generating the first checkpoint summary.
pub const SUMMARIZATION_PROMPT: &str = r#"上方的消息是一段需要摘要的对话。请创建一份结构化的上下文检查点摘要，供另一个 LLM 用来接续工作。

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

保持每个章节简洁。必须保留完整的文件路径、函数名和错误信息原文。"#;

/// Prompt used when a previous checkpoint summary already exists.
pub const UPDATE_SUMMARIZATION_PROMPT: &str = r#"上方的消息是需要合入现有摘要的新对话内容。现有摘要在 <previous-summary> 标签中提供。

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

保持每个章节简洁。必须保留完整的文件路径、函数名和错误信息原文。"#;

/// Prompt used when compaction has to split the middle of one user turn.
pub const TURN_PREFIX_SUMMARIZATION_PROMPT: &str = r#"这是一个因过长而被截断的轮次的前半段。后半段（近期工作）已被保留。

请对前半段生成摘要，为保留的后半段提供上下文：

## 原始请求
[用户在这个轮次中要求做什么？]

## 前期进展
- [前半段中的关键决策和已完成的工作]

## 后半段所需上下文
- [理解保留的后半段内容所需的信息]

保持简洁。只聚焦于理解后半段所必需的信息。"#;

/// Small constant used to represent top-level JSON request wrapper overhead
/// that is not captured by per-message or per-tool estimates.
const REQUEST_WRAPPER_TOKENS: usize = 32;

/// Default compaction settings for SA.
///
/// These numbers are intentionally conservative because SA only receives
/// authoritative usage snapshots on successful assistant turns. Everything
/// after the latest assistant usage still relies on heuristics.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct CompactionConfig {
    /// Master kill-switch.
    #[serde(default = "default_compaction_enabled")]
    pub enabled: bool,
    /// Approximate total request size at which compaction should trigger.
    ///
    /// This includes:
    /// - the stable system/developer prompt
    /// - any existing compaction summary
    /// - the retained conversation messages
    /// - tool definitions sent with the next main model call
    #[serde(default = "default_compaction_trigger_tokens")]
    pub trigger_tokens: usize,
    /// Approximate token budget to preserve as the recent suffix.
    #[serde(default = "default_compaction_keep_recent_tokens")]
    pub keep_recent_tokens: usize,
    /// Max tokens reserved for the summarization response itself.
    #[serde(default = "default_compaction_reserve_summary_tokens")]
    pub reserve_summary_tokens: usize,
    /// Refuse to compact extremely short conversations.
    #[serde(default = "default_compaction_min_messages_to_compact")]
    pub min_messages_to_compact: usize,
}

impl Default for CompactionConfig {
    fn default() -> Self {
        Self {
            enabled: default_compaction_enabled(),
            trigger_tokens: default_compaction_trigger_tokens(),
            keep_recent_tokens: default_compaction_keep_recent_tokens(),
            reserve_summary_tokens: default_compaction_reserve_summary_tokens(),
            min_messages_to_compact: default_compaction_min_messages_to_compact(),
        }
    }
}

/// Default value for `CompactionConfig.enabled`.
const fn default_compaction_enabled() -> bool {
    true
}

/// Default value for `CompactionConfig.trigger_tokens`.
const fn default_compaction_trigger_tokens() -> usize {
    180_000
}

/// Default value for `CompactionConfig.keep_recent_tokens`.
const fn default_compaction_keep_recent_tokens() -> usize {
    8_000
}

/// Default value for `CompactionConfig.reserve_summary_tokens`.
const fn default_compaction_reserve_summary_tokens() -> usize {
    4_096
}

/// Default value for `CompactionConfig.min_messages_to_compact`.
const fn default_compaction_min_messages_to_compact() -> usize {
    8
}

/// Persistent in-memory state carried across compaction runs in one SA task.
#[derive(Debug, Clone, Default)]
pub struct CompactionState {
    /// The latest checkpoint summary, if compaction has happened before.
    summary: Option<String>,
}

impl CompactionState {
    /// Return the currently active checkpoint summary.
    pub fn summary(&self) -> Option<&str> {
        self.summary.as_deref()
    }

    /// Replace the stored checkpoint summary.
    fn set_summary(&mut self, summary: String) {
        self.summary = Some(summary);
    }
}

/// Human-readable report returned after one successful compaction.
#[derive(Debug, Clone)]
pub struct CompactionReport {
    /// Estimated total request tokens before compaction.
    pub tokens_before: usize,
    /// Estimated total request tokens after compaction.
    pub tokens_after: usize,
    /// Number of real conversation messages summarized away across all
    /// compaction passes performed in one call.
    pub summarized_messages: usize,
    /// Number of real conversation messages still kept verbatim.
    pub kept_messages: usize,
    /// Whether the cut happened in the middle of a user turn.
    pub split_turn: bool,
}

/// Internal preparation result used before the summarization model call.
#[derive(Debug, Clone)]
struct PreparedCompaction {
    /// Messages whose content should be folded into the main checkpoint summary.
    messages_to_summarize: Vec<ChatMessage>,
    /// Messages retained verbatim after compaction.
    kept_messages: Vec<ChatMessage>,
    /// Prefix of a split turn that must be summarized separately.
    turn_prefix_messages: Vec<ChatMessage>,
    /// Previous checkpoint summary, if any.
    previous_summary: Option<String>,
    /// Whether we had to split one turn.
    split_turn: bool,
}

/// Cut point result returned by the suffix-preservation algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CutPointResult {
    /// First real conversation message that should remain verbatim.
    first_kept_index: usize,
    /// Start of the turn that contains `first_kept_index`, when split.
    turn_start_index: Option<usize>,
    /// Whether the cut is in the middle of a turn.
    split_turn: bool,
}

/// Build the exact messages that should be sent to the provider for the next
/// model request.
///
/// The stable system/developer prompt always stays first. If history has been
/// compacted already, we inject a synthetic user summary right after it, then we
/// append the retained real conversation messages.
pub fn build_request_messages(
    system_message: &ChatMessage,
    state: &CompactionState,
    conversation_messages: &[ChatMessage],
) -> Vec<ChatMessage> {
    let mut request_messages = Vec::with_capacity(
        1 + conversation_messages.len() + usize::from(state.summary().is_some()),
    );
    request_messages.push(system_message.clone());

    if let Some(summary) = state.summary() {
        request_messages.push(build_compaction_summary_message(summary));
    }

    request_messages.extend(conversation_messages.iter().cloned());
    request_messages
}

/// Create the synthetic user message that carries the compaction checkpoint.
pub fn build_compaction_summary_message(summary: &str) -> ChatMessage {
    ChatMessage::text(
        "user",
        format!("{COMPACTION_SUMMARY_PREFIX}{summary}{COMPACTION_SUMMARY_SUFFIX}"),
    )
}

/// Opportunistically compact the conversation history when it becomes large.
///
/// The mutable `conversation_messages` slice must contain only the real
/// conversation after the stable system/developer prompt. This function never
/// mutates that leading prompt and never injects synthetic summary messages into
/// the stored conversation state. Synthetic messages are created only when the
/// caller later builds the actual model request via [`build_request_messages`].
pub async fn maybe_compact_history(
    llm: &OpenAiClient,
    model: &str,
    system_role_name: &str,
    system_message: &ChatMessage,
    reasoning_effort: Option<&str>,
    state: &mut CompactionState,
    conversation_messages: &mut Vec<ChatMessage>,
    tool_definitions: &[ToolDefinition],
    cfg: &CompactionConfig,
    cancel: &CancelToken,
) -> anyhow::Result<Option<CompactionReport>> {
    if !cfg.enabled {
        return Ok(None);
    }

    let mut initial_tokens_before = estimate_request_tokens(
        system_message,
        state,
        conversation_messages,
        tool_definitions,
    );
    if initial_tokens_before < cfg.trigger_tokens {
        return Ok(None);
    }

    let mut total_summarized_messages = 0usize;
    let mut split_turn = false;

    loop {
        let total_tokens_before = estimate_request_tokens(
            system_message,
            state,
            conversation_messages,
            tool_definitions,
        );
        if total_tokens_before < cfg.trigger_tokens {
            return Ok(Some(CompactionReport {
                tokens_before: initial_tokens_before,
                tokens_after: total_tokens_before,
                summarized_messages: total_summarized_messages,
                kept_messages: conversation_messages.len(),
                split_turn,
            }));
        }

        let prepared =
            prepare_compaction(conversation_messages, state, cfg).ok_or_else(|| {
                anyhow::anyhow!(
                    "Estimated request size is {total_tokens_before} tokens, but no more compactable history remains"
                )
            })?;

        let history_summary = if prepared.messages_to_summarize.is_empty() {
            prepared
                .previous_summary
                .clone()
                .unwrap_or_else(|| "No prior history.".to_string())
        } else {
            generate_summary(
                llm,
                model,
                system_role_name,
                reasoning_effort,
                &prepared.messages_to_summarize,
                cfg.reserve_summary_tokens,
                cancel,
                prepared.previous_summary.as_deref(),
                None,
            )
            .await?
        };

        let final_summary = if prepared.split_turn && !prepared.turn_prefix_messages.is_empty() {
            let turn_prefix_summary = generate_turn_prefix_summary(
                llm,
                model,
                system_role_name,
                reasoning_effort,
                &prepared.turn_prefix_messages,
                cfg.reserve_summary_tokens,
                cancel,
            )
            .await?;

            format!(
                "{history_summary}\n\n---\n\n**Turn Context (split turn):**\n\n{turn_prefix_summary}"
            )
        } else {
            history_summary
        };

        let mut kept_messages = prepared.kept_messages;
        clear_stale_usage_snapshots(&mut kept_messages);

        let mut next_state = state.clone();
        next_state.set_summary(final_summary.clone());
        let total_tokens_after = estimate_request_tokens(
            system_message,
            &next_state,
            &kept_messages,
            tool_definitions,
        );
        if total_tokens_after >= total_tokens_before {
            anyhow::bail!(
                "Compaction made no progress: estimated request size stayed at {} -> {} tokens",
                total_tokens_before,
                total_tokens_after
            );
        }

        total_summarized_messages +=
            prepared.messages_to_summarize.len() + prepared.turn_prefix_messages.len();
        split_turn |= prepared.split_turn;
        *conversation_messages = kept_messages;
        state.set_summary(final_summary);
        initial_tokens_before = initial_tokens_before.max(total_tokens_before);
    }
}

/// Decide whether compaction is needed and, if so, which slices should be
/// summarized versus retained.
fn prepare_compaction(
    conversation_messages: &[ChatMessage],
    state: &CompactionState,
    cfg: &CompactionConfig,
) -> Option<PreparedCompaction> {
    if conversation_messages.len() < cfg.min_messages_to_compact {
        return None;
    }

    let cut = find_cut_point(conversation_messages, cfg.keep_recent_tokens);
    if cut.first_kept_index == 0 {
        return None;
    }

    let history_end = if cut.split_turn {
        cut.turn_start_index?
    } else {
        cut.first_kept_index
    };

    let messages_to_summarize = conversation_messages[..history_end].to_vec();
    let turn_prefix_messages = if cut.split_turn {
        conversation_messages[history_end..cut.first_kept_index].to_vec()
    } else {
        Vec::new()
    };
    let kept_messages = conversation_messages[cut.first_kept_index..].to_vec();

    if messages_to_summarize.is_empty() && turn_prefix_messages.is_empty() {
        return None;
    }

    Some(PreparedCompaction {
        messages_to_summarize,
        kept_messages,
        turn_prefix_messages,
        previous_summary: state.summary().map(str::to_string),
        split_turn: cut.split_turn,
    })
}

/// Serialize the simplified SA conversation into plain text so the model treats
/// it as data to summarize instead of an active conversation to continue.
pub fn serialize_conversation(messages: &[ChatMessage]) -> String {
    let mut parts = Vec::<String>::new();

    for message in messages {
        match message.role.as_str() {
            "user" => {
                if let Some(content) = non_empty_text(message.content.as_deref()) {
                    parts.push(format!("[User]: {content}"));
                }
            }
            "assistant" => {
                if let Some(content) = non_empty_text(message.content.as_deref()) {
                    parts.push(format!("[Assistant]: {content}"));
                }

                if let Some(tool_calls) = message.tool_calls.as_ref() {
                    let formatted_calls = tool_calls
                        .iter()
                        .map(format_tool_call)
                        .collect::<Vec<_>>()
                        .join("; ");
                    if !formatted_calls.is_empty() {
                        parts.push(format!("[Assistant tool calls]: {formatted_calls}"));
                    }
                }
            }
            "tool" => {
                if let Some(content) = non_empty_text(message.content.as_deref()) {
                    parts.push(format!("[Tool result]: {content}"));
                }
            }
            role if role == "system" || role == "developer" => {
                if let Some(content) = non_empty_text(message.content.as_deref()) {
                    parts.push(format!("[Context]: {content}"));
                }
            }
            other => {
                if let Some(content) = non_empty_text(message.content.as_deref()) {
                    parts.push(format!("[{other}]: {content}"));
                }
            }
        }
    }

    parts.join("\n\n")
}

/// Estimate the approximate token size of a whole message list.
pub fn estimate_context_tokens(messages: &[ChatMessage]) -> usize {
    messages.iter().map(estimate_tokens).sum()
}

/// Estimate the approximate token size of serialized tool definitions.
pub fn estimate_tool_definitions_tokens(tool_definitions: &[ToolDefinition]) -> usize {
    if tool_definitions.is_empty() {
        return 0;
    }

    match serde_json::to_string(tool_definitions) {
        Ok(serialized) => serialized.len().div_ceil(4),
        Err(_) => 0,
    }
}

/// Estimate the approximate token size of the full model request that SA would
/// send on the next `/v1/chat/completions` call.
///
/// This mirrors OpenClaw's high-level hybrid strategy:
/// - if we have a recent assistant usage snapshot, treat it as the best
///   estimate for the request up to that assistant turn
/// - estimate only the messages after that turn
/// - if there is no usage snapshot, fall back to a full heuristic estimate
///
/// Tool definitions and wrapper overhead are only added in the full-heuristic
/// branch. When a usage snapshot exists, it already came from a real provider
/// request that included those fields.
pub fn estimate_request_tokens(
    system_message: &ChatMessage,
    state: &CompactionState,
    conversation_messages: &[ChatMessage],
    tool_definitions: &[ToolDefinition],
) -> usize {
    let request_messages = build_request_messages(system_message, state, conversation_messages);

    if let Some(estimate) = estimate_request_tokens_from_last_usage(&request_messages) {
        return estimate;
    }

    REQUEST_WRAPPER_TOKENS
        + estimate_context_tokens(&request_messages)
        + estimate_tool_definitions_tokens(tool_definitions)
}

/// Usage snapshot metadata for the newest assistant turn that has one.
#[derive(Debug, Clone, Copy)]
struct AssistantUsageInfo<'a> {
    /// Usage payload attached to the assistant turn.
    usage: &'a ChatUsage,
    /// Message index of the assistant turn.
    index: usize,
}

/// Estimate request size using the last assistant usage snapshot plus trailing
/// heuristic tokens.
fn estimate_request_tokens_from_last_usage(messages: &[ChatMessage]) -> Option<usize> {
    let usage_info = get_last_assistant_usage_info(messages)?;
    let usage_tokens = calculate_request_tokens_from_usage(usage_info.usage)?;
    let trailing_tokens = estimate_context_tokens(&messages[usage_info.index + 1..]);
    Some(usage_tokens.saturating_add(trailing_tokens))
}

/// Walk the message list backwards and find the newest assistant turn with
/// provider-reported usage.
fn get_last_assistant_usage_info(messages: &[ChatMessage]) -> Option<AssistantUsageInfo<'_>> {
    for (index, message) in messages.iter().enumerate().rev() {
        if message.role != "assistant" {
            continue;
        }

        let usage = message.usage.as_ref()?;
        return Some(AssistantUsageInfo { usage, index });
    }

    None
}

/// Convert provider-reported usage into the conservative "whole request"
/// estimate style used by OpenClaw.
///
/// Adaptation note:
/// - OpenAI-style `prompt_tokens_details.cached_tokens` is usually a subset of
///   `prompt_tokens`, so we must not blindly add it again when prompt/input
///   tokens are already present.
fn calculate_request_tokens_from_usage(usage: &ChatUsage) -> Option<usize> {
    if let Some(total) = usage.total_tokens {
        return Some(saturating_u64_to_usize(total));
    }

    let input = usage.input_tokens.unwrap_or(0);
    let output = usage.output_tokens.unwrap_or(0);
    if input > 0 || output > 0 {
        return Some(saturating_u64_to_usize(input.saturating_add(output)));
    }

    let sum = usage
        .cache_read_tokens()
        .saturating_add(usage.cache_write_tokens());

    if sum == 0 {
        return None;
    }

    Some(saturating_u64_to_usize(sum))
}

/// Convert a possibly large `u64` token counter to `usize` without panicking on
/// narrower targets.
fn saturating_u64_to_usize(value: u64) -> usize {
    usize::try_from(value).unwrap_or(usize::MAX)
}

/// Estimate the approximate token size of one chat message.
///
/// This intentionally follows the upstream chars/4 heuristic rather than a
/// tokenizer dependency, which keeps SA lightweight and provider-agnostic.
pub fn estimate_tokens(message: &ChatMessage) -> usize {
    let mut chars = 0usize;

    chars += message.role.len();
    chars += message.content.as_deref().map_or(0, str::len);

    if let Some(tool_call_id) = message.tool_call_id.as_deref() {
        chars += tool_call_id.len();
    }

    if let Some(tool_calls) = message.tool_calls.as_ref() {
        for tool_call in tool_calls {
            chars += tool_call.id.len();
            chars += tool_call.kind.len();
            chars += tool_call.function.name.len();
            chars += tool_call.function.arguments.len();
        }
    }

    chars.div_ceil(4)
}

/// Drop assistant usage snapshots after compaction rewrites the earlier
/// conversation prefix.
///
/// Reusing those old snapshots after history has been summarized would cause
/// stale context sizes to leak into future estimates.
fn clear_stale_usage_snapshots(messages: &mut [ChatMessage]) {
    for message in messages {
        if message.role == "assistant" {
            message.usage = None;
        }
    }
}

/// Find all indices where a cut is legal.
///
/// We never cut at a tool result because tool outputs must stay attached to the
/// assistant message that requested them.
fn find_valid_cut_points(messages: &[ChatMessage]) -> Vec<usize> {
    messages
        .iter()
        .enumerate()
        .filter_map(|(index, message)| match message.role.as_str() {
            "tool" => None,
            _ => Some(index),
        })
        .collect()
}

/// Find the user message that started the turn containing `entry_index`.
fn find_turn_start_index(
    messages: &[ChatMessage],
    entry_index: usize,
    start_index: usize,
) -> Option<usize> {
    for index in (start_index..=entry_index).rev() {
        if messages[index].role == "user" {
            return Some(index);
        }
    }
    None
}

/// Choose the point from which the newest suffix should remain verbatim.
fn find_cut_point(messages: &[ChatMessage], keep_recent_tokens: usize) -> CutPointResult {
    let cut_points = find_valid_cut_points(messages);
    if cut_points.is_empty() {
        return CutPointResult {
            first_kept_index: 0,
            turn_start_index: None,
            split_turn: false,
        };
    }

    let mut accumulated_tokens = 0usize;
    let mut cut_index = 0usize;
    let mut hit_budget = false;

    for index in (0..messages.len()).rev() {
        accumulated_tokens += estimate_tokens(&messages[index]);
        if accumulated_tokens >= keep_recent_tokens {
            cut_index = cut_points
                .iter()
                .copied()
                .find(|candidate| *candidate >= index)
                .unwrap_or(0);
            hit_budget = true;
            break;
        }
    }

    if !hit_budget {
        return CutPointResult {
            first_kept_index: 0,
            turn_start_index: None,
            split_turn: false,
        };
    }

    let split_turn = messages[cut_index].role != "user";
    let turn_start_index = if split_turn {
        find_turn_start_index(messages, cut_index, 0)
    } else {
        None
    };

    CutPointResult {
        first_kept_index: cut_index,
        turn_start_index,
        split_turn: split_turn && turn_start_index.is_some(),
    }
}

/// Build the exact prompt text passed to the summarizer model call.
fn build_summarization_prompt(
    conversation_text: &str,
    previous_summary: Option<&str>,
    custom_focus: Option<&str>,
) -> String {
    let mut prompt_text = String::new();
    let _ = writeln!(prompt_text, "<conversation>");
    let _ = writeln!(prompt_text, "{conversation_text}");
    let _ = writeln!(prompt_text, "</conversation>");
    let _ = writeln!(prompt_text);

    if let Some(previous_summary) = previous_summary {
        let previous_summary = previous_summary.trim();
        if !previous_summary.is_empty() {
            let _ = writeln!(prompt_text, "<previous-summary>");
            let _ = writeln!(prompt_text, "{previous_summary}");
            let _ = writeln!(prompt_text, "</previous-summary>");
            let _ = writeln!(prompt_text);
        }
    }

    prompt_text.push_str(if previous_summary.is_some() {
        UPDATE_SUMMARIZATION_PROMPT
    } else {
        SUMMARIZATION_PROMPT
    });

    if let Some(custom_focus) = custom_focus {
        let custom_focus = custom_focus.trim();
        if !custom_focus.is_empty() {
            let _ = write!(prompt_text, "\n\nAdditional focus: {custom_focus}");
        }
    }

    prompt_text
}

/// Build the prompt text for split-turn prefix summarization.
fn build_turn_prefix_prompt(conversation_text: &str) -> String {
    format!(
        "<conversation>\n{conversation_text}\n</conversation>\n\n{TURN_PREFIX_SUMMARIZATION_PROMPT}"
    )
}

/// Call the provider once to summarize older history.
async fn generate_summary(
    llm: &OpenAiClient,
    model: &str,
    system_role_name: &str,
    reasoning_effort: Option<&str>,
    current_messages: &[ChatMessage],
    reserve_summary_tokens: usize,
    cancel: &CancelToken,
    previous_summary: Option<&str>,
    custom_focus: Option<&str>,
) -> anyhow::Result<String> {
    let conversation_text = serialize_conversation(current_messages);
    let prompt_text =
        build_summarization_prompt(&conversation_text, previous_summary, custom_focus);
    let req = build_summarization_request(
        system_role_name,
        model,
        reasoning_effort,
        prompt_text,
        validated_completion_budget(reserve_summary_tokens, 0.8)?,
    );

    let response = chat_completions_with_retry(llm, &req, cancel, "Summarization").await?;

    let choice = response
        .first_choice()
        .context("Summarization response contained no choices")?;
    let summary = choice
        .message
        .content
        .as_deref()
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .ok_or_else(|| anyhow::anyhow!("Summarization response contained empty content"))?;

    Ok(summary.to_string())
}

/// Call the provider once to summarize the prefix of a split turn.
async fn generate_turn_prefix_summary(
    llm: &OpenAiClient,
    model: &str,
    system_role_name: &str,
    reasoning_effort: Option<&str>,
    current_messages: &[ChatMessage],
    reserve_summary_tokens: usize,
    cancel: &CancelToken,
) -> anyhow::Result<String> {
    let conversation_text = serialize_conversation(current_messages);
    let prompt_text = build_turn_prefix_prompt(&conversation_text);
    let req = build_summarization_request(
        system_role_name,
        model,
        reasoning_effort,
        prompt_text,
        validated_completion_budget(reserve_summary_tokens, 0.5)?,
    );

    let response =
        chat_completions_with_retry(llm, &req, cancel, "Turn-prefix summarization").await?;

    let choice = response
        .first_choice()
        .context("Turn-prefix summarization response contained no choices")?;
    let summary = choice
        .message
        .content
        .as_deref()
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .ok_or_else(|| {
            anyhow::anyhow!("Turn-prefix summarization response contained empty content")
        })?;

    Ok(summary.to_string())
}

/// Build one summary-model request with the dedicated compaction system prompt.
fn build_summarization_request(
    system_role_name: &str,
    model: &str,
    reasoning_effort: Option<&str>,
    prompt_text: String,
    max_tokens: u32,
) -> ChatCompletionsRequest {
    ChatCompletionsRequest {
        model: model.to_string(),
        messages: vec![
            ChatMessage::text(system_role_name, SUMMARIZATION_SYSTEM_PROMPT),
            ChatMessage::text("user", prompt_text),
        ],
        max_tokens: Some(max_tokens),
        reasoning_effort: reasoning_effort.map(str::to_string),
        tools: None,
        tool_choice: None,
        stream: Some(false),
    }
}

/// Convert a reserved summary budget into a provider request field.
fn validated_completion_budget(reserve_summary_tokens: usize, ratio: f64) -> anyhow::Result<u32> {
    let max_tokens = ((reserve_summary_tokens as f64) * ratio).floor() as usize;
    if max_tokens == 0 {
        anyhow::bail!("Invalid compaction config: reserve_summary_tokens must be greater than 0");
    }

    u32::try_from(max_tokens)
        .context("Invalid compaction config: reserve_summary_tokens exceeds u32 range")
}

/// Send one compaction-related chat completion request with retry handling.
async fn chat_completions_with_retry(
    llm: &OpenAiClient,
    req: &ChatCompletionsRequest,
    cancel: &CancelToken,
    label: &str,
) -> anyhow::Result<crate::openai::ChatCompletionsResponse> {
    let mut error_count = 0u32;

    loop {
        let response = tokio::select! {
            _ = cancel.cancelled() => {
                anyhow::bail!("{label} cancelled before completion")
            }
            response = llm.chat_completions(req) => response,
        };

        match response {
            Ok(resp) => return Ok(resp),
            Err(err) if err.is_retriable() => {
                error_count = error_count.saturating_add(1);
                let delay = retry_delay(error_count);
                tokio::select! {
                    _ = cancel.cancelled() => {
                        anyhow::bail!("{label} cancelled during retry backoff")
                    }
                    _ = tokio::time::sleep(delay) => {}
                }
            }
            Err(err) => return Err(map_summarization_error(label, err)),
        }
    }
}

/// Render one assistant tool call into a compact, readable form for summary
/// generation.
fn format_tool_call(tool_call: &ToolCall) -> String {
    match serde_json::from_str::<serde_json::Value>(&tool_call.function.arguments) {
        Ok(serde_json::Value::Object(arguments)) => {
            let mut parts = arguments
                .iter()
                .map(|(key, value)| format!("{key}={value}"))
                .collect::<Vec<_>>();
            parts.sort();
            format!("{}({})", tool_call.function.name, parts.join(", "))
        }
        _ => format!(
            "{}({})",
            tool_call.function.name, tool_call.function.arguments
        ),
    }
}

/// Strip and reject empty text values.
fn non_empty_text(text: Option<&str>) -> Option<&str> {
    let text = text?.trim();
    if text.is_empty() {
        return None;
    }
    Some(text)
}

/// Convert transport-layer summarization failures into stable `anyhow` errors.
fn map_summarization_error(prefix: &str, err: ChatCompletionsError) -> anyhow::Error {
    anyhow::anyhow!("{prefix}: {err}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::openai::{
        ChatUsage, PromptTokenDetails, ToolCall, ToolDefinition, ToolFunctionCall,
        ToolFunctionDefinition,
    };

    /// Helper used by several tests.
    fn assistant_with_tool_call(name: &str, arguments: serde_json::Value) -> ChatMessage {
        ChatMessage {
            role: "assistant".to_string(),
            content: Some("准备调用工具。".to_string()),
            tool_calls: Some(vec![ToolCall {
                id: "call_1".to_string(),
                kind: "function".to_string(),
                function: ToolFunctionCall {
                    name: name.to_string(),
                    arguments: arguments.to_string(),
                },
            }]),
            tool_call_id: None,
            usage: None,
        }
    }

    #[test]
    fn serialize_conversation_includes_tool_calls_and_tool_results() {
        let messages = vec![
            ChatMessage::text("user", "请读取 Cargo.toml"),
            assistant_with_tool_call("Read", serde_json::json!({ "path": "Cargo.toml" })),
            ChatMessage::tool_result("call_1", "[file contents]"),
        ];

        let serialized = serialize_conversation(&messages);
        assert!(serialized.contains("[User]: 请读取 Cargo.toml"));
        assert!(serialized.contains("[Assistant]: 准备调用工具。"));
        assert!(serialized.contains("[Assistant tool calls]: Read(path=\"Cargo.toml\")"));
        assert!(serialized.contains("[Tool result]: [file contents]"));
    }

    #[test]
    fn build_request_messages_injects_summary_after_system_message() {
        let system_message = ChatMessage::text("developer", "system prompt");
        let mut state = CompactionState::default();
        state.set_summary("历史摘要".to_string());

        let request_messages = build_request_messages(
            &system_message,
            &state,
            &[ChatMessage::text("user", "最新消息")],
        );

        assert_eq!(request_messages.len(), 3);
        assert_eq!(request_messages[0].role, "developer");
        assert_eq!(request_messages[1].role, "user");
        assert!(
            request_messages[1]
                .content
                .as_deref()
                .unwrap_or_default()
                .contains("历史摘要")
        );
        assert_eq!(request_messages[2].content.as_deref(), Some("最新消息"));
    }

    #[test]
    fn build_summarization_prompt_uses_update_template_when_previous_summary_exists() {
        let prompt = build_summarization_prompt("[User]: hi", Some("旧摘要"), Some("聚焦测试"));
        assert!(prompt.contains("<conversation>"));
        assert!(prompt.contains("<previous-summary>"));
        assert!(prompt.contains("旧摘要"));
        assert!(prompt.contains("根据新信息更新现有的结构化摘要"));
        assert!(prompt.contains("Additional focus: 聚焦测试"));
    }

    #[test]
    fn build_summarization_request_includes_system_prompt_and_budget() {
        let req = build_summarization_request(
            "developer",
            "gpt-5.2",
            Some("high"),
            "需要摘要的内容".to_string(),
            1024,
        );

        assert_eq!(req.model, "gpt-5.2");
        assert_eq!(req.max_tokens, Some(1024));
        assert_eq!(req.messages.len(), 2);
        assert_eq!(req.messages[0].role, "developer");
        assert_eq!(
            req.messages[0].content.as_deref(),
            Some(SUMMARIZATION_SYSTEM_PROMPT)
        );
        assert_eq!(req.messages[1].role, "user");
        assert_eq!(req.reasoning_effort.as_deref(), Some("high"));
        assert!(req.tools.is_none());
    }

    #[test]
    fn estimate_request_tokens_counts_system_summary_and_tools() {
        let system_message = ChatMessage::text("developer", "system prompt");
        let mut state = CompactionState::default();
        state.set_summary("历史摘要".to_string());
        let tool_definitions = vec![ToolDefinition {
            kind: "function".to_string(),
            function: ToolFunctionDefinition {
                name: "Read".to_string(),
                description: "读取文件".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "path": { "type": "string" }
                    }
                }),
            },
        }];

        let tokens = estimate_request_tokens(
            &system_message,
            &state,
            &[ChatMessage::text("user", "最新消息")],
            &tool_definitions,
        );

        assert!(tokens > estimate_context_tokens(&[ChatMessage::text("user", "最新消息")]));
        assert!(estimate_tool_definitions_tokens(&tool_definitions) > 0);
    }

    #[test]
    fn estimate_request_tokens_prefers_last_assistant_usage_snapshot() {
        let system_message = ChatMessage::text("developer", "system prompt");
        let tool_definitions = vec![ToolDefinition {
            kind: "function".to_string(),
            function: ToolFunctionDefinition {
                name: "VeryLargeTool".to_string(),
                description: "x".repeat(2_000),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "path": { "type": "string" },
                        "content": { "type": "string" }
                    }
                }),
            },
        }];

        let mut assistant = ChatMessage::text("assistant", "前一轮回答");
        assistant.usage = Some(ChatUsage {
            input_tokens: Some(900),
            output_tokens: Some(100),
            total_tokens: Some(1_000),
            prompt_tokens_details: None,
            cache_read_input_tokens: None,
            cache_creation_input_tokens: None,
        });
        let trailing_user = ChatMessage::text("user", "新的追问");

        let tokens = estimate_request_tokens(
            &system_message,
            &CompactionState::default(),
            &[
                ChatMessage::text("user", "初始请求"),
                assistant,
                trailing_user.clone(),
            ],
            &tool_definitions,
        );

        assert_eq!(tokens, 1_000 + estimate_tokens(&trailing_user));
    }

    #[test]
    fn find_cut_point_never_keeps_from_tool_result() {
        let messages = vec![
            ChatMessage::text("user", "step 1"),
            assistant_with_tool_call("Read", serde_json::json!({ "path": "a.txt" })),
            ChatMessage::tool_result("call_1", "tool output"),
            ChatMessage::text("user", "step 2"),
            ChatMessage::text("assistant", "done"),
        ];

        let cut = find_cut_point(&messages, 4);
        assert_ne!(messages[cut.first_kept_index].role, "tool");
    }

    #[test]
    fn prepare_compaction_detects_split_turn() {
        let cfg = CompactionConfig {
            enabled: true,
            trigger_tokens: 1,
            keep_recent_tokens: 3,
            reserve_summary_tokens: 128,
            min_messages_to_compact: 1,
        };
        let state = CompactionState::default();
        let messages = vec![
            ChatMessage::text("user", "这是一个很长的请求，需要很多上下文。"),
            ChatMessage::text("assistant", "先读取文件。"),
            ChatMessage::tool_result("call_1", "工具结果 1"),
            ChatMessage::text("assistant", "继续处理。"),
            ChatMessage::text("assistant", "最后再补充一段近期处理状态。"),
        ];

        let prepared =
            prepare_compaction(&messages, &state, &cfg).expect("compaction should trigger");
        assert!(prepared.split_turn);
        assert!(!prepared.turn_prefix_messages.is_empty());
        assert!(!prepared.kept_messages.is_empty());
    }

    #[test]
    fn estimate_tokens_counts_tool_call_arguments() {
        let message = assistant_with_tool_call(
            "Edit",
            serde_json::json!({ "path": "src/main.rs", "old": "a", "new": "b" }),
        );

        assert!(estimate_tokens(&message) > 0);
    }

    #[test]
    fn validated_completion_budget_rejects_zero_budget() {
        let err =
            validated_completion_budget(0, 0.8).expect_err("zero summary budget must be rejected");
        assert!(
            err.to_string()
                .contains("reserve_summary_tokens must be greater than 0")
        );
    }

    #[test]
    fn calculate_request_tokens_from_usage_prefers_input_and_output_without_double_counting_cache_details()
     {
        let usage = ChatUsage {
            input_tokens: Some(1_200),
            output_tokens: Some(80),
            total_tokens: None,
            prompt_tokens_details: Some(PromptTokenDetails {
                cached_tokens: Some(500),
            }),
            cache_read_input_tokens: None,
            cache_creation_input_tokens: Some(40),
        };

        assert_eq!(calculate_request_tokens_from_usage(&usage), Some(1_280));
    }

    #[test]
    fn clear_stale_usage_snapshots_removes_assistant_usage_only() {
        let mut assistant = ChatMessage::text("assistant", "历史回答");
        assistant.usage = Some(ChatUsage {
            input_tokens: Some(10),
            output_tokens: Some(5),
            total_tokens: Some(15),
            prompt_tokens_details: None,
            cache_read_input_tokens: None,
            cache_creation_input_tokens: None,
        });
        let user = ChatMessage::text("user", "后续问题");
        let mut messages = vec![assistant, user.clone()];

        clear_stale_usage_snapshots(&mut messages);

        assert!(messages[0].usage.is_none());
        assert_eq!(messages[1].content, user.content);
        assert!(messages[1].usage.is_none());
    }
}
