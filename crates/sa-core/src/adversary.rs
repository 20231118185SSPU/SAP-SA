//! Adversarial quality review: a second LLM call that reviews the agent's
//! recent output for quality issues. Inspired by GenericAgent's adversary
//! pattern.

use crate::openai::{ChatCompletionsRequest, ChatMessage, MessageContent, OpenAiClient};
use serde::Deserialize;

/// Result of one adversarial review pass.
#[derive(Debug, Deserialize)]
pub struct AdversaryReview {
    /// List of detected quality issues (empty if none found).
    pub issues: Vec<String>,
    /// Overall quality score (0-100).
    pub score: u8,
}

/// Run an adversarial quality review on the last few turns of conversation.
///
/// Uses a separate LLM call with low temperature to produce consistent,
/// structured evaluations. Designed to be called when the agent finishes or
/// pauses a task.
pub async fn run_adversary_review(
    llm: &OpenAiClient,
    model: &str,
    recent_messages: &[ChatMessage],
) -> anyhow::Result<AdversaryReview> {
    // Build a concise review prompt with the last few messages.
    let mut review_prompt = String::from(
        "你是质量审查员。审查以下 agent 对话的最后几轮，检测质量问题。\n\
         检查项：\n\
         1. 是否存在幻觉（声称做了但实际没做的事）\n\
         2. 是否存在逻辑矛盾\n\
         3. 是否遗漏了关键信息\n\
         4. 是否过度重复相同操作\n\
         5. 输出是否包含 <summary> 标签\n\n\
         用 JSON 回复，格式：\n\
         {\"issues\": [\"问题描述1\", \"问题描述2\"], \"score\": 85}\n\
         score 范围 0-100（100=完美）。如果没有问题，issues 为空数组。\n\n\
         审查对象：\n",
    );

    for (i, msg) in recent_messages.iter().enumerate() {
        let role = &msg.role;
        let content = match &msg.content {
            Some(MessageContent::Text(text)) => text.as_str(),
            _ => "[non-text content]",
        };
        // Truncate long messages to keep the review prompt concise.
        let truncated = if content.len() > 500 {
            format!("{}...（已截断，共{}字符）", &content[..500], content.len())
        } else {
            content.to_string()
        };
        review_prompt.push_str(&format!("#{i} [{role}]: {truncated}\n"));
    }

    let review_message = ChatMessage::text("user", &review_prompt);

    let req = ChatCompletionsRequest {
        model: model.to_string(),
        messages: vec![review_message],
        max_tokens: Some(512),
        reasoning_effort: None,
        tools: None,
        tool_choice: None,
        stream: Some(false),
        temperature: Some(0.2),
        top_p: None,
    };

    let resp = llm.chat_completions(&req).await?;
    let choice = resp.first_choice()?;
    let text = choice.message.text_content().unwrap_or_default();

    // Parse JSON from the response. The model might wrap it in markdown code
    // blocks.
    let json_str = text
        .trim()
        .strip_prefix("```json")
        .and_then(|s| s.strip_suffix("```"))
        .map(str::trim)
        .unwrap_or_else(|| text.trim());

    let review: AdversaryReview =
        serde_json::from_str(json_str).unwrap_or_else(|_| AdversaryReview {
            issues: vec![format!("Failed to parse adversary review response: {text}")],
            score: 0,
        });

    Ok(review)
}
